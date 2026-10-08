"""Bounded native-transfer numeric/launch checks for the two execution optimizations."""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from splendor.benchmark_search import fixed_requests
from splendor.mcts_policy import MCTSConfig
from splendor.multiworker_gpu import _OutputSink
from splendor.native_policy import _default_catalog, _extension, _native_catalog
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.search_inference import FrozenSearchInference, InferenceConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=['baseline', 'static', 'direct'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    checkpoint = Path('runs/entity_attention_offline/checkpoints/final.pt')
    model = load_frozen_actor_critic_checkpoint(checkpoint, device='cuda')
    options = dict(mode='compiled', actor_projection=False, candidate_buckets=(16, 32, 48, 64),
                   compile_failure_policy='fail_fast')
    if args.variant == 'static':
        options.update(batch_buckets=(256, 512), compile_mode='reduce-overhead')
    if args.variant == 'direct':
        options.update(batch_buckets=(64, 256, 500), compile_mode='native-cudagraph')
    evaluator = FrozenSearchInference(model, device='cuda', config=InferenceConfig(**options))
    catalog = _native_catalog(_default_catalog())
    requests = [item[1] for item in fixed_requests()[:2]]
    native = _extension().NativeBatch(catalog, False, 2, 32)
    jobs = [_extension().SearchJob(catalog, request, MCTSConfig(simulations=2, determinizations=1,
             root_noise_fraction=0.0), 919 + i, 101, False, 0.0) for i, request in enumerate(requests)]
    assert sum(native.fill(job) for job in jobs) == 2
    source = native.arrays()
    transfer = evaluator.new_native_transfer(state_only=False, capacity=500, width=32,
                 float_stride=native.float_stride(), mask_stride=native.mask_stride())
    layout = native.packed()
    floats = transfer.float_staging.reshape(-1, native.float_stride())
    masks = transfer.mask_staging.reshape(-1, native.mask_stride())
    for name, offset, count, shape in layout['float_fields']:
        floats[:500, offset:offset + count] = np.resize(np.asarray(source[name]).reshape(2, count), (500, count))
    for name, offset, count, shape in layout['mask_fields']:
        masks[:500, offset:offset + count] = np.resize(np.asarray(source[name]).reshape(2, count), (500, count))
    float_source, mask_source = floats[:500].copy(), masks[:500].copy()
    def refill():
        # Collection gathers every used row anew. Static padding clears the
        # unused suffix, so a later larger batch must refill it as well.
        floats[:500] = float_source
        masks[:500] = mask_source
    started = time.perf_counter()
    # Warm the actual native strides and both configured static B shapes.
    for rows in (128, 388, 500):
        refill()
        for _ in range(3):
            transfer.begin_policy_value(_OutputSink(rows, layout)).finish()
    compile_warmup = time.perf_counter() - started
    reference = FrozenSearchInference(model, device='cuda', config=InferenceConfig(actor_projection=False))
    ref_logits, ref_values = reference.policy_value_native(source)
    numbers = []
    for rows in (128, 388, 500, 3, 128):
        refill()
        sink = _OutputSink(rows, layout)
        transfer.begin_policy_value(sink).finish()
        expected_logits = np.resize(ref_logits, (rows, 32))
        expected_values = np.resize(ref_values, rows)
        np.testing.assert_allclose(sink.output[:, :32], expected_logits, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(sink.output[:, 32], expected_values, atol=1e-5, rtol=1e-5)
        start = time.perf_counter()
        for _ in range(40):
            transfer.begin_policy_value(_OutputSink(rows, layout)).finish()
        numbers.append({'rows': rows, 'milliseconds_per_batch': (time.perf_counter()-start)*1000/40,
                        'max_value_error': float(np.max(np.abs(sink.output[:, 32]-expected_values)))})
    state_probe = _extension().NativeBatch(catalog, True, 1, 1)
    state_transfer = evaluator.new_native_transfer(state_only=True, capacity=500, width=0,
                     float_stride=state_probe.float_stride(), mask_stride=state_probe.mask_stride())
    state_layout = state_probe.packed()
    state_floats = state_transfer.float_staging.reshape(-1, state_probe.float_stride())
    state_masks = state_transfer.mask_staging.reshape(-1, state_probe.mask_stride())
    value_checks = []
    for rows in (388, 3, 500):
        for name, offset, count, shape in state_layout['float_fields']:
            state_floats[:rows, offset:offset+count] = np.resize(np.asarray(source[name]).reshape(2, count), (rows, count))
        for name, offset, count, shape in state_layout['mask_fields']:
            state_masks[:rows, offset:offset+count] = np.resize(np.asarray(source[name]).reshape(2, count), (rows, count))
        sink = _OutputSink(rows, state_layout)
        state_transfer.begin_value_only(sink).finish()
        expected_values = np.resize(ref_values, rows)
        np.testing.assert_allclose(sink.output, expected_values, atol=1e-5, rtol=1e-5)
        value_checks.append({'rows':rows, 'max_error':float(np.max(np.abs(sink.output-expected_values)))})
    trace = Path('runtime/entity_attention_execution_speed_20261005') / (args.variant + '-trace.json')
    refill()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA]) as profiler:
        for _ in range(10):
            transfer.begin_policy_value(_OutputSink(388, layout)).finish()
    profiler.export_chrome_trace(str(trace))
    events = json.loads(trace.read_text())['traceEvents']
    launch = Counter(e['name'] for e in events if e.get('cat') == 'cuda_runtime')
    output = {'variant': args.variant, 'checkpoint': str(checkpoint), 'config': evaluator.config.to_dict(),
              'seed': 919, 'numeric_checks': 'passed', 'warmup_seconds': compile_warmup,
              'batches': numbers, 'cuda_runtime_counts_for_10_batches': dict(launch),
              'value_only':value_checks,
              'inference': evaluator.stats.to_dict(), 'trace': str(trace)}
    args.output.write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps({k: v for k,v in output.items() if k != 'inference'}), flush=True)

if __name__ == '__main__':
    main()
