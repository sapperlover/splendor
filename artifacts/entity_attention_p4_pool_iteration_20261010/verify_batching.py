"""Bounded native CUDA Graph checks for the revised formal batch buckets."""
import json
import time
from pathlib import Path

import numpy as np
import torch

from splendor.alphazero_recording import read_json, sha256
from splendor.benchmark_search import micro_requests
from splendor.mcts_policy import MCTSConfig
from splendor.multiworker_gpu import _OutputSink
from splendor.native_policy import _default_catalog, _extension, _native_catalog
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.search_inference import FrozenSearchInference, InferenceConfig


torch.set_num_threads(1)
torch.manual_seed(919)
config_path = Path('configs/entity_attention_v1/hybrid_p4_pool_20261010.json')
raw = read_json(config_path)
execution = raw['selfplay']['execution']
checkpoint = Path(raw['initialization']['checkpoint']['path'])
model = load_frozen_actor_critic_checkpoint(checkpoint, device='cuda')
options = InferenceConfig(**{key: execution[key] for key in (
    'actor_projection', 'candidate_buckets', 'compile_mode', 'compile_failure_policy',
    'native_transport', 'batch_buckets')}, mode=execution['inference_mode'])
compiled = FrozenSearchInference(model, device='cuda', config=options)
reference = FrozenSearchInference(model, device='cuda', config=InferenceConfig(actor_projection=False))
catalog = _native_catalog(_default_catalog())
requests = [r for r in micro_requests() if 2 <= len(r.legal_actions) <= 16][:2]
assert len(requests) == 2
rows_to_check = (1, 17, 33, 65, 129, 193, 257, 385, 500, 3)
checks, transfers = [], []
started = time.perf_counter()
torch.cuda.reset_peak_memory_stats()


def check_transfer(native, source, width, state_only):
    layout = native.packed()
    transfer = compiled.new_native_transfer(state_only=state_only, capacity=500, width=width,
        float_stride=native.float_stride(), mask_stride=native.mask_stride())
    transfers.append(transfer)  # Keep all graphs resident, as in one collection owner.
    floats = transfer.float_staging.reshape(500, native.float_stride())
    masks = transfer.mask_staging.reshape(500, native.mask_stride())
    expected = (reference.value_only_native(source) if state_only
                else reference.policy_value_native(source))
    for rows in rows_to_check:
        # A padded small request clears its suffix; refill on every later request.
        for name, offset, count, shape in layout['float_fields']:
            floats[:, offset:offset + count] = np.resize(np.asarray(source[name]).reshape(2, count), (500, count))
        for name, offset, count, shape in layout['mask_fields']:
            masks[:, offset:offset + count] = np.resize(np.asarray(source[name]).reshape(2, count), (500, count))
        sink = _OutputSink(rows, layout)
        (transfer.begin_value_only(sink) if state_only else transfer.begin_policy_value(sink)).finish()
        if state_only:
            target = np.resize(expected, rows)
            np.testing.assert_allclose(sink.output, target, atol=1e-5, rtol=1e-5)
            error = float(np.max(np.abs(sink.output - target)))
        else:
            logits, values = expected
            target_logits, target_values = np.resize(logits, (rows, width)), np.resize(values, rows)
            np.testing.assert_allclose(sink.output[:, :width], target_logits, atol=1e-5, rtol=1e-5)
            np.testing.assert_allclose(sink.output[:, width], target_values, atol=1e-5, rtol=1e-5)
            error = float(np.max(np.abs(sink.output[:, width] - target_values)))
        checks.append({'kind': 'value' if state_only else 'policy_value', 'candidate_width': width,
            'rows': rows, 'padded_rows': compiled.compiled.batch_for(rows), 'max_value_error': error})
    print(f'passed width={width} state_only={state_only}', flush=True)


for width in execution['candidate_buckets']:
    native = _extension().NativeBatch(catalog, False, 2, width)
    jobs = [_extension().SearchJob(catalog, request, MCTSConfig(simulations=2, determinizations=1,
        root_noise_fraction=0.0), 919 + i, 101, False, 0.0) for i, request in enumerate(requests)]
    assert sum(native.fill(job) for job in jobs) == 2
    source = native.arrays()
    check_transfer(native, source, width, False)
state_native = _extension().NativeBatch(catalog, True, 1, 1)
check_transfer(state_native, source, 0, True)
torch.cuda.synchronize()
stats = compiled.stats.to_dict()
assert stats['native_graph_captures'] == 5 * len(execution['batch_buckets'])
assert not stats['compile']['compile_failures'] and not stats['compile']['disabled_graphs']
output = {'scope': 'bounded inference numeric/memory validation; no selfplay throughput or training',
    'config_path': str(config_path), 'config_sha256': sha256(config_path),
    'script_sha256': sha256(Path(__file__)), 'checkpoint_sha256': sha256(checkpoint),
    'device': torch.cuda.get_device_name(), 'torch': torch.__version__, 'seed': 919,
    'config': options.to_dict(), 'checks': checks, 'inference': stats,
    'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
    'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'seconds': time.perf_counter() - started}
path = Path('runtime/entity_attention_p4_pool_iteration_20261010/gpu_progressive_batching.json')
path.write_text(json.dumps(output, indent=2) + '\n')
print(json.dumps({k: v for k, v in output.items() if k not in ('checks', 'inference')}, indent=2), flush=True)
