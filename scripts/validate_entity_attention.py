"""The prompt11 finite integration smoke and two representative GPU shapes."""
from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

from splendor.actor_critic import tensorize_batch
from splendor.alphazero_checkpoint import load_alphazero_checkpoint
from splendor.alphazero_recording import read_json, source_identity, write_json
from splendor.alphazero_train import collate_training_samples
from splendor.evaluation import FixedEvaluationConfig, run_fixed_evaluation
from splendor.offline_train import make_index
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.search_inference import CompiledForward, FrozenSearchInference, InferenceConfig
from splendor.selfplay_fast import FastSelfPlayConfig, generate


def smoke(root: Path):
    base = Path('configs/entity_attention_v1')
    offline = root / 'offline'
    command = [sys.executable, '-m', 'splendor.offline_train', '--config', str(base / 'offline_smoke.json'), '--output', str(offline)]
    subprocess.run(command + ['--max-steps-this-session', '4'], check=True)
    first = load_alphazero_checkpoint(offline / 'checkpoints/latest.pt')
    if first.round_train_step != 4:
        raise RuntimeError('smoke first segment must commit four updates')
    subprocess.run(command + ['--resume'], check=True)
    exported = offline / 'checkpoints/final.pt'
    final = load_alphazero_checkpoint(exported)
    if final.round_train_step != 8 or final.total_train_step != 8:
        raise RuntimeError('smoke resume must finish at eight updates')
    selfplay = read_json(base / 'selfplay_smoke.json')
    selfplay['checkpoint'] = str(exported)
    write_json(root / 'selfplay_config.json', selfplay)
    collection = generate(FastSelfPlayConfig.from_dict(selfplay), root / 'selfplay')
    if collection['completed_games'] != 2 or collection['exceptions'] or collection['engineering_truncations']:
        raise RuntimeError('new compiled GPU pool smoke did not complete two games')
    evaluation = read_json(base / 'evaluation_smoke.json')
    evaluation['policy_a']['checkpoint'] = str(exported)
    write_json(root / 'evaluation_config.json', evaluation)
    result = run_fixed_evaluation(FixedEvaluationConfig.from_dict(evaluation), root / 'evaluation.json')
    if not result.summary['passed'] or result.summary['completed_pairs'] != 1:
        raise RuntimeError('mixed architecture smoke evaluation did not complete one pair')
    owner = read_json(root / 'selfplay/status.json')['gpu_owner']
    compiled = owner['inference']['compile']
    if not compiled['compiled_calls'] or compiled['compile_failure_fallbacks']:
        raise RuntimeError('smoke must actually execute compiled inference')
    routes = result.summary['execution_stats']['models']
    if len(routes) != 2 or any(not route['inference']['compile']['compiled_calls'] for route in routes.values()):
        raise RuntimeError('both mixed architecture routes must execute compiled inference')
    return {'offline_updates': 8, 'first_segment_updates': 4, 'resumed_updates': 4,
            'new_selfplay_games': 2, 'mixed_evaluation_pairs': 1,
            'collection_compile': compiled,
            'mixed_model_inference': [route['inference'] for route in routes.values()],
            'note': 'random model smoke verifies integration; its score is not evidence of strength'}


def benchmark(root: Path):
    index = make_index(read_json(root / 'offline/dataset.json'), 64 * 1024**2)
    samples = [index.cache.sample(g, i) for g in index.split_games('train') for i in range(min(g.decision_count, 32))]
    cases = []
    for rows, width in ((64, 32), (500, 64)):
        available = [sample for sample in samples if int(sample.arrays['candidate_offsets'][sample.index + 1]
                                                        - sample.arrays['candidate_offsets'][sample.index]) <= width]
        if not available:
            raise RuntimeError(f'no real source rows fit candidate width {width}')
        selected = [available[i % len(available)] for i in range(rows)]
        batch = tensorize_batch(collate_training_samples(selected).decisions, device='cuda')
        cases.append((rows, width, CompiledForward._pad_width(batch, width)))
    records = []
    for architecture, checkpoint, projection in (
        ('deep_sets_v1', Path('runs/az_loop_seed1/rounds/round-004/checkpoints/final.pt'), True),
        ('entity_attention_v1', root / 'offline/checkpoints/final.pt', False),
    ):
        model = load_frozen_actor_critic_checkpoint(checkpoint, device='cuda')
        evaluator = FrozenSearchInference(model, device='cuda', config=InferenceConfig(
            mode='compiled', actor_projection=projection, candidate_buckets=(16,32,48,64), compile_failure_policy='fail_fast'))
        for rows, width, batch in cases:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            first = time.perf_counter()
            with torch.inference_mode():
                actual = evaluator._forward(batch)
            torch.cuda.synchronize()
            first_seconds = time.perf_counter() - first
            with torch.inference_mode():
                reference = evaluator.model._forward_prepared(batch)
                for left, right in zip(actual, reference):
                    torch.testing.assert_close(left, right, atol=2e-5, rtol=2e-4)
                for _ in range(3):
                    evaluator._forward(batch)
                torch.cuda.synchronize()
                durations = []
                for _ in range(10):
                    tick = time.perf_counter()
                    evaluator._forward(batch)
                    torch.cuda.synchronize()
                    durations.append(time.perf_counter() - tick)
            records.append({'architecture': architecture, 'rows': rows, 'candidate_width': width,
                'actor_projection': projection, 'first_call_seconds': first_seconds,
                'warm_calls': len(durations), 'warm_mean_ms': sum(durations) / len(durations) * 1000,
                'allocated_bytes': torch.cuda.memory_allocated(), 'reserved_bytes': torch.cuda.memory_reserved(),
                'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                'compile': evaluator.compiled.stats.to_dict()})
            print(json.dumps({k: v for k, v in records[-1].items() if k != 'compile'}), flush=True)
        del actual, reference, evaluator, model
        gc.collect()
    for record in records:
        if record['architecture'] == 'entity_attention_v1':
            old = next(r for r in records if r['architecture'] == 'deep_sets_v1' and r['rows'] == record['rows'])
            record['warm_latency_relative_to_old'] = record['warm_mean_ms'] / old['warm_mean_ms']
    return {'device': torch.cuda.get_device_name(), 'torch': torch.__version__, 'records': records,
            'scope': 'real archived states/legally ordered candidates, repeated to fill B and padded to C; FP32 compiled',
            'memory_scope': 'same process, includes resident input cases; allocator reserved persists between models',
            'first_call_scope': 'actual first call, not pure compile time; compilation cache was not cleared',
            'interpretation': 'forward ratio does not measure end-to-end collection ratio'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('runtime/entity_attention_v1_smoke'))
    parser.add_argument('--artifacts', type=Path, default=Path('artifacts/entity_attention_v1_20261004'))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('this validation requires a real CUDA device')
    torch.set_num_threads(1)
    integration = smoke(args.output)
    write_json(args.artifacts / 'integration.json', integration)
    shapes = benchmark(args.output)
    write_json(args.artifacts / 'gpu_shapes.json', shapes)
    write_json(args.artifacts / 'source_identity.json', source_identity(Path.cwd()))


if __name__ == '__main__':
    main()
