"""Bounded engineering check, never a strength panel or training launcher."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

from splendor.alphazero_recording import source_identity, write_json
from splendor.benchmark_search import fixed_requests
from splendor.evaluation import FixedEvaluationConfig, run_fixed_evaluation
from splendor.evaluation_pool import pool_inference_config
from splendor.mcts_policy import MCTSConfig
from splendor.native_policy import _default_catalog, _extension, _native_catalog
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.search_inference import FrozenSearchInference, InferenceConfig
from splendor.selfplay_fast import ExecutionConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source_code = source_identity(Path.cwd(), args.output / 'source')
    shutil.copyfile(__file__, args.output / 'smoke.py')
    torch.set_num_threads(1)
    path = Path('configs/entity_attention_v1/random_budget_37423_p6k_v24k_20261019.json')
    source = json.loads(path.read_text())
    raw = json.loads(json.dumps(source['search_evaluation']))
    raw['policy_a']['checkpoint'] = (
        'runs/entity_attention_37423_random_budget_p6k_v24k_20261019/rounds/round-016/checkpoints/final.pt')
    raw['pairs'] = 2
    for name in ('policy_a', 'policy_b'):
        raw[name]['search_config']['simulations'] = 3
    raw['execution'].update(search_workers=2, total_active_games=4)
    write_json(args.output / 'config.json', raw)
    result = run_fixed_evaluation(FixedEvaluationConfig.from_dict(raw), args.output / 'arena.json')
    assert result.summary['passed'] and result.summary['completed_games'] == 4
    arena_stats = result.summary['execution_stats']
    assert len(arena_stats['models']) == 2
    assert arena_stats['owner_pipeline_slots'] == raw['execution']['pipeline_slots']
    assert all(not worker['cuda_initialized'] for worker in arena_stats['workers'])
    assert all(model['inference']['native_graph_captures'] > 0
               for model in arena_stats['models'].values())
    gc.collect()
    torch.cuda.empty_cache()

    # Probe the actual B500 limit for both network architectures and both
    # transfer lanes. Only one repeated root and width64, not 1000 full games.
    execution = ExecutionConfig(**source['selfplay']['execution'])
    models = [load_frozen_actor_critic_checkpoint(Path(raw[name]['checkpoint']), device='cuda')
              for name in ('policy_a', 'policy_b')]
    evaluators = [FrozenSearchInference(model, device='cuda', config=pool_inference_config(execution))
                  for model in models]
    native = _extension()
    catalog = _native_catalog(_default_catalog())
    request = fixed_requests()[0][1]
    width, rows = 64, 500
    transfers, comparisons = [], []
    torch.cuda.reset_peak_memory_stats()
    for evaluator in evaluators:
        eager = FrozenSearchInference(evaluator.model, device='cuda', config=InferenceConfig(mode='eager'))
        expected_logits, expected_value = eager.policy_value((request,))[0]
        for lane in (0, 1):
            probe = native.NativeBatch(catalog, False, 1, width)
            transfer = evaluator.new_native_transfer(state_only=False, capacity=rows, width=width,
                float_stride=probe.float_stride(), mask_stride=probe.mask_stride())
            transfers.append(transfer)
            batch = native.NativeBatch(catalog, False, rows, width,
                transfer.float_staging, transfer.mask_staging)
            jobs = [native.SearchJob(catalog, request, MCTSConfig(simulations=1, determinizations=1),
                                    index + 123, 101, False, 0.) for index in range(rows)]
            assert sum(batch.fill(job) for job in jobs) == rows
            transfer.begin_policy_value(batch).finish()
            actual = transfer.out_numpy[:rows]
            logits = actual[:, :len(expected_logits)]
            values = actual[:, width]
            np.testing.assert_allclose(logits, np.broadcast_to(expected_logits, logits.shape), rtol=2e-4, atol=2e-4)
            np.testing.assert_allclose(values, expected_value, rtol=2e-4, atol=2e-4)
            comparisons.append({'lane': lane, 'rows': rows, 'width': width,
                'max_logit_abs_error': float(np.max(np.abs(logits - expected_logits))),
                'max_value_abs_error': float(np.max(np.abs(values - expected_value)))})
    report = {'scope': 'engineering only; 2 pairs S3 and B500 repeated-root forward checks; no strength claim',
        'passed': True, 'arena_completed_games': 4,
        'arena_gpu_peak_bytes': arena_stats['gpu_peak_bytes'],
        'b500_two_models_two_lanes_gpu_peak_bytes': torch.cuda.max_memory_allocated(),
        'b500_comparisons': comparisons,
        'gpu': torch.cuda.get_device_name(),
        'command': sys.argv, 'source_identity': source_code,
        'source_config_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'models': result.summary['checkpoint_identity'],
        'arena_inference': arena_stats['inference'],
        'batch_probe_inference': [evaluator.stats.to_dict() for evaluator in evaluators]}
    write_json(args.output / 'verification.json', report)
    print(json.dumps({'passed': True, 'verification': str(args.output / 'verification.json'),
        'arena_completed_games': 4, 'b500_peak_mib': report['b500_two_models_two_lanes_gpu_peak_bytes'] / 1024**2}))


if __name__ == '__main__':
    main()
