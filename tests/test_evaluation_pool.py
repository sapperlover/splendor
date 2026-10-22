from dataclasses import replace
import hashlib
from pathlib import Path

import pytest
import torch

from splendor.evaluation import FixedEvaluationConfig, SerializablePolicySpec, run_fixed_evaluation
from test_speed_optimization import _weights, _entry


def _config(tmp_path, *, pipeline_slots=2, chance=False):
    a, b = _weights(tmp_path / 'a.pt', 31), _weights(tmp_path / 'b.pt', 32)
    search = {'simulations': 3, 'determinizations': 1, 'root_noise_fraction': 0., 'seed': 123,
              'fold_forced_phases': True, 'max_simulation_transitions': 128}
    def spec(name, path):
        return SerializablePolicySpec.mcts_checkpoint(name, path, search_config=search,
            search_backend='native_chance' if chance else 'native',
            chance_widening={'k': .25, 'alpha': .5} if chance else None)
    return FixedEvaluationConfig(spec('A', a), spec('B', b), (91, 92),
        policy_seed_base=52, bootstrap_resamples=50, bootstrap_seed=9,
        execution={'backend': 'native_cuda_pool', 'search_workers': 2, 'total_active_games': 3,
            'batch_interface': 'native', 'max_batch_size': 4, 'local_microbatch': 2,
            'shared_slots': 2, 'shared_candidate_width': 128, 'max_batch_wait_ms': 1,
            'pipeline_slots': pipeline_slots, 'inference_mode': 'eager'})


@pytest.mark.parametrize('pipeline_slots,chance', [(1, False), (2, False), (2, True)])
def test_pool_dual_models_match_checked_serial_and_resume(tmp_path, monkeypatch, pipeline_slots, chance):
    torch.set_num_threads(1)
    config = _config(tmp_path, pipeline_slots=pipeline_slots, chance=chance)
    callbacks = []
    pooled = run_fixed_evaluation(config, tmp_path / 'pool.json', pair_callback=callbacks.append)
    reference = run_fixed_evaluation(replace(config, execution=None), tmp_path / 'serial.json')
    assert pooled.summary['passed'] and reference.summary['passed']
    assert len(callbacks) == 2
    stats = pooled.summary['execution_stats']
    assert stats['owner_pipeline_slots'] == pipeline_slots
    assert stats['actual_search_workers'] == 2
    assert stats['actual_active_limit'] == 3
    assert all(not worker['cuda_initialized'] for worker in stats['workers'])
    expected = {hashlib.sha256(Path(spec.checkpoint).read_bytes()).hexdigest()
                for spec in (config.policy_a, config.policy_b)}
    assert set(stats['models']) == expected
    assert all(model['inference']['logical_model_positions'] > 0 for model in stats['models'].values())
    assert stats['inference']['logical_model_positions'] == sum(
        model['inference']['logical_model_positions'] for model in stats['models'].values())
    assert sum(game['simulations'] for game in stats['games']) > 0
    for pair_id in (0, 1):
        actual = sorted(pooled.pair_records[pair_id], key=lambda record: record['game_in_pair'])
        expected = sorted(reference.pair_records[pair_id], key=lambda record: record['game_in_pair'])
        for a, b in zip(actual, expected):
            # Timings vary; game/search outputs must be identical.
            for key in ('policy_seeds', 'first_player', 'game_seed', 'final_state_digest',
                        'status', 'decision_count', 'outcome_for_a', 'scores'):
                assert a[key] == b[key], key
    before = {p.name: p.read_bytes() for p in pooled.game_log_root.glob('pair-*.jsonl')}
    from splendor import multiworker_gpu
    monkeypatch.setattr(multiworker_gpu, 'collect', lambda *a, **kw: pytest.fail('replayed finished games'))
    resumed = run_fixed_evaluation(config, pooled.output_path, resume=True)
    assert resumed.summary['execution_stats'] == stats
    assert before == {p.name: p.read_bytes() for p in pooled.game_log_root.glob('pair-*.jsonl')}


def test_model_batches_do_not_coalesce_across_models():
    from collections import deque
    from splendor.multiworker_gpu import _batch_trigger
    a, b = _entry(0), _entry(1)
    a['event'] = ('batch', (*a['event'][1][:-1], 'model-a'))
    b['event'] = ('batch', (*b['event'][1][:-1], 'model-b'))
    assert _batch_trigger(deque([a, b]), {0, 1}, {}, 4, 64, 0., 1.) is None


def test_pool_configuration_rejects_ignored_inference_and_non_native(tmp_path):
    config = _config(tmp_path)
    with pytest.raises(ValueError, match='conflicts'):
        replace(config, policy_a=replace(config.policy_a, inference={'actor_projection': True}))
    with pytest.raises(ValueError, match='native MCTS or greedy'):
        replace(config, policy_a=SerializablePolicySpec.baseline('random'))


@pytest.mark.parametrize('greedy_side', ['a', 'b', 'both'])
def test_pool_greedy_matches_serial_argmax_and_has_zero_search(tmp_path, greedy_side):
    torch.set_num_threads(1)
    config = _config(tmp_path, pipeline_slots=1)
    search = {**config.policy_a.search_config, 'simulations': 8, 'determinizations': 4}
    a = replace(config.policy_a, search_config=search)
    b = replace(config.policy_b, search_config=search)
    if greedy_side in ('a', 'both'):
        a = SerializablePolicySpec.greedy_checkpoint('A', a.checkpoint)
    if greedy_side in ('b', 'both'):
        b = SerializablePolicySpec.greedy_checkpoint('B', b.checkpoint)
    config = replace(config, policy_a=a, policy_b=b)
    pooled = run_fixed_evaluation(config, tmp_path/'pool.json')
    reference = run_fixed_evaluation(replace(config, execution=None), tmp_path/'serial.json')
    assert pooled.summary['passed'] and reference.summary['passed']
    for pair_id in (0, 1):
        actual = sorted(pooled.pair_records[pair_id], key=lambda r:r['game_in_pair'])
        expected = sorted(reference.pair_records[pair_id], key=lambda r:r['game_in_pair'])
        for observed, original in zip(actual, expected, strict=True):
            for field in ('policy_seeds','first_player','game_seed','final_state_digest',
                          'status','decision_count','outcome_for_a','scores'):
                assert observed[field] == original[field], field
            for side, spec in (('a', a), ('b', b)):
                stats = observed[f'policy_{side}_stats']
                if spec.kind == 'greedy_checkpoint':
                    assert stats['simulations'] == stats['determinizations'] == stats['state_transitions'] == 0
                    assert stats['searchable_decisions'] == 0
                else:
                    assert stats['simulations'] == 8*stats['searchable_decisions']
                    assert stats['determinizations'] == 4*stats['searchable_decisions']


def test_pool_checked_watchdog_is_not_scored(tmp_path):
    config = replace(_config(tmp_path), game_seeds=(91,), max_completed_turns=1)
    result = run_fixed_evaluation(config, tmp_path / 'truncated.json')
    assert not result.summary['passed']
    assert result.summary['completed_games'] == 2
    assert all(record['outcome_for_a'] is None for record in result.pair_records[0])


def test_pool_interrupt_keeps_finished_games_and_resumes(tmp_path):
    config = _config(tmp_path)
    output = tmp_path / 'interrupted.json'
    def stop(_result):
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        run_fixed_evaluation(config, output, pair_callback=stop)
    import json
    partial = json.loads(output.read_text())
    assert partial['completed_pairs'] >= 1
    result = run_fixed_evaluation(config, output, resume=True, require_scored_pairs=True)
    assert result.summary['passed'] and result.summary['completed_pairs'] == 2
    assert all(len(records) == 2 for records in result.pair_records.values())
