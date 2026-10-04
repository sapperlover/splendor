from collections import deque
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch

from splendor.actor_critic import ActorCritic
from splendor.evaluation import FixedEvaluationConfig, SerializablePolicySpec, run_fixed_evaluation
from splendor.features import FEATURE_SCHEMA_VERSION
from splendor.multiworker_gpu import _batch_trigger, _gather
from splendor.ppo import TRAINING_CHECKPOINT_FORMAT_VERSION
from splendor.replay import RULESET_ID


def _entry(wid, slot=0, generation=1, rows=2, kind='policy', width=32, offset=0):
    return {'event': ('batch', (wid, slot, generation, rows, kind, width, 0)), 'offset': offset}


def test_batch_triggers_track_supply_not_starting_worker_count():
    pending = deque([_entry(0)])
    assert _batch_trigger(pending, {0}, {0: ((0, 1),)}, 500, 64, 1., 2.) == 'all_supply_arrived'
    assert _batch_trigger(pending, {0, 1}, {0: ((0, 1),)}, 500, 64, 1., 2.) is None
    assert _batch_trigger(pending, {0, 1}, {0: ((0, 1),)}, 500, 64, 2., 2.) == 'deadline'
    assert _batch_trigger(pending, set(), {}, 500, 64, 1., 2.) == 'producers_exhausted'
    assert _batch_trigger(pending, {0}, {}, 2, 64, 1., 2.) == 'full'


def test_batch_supply_multiple_slots_split_and_mixed_kinds():
    pending = deque([_entry(0, offset=1), _entry(0, slot=1, kind='value')])
    blocked = {0: ((0, 1), (1, 1))}
    assert _batch_trigger(pending, {0}, blocked, 500, 64, 1., 2.) == 'all_supply_arrived'
    pending.pop()
    assert _batch_trigger(pending, {0}, blocked, 500, 64, 1., 2.) is None
    pending.append(_entry(0, slot=1, generation=2))
    assert _batch_trigger(pending, {0}, blocked, 500, 64, 1., 2.) is None
    # A wide descriptor does not fill a narrow batch.
    pending = deque([_entry(0), _entry(1, rows=500, width=128)])
    assert _batch_trigger(pending, {0, 1}, {}, 500, 64, 1., 2.) is None


@pytest.mark.parametrize('src_width,dst_width,kind', [(3, 3, 'policy'), (2, 3, 'policy'), (0, 0, 'value')])
def test_gather_nonzero_offsets_and_tail(src_width, dst_width, kind):
    from types import SimpleNamespace
    base_float, base_mask, action_dim = 7, 4, 2
    sf, sm = base_float + src_width * action_dim, base_mask + src_width
    df, dm = base_float + dst_width * action_dim, base_mask + dst_width
    source = SimpleNamespace(floats=np.arange(4 * sf, dtype=np.float32),
                             masks=np.arange(4 * sm, dtype=np.uint8) % 2)
    dest = SimpleNamespace(float_staging=np.full(5 * df, -99., dtype=np.float32),
                           mask_staging=np.ones(5 * dm, dtype=np.bool_))
    _gather(source, src_width, dest, kind, 1, 2, 2, sf, sm, df, dm, base_float, base_mask, action_dim)
    actual_f, actual_m = dest.float_staging.reshape(5, df), dest.mask_staging.reshape(5, dm)
    np.testing.assert_array_equal(actual_f[2:4, :sf], source.floats.reshape(4, sf)[1:3])
    np.testing.assert_array_equal(actual_m[2:4, :sm], source.masks.reshape(4, sm)[1:3])
    assert not actual_f[2:4, sf:].any() and not actual_m[2:4, sm:].any()
    assert (actual_f[[0, 1, 4]] == -99).all()


def _weights(path: Path, seed: int):
    torch.manual_seed(seed)
    model = ActorCritic()
    torch.save({'format_version': TRAINING_CHECKPOINT_FORMAT_VERSION,
                'feature_schema_version': FEATURE_SCHEMA_VERSION,
                'metadata': {'rules_version': RULESET_ID}, 'model_config': asdict(model.config),
                'model_state_dict': model.state_dict()}, path)
    return path


def test_batched_evaluation_two_models_reference_and_resume(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    a, b = _weights(tmp_path / 'a.pt', 1), _weights(tmp_path / 'b.pt', 2)
    search = {'simulations': 3, 'determinizations': 1, 'root_noise_fraction': 0., 'seed': 123}
    config = FixedEvaluationConfig(
        SerializablePolicySpec.mcts_checkpoint('A', a, search_config=search, search_backend='native'),
        SerializablePolicySpec.mcts_checkpoint('B', b, search_config=search, search_backend='native'),
        (91,), policy_seed_base=52, max_completed_turns=400,
        execution={'backend': 'native_batched', 'active_games': 2, 'max_batch_size': 2,
                   'inference_mode': 'eager', 'actor_projection': False})
    from splendor.native_policy import NativeBatchRunner
    original_run = NativeBatchRunner.run
    routed = set()
    def checked_run(self, jobs, **kwargs):
        assert all(job.policy.evaluator is self.evaluator for job in jobs)
        routed.update(job.policy.model_version for job in jobs)
        return original_run(self, jobs, **kwargs)
    monkeypatch.setattr(NativeBatchRunner, 'run', checked_run)
    pooled = run_fixed_evaluation(config, tmp_path / 'batch.json')
    assert len(routed) == 2
    reference = run_fixed_evaluation(replace(config, execution=None), tmp_path / 'reference.json')
    assert pooled.summary['passed'] and reference.summary['passed']
    assert len(pooled.summary['execution_stats']['models']) == 2
    for actual, expected in zip(pooled.pair_records[0], reference.pair_records[0]):
        for key in ('policy_seeds', 'first_player', 'game_seed', 'final_state_digest', 'status', 'decision_count'):
            assert actual[key] == expected[key], key
    files = list(pooled.game_log_root.glob('pair-*.jsonl'))
    before = [p.read_bytes() for p in files]
    resumed = run_fixed_evaluation(config, pooled.output_path, resume=True)
    assert resumed.summary['completed_games'] == 2
    assert before == [p.read_bytes() for p in files]
    assert resumed.summary['execution_stats'] == pooled.summary['execution_stats']
    with pytest.raises(ValueError, match='manifest|identity|configuration'):
        run_fixed_evaluation(replace(config, execution={**config.execution, 'max_batch_size': 1}),
                             pooled.output_path, resume=True)


def test_batched_failure_and_partial_pair_recovery(tmp_path, monkeypatch):
    from splendor.native_policy import NativeBatchRunner
    a = _weights(tmp_path / 'a.pt', 3)
    search = {'simulations': 2, 'determinizations': 1, 'root_noise_fraction': 0.}
    spec = SerializablePolicySpec.mcts_checkpoint('A', a, search_config=search, search_backend='native')
    config = FixedEvaluationConfig(spec, replace(spec, name='B'), (52,),
        execution={'active_games': 2, 'max_batch_size': 2, 'inference_mode': 'eager'})
    original = NativeBatchRunner.run
    def fail(*args, **kwargs):
        raise RuntimeError('controlled batch failure')
    monkeypatch.setattr(NativeBatchRunner, 'run', fail)
    result = run_fixed_evaluation(config, tmp_path / 'failure.json')
    assert result.summary['exceptions'] == 2 and not result.summary['passed']
    assert list((tmp_path / 'failures').rglob('batch-*.json'))
    monkeypatch.setattr(NativeBatchRunner, 'run', original)
    # Simulate interruption after just one game; normal resume must replay
    # both original seats and leave exactly two authoritative lines.
    file = next(result.game_log_root.glob('pair-*.jsonl'))
    file.write_text(file.read_text().splitlines()[0] + '\n')
    recovered = run_fixed_evaluation(config, result.output_path, resume=True)
    assert recovered.summary['passed'] and len(file.read_text().splitlines()) == 2


def test_batched_execution_rejects_unsupported_policy_and_participates_in_identity():
    base = SerializablePolicySpec.baseline('score')
    with pytest.raises(ValueError, match='native mcts_checkpoint'):
        FixedEvaluationConfig(base, base, (1,), execution={})
    from splendor.evaluation_batch import BatchEvaluationConfig
    with pytest.raises(ValueError, match='wait=0'):
        BatchEvaluationConfig(max_batch_wait_ms=5)


def test_batched_cycle_adjudication_and_turn_watchdog(tmp_path, monkeypatch):
    from splendor import runner
    checkpoint = _weights(tmp_path / 'weights.pt', 5)
    spec = SerializablePolicySpec.mcts_checkpoint('A', checkpoint, search_backend='native',
        search_config={'simulations': 1, 'determinizations': 1, 'root_noise_fraction': 0.})
    config = FixedEvaluationConfig(spec, replace(spec, name='B'), (33,), execution={
        'active_games': 2, 'max_batch_size': 2, 'inference_mode': 'eager'})
    # The actual repeated-state fixture is covered by test_evaluation_3a;
    # force signatures here to exercise both seats through the batch scheduler.
    with monkeypatch.context() as context:
        context.setattr(runner, 'canonical_state_key', lambda *a: ('repeat',))
        cycle = run_fixed_evaluation(config, tmp_path / 'cycle.json')
    assert cycle.summary['passed'] and cycle.summary['policy_cycle_draws'] == 2
    truncated = run_fixed_evaluation(replace(config, max_completed_turns=1,
        adjudicate_deterministic_policy_cycles=False), tmp_path / 'truncated.json')
    assert truncated.summary['engineering_truncations'] == 2
    assert not truncated.summary['passed'] and truncated.summary['exceptions'] == 0
