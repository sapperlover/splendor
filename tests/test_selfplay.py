from __future__ import annotations

import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch

from splendor.actor_critic import ActorCritic
from splendor.contracts import DecisionRequest
from splendor.engine import GameEngine
from splendor.features import FEATURE_SCHEMA_VERSION, encode_decision
from splendor.mcts_policy import MCTSConfig, MCTSSearchResult
from splendor.ppo import TRAINING_CHECKPOINT_FORMAT_VERSION
from splendor.replay import RULESET_ID
from splendor.result import TerminalResult
from splendor.selfplay import (
    PROTOCOL, SelfPlayConfig, TemperatureSchedule, _Recorder, derive_seed,
    generate, label_terminal_values, search_seeds, split_for, validate_run,
)
from splendor.selfplay_data import (
    STATE_FIELDS, SelfPlayDataset, collate_selfplay_samples,
    game_path, load_game, validate_game_arrays,
)


def config(checkpoint: Path, *, workers=1, games=2, simulations=3, determinizations=2):
    return SelfPlayConfig(
        protocol=PROTOCOL, run_id='test-round', round_index=0,
        checkpoint=str(checkpoint), num_games=games, seed=12001,
        search_config=MCTSConfig(simulations=simulations, determinizations=determinizations),
        workers=workers, split_seed=12002,
        temperature_schedule=TemperatureSchedule(opening_completed_turns=10),
    )


def checkpoint(tmp_path: Path) -> Path:
    model = ActorCritic().eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    path = tmp_path / 'zero.pt'
    torch.save({
        'format_version': TRAINING_CHECKPOINT_FORMAT_VERSION,
        'feature_schema_version': FEATURE_SCHEMA_VERSION,
        'metadata': {'rules_version': RULESET_ID},
        'model_config': asdict(model.config),
        'model_state_dict': model.state_dict(),
    }, path)
    return path


@pytest.mark.parametrize('field,value', [
    ('num_games', True), ('num_games', 0), ('seed', 1.2),
    ('workers', 0), ('validation_fraction', math.nan),
    ('validation_fraction', 1), ('add_root_noise', 1),
    ('device', 'cuda'),
])
def test_config_rejects_bad_values(tmp_path, field, value):
    path = checkpoint(tmp_path)
    with pytest.raises(ValueError):
        replace(config(path), **{field: value})
    with pytest.raises(ValueError):
        SelfPlayConfig.from_dict(asdict(config(path)) | {'unknown': 1})


def test_seed_domains_split_and_temperature(tmp_path):
    path = checkpoint(tmp_path)
    cfg = config(path)
    assert derive_seed(cfg.seed, 0, 0, 'game') != derive_seed(cfg.seed, 0, 0, 'search_world', 0)
    assert search_seeds(cfg, 0)[0] != search_seeds(cfg, 0)[1]
    assert split_for(cfg, 0) == split_for(replace(cfg, workers=4), 0)
    request = GameEngine.new(17).decision()
    assert isinstance(request, DecisionRequest)
    schedule = cfg.temperature_schedule
    nine = replace(request, observation=replace(request.observation, own_completed_turns=5, opponent_completed_turns=4))
    ten = replace(request, observation=replace(request.observation, own_completed_turns=5, opponent_completed_turns=5))
    assert schedule.for_request(nine) == 1.0
    assert schedule.for_request(ten) == 0.0


def test_recorder_uses_one_exploratory_search_and_preserves_soft_target(tmp_path):
    request = GameEngine.new(23).decision()
    assert isinstance(request, DecisionRequest)
    cfg = config(checkpoint(tmp_path))
    calls = []
    class FakePolicy:
        def search(self, request, **kwargs):
            calls.append(kwargs)
            n = len(request.legal_actions)
            target = (0.25, 0.75) + (0.0,) * (n-2)
            return MCTSSearchResult(
                1, True, tuple([1/n]*n), tuple([1/n]*n),
                (2, 1), ((2, 0)+(0,)*(n-2), (0, 1)+(0,)*(n-2)),
                (2, 1)+(0,)*(n-2), target, (0.0, 1.0)+(0.0,)*(n-2),
                3, 1, 0,
            )
    records = []
    recorder = _Recorder(0, FakePolicy(), cfg, 7, records)
    assert recorder.choose(request) is request.legal_actions[1]
    assert len(calls) == len(records) == 1
    assert calls[0]['add_root_noise'] is True
    assert calls[0]['temperature'] == 1.0
    assert records[0]['policy_target'][:2] == (0.25, 0.75)
    assert records[0]['behavior_policy'][:2] == (0.0, 1.0)
    assert records[0]['chosen_candidate_index'] == 1
    recorder.choose(request)
    assert records[1]['decision_index'] == 1
    assert records[0]['sampling_seed'] != records[1]['sampling_seed']


@pytest.mark.parametrize('winners,expected', [((0,), (1, -1, 1)), ((1,), (-1, 1, -1)), ((0, 1), (0, 0, 0))])
def test_formal_terminal_labels_follow_real_seat_not_decision_parity(winners, expected):
    records = [{'player_id': p} for p in (0, 1, 0)]
    terminal = TerminalResult((15, 15), (5, 5), winners, (20, 20))
    label_terminal_values(records, terminal)
    assert tuple(r['value_target'] for r in records) == expected


def _stable_archive(root: Path, game_id: int):
    arrays, metadata = load_game(game_path(root, game_id))
    metadata.pop('collection_seconds')
    for stats in metadata['mcts_stats']:
        for name in ('feature_seconds', 'inference_seconds', 'total_seconds'):
            stats.pop(name)
    return arrays, metadata


def test_serial_parallel_archive_resume_and_model_batch(tmp_path):
    path = checkpoint(tmp_path)
    single = config(path, workers=1)
    multi = replace(single, workers=2)
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        serial_root = tmp_path / 'serial'
        parallel_root = tmp_path / 'parallel'
        assert generate(single, serial_root)['completed_games'] == 2
        assert generate(multi, parallel_root)['completed_games'] == 2
    finally:
        torch.set_num_threads(previous_threads)
    for game_id in (0, 1):
        left, lm = _stable_archive(serial_root, game_id)
        right, rm = _stable_archive(parallel_root, game_id)
        assert lm == rm
        assert left.keys() == right.keys()
        for name in left:
            np.testing.assert_array_equal(left[name], right[name])
    result = validate_run(serial_root)
    assert result['completed_games'] == 2 and result['missing_games'] == 0
    assert result['value_samples'] == result['samples']
    assert result['policy_samples'] > 0
    assert result['first_player_counts'] == [1, 1]
    original = game_path(serial_root, 0).read_bytes()
    (serial_root / 'summary.json').unlink()
    (serial_root / 'games' / '.game-000001.npz.interrupted.tmp').write_bytes(b'partial')
    assert generate(multi, serial_root, resume=True)['completed_games'] == 2
    assert game_path(serial_root, 0).read_bytes() == original
    original_second, original_second_meta = _stable_archive(serial_root, 1)
    game_path(serial_root, 1).unlink()
    assert generate(multi, serial_root, resume=True)['completed_games'] == 2
    restored_second, restored_second_meta = _stable_archive(serial_root, 1)
    assert original_second_meta == restored_second_meta
    for name in original_second:
        np.testing.assert_array_equal(original_second[name], restored_second[name])
    dataset = SelfPlayDataset(serial_root)
    assert len(SelfPlayDataset(serial_root, split='train')) + len(SelfPlayDataset(serial_root, split='validation')) == len(dataset)
    batch = collate_selfplay_samples((dataset[0], dataset[1]))
    assert len(dataset) == result['samples']
    assert batch.decisions.candidate_mask.dtype == np.bool_
    assert batch.policy_targets.shape == batch.decisions.candidate_mask.shape
    assert not np.any(batch.policy_targets[~batch.decisions.candidate_mask])
    model = ActorCritic().eval()
    with torch.inference_mode():
        logits, values = model(batch.decisions)
    assert torch.isfinite(logits[batch.decisions.candidate_mask]).all()
    assert torch.isfinite(values).all()
    assert torch.isfinite(-(torch.as_tensor(batch.policy_targets) * torch.log_softmax(logits, -1).nan_to_num()).sum(-1)).all()
    game_seed = derive_seed(single.seed, 0, 0, 'game')
    live = encode_decision(GameEngine.new(game_seed, first_player=0).decision())
    first_arrays, _ = load_game(game_path(serial_root, 0))
    for name in STATE_FIELDS:
        np.testing.assert_array_equal(first_arrays[name][0], getattr(live, name))
    np.testing.assert_array_equal(first_arrays['candidate_features'][:int(first_arrays['candidate_offsets'][1])], live.candidate_features)
    with pytest.raises(ValueError, match='fingerprint'):
        generate(replace(single, search_config=replace(single.search_config, simulations=4)), serial_root, resume=True)
    with pytest.raises(ValueError, match='fingerprint'):
        generate(replace(single, temperature_schedule=TemperatureSchedule(opening_temperature=0.5)), serial_root, resume=True)
    game_path(serial_root, 0).write_bytes(b'broken')
    with pytest.raises(ValueError, match='invalid game archive'):
        validate_run(serial_root)
    with pytest.raises(ValueError, match='invalid game archive'):
        SelfPlayDataset(serial_root)


def test_single_candidate_record_and_temperature_boundary(tmp_path):
    request = GameEngine.new(24).decision()
    assert isinstance(request, DecisionRequest)
    forced = replace(request, legal_actions=request.legal_actions[:1])
    cfg = config(checkpoint(tmp_path))
    calls = []
    class FakePolicy:
        def search(self, request, **kwargs):
            calls.append(kwargs)
            return MCTSSearchResult(0, False, (1.0,), (1.0,), (), (), (0,), (1.0,), (1.0,), 0, 0, 0)
    records = []
    wrapper = _Recorder(1, FakePolicy(), cfg, 0, records)
    assert wrapper.choose(forced) is forced.legal_actions[0]
    assert records[0]['policy_weight'] == 0
    assert records[0]['value_weight'] == 1
    assert records[0]['simulations'] == 0
    assert records[0]['valid_world_count'] == 0
    assert records[0]['decision_index'] == 0
    late = replace(forced, observation=replace(forced.observation, own_completed_turns=5, opponent_completed_turns=5))
    wrapper.choose(late)
    assert calls[0]['temperature'] == 1.0
    assert calls[1]['temperature'] == 0.0
    assert calls[1]['add_root_noise'] is True


def test_archive_semantic_corruption_and_engineering_truncation(tmp_path):
    path = checkpoint(tmp_path)
    cfg = config(path, games=1, simulations=2, determinizations=1)
    root = tmp_path / 'valid'
    generate(cfg, root)
    arrays, metadata = load_game(game_path(root, 0))
    common = dict(expected_game_id=0, fingerprint=metadata['semantic_fingerprint'], simulations=2, worlds=1, split=metadata['split'])
    validate_game_arrays(arrays, metadata, **common)
    wrong = {name: value.copy() for name, value in arrays.items()}
    wrong['candidate_offsets'][1] = 0
    with pytest.raises(ValueError, match='offsets'):
        validate_game_arrays(wrong, metadata, **common)
    wrong = {name: value.copy() for name, value in arrays.items()}
    wrong['value_target'][0] *= -1
    if wrong['value_target'][0] == arrays['value_target'][0]:
        wrong['value_target'][0] = 0 if arrays['value_target'][0] else 1
    with pytest.raises(ValueError, match='terminal value'):
        validate_game_arrays(wrong, metadata, **common)
    wrong = {name: value.copy() for name, value in arrays.items()}
    wrong['policy_target'][0] = np.nan
    with pytest.raises(ValueError, match='policy_target'):
        validate_game_arrays(wrong, metadata, **common)
    with pytest.raises(ValueError, match='schema'):
        validate_game_arrays(arrays, metadata | {'data_schema': 'unknown'}, **common)
    truncated = replace(cfg, run_id='truncated', max_completed_turns=1)
    truncated_root = tmp_path / 'truncated'
    with pytest.raises(RuntimeError, match='engineering_truncation'):
        generate(truncated, truncated_root)
    assert not list((truncated_root / 'games').glob('game-*.npz'))
    assert (truncated_root / 'failures' / 'game-000000.json').exists()
    with pytest.raises(RuntimeError, match='engineering_truncation'):
        generate(truncated, truncated_root, resume=True)
    assert not list((truncated_root / 'games').glob('game-*.npz'))


def test_lock_and_checkpoint_content_identity(tmp_path):
    import fcntl
    from splendor.selfplay import _lock

    root = tmp_path / 'locked'
    descriptor = _lock(root)
    try:
        with pytest.raises(RuntimeError, match='already active'):
            _lock(root)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        import os
        os.close(descriptor)
    path = checkpoint(tmp_path)
    cfg = config(path, games=1, simulations=2, determinizations=1)
    output = tmp_path / 'frozen'
    generate(cfg, output)
    payload = torch.load(path, weights_only=True)
    first = next(iter(payload['model_state_dict']))
    payload['model_state_dict'][first] = payload['model_state_dict'][first] + 1
    torch.save(payload, path)
    with pytest.raises(ValueError, match='fingerprint'):
        generate(cfg, output, resume=True)
    with pytest.raises(ValueError, match='checkpoint contents changed'):
        validate_run(output)
