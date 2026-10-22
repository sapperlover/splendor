"""Finite checks for sparse EMA time scale, validation identity and direct arenas."""
import importlib.util
from dataclasses import asdict
import json
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'artifacts/entity_attention_ema_17_19_20261020/run.py'
spec = importlib.util.spec_from_file_location('checkpoint_ema_probe', PATH)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_checkpoint_half_life_is_measured_in_optimizer_updates():
    state = {'weight': torch.ones(3)}
    average = {'weight': torch.zeros(3, dtype=torch.float64)}
    for _ in range(10):
        probe.ema_update(average, state, 50, 500)
    torch.testing.assert_close(average['weight'], torch.full((3,), .5, dtype=torch.float64))
    with pytest.raises(ValueError):
        probe.ema_update(average, {'different_key': torch.ones(3)}, 50, 500)


def test_trajectory_does_not_double_count_round_boundary_aliases():
    config = json.loads(probe.CONFIG.read_text())
    sequence = list(probe.trajectory(config))
    assert len(sequence) == 160
    assert sequence[0][:2] == (12, 50)
    assert sequence[-1][:2] == (19, 1000)
    assert len({str(p) for _, _, p in sequence}) == 160
    assert all(p.name not in ('initial.pt', 'final.pt') for _, _, p in sequence)


def test_holdout_identity_preserves_duplicate_draws_and_excludes_train():
    game = {'round_index': 61, 'game_id': 2, 'run_id': 'r18', 'split': 'validation', 'decision_count': 50}
    ref = {'round_index': 61, 'game_id': 2, 'run_id': 'r18', 'decision_index': 11}
    refs = [ref.copy(), ref.copy()]
    assert probe.validation_records({'games': [game]}, refs) == [game]
    assert refs == [ref, ref]  # only archives are deduplicated, not sampled positions
    for bad in ({**game, 'split': 'train'}, {**game, 'run_id': 'different_run'}):
        with pytest.raises(ValueError):
            probe.validation_records({'games': [bad]}, refs)


def test_one_half_life_selected_across_all_three_rounds():
    deltas = {250: [-.020, .015, .015], 500: [-.002, -.002, -.002], 1000: [-.010, .003, .003]}
    groups = []
    for i, iteration in enumerate((17, 18, 19)):
        points = [{'half_life_steps': None, 'metrics': {'policy_kl': .4, 'value_mse': .75, 'kl_plus_mse': 1.15}}]
        for half, values in deltas.items():
            points.append({'half_life_steps': half, 'metrics': {'policy_kl': .4,
                'value_mse': .75+values[i], 'kl_plus_mse': 1.15+values[i]}})
        groups.append({'iteration': iteration, 'points': points})
    selection = probe.select_half_life({'groups': groups})
    assert selection['half_life_steps'] == 500
    assert all(len(c['per_round']) == 3 for c in selection['candidates'])


def test_export_is_a_frozen_model_not_a_mismatched_optimizer_checkpoint(tmp_path):
    from splendor.entity_attention import EntityAttention, EntityAttentionConfig
    from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
    from splendor.alphazero_checkpoint import load_alphazero_checkpoint

    config = EntityAttentionConfig(hidden=8, heads=2, layers=1, ffn_hidden=16,
        market_relations=True, value_readout='global_players_market')
    model = EntityAttention(config)
    state = {k: v.to(torch.float64) for k, v in model.state_dict().items()}
    path = tmp_path / 'ema.pt'
    record = probe.save_frozen(path, asdict(config), state, {'half_life_steps': 500})
    loaded = load_frozen_actor_critic_checkpoint(path)
    for name, value in model.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[name])
    assert not any(p.requires_grad for p in loaded.parameters())
    assert record['sha256'] == probe.sha256(path)
    with pytest.raises(ValueError, match='not an AlphaZero checkpoint'):
        load_alphazero_checkpoint(path)


@pytest.mark.parametrize('kind,pairs', [('greedy', 1000), ('search', 500)])
def test_direct_arena_uses_fresh_seeds_and_collector_execution(kind, pairs):
    config = json.loads(probe.CONFIG.read_text())
    source = json.loads((ROOT / config['source_config']).read_text())
    plan = {'config': config, 'source_config': source}
    value = probe.evaluation_config(plan, '/tmp/ema-r19.pt', '/tmp/raw-r19.pt', kind)
    from splendor.evaluation import FixedEvaluationConfig
    parsed = FixedEvaluationConfig.from_dict(value)
    assert len(parsed.game_seeds) == pairs
    assert value['policy_b']['checkpoint'] == '/tmp/raw-r19.pt'
    assert value['policy_a']['checkpoint'] == '/tmp/ema-r19.pt'
    assert value['seed_start'] != source[f'{kind}_evaluation']['seed_start']
    if kind == 'search':
        assert value['execution'] == {**source['selfplay']['execution'], 'pipeline_slots': 1}
        for side in ('policy_a', 'policy_b'):
            assert value[side]['search_config']['simulations'] == 1024
            assert value[side]['search_config']['determinizations'] == 1
            assert value[side]['search_config']['root_noise_fraction'] == 0
