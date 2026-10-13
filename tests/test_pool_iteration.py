from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pytest
import torch

from test_hybrid_training import source, small_config
from splendor import pool_iteration
from splendor.alphazero_checkpoint import load_alphazero_checkpoint
from splendor.alphazero_recording import read_json, sha256, write_json
from splendor.hybrid_train import HybridConfig, frozen_offline_dataset, run_hybrid
from splendor.offline_train import freeze_dataset
from splendor.compact_game_cache import DictionaryArray, CompactGameCache
from splendor.alphazero_replay import ReplayGame, _GameCache
from splendor.selfplay import derive_seed
from splendor.alphazero_loop import LoopSeeds
from splendor.selfplay_data import collate_selfplay_samples


def test_formal_progressive_plan_and_collection_batching():
    plan, config = pool_iteration.load_plan(
        Path('configs/entity_attention_v1/p4_pool_iteration_20261010.json'))
    assert plan['rounds'] == 9
    assert config.new_games_per_round == 1000 and config.replay_max_games == 6000
    assert [config.quotas(r) for r in range(9)] == [
        (224, 32), (186, 70), (147, 109), (109, 147), (70, 186), (32, 224),
        (0, 256), (0, 256), (0, 256)]
    assert config.quotas(1000) == (0, 256)
    assert config.selfplay.execution['max_batch_size'] == 500
    assert tuple(config.selfplay.execution['batch_buckets']) == (16, 32, 64, 128, 192, 256, 384, 500)


def setup_plan(tmp_path, source):
    base = small_config(source)
    parent = tmp_path / 'parent'
    state = run_hybrid(base, parent, online_rounds=0)
    initial = state['latest_checkpoint']
    records = freeze_dataset(base.offline)['games']
    for r in records:
        r['source_id'] = 'old'
    manifest = tmp_path / 'pool.jsonl'
    manifest.write_text(''.join(json.dumps(r) + '\n' for r in records))
    split_base = next(seed for seed in range(30710000, 30711000) if all(
        len({derive_seed(LoopSeeds(0, seed, 0).split_for(r), r, gid, 'split') < (1 << 63)
             for gid in range(2)}) == 2 for r in range(3)))
    evaluation = {'device': 'cpu', 'workers': 1, 'pairs': 1, 'seed_start': 30900000,
        'policy_seed_base': 30910000, 'bootstrap_seed': 30920000, 'bootstrap_resamples': 100,
        'policy_a': {'name': 'new', 'kind': 'greedy_checkpoint', 'checkpoint': 'generated-at-runtime'},
        'policy_b': {'name': 'P4V4', 'kind': 'greedy_checkpoint', 'checkpoint': initial['path']}}
    config = replace(base, run_id='p4-tiny', new_games_per_round=2, replay_max_games=4,
        online_validation_positions=2, seeds=replace(base.seeds, split=split_base),
        selfplay=replace(base.selfplay, validation_fraction=.5),
        mix_schedule=((1, 3), (2, 1)), evaluation_every_rounds=3, evaluation=evaluation,
        evaluation_anchor={'path': initial['path'], 'sha256': initial['sha256']},
        initialization={'checkpoint': {'path': initial['path'], 'sha256': initial['sha256']},
            'manifest': {'path': str(manifest), 'sha256': sha256(manifest)}},
        fitting_diagnostics={'steps': [0, 1, 2], 'seed': 30740000,
            'train_positions_per_generation': 4, 'offline_positions_per_source_split': 4})
    hybrid_path = tmp_path / 'hybrid.json'
    write_json(hybrid_path, config.to_dict())
    plan = {'protocol': pool_iteration.PROTOCOL, 'rounds': 3,
        'hybrid_config': str(hybrid_path), 'output': str(tmp_path / 'run'),
        'artifacts': str(tmp_path / 'artifacts'), 'greedy_evaluation': evaluation,
        'strong_anchor': {'path': initial['path'], 'sha256': initial['sha256']}}
    plan_path = tmp_path / 'plan.json'
    write_json(plan_path, plan)
    return plan, config, plan_path


def test_warm_start_resume_diagnostics_control_and_unconditional_teacher(tmp_path, source, monkeypatch):
    torch.set_num_threads(1)
    plan, config, plan_path = setup_plan(tmp_path, source)
    # Exercise the real entry point, including progressive quotas and pure online.
    assert pool_iteration.load_plan(plan_path)[1].to_dict() == config.to_dict()
    checked = pool_iteration.check(plan, config)
    assert checked['quotas_by_round'] == [
        {'round': 1, 'offline': 3, 'online': 1},
        {'round': 2, 'offline': 1, 'online': 3},
        {'round': 3, 'offline': 0, 'online': 4}]
    seen = []
    def bad_score(cfg, path, **kwargs):
        # Valid engineering outcome, deliberately losing score. Cannot reject teachers.
        summary = {'passed': True, 'win_rate_score': 0.0, 'completed_pairs': 1,
                   'total_pairs': 1, 'paired_bootstrap': {'lower': 0.0, 'upper': 0.0}}
        seen.append((cfg.policy_a.checkpoint, cfg.policy_b.checkpoint))
        write_json(path, summary)
        return SimpleNamespace(summary=summary)
    monkeypatch.setattr(pool_iteration, 'run_fixed_evaluation', bad_score)
    monkeypatch.setattr('splendor.hybrid_train.run_fixed_evaluation', bad_score)
    bounded = pool_iteration.run(plan, config, plan_path, max_steps_this_session=1)
    assert bounded['completed_rounds'] == 0
    assert not (Path(plan['output']) / 'offline_control').exists()
    resumed = pool_iteration.run(plan, config, plan_path)
    assert resumed['status'] == 'complete' and resumed['completed_rounds'] == 3
    assert len(resumed['panels']) == 6
    root = Path(plan['output'])
    for i in range(3):
        stage = root / 'rounds' / f'round-{i:03}'
        points = [read_json(stage / 'fitting' / f'step-{s:09}.json') for s in (0, 1, 2)]
        assert points[0]['online_train_fixed']['set_digest'] == points[-1]['online_train_fixed']['set_digest']
        assert points[-1]['online_validation_full']['scope'] == 'complete original validation games'
        usage = read_json(stage / 'source_usage.json')
        offline_count, online_count = config.quotas(i)
        assert sum(g['actual_draws'] for g in usage['online_by_generation']) == 2 * online_count
        assert usage['offline']['draws'] == 2 * offline_count
        assert read_json(stage / 'replay_snapshot.json')['quotas'] == [offline_count, online_count]
        if i:
            teacher = read_json(stage / 'replay_snapshot.json')['teacher']
            assert teacher['sha256'] == sha256(root / 'rounds' / f'round-{i-1:03}' / 'checkpoints/final.pt')
    snapshot = read_json(root / 'rounds/round-002/replay_snapshot.json')
    assert len(snapshot['games']) == 4 and {g['round_index'] for g in snapshot['games']} == {1, 2}
    fixed = [read_json(root / 'rounds' / f'round-{i:03}' / 'fitting/train_references.json')['references'] for i in range(3)]
    assert fixed[0] == [r for r in fixed[1] if r['round_index'] == 0]
    assert [r for r in fixed[1] if r['round_index'] == 1] == [r for r in fixed[2] if r['round_index'] == 1]
    final = load_alphazero_checkpoint(root / 'rounds/round-002/checkpoints/final.pt')
    control = load_alphazero_checkpoint(root / 'offline_control/checkpoints/final.pt')
    parent = load_alphazero_checkpoint(config.initialization['checkpoint']['path'])
    assert final.total_train_step == control.total_train_step == parent.total_train_step + 6
    assert {int(s['step']) for s in final.optimizer_state_dict['state'].values()} == {8}
    assert len(seen) == 6  # 3 greedy + 1 periodic + 2 final panels
    assert {g['split'] for g in frozen_offline_dataset(config)['games']} == {'train', 'validation'}

    continuous_plan = {**plan, 'output': str(tmp_path / 'continuous'), 'artifacts': str(tmp_path / 'continuous_artifacts')}
    continuous_path = tmp_path / 'continuous_plan.json'
    write_json(continuous_path, continuous_plan)
    pool_iteration.run(continuous_plan, config, continuous_path)
    uninterrupted = load_alphazero_checkpoint(Path(continuous_plan['output']) / 'rounds/round-002/checkpoints/final.pt')
    assert final.sampling_rng_state == uninterrupted.sampling_rng_state
    assert torch.equal(final.torch_cpu_rng_state, uninterrupted.torch_cpu_rng_state)
    for k, value in final.model_state_dict.items():
        torch.testing.assert_close(value, uninterrupted.model_state_dict[k], rtol=0, atol=0)
    for k, params in final.optimizer_state_dict['state'].items():
        for name, value in params.items():
            torch.testing.assert_close(value, uninterrupted.optimizer_state_dict['state'][k][name], rtol=0, atol=0)
    # Re-running a fully completed command consumes no optimizer updates.
    final_hash = sha256(root / 'rounds/round-002/checkpoints/final.pt')
    pool_iteration.run(plan, config, plan_path)
    assert sha256(root / 'rounds/round-002/checkpoints/final.pt') == final_hash


def test_compact_cache_preserves_float_bits_and_training_batch(source):
    values = np.array([0., -0., 1., -.125], dtype=np.float32)
    packed = DictionaryArray(values)
    np.testing.assert_array_equal(packed[:].view(np.uint32), values.view(np.uint32))
    base = small_config(source)
    record = freeze_dataset(base.offline)['games'][0]
    game = ReplayGame(**{k: record[k] for k in ('round_index', 'run_id', 'game_id', 'split', 'decision_count', 'policy_samples')}, path=Path(record['path']))
    plain, compact = _GameCache(16 * 1024**2), CompactGameCache(16 * 1024**2)
    a = collate_selfplay_samples([plain.sample(game, i) for i in range(min(3, game.decision_count))])
    b = collate_selfplay_samples([compact.sample(game, i) for i in range(min(3, game.decision_count))])
    for key in (f.name for f in fields(a.decisions)):
        np.testing.assert_array_equal(getattr(a.decisions, key), getattr(b.decisions, key))
    for key in ('policy_targets', 'value_targets', 'policy_weights', 'value_weights'):
        np.testing.assert_array_equal(getattr(a, key), getattr(b, key))


def test_real_greedy_panel_resumes(source, tmp_path):
    plan, config, _ = setup_plan(tmp_path, source)
    p = config.initialization['checkpoint']['path']
    out = tmp_path / 'panel.json'
    result = pool_iteration.panel(plan['greedy_evaluation'], p, p, out)
    first = read_json(out)
    assert result.summary['passed']
    pool_iteration.panel(plan['greedy_evaluation'], p, p, out)
    resumed = read_json(out)
    assert resumed['win_rate_score'] == first['win_rate_score']
    assert resumed['pairs'] == first['pairs']


@pytest.mark.parametrize('policy_games', [None, 2])
def test_online_forever_inherits_window_resumes_lr_and_unconditionally_upgrades(tmp_path, source, monkeypatch,
                                                                             policy_games):
    torch.set_num_threads(1)
    old_plan, parent_cfg, _ = setup_plan(tmp_path, source)
    split_base = next(seed for seed in range(30710000, 30720000) if all(
        len({derive_seed(LoopSeeds(0, seed, 0).split_for(r), r, gid, 'split') < (1 << 63)
             for gid in range(2)}) == 2 for r in range(6)))
    parent_cfg = replace(parent_cfg, seeds=replace(parent_cfg.seeds, split=split_base))
    seen = []
    def bad_score(cfg, path, **kwargs):
        seen.append((Path(path).name, cfg.policy_a.checkpoint, cfg.policy_b.checkpoint))
        summary = {'passed': True, 'win_rate_score': 0.0, 'completed_pairs': 1,
            'total_pairs': 1, 'paired_bootstrap': {'lower': 0.0, 'upper': 0.0}}
        write_json(path, summary)
        return SimpleNamespace(summary=summary)
    monkeypatch.setattr('splendor.hybrid_train.run_fixed_evaluation', bad_score)
    monkeypatch.setattr(pool_iteration, 'run_fixed_evaluation', bad_score)
    parent = tmp_path / 'online_parent'
    parent_state = run_hybrid(parent_cfg, parent, online_rounds=3)
    parent_final = Path(parent_state['latest_checkpoint']['path'])
    parent_hash = sha256(parent_final)
    snapshot = read_json(parent / 'rounds/round-002/replay_snapshot.json')
    fixed = read_json(parent / 'rounds/round-002/fitting/train_references.json')
    source_payload = load_alphazero_checkpoint(parent_final)
    config = replace(parent_cfg, run_id='online-forever', online_only=True,
        initialization=None, mix_schedule=(), learning_rate_override=2e-5,
        policy_replay_max_games=policy_games,
        # Old source paths must not be consulted, even for checks or validation.
        offline=replace(parent_cfg.offline, source_run='unused-no-offline-source',
                        source_artifacts='unused-no-offline-artifacts'))
    config_path = tmp_path / 'online_hybrid.json'
    write_json(config_path, config.to_dict())
    plan = {'protocol': pool_iteration.PROTOCOL, 'forever': True, 'hybrid_config': str(config_path),
        'output': str(tmp_path / 'forever'), 'artifacts': str(tmp_path / 'forever_artifacts'),
        'greedy_evaluation': old_plan['greedy_evaluation'],
        'fork_from': {'run': str(parent), 'round': 3, 'checkpoint_sha256': parent_hash,
                     'replay_snapshot_sha256': sha256(parent / 'rounds/round-002/replay_snapshot.json')}}
    plan_path = tmp_path / 'online_plan.json'
    write_json(plan_path, plan)
    assert pool_iteration.load_plan(plan_path)[1].to_dict() == config.to_dict()
    checked = pool_iteration.check(plan, config)
    assert checked['initial_step'] == source_payload.total_train_step
    assert checked['source_games'] == 4 and checked['first_absolute_round'] == 4
    def forbidden(*args, **kwargs):
        raise AssertionError('online-only run touched old offline pool or final control')
    monkeypatch.setattr('splendor.hybrid_train.frozen_offline_dataset', forbidden)
    monkeypatch.setattr('splendor.hybrid_train.freeze_dataset', forbidden)
    monkeypatch.setattr('splendor.hybrid_train.run_offline', forbidden)
    monkeypatch.setattr(pool_iteration, 'offline_control', forbidden)
    monkeypatch.setattr(pool_iteration, 'final_offline_holdouts', forbidden)
    seen.clear()
    partial = pool_iteration.run(plan, config, plan_path, max_steps_this_session=1)
    root = Path(plan['output'])
    assert partial['completed_rounds'] == 0
    assert pool_iteration.status(plan, config)['current_round'] == 4
    initial = load_alphazero_checkpoint(root / 'rounds/round-000/checkpoints/initial.pt')
    assert initial.total_train_step == source_payload.total_train_step
    assert initial.sampling_rng_state == source_payload.sampling_rng_state
    for k, v in source_payload.model_state_dict.items():
        torch.testing.assert_close(initial.model_state_dict[k], v, rtol=0, atol=0)
    for k, params in source_payload.optimizer_state_dict['state'].items():
        for name, value in params.items():
            torch.testing.assert_close(initial.optimizer_state_dict['state'][k][name], value, rtol=0, atol=0)
    assert {g['lr'] for g in initial.optimizer_state_dict['param_groups']} == {2e-5}
    assert not (root / 'offline').exists()
    resumed = pool_iteration.run(plan, config, plan_path, max_steps_this_session=5)
    assert resumed['completed_rounds'] == 3 and resumed['last_absolute_round'] == 6
    assert resumed['target_rounds'] is None and resumed['status'] == 'in_progress'
    assert len(seen) == 4  # Absolute round 6 search, plus greedy every round, all losing.
    assert all(Path(opponent) == Path(config.evaluation_anchor['path']).resolve() for _, _, opponent in seen)
    assert pool_iteration.status(plan, config)['current_round'] == 7
    assert not (root / 'offline_control').exists() and not (root / 'complete.json').exists()
    for local in range(3):
        stage = root / 'rounds' / f'round-{local:03d}'
        snap = read_json(stage / 'replay_snapshot.json')
        assert snap['quotas'] == [0, 4]
        assert {g['round_index'] for g in snap['games']} == {local+2, local+3}
        assert len(snap['games']) == 4
        if local:
            assert snap['teacher']['sha256'] == sha256(root / 'rounds' / f'round-{local-1:03d}' / 'checkpoints/final.pt')
        else:
            assert snap['teacher']['sha256'] == parent_hash
            inherited = [g for g in snap['games'] if g['round_index'] == 2]
            original = [g for g in snapshot['games'] if g['round_index'] == 2]
            assert inherited == original
            new_refs = read_json(stage / 'fitting/train_references.json')['references']
            assert [r for r in new_refs if r['round_index'] == 2] == [r for r in fixed['references'] if r['round_index'] == 2]
        for step in (0, 1, 2):
            point = read_json(stage / 'fitting' / f'step-{step:09d}.json')
            assert point['offline_fixed'] is None
            assert f'g{local+4:03d}/validation/all' in point['online_validation_full']['metrics']
        usage = read_json(stage / 'source_usage.json')
        if policy_games is None:
            assert sum(g['actual_draws'] for g in usage['online_by_generation']) == 8
        else:
            pools = snap['joint_head_pools']
            assert pools['mode'] == 'joint_separate_batches'
            assert pools['policy_replay_max_games'] == 2 and pools['value_replay_max_games'] == 4
            assert [g['round_index'] for g in pools['generations'] if g['policy_expected_fraction']] == [local+3]
            assert usage['policy']['draws'] == usage['value']['draws'] == 8
            draws = usage['joint_head_pools']['actual_draws_by_head_round']
            assert draws[f'policy_round_{local+3}'] == 8
            assert sum(n for k, n in draws.items() if k.startswith('value_round_')) == 8
            assert sum(usage['online_replacement']['actual_draws'].values()) == 16
            import json
            records = [json.loads(line) for line in (stage / 'training_metrics.jsonl').read_text().splitlines()]
            steps = [row for row in records if row['event'] == 'train_step']
            assert all(row['training_mode'] == 'joint_separate_batches' for row in steps)
            assert all(row['policy_batch_size'] == row['value_batch_size'] == 4 for row in steps)
            assert all('policy_grad_norm' not in row and 'value_grad_norm' not in row for row in steps)
            for head in ('policy', 'value'):
                assert all(read_json(p)['batch_size'] == 4 for p in (stage / 'usage' / head).glob('*.json'))
    final = load_alphazero_checkpoint(root / 'rounds/round-002/checkpoints/final.pt')
    assert final.total_train_step == source_payload.total_train_step+6
    assert final.learner_config['learning_rate'] == 2e-5
    assert sha256(parent_final) == parent_hash
    # A continuous fork must match the interrupted run exactly, including AdamW and RNG.
    continuous = {**plan, 'output': str(tmp_path / 'online_continuous'),
                  'artifacts': str(tmp_path / 'online_continuous_artifacts')}
    continuous_path = tmp_path / 'online_continuous_plan.json'
    write_json(continuous_path, continuous)
    pool_iteration.run(continuous, config, continuous_path, max_steps_this_session=6)
    uninterrupted = load_alphazero_checkpoint(Path(continuous['output']) / 'rounds/round-002/checkpoints/final.pt')
    assert final.sampling_rng_state == uninterrupted.sampling_rng_state
    assert torch.equal(final.torch_cpu_rng_state, uninterrupted.torch_cpu_rng_state)
    for k, value in final.model_state_dict.items():
        torch.testing.assert_close(value, uninterrupted.model_state_dict[k], rtol=0, atol=0)
    for k, params in final.optimizer_state_dict['state'].items():
        for name, value in params.items():
            torch.testing.assert_close(value, uninterrupted.optimizer_state_dict['state'][k][name], rtol=0, atol=0)
    # Summary recovery uses committed generations, including correct absolute column names.
    (Path(plan['artifacts']) / 'rounds/round-000006.json').unlink()
    rebuilt = pool_iteration.analyze(plan)
    assert rebuilt['latest_round']['round'] == 6
    assert len(list((Path(plan['artifacts']) / 'rounds').glob('*.json'))) == 3


def test_formal_online_forever_plan():
    plan, config = pool_iteration.load_plan(
        Path('configs/entity_attention_v1/r9_online_lr2e5_1000_forever_20261011.json'))
    assert plan['forever'] and 'rounds' not in plan and 'strong_anchor' not in plan
    assert plan['fork_from']['round'] == 9
    assert config.online_only and config.initialization is None
    assert config.quotas(0) == config.quotas(10000) == (0, 256)
    assert config.new_games_per_round == config.updates_per_round == 1000
    assert config.replay_max_games == 6000 and config.learning_rate_override == 2e-5
    assert config.offline.learner.learning_rate == 1e-4
    assert config.offline.replay_cache_bytes == 8*1024**3
    assert config.fitting_diagnostics['steps'] == [0, 100, 250, 500, 1000]
    assert config.evaluation_every_rounds == 3 and config.evaluation['pairs'] == 200
