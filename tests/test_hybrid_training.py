from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch

from splendor import alphazero_replay, hybrid_train
from splendor.actor_critic import ActorCritic
from splendor.alphazero_checkpoint import load_alphazero_checkpoint
from splendor.alphazero_loop import LoopSeeds, SelfPlaySection
from splendor.alphazero_recording import read_json, sha256, write_json
from splendor.alphazero_replay import PositionPool, ReplayGame, ReplayIndex, ReplayRun, build_replay_index
from splendor.alphazero_train import AlphazeroLearner, LearnerConfig
from splendor.hybrid_train import (HybridConfig, SplitTrainingConfig, draw_mixed_batch,
                                   draw_split_batch, prepare_fork, run_hybrid, split_replay_pools)
from splendor.mcts_policy import MCTSConfig
from splendor.offline_train import OfflineConfig
from splendor.rng import DeterministicRng
from splendor.selfplay import derive_seed
from splendor.selfplay_fast import ExecutionConfig, FastSelfPlayConfig, generate


def test_formal_schedule_and_execution():
    config = HybridConfig.from_dict(read_json(Path('configs/entity_attention_v1/hybrid_static_graph.json')))
    assert [config.quotas(r) for r in (0, 1, 2, 3, 5, 7, 10, 15, 1000)] == [
        (224, 32), (192, 64), (160, 96), (128, 128), (96, 160), (64, 192), (32, 224), (0, 256), (0, 256)]
    execution = ExecutionConfig(**config.selfplay.execution)
    assert execution.shared_slots == 2 and execution.pipeline_slots == 1
    assert execution.compile_mode == 'native-cudagraph'
    assert execution.batch_buckets == [64, 256, 500] or tuple(execution.batch_buckets) == (64, 256, 500)
    assert config.to_dict() == HybridConfig.from_dict(config.to_dict()).to_dict()


def test_formal_fork_configs_and_replacement_schedule():
    original = HybridConfig.from_dict(read_json(Path('configs/entity_attention_v1/hybrid_static_graph.json')))
    configs = []
    for simulations in (512, 1024):
        config = HybridConfig.from_dict(read_json(Path(f'configs/entity_attention_v1/hybrid_fork_s{simulations}_n2.json')))
        assert config.offline == original.offline
        assert config.new_games_per_round == config.updates_per_round == 100
        assert config.replay_max_games == 2000 and config.evaluation_every_rounds == 10
        assert [config.quotas(r) for r in (0, 9, 10, 59, 60, 10000)] == [
            (64, 192), (64, 192), (32, 224), (32, 224), (0, 256), (0, 256)]
        assert config.selfplay.search_config['simulations'] == simulations
        assert config.selfplay.search_config['determinizations'] == 2
        execution = ExecutionConfig(**config.selfplay.execution)
        assert execution.total_active_games == 100 and execution.max_batch_size == 128
        assert execution.shared_slots == 2 and execution.pipeline_slots == 1
        assert execution.compile_mode == 'native-cudagraph'
        configs.append(config.to_dict())
    for raw in configs:
        raw.pop('run_id')
        raw['selfplay']['search_config'].pop('simulations')
    assert configs[0] == configs[1]


def test_same_game_numbers_are_isolated_and_zero_pool_does_not_draw(monkeypatch):
    def load(path):
        return {'value_targets': np.array([1 if str(path) == 'old' else -1])}, {'run_id': str(path)}
    monkeypatch.setattr(alphazero_replay, 'load_game', load)
    pools = []
    for name in ('old', 'new'):
        index = ReplayIndex([ReplayGame(0, name, 0, 'train', 1, 1, Path(name))], cache_bytes=1024)
        pools.append(PositionPool(index.games, index.cache, name))
    batch = draw_mixed_batch(*pools, DeterministicRng(7), 2, 2)
    assert [ref.run_id for ref in batch.references] == ['old', 'old', 'new', 'new']
    assert [int(s.arrays['value_targets'][s.index]) for s in batch.samples] == [1, 1, -1, -1]
    assert batch.source_positions == {'offline': 2, 'online': 2}
    a, b = DeterministicRng(9), DeterministicRng(9)
    draw_mixed_batch(None, pools[1], a, 0, 4)
    for _ in range(4):
        pools[1].sample(b)
    assert a.snapshot() == b.snapshot()


@pytest.fixture(scope='module')
def source(tmp_path_factory):
    root = tmp_path_factory.mktemp('hybrid-source')
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(31)
    base = root / 'base.pt'
    AlphazeroLearner(ActorCritic(), LearnerConfig(device='cpu'), sampling_rng=DeterministicRng(0)).save(base)
    source_run, artifacts = root / 'source', root / 'artifacts'
    round_root = source_run / 'rounds/round-000'
    split = next(seed for seed in range(1000) if len({derive_seed(seed, 0, gid, 'split') < (1 << 63)
                                                   for gid in range(2)}) == 2)
    collection = FastSelfPlayConfig(protocol='mcts-selfplay-v2', run_id='old-round-000', round_index=0,
        checkpoint=str(base), num_games=2, seed=31001, search_config=MCTSConfig(simulations=2),
        execution=ExecutionConfig(backend='batched_cpu', engine_mode='fast', max_batch_size=1),
        validation_fraction=.5, split_seed=split)
    generate(collection, round_root / 'selfplay')
    index = build_replay_index([ReplayRun(0, collection.run_id, round_root / 'selfplay')], max_games=2)
    games = [g.to_dict() for g in index.games]
    write_json(round_root / 'replay_snapshot.json', {'games': games})
    write_json(source_run / 'loop_state.json', {'completed_rounds': [0]})
    write_json(artifacts / 'round-000/dataset.json', {'games': [
        {**g, 'status': 'natural_terminal', 'archive_sha256': sha256(Path(g['path'])),
         'checkpoint_sha256': sha256(base)} for g in games]})
    yield source_run, artifacts
    torch.set_num_threads(threads)


def small_config(source):
    run, artifacts = source
    offline = OfflineConfig(source_run=str(run), source_artifacts=str(artifacts), rounds=(0,),
        model_config={'architecture': 'entity_attention_v1'}, max_steps=2, validation_positions=8,
        replay_cache_bytes=16 * 1024**2, full_validation_at_end=False,
        learner=LearnerConfig(device='cpu', batch_size=4, history_batch_size=2,
                              checkpoint_every_steps=1, validation_batch_size=4))
    return HybridConfig(run_id='hybrid-test', offline=offline,
        selfplay=SelfPlaySection(search_config=asdict(MCTSConfig(simulations=2)),
            execution=asdict(ExecutionConfig(backend='batched_cpu', engine_mode='fast', max_batch_size=1,
                                            actor_projection=False)),
            add_root_noise=True, temperature_schedule={}, validation_fraction=0., max_completed_turns=400),
        seeds=LoopSeeds(collection=7700000, split=7701000, sampling=offline.sampling_seed),
        new_games_per_round=1, updates_per_round=2, replay_max_games=2,
        online_validation_positions=0, mix_schedule=((1, 3), (2, 2)), evaluation_every_rounds=0)


def test_continuous_resume_optimizer_rng_and_latest_teacher(tmp_path, source, monkeypatch):
    config = small_config(source)
    # Deliberately poor score is observational and cannot reject the new model.
    seen = []
    monkeypatch.setattr(hybrid_train, '_evaluation', lambda cfg, cp, anchor, stage: seen.append(cp))
    config = replace(config, evaluation_every_rounds=1, evaluation={'test_score': 0.0})
    uninterrupted = tmp_path / 'continuous'
    run_hybrid(config, uninterrupted, online_rounds=2)
    segmented = tmp_path / 'resumed'
    run_hybrid(config, segmented, online_rounds=2, max_steps_this_session=1)
    run_hybrid(config, segmented, online_rounds=2, resume=True, max_steps_this_session=2)
    initial = load_alphazero_checkpoint(segmented / 'rounds/round-000/checkpoints/initial.pt')
    assert initial.total_train_step == 2
    assert {int(s['step']) for s in initial.optimizer_state_dict['state'].values()} == {2}
    frozen_hash = sha256(segmented / 'config.json')
    state = run_hybrid(config, segmented, online_rounds=2, resume=True,
                       replay_cache_bytes_override=512 * 1024)
    assert sha256(segmented / 'config.json') == frozen_hash
    sessions = [read_json(path) for path in (segmented / 'source').glob('*/identity.json')]
    assert 512 * 1024 in {session['replay_cache_capacity_bytes'] for session in sessions}
    a = load_alphazero_checkpoint(uninterrupted / 'rounds/round-001/checkpoints/final.pt')
    b = load_alphazero_checkpoint(state['latest_checkpoint']['path'])
    assert b.total_train_step == 6
    assert {int(s['step']) for s in b.optimizer_state_dict['state'].values()} == {6}
    assert a.sampling_rng_state == b.sampling_rng_state
    assert torch.equal(a.torch_cpu_rng_state, b.torch_cpu_rng_state)
    for key in a.model_state_dict:
        torch.testing.assert_close(a.model_state_dict[key], b.model_state_dict[key], rtol=0, atol=0)
    for key, params in a.optimizer_state_dict['state'].items():
        for name in params:
            torch.testing.assert_close(params[name], b.optimizer_state_dict['state'][key][name], rtol=0, atol=0)
    snapshot = read_json(segmented / 'rounds/round-001/replay_snapshot.json')
    assert snapshot['teacher']['sha256'] == sha256(segmented / 'rounds/round-000/checkpoints/final.pt')
    assert len(seen) == 4
    # Same frozen algorithm config can continue into the pure-online tier.
    continued = run_hybrid(config, segmented, online_rounds=3, resume=True)
    assert continued['completed_rounds'] == [0, 1, 2]
    assert read_json(segmented / 'rounds/round-002/replay_snapshot.json')['quotas'] == [0, 4]
    assert len(read_json(segmented / 'rounds/round-002/replay_snapshot.json')['games']) == 2


def test_fixed_online_holdout_survives_replay_eviction(tmp_path, source):
    config = small_config(source)
    # Both tiny generations must contain a whole train game and a validation game.
    split_base = next(base for base in range(14610000, 14611000) if all(
        len({derive_seed(LoopSeeds(0, base, 0).split_for(r), r, gid, 'split') < (1 << 63)
             for gid in range(2)}) == 2 for r in range(2)))
    config = replace(config, new_games_per_round=2, replay_max_games=2, online_validation_positions=8,
                     seeds=replace(config.seeds, split=split_base),
                     selfplay=replace(config.selfplay, validation_fraction=.5))
    output = tmp_path / 'holdout'
    run_hybrid(config, output, online_rounds=2)
    snapshot = read_json(output / 'rounds/round-001/replay_snapshot.json')
    assert {game['round_index'] for game in snapshot['games']} == {1}
    assert {ref['round_index'] for ref in read_json(output / 'fixed_online_validation.json')['references']} == {0}
    metrics = read_json(output / 'rounds/round-001/validation_after.json')
    assert metrics['online_fixed']['positions'] == 8 and metrics['online_fixed']['available']
    assert metrics['current']['available']


def test_cross_round_caches_drop_evicted_games_and_retire_offline_train(tmp_path, source, monkeypatch):
    config = small_config(source)
    split_base = next(base for base in range(14610000, 14611000) if all(
        len({derive_seed(LoopSeeds(0, base, 0).split_for(r), r, gid, 'split') < (1 << 63)
             for gid in range(2)}) == 2 for r in range(4)))
    config = replace(config, new_games_per_round=2, replay_max_games=2,
                     online_validation_positions=8, seeds=replace(config.seeds, split=split_base),
                     selfplay=replace(config.selfplay, validation_fraction=.5))
    output = tmp_path / 'cache-lifetime'
    offline_indices, online_indices, before_collection = [], [], []
    original_make, original_build, original_generate = (
        hybrid_train.make_index, hybrid_train.build_replay_index, hybrid_train.generate)

    def make(dataset, capacity):
        index = original_make(dataset, capacity)
        if not offline_indices:
            offline_indices.append(index)
        return index

    def build(*args, **kwargs):
        index = original_build(*args, **kwargs)
        online_indices.append(index)
        return index

    def collect(*args, **kwargs):
        before_collection.append(set(offline_indices[0].cache._entries))
        return original_generate(*args, **kwargs)

    monkeypatch.setattr(hybrid_train, 'make_index', make)
    monkeypatch.setattr(hybrid_train, 'build_replay_index', build)
    monkeypatch.setattr(hybrid_train, 'generate', collect)
    state = run_hybrid(config, output, online_rounds=4)
    assert state['completed_rounds'] == [0, 1, 2, 3]
    off_index = offline_indices[0]
    offline_train = {g.key for g in off_index.split_games('train')}
    offline_holdout = {g.key for g in off_index.split_games('validation')}
    assert before_collection[1] & offline_train
    assert before_collection[2] <= offline_holdout  # Released before pure-online collection.
    assert set(off_index.cache._entries) <= offline_holdout
    cache = online_indices[-1].cache
    assert all(index.cache is cache for index in online_indices)
    fixed_games = read_json(output / 'fixed_online_dataset.json')['games']
    keep = {g.key for g in online_indices[-1].games} | {
        (g['round_index'], g['game_id']) for g in fixed_games}
    assert set(cache._entries) == keep
    assert cache.entries == 3  # Two active games and one fixed holdout outside the window.
    assert read_json(output / 'rounds/round-003/validation_after.json')['online_fixed']['positions'] == 8


def test_phase_cleanup_releases_compiled_inference_cycles_without_changing_rng(monkeypatch):
    import weakref

    from splendor.benchmark_search import fixed_requests
    from splendor.features import encode_decision
    from splendor.search_inference import FrozenSearchInference, InferenceConfig

    original = torch.compile
    monkeypatch.setattr(torch, 'compile', lambda function, **kw:
                        original(function, backend='eager', dynamic=kw.get('dynamic')))
    refs = []
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        encoded = encode_decision(fixed_requests()[0][1])
        for _ in range(3):
            model = ActorCritic().eval().requires_grad_(False)
            evaluator = FrozenSearchInference(model, device='cpu', config=InferenceConfig(
                mode='compiled', actor_projection=False, candidate_buckets=(64,), batch_buckets=(1,)))
            expected = evaluator.policy_value_encoded([encoded])
            refs.append(weakref.ref(evaluator))
            del evaluator, model
            rng = torch.get_rng_state().clone()
            hybrid_train._release_inference_memory()
            assert torch.equal(torch.get_rng_state(), rng)
            assert all(ref() is None for ref in refs)
            assert all(np.isfinite(row).all() and np.isfinite(value) for row, value in expected)
    finally:
        torch.set_num_threads(threads)


def test_batch_tuning_finishes_existing_collection_before_using_new_size(tmp_path, source, monkeypatch):
    config = small_config(source)
    output = tmp_path / 'batch-tuning'
    run_hybrid(config, output, online_rounds=1, max_steps_this_session=3)
    frozen_hash = sha256(output / 'config.json')
    manifest_path = output / 'rounds/round-000/selfplay/manifest.json'
    manifest_hash = sha256(manifest_path)
    first_game_hashes = {str(p): sha256(p) for p in manifest_path.parent.rglob('*.npz')}
    tuned = replace(config, selfplay=replace(config.selfplay,
                    execution={**config.selfplay.execution, 'max_batch_size': 2}))
    calls = []
    original_generate = hybrid_train.generate

    def collect(cfg, root, **kwargs):
        calls.append(cfg.execution.max_batch_size)
        return original_generate(cfg, root, **kwargs)

    monkeypatch.setattr(hybrid_train, 'generate', collect)
    state = run_hybrid(tuned, output, resume=True, online_rounds=2)
    assert calls == [1, 2]
    assert state['completed_rounds'] == [0, 1]
    assert state['latest_checkpoint']['total_train_step'] == 6
    assert sha256(output / 'config.json') == frozen_hash
    assert sha256(manifest_path) == manifest_hash
    assert all(sha256(Path(p)) == digest for p, digest in first_game_hashes.items())
    assert read_json(output / 'rounds/round-001/selfplay/manifest.json')['config']['execution']['max_batch_size'] == 2
    identities = [read_json(p) for p in (output / 'source').glob('*/identity.json')]
    assert sorted(i['collection_execution']['max_batch_size'] for i in identities) == [1, 2]
    assert all(i['frozen_config_sha256'] == frozen_hash for i in identities)


def test_resume_can_tune_graph_buckets_but_not_teacher_or_training(tmp_path, source):
    base = small_config(source)
    config = replace(base, selfplay=replace(base.selfplay,
        search_config={**base.selfplay.search_config, 'determinizations': 1}, execution={
        **base.selfplay.execution, 'inference_mode': 'compiled',
        'max_batch_size': 2, 'batch_buckets': [1, 2]}))
    output = tmp_path / 'graph-buckets'
    initial = run_hybrid(config, output, online_rounds=0)
    checkpoint_hash = sha256(Path(initial['latest_checkpoint']['path']))
    tuned = replace(config, selfplay=replace(config.selfplay, execution={
        **config.selfplay.execution, 'max_batch_size': 3, 'batch_buckets': [1, 2, 3]}))
    resumed = run_hybrid(tuned, output, resume=True, online_rounds=0)
    assert resumed == initial
    assert sha256(Path(resumed['latest_checkpoint']['path'])) == checkpoint_hash
    for changed in (
        replace(tuned, updates_per_round=3),
        replace(tuned, replay_max_games=3),
        replace(tuned, selfplay=replace(tuned.selfplay,
                search_config={**tuned.selfplay.search_config, 'determinizations': 2})),
        replace(tuned, selfplay=replace(tuned.selfplay,
                execution={**tuned.selfplay.execution, 'max_batch_wait_ms': 3})),
    ):
        with pytest.raises(ValueError):
            run_hybrid(changed, output, resume=True, online_rounds=0)
    with pytest.raises(ValueError, match='cover max_batch_size'):
        run_hybrid(replace(tuned, selfplay=replace(tuned.selfplay,
            execution={**tuned.selfplay.execution, 'max_batch_size': 4})),
            output, resume=True, online_rounds=0)


def test_resume_after_evaluation_failure_does_not_repeat_training(tmp_path, source, monkeypatch):
    config = replace(small_config(source), evaluation_every_rounds=1, evaluation={'test_score': 0.0})
    output = tmp_path / 'evaluation-failure'

    def failed(*args):
        raise RuntimeError('injected evaluation failure')

    monkeypatch.setattr(hybrid_train, '_evaluation', failed)
    with pytest.raises(RuntimeError, match='injected evaluation failure'):
        run_hybrid(config, output, online_rounds=1)
    state = read_json(output / 'hybrid_state.json')
    assert state['completed_rounds'] == [] and state['latest_checkpoint']['total_train_step'] == 2
    final = output / 'rounds/round-000/checkpoints/final.pt'
    before = load_alphazero_checkpoint(final)
    checksum = sha256(final)
    assert before.total_train_step == 4 and before.round_train_step == config.updates_per_round
    snapshot = read_json(output / 'rounds/round-000/replay_snapshot.json')
    archive_hashes = {game['path']: sha256(Path(game['path'])) for game in snapshot['games']}
    seen = []
    monkeypatch.setattr(hybrid_train, '_evaluation', lambda cfg, cp, anchor, stage: seen.append(cp))
    state = run_hybrid(config, output, resume=True, online_rounds=1)
    assert state['completed_rounds'] == [0] and state['latest_checkpoint']['total_train_step'] == 4
    assert seen == [final.resolve()]
    assert sha256(final) == checksum
    assert all(sha256(Path(path)) == digest for path, digest in archive_hashes.items())


def test_evaluation_resumes_existing_sidecar_and_does_not_gate_score(tmp_path, monkeypatch):
    from types import SimpleNamespace
    config = HybridConfig.from_dict(read_json(Path('configs/entity_attention_v1/hybrid_static_graph.json')))
    path = tmp_path / 'evaluation/mcts_vs_theta5000.json'
    Path(str(path) + '.games').mkdir(parents=True)
    calls = []

    def evaluate(cfg, output, *, resume, require_scored_pairs):
        calls.append(resume)
        assert require_scored_pairs
        assert cfg.policy_a.checkpoint.endswith('candidate.pt')
        assert cfg.policy_b.checkpoint.endswith('anchor.pt')
        assert cfg.policy_b.inference['actor_projection'] is False
        return SimpleNamespace(summary={'passed': True, 'win_rate_score': 0.0})

    monkeypatch.setattr(hybrid_train, 'run_fixed_evaluation', evaluate)
    hybrid_train._evaluation(config, tmp_path / 'candidate.pt', tmp_path / 'anchor.pt', tmp_path)
    assert calls == [True]


def test_fork_preserves_optimizer_rng_and_replaces_replay_on_resume(tmp_path, source, monkeypatch):
    parent_cfg = replace(small_config(source), new_games_per_round=2, updates_per_round=3)
    parent = tmp_path / 'parent'
    run_hybrid(parent_cfg, parent, online_rounds=1)
    hashes = {str(p.relative_to(parent)): sha256(p) for p in parent.rglob('*') if p.is_file()}
    final = parent / 'rounds/round-000/checkpoints/final.pt'
    parent_payload = load_alphazero_checkpoint(final)
    assert parent_payload.total_train_step == 5
    branch_cfg = replace(parent_cfg, run_id='fork-test', new_games_per_round=1, updates_per_round=2,
        mix_schedule=((1, 1),), selfplay=replace(parent_cfg.selfplay,
            search_config=asdict(MCTSConfig(simulations=4, determinizations=2))))

    def forbidden(*args, **kwargs):
        raise AssertionError('fork must not rerun offline fitting')

    monkeypatch.setattr(hybrid_train, 'run_offline', forbidden)
    continuous, segmented = tmp_path / 'continuous-fork', tmp_path / 'segmented-fork'
    run_hybrid(branch_cfg, continuous, fork_from=parent, fork_round=1, online_rounds=2)
    initialized = run_hybrid(branch_cfg, segmented, fork_from=parent, fork_round=1, online_rounds=0)
    assert initialized['completed_rounds'] == [] and initialized['online_round_offset'] == 1
    assert initialized['latest_checkpoint']['total_train_step'] == 5
    assert sha256(segmented / 'checkpoints/fork_initial.pt') == sha256(final)
    assert sha256(segmented / 'checkpoints/theta_5000.pt') == sha256(parent / 'checkpoints/theta_5000.pt')
    assert sha256(segmented / 'offline/fixed_validation.json') == sha256(parent / 'offline/fixed_validation.json')

    run_hybrid(branch_cfg, segmented, resume=True, online_rounds=2, max_steps_this_session=1)
    initial = load_alphazero_checkpoint(segmented / 'rounds/round-000/checkpoints/initial.pt')
    assert initial.total_train_step == 5 and initial.round_index == 2 and initial.round_steps_target == 2
    assert initial.sampling_rng_state == parent_payload.sampling_rng_state
    for key, params in parent_payload.optimizer_state_dict['state'].items():
        for name in params:
            torch.testing.assert_close(params[name], initial.optimizer_state_dict['state'][key][name], rtol=0, atol=0)
    first = read_json(segmented / 'rounds/round-000/replay_snapshot.json')
    assert {(g['round_index'], g['game_id']) for g in first['games']} == {(0, 1), (1, 0)}
    assert first['replacement']['inherited_games'] == first['replacement']['branch_games'] == 1

    state = run_hybrid(branch_cfg, segmented, resume=True, online_rounds=2)
    assert state['completed_rounds'] == [0, 1] and state['latest_checkpoint']['total_train_step'] == 9
    a = load_alphazero_checkpoint(continuous / 'rounds/round-001/checkpoints/final.pt')
    b = load_alphazero_checkpoint(state['latest_checkpoint']['path'])
    assert b.round_index == 3 and a.sampling_rng_state == b.sampling_rng_state
    assert torch.equal(a.torch_cpu_rng_state, b.torch_cpu_rng_state)
    for key in a.model_state_dict:
        torch.testing.assert_close(a.model_state_dict[key], b.model_state_dict[key], rtol=0, atol=0)
    for key, params in a.optimizer_state_dict['state'].items():
        for name in params:
            torch.testing.assert_close(params[name], b.optimizer_state_dict['state'][key][name], rtol=0, atol=0)
    second = read_json(segmented / 'rounds/round-001/replay_snapshot.json')
    assert {(g['round_index'], g['game_id']) for g in second['games']} == {(1, 0), (2, 0)}
    assert second['replacement']['inherited_games'] == 0 and second['replacement']['branch_games'] == 2
    usage = read_json(segmented / 'rounds/round-001/source_usage.json')
    assert usage['online_replacement']['actual_draws'] == {'inherited_online': 0, 'branch_online': 8}
    assert hashes == {str(p.relative_to(parent)): sha256(p) for p in parent.rglob('*') if p.is_file()}
    with pytest.raises(ValueError, match='identical frozen'):
        run_hybrid(replace(branch_cfg, updates_per_round=1), segmented, resume=True, online_rounds=3)
    with pytest.raises(ValueError, match='preserve.*learner'):
        prepare_fork(replace(branch_cfg, offline=replace(branch_cfg.offline,
            learner=replace(branch_cfg.offline.learner, learning_rate=2e-4))), parent)
    with pytest.raises(ValueError, match='completed human-numbered'):
        prepare_fork(branch_cfg, parent, 2)
    with pytest.raises(ValueError, match='only to initialize'):
        run_hybrid(branch_cfg, segmented, resume=True, fork_from=parent, online_rounds=3)


def test_fork_does_not_reintroduce_parent_evictions_and_preserves_holdout(tmp_path, source):
    cfg = small_config(source)
    # Three committed games, only the last two retained by the parent. Those
    # two include one train game and one historical validation game.
    split_base = next(base for base in range(14610000, 14611000) if
        derive_seed(LoopSeeds(0, base, 0).split_for(0), 0, 2, 'split') >= (1 << 63) and
        derive_seed(LoopSeeds(0, base, 0).split_for(0), 0, 1, 'split') < (1 << 63))
    parent_cfg = replace(cfg, new_games_per_round=3, replay_max_games=2,
        online_validation_positions=8, seeds=replace(cfg.seeds, split=split_base),
        selfplay=replace(cfg.selfplay, validation_fraction=.5))
    from splendor.selfplay_data import load_game
    parent = tmp_path / 'partial-parent'
    run_hybrid(parent_cfg, parent, online_rounds=1)
    branch_cfg = replace(parent_cfg, run_id='partial-fork', new_games_per_round=1,
        updates_per_round=1, replay_max_games=4, mix_schedule=((1, 1),),
        selfplay=replace(parent_cfg.selfplay, validation_fraction=0.))
    branch = tmp_path / 'partial-fork'
    run_hybrid(branch_cfg, branch, fork_from=parent, online_rounds=1)
    games = read_json(branch / 'rounds/round-000/replay_snapshot.json')['games']
    assert {(g['round_index'], g['game_id']) for g in games} == {(0, 1), (0, 2), (1, 0)}
    assert {(g['round_index'], g['game_id']) for g in read_json(branch / 'online_replay_index.json')['games']} == {
        (0, 1), (0, 2), (1, 0)}
    assert load_game(Path(games[-1]['path']))[1]['round_index'] == 1
    assert sha256(branch / 'fixed_online_validation.json') == sha256(parent / 'fixed_online_validation.json')
    assert read_json(branch / 'rounds/round-000/validation_after.json')['online_fixed']['positions'] == 8
    assert hybrid_train._load_fork(branch, read_json(branch / 'hybrid_state.json'), branch_cfg)['games']


def test_split_config_and_teacher_windows(source):
    formal = HybridConfig.from_dict(read_json(Path('configs/entity_attention_v1/hybrid_split_policy_value.json')))
    assert formal.to_dict() == HybridConfig.from_dict(formal.to_dict()).to_dict()
    assert formal.mix_schedule == () and formal.replay_max_games == 3000
    assert formal.split_training == SplitTrainingConfig()
    assert formal.new_games_per_round == 200 and formal.updates_per_round == 100
    assert formal.evaluation['pairs'] == 100
    assert not formal.evaluation['policy_a']['inference']['actor_projection']
    assert formal.evaluation['policy_b']['inference']['actor_projection']
    with pytest.raises(ValueError, match='pure online'):
        replace(formal, mix_schedule=((1, 32),))
    with pytest.raises(ValueError, match='nonempty history'):
        replace(formal, split_training=replace(formal.split_training, value_recent_positions=256))
    source_game = build_replay_index([ReplayRun(0, 'old-round-000', source[0] / 'rounds/round-000/selfplay')],
                                   max_games=2).split_games('train')[0]
    games = [replace(source_game, round_index=r, run_id=f'teacher-{r}', game_id=0, decision_count=8)
             for r in (10, 12, 20, 21, 24)]
    games.append(replace(source_game, round_index=24, game_id=1, split='validation'))
    index = ReplayIndex(games)
    recent, history, rounds = split_replay_pools(index, formal)
    assert rounds == [12, 20, 21, 24]
    assert {g.round_index for g in history.games} == {10}
    assert all(g.split == 'train' for pool in (recent, history) for g in pool.games)
    rng = DeterministicRng(221)
    drawn = draw_split_batch(recent, history, rng, 256, 128, inherited_run_ids={'teacher-10'})
    assert len(drawn.samples) == len(drawn.value_samples) == 256
    assert len(drawn.references) == 512
    assert all(ref.round_index in rounds for ref in drawn.references[:384])
    assert all(ref.round_index == 10 for ref in drawn.references[384:])
    assert drawn.source_positions['inherited_online'] == 128
    assert sum(v for k, v in drawn.source_positions.items() if k.startswith('policy_round_')) == 256
    assert sum(v for k, v in drawn.source_positions.items() if k.startswith('value_round_')) == 256
    again = draw_split_batch(recent, history, DeterministicRng(221), 256, 128,
                            inherited_run_ids={'teacher-10'})
    assert drawn.references == again.references
    with pytest.raises(ValueError, match='older inherited'):
        split_replay_pools(ReplayIndex(games[1:]), formal)


def test_split_value_does_not_change_policy_gradient_or_clipping(source):
    import copy
    from splendor.entity_attention import EntityAttention
    from splendor.alphazero_train import alphazero_loss_sums, collate_training_samples, _torch_batch
    index = build_replay_index([ReplayRun(0, 'old-round-000', source[0] / 'rounds/round-000/selfplay')], max_games=2)
    game = index.split_games('train')[0]
    samples = [index.cache.sample(game, i) for i in range(min(4, game.decision_count))]
    batch = collate_training_samples(samples)
    assert batch.policy_weights.sum() > 0
    extreme = replace(batch, value_targets=np.full_like(batch.value_targets, 1000.0))
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(82)
        model = EntityAttention()
        cfg = LearnerConfig(device='cpu', batch_size=4, history_batch_size=2)
        manual = AlphazeroLearner(copy.deepcopy(model), cfg, sampling_rng=DeterministicRng(8))
        split = AlphazeroLearner(copy.deepcopy(model), cfg, sampling_rng=DeterministicRng(8))
        tensors = _torch_batch(batch, manual.device)
        logits = manual.model.policy_logits(batch.decisions)
        loss = alphazero_loss_sums(logits, torch.zeros_like(tensors['value_targets']),
            candidate_mask=tensors['candidate_mask'], policy_targets=tensors['policy_targets'],
            value_targets=tensors['value_targets'], policy_weights=tensors['policy_weights'],
            value_weights=torch.zeros_like(tensors['value_weights'])).policy_loss
        loss.backward()
        policy_params = [p for name, p in manual.model.named_parameters() if not name.startswith('critic.')]
        torch.nn.utils.clip_grad_norm_(policy_params, 1.)
        manual.optimizer.step()
        record = split.train_split_step(batch, extreme, references=(), policy_max_grad_norm=1., value_max_grad_norm=1.)
        assert record['value_grad_norm'] > 10 * record['policy_grad_norm']
        assert record['value_grad_clipped'] and record['batch_size'] == 8
        for name, param in manual.model.named_parameters():
            other = dict(split.model.named_parameters())[name]
            if not name.startswith('critic.'):
                torch.testing.assert_close(param.grad, other.grad, rtol=0, atol=0)
                torch.testing.assert_close(param, other, rtol=0, atol=0)
            else:
                assert param.grad is None and other.grad is not None
        assert record['value_target_sq_sum'] == 1e6 * extreme.value_weights.sum()
        assert record['value_positions'] == np.count_nonzero(extreme.value_weights)
    finally:
        torch.set_num_threads(threads)


def test_split_fork_continuous_matches_partial_resume(tmp_path, source):
    parent_cfg = replace(small_config(source), mix_schedule=(), replay_max_games=3, updates_per_round=1)
    parent = tmp_path / 'split-parent'
    run_hybrid(parent_cfg, parent, online_rounds=3)
    parent_final = parent / 'rounds/round-002/checkpoints/final.pt'
    before = load_alphazero_checkpoint(parent_final)
    parent_hashes = {str(p): sha256(p) for p in parent.rglob('*') if p.is_file()}
    cfg = replace(parent_cfg, run_id='split-child', updates_per_round=2,
                  split_training=SplitTrainingConfig(recent_rounds=1, value_recent_positions=2))
    continuous, segmented = tmp_path / 'split-continuous', tmp_path / 'split-segmented'
    run_hybrid(cfg, continuous, fork_from=parent, fork_round=3, online_rounds=2)
    run_hybrid(cfg, segmented, fork_from=parent, fork_round=3, online_rounds=2, max_steps_this_session=1)
    initial = load_alphazero_checkpoint(segmented / 'rounds/round-000/checkpoints/initial.pt')
    assert initial.sampling_rng_state == before.sampling_rng_state
    assert torch.equal(initial.torch_cpu_rng_state, before.torch_cpu_rng_state)
    for key, params in before.optimizer_state_dict['state'].items():
        for name in params:
            torch.testing.assert_close(params[name], initial.optimizer_state_dict['state'][key][name], rtol=0, atol=0)
    run_hybrid(cfg, segmented, resume=True, online_rounds=2)
    a = load_alphazero_checkpoint(continuous / 'rounds/round-001/checkpoints/final.pt')
    b = load_alphazero_checkpoint(segmented / 'rounds/round-001/checkpoints/final.pt')
    assert a.sampling_rng_state == b.sampling_rng_state and torch.equal(a.torch_cpu_rng_state, b.torch_cpu_rng_state)
    for key in a.model_state_dict:
        torch.testing.assert_close(a.model_state_dict[key], b.model_state_dict[key], rtol=0, atol=0)
    for key, params in a.optimizer_state_dict['state'].items():
        for name in params:
            torch.testing.assert_close(params[name], b.optimizer_state_dict['state'][key][name], rtol=0, atol=0)
    usage = read_json(segmented / 'rounds/round-001/source_usage.json')
    assert usage['policy_recent']['draws'] == 8 and usage['value_recent']['draws'] == 4
    assert usage['value_history']['draws'] == 4
    assert usage['split_training']['recent_round_indices'] == [4]
    rows = [read_json(p) for p in (segmented / 'rounds/round-001/usage/policy_recent').glob('*.json')]
    assert all(row['batch_size'] == 4 for row in rows)
    assert parent_hashes == {str(p): sha256(p) for p in parent.rglob('*') if p.is_file()}
    with pytest.raises(ValueError, match='identical frozen'):
        run_hybrid(replace(cfg, split_training=replace(cfg.split_training, value_recent_positions=1)),
                   segmented, resume=True, online_rounds=3)


def test_external_evaluation_preserves_opponent_inference_and_hash(tmp_path, monkeypatch):
    from types import SimpleNamespace
    config = HybridConfig.from_dict(read_json(Path('configs/entity_attention_v1/hybrid_split_policy_value.json')))
    anchor = tmp_path / 'ppo.pt'
    anchor.write_bytes(b'fixed-ppo')
    raw = config.to_dict()
    raw['evaluation']['policy_b']['checkpoint'] = str(anchor)
    raw['evaluation_anchor'] = {'path': str(anchor), 'sha256': sha256(anchor)}
    config = HybridConfig.from_dict(raw)
    calls = []
    def evaluate(cfg, path, *, resume, require_scored_pairs):
        assert cfg.policy_b.name == config.evaluation['policy_b']['name']
        assert cfg.policy_b.inference['actor_projection']
        assert not cfg.policy_a.inference['actor_projection']
        assert len(cfg.game_seeds) == 100 and require_scored_pairs
        calls.append(path)
        return SimpleNamespace(summary={'passed': True, 'win_rate_score': .0})
    monkeypatch.setattr(hybrid_train, 'run_fixed_evaluation', evaluate)
    hybrid_train._evaluation(config, tmp_path / 'candidate.pt', anchor, tmp_path)
    assert calls == [tmp_path / 'evaluation/mcts_vs_fixed_anchor.json']
    anchor.write_bytes(b'changed')
    with pytest.raises(ValueError, match='checkpoint changed'):
        hybrid_train._evaluation(config, tmp_path / 'candidate.pt', anchor, tmp_path)


@pytest.mark.parametrize("label_workers", [1, 2])
def test_behavior_plus3_real_hybrid_training_and_partial_resume(tmp_path, source, label_workers):
    from splendor.policy_averaging import PolicyAveragingConfig
    parent_cfg = replace(small_config(source), mix_schedule=(), replay_max_games=3, updates_per_round=1,
        selfplay=replace(small_config(source).selfplay, search_config=asdict(MCTSConfig(simulations=5, determinizations=1))))
    parent = tmp_path / 'm4-parent'
    run_hybrid(parent_cfg, parent, online_rounds=3)
    cfg = replace(parent_cfg, run_id='m4-child', updates_per_round=2,
                  split_training=SplitTrainingConfig(recent_rounds=1, value_recent_positions=2),
                  policy_averaging=PolicyAveragingConfig(active_roots=4, games_per_block=1,
                      search_workers=label_workers, max_batch_size=4, local_microbatch=2))
    continuous, segmented = tmp_path / 'm4-continuous', tmp_path / 'm4-segmented'
    run_hybrid(cfg, continuous, fork_from=parent, fork_round=3, online_rounds=1)
    run_hybrid(cfg, segmented, fork_from=parent, fork_round=3, online_rounds=1, max_steps_this_session=1)
    labels = segmented / 'rounds/round-000/policy_labels/manifest.json'
    label_hash = sha256(labels)
    run_hybrid(cfg, segmented, resume=True, online_rounds=1)
    assert sha256(labels) == label_hash
    a = load_alphazero_checkpoint(continuous / 'rounds/round-000/checkpoints/final.pt')
    b = load_alphazero_checkpoint(segmented / 'rounds/round-000/checkpoints/final.pt')
    assert a.sampling_rng_state == b.sampling_rng_state
    for key in a.model_state_dict:
        torch.testing.assert_close(a.model_state_dict[key], b.model_state_dict[key], rtol=0, atol=0)
    for key, params in a.optimizer_state_dict['state'].items():
        for name in params:
            torch.testing.assert_close(params[name], b.optimizer_state_dict['state'][key][name], rtol=0, atol=0)
    usage = read_json(segmented / 'rounds/round-000/source_usage.json')
    assert usage['policy_averaging']['actual_policy_m4_fraction'] == 1
    assert usage['policy_recent']['draws'] == 8 and usage['value_history']['draws'] == 4
    before = read_json(segmented / 'rounds/round-000/validation_before.json')
    assert before['policy_target_scope'].startswith('current=M4')
    state = run_hybrid(cfg, segmented, resume=True, online_rounds=1)
    assert state['completed_rounds'] == [0]


def test_policy_window_changes_only_policy_draws(source):
    formal = HybridConfig.from_dict(read_json(Path('configs/entity_attention_v1/hybrid_split_policy_value.json')))
    assert 'policy_recent_rounds' not in formal.to_dict()['split_training']
    assert 'learning_rate_override' not in formal.to_dict()
    game = build_replay_index([ReplayRun(0, 'old-round-000', source[0] / 'rounds/round-000/selfplay')],
                             max_games=2).split_games('train')[0]
    index = ReplayIndex([replace(game, round_index=r, run_id=f't-{r}') for r in range(5)])
    cfg = replace(formal, split_training=replace(formal.split_training, policy_recent_rounds=1))
    recent, history, rounds = split_replay_pools(index, cfg)
    policy = hybrid_train.split_policy_pool(index, cfg, recent)
    assert rounds == [1, 2, 3, 4] and {g.round_index for g in policy.games} == {4}
    a = draw_split_batch(recent, history, DeterministicRng(19), 256, 128, policy_pool=policy)
    b = draw_split_batch(recent, history, DeterministicRng(19), 256, 128)
    assert all(ref.round_index == 4 for ref in a.references[:256])
    # randbelow sizes consume different RNG streams; assert value buckets, not identical draws.
    assert all(ref.round_index in rounds for ref in a.references[256:384])
    assert all(ref.round_index == 0 for ref in a.references[384:])
    assert a.source_positions['value_recent'] == b.source_positions['value_recent'] == 128
    assert a.source_positions['value_history'] == b.source_positions['value_history'] == 128
    with pytest.raises(ValueError, match='1..recent_rounds'):
        replace(cfg.split_training, policy_recent_rounds=5)


def test_lr_fork_records_effective_config_and_resume_is_identical(tmp_path, source):
    parent_cfg = replace(small_config(source), mix_schedule=(), replay_max_games=3, updates_per_round=1)
    parent = tmp_path / 'lr-parent'
    run_hybrid(parent_cfg, parent, online_rounds=3)
    before = load_alphazero_checkpoint(parent / 'rounds/round-002/checkpoints/final.pt')
    cfg = replace(parent_cfg, run_id='lr-child', updates_per_round=2, learning_rate_override=2.5e-5,
                  split_training=SplitTrainingConfig(recent_rounds=1, value_recent_positions=2,
                                                     policy_recent_rounds=1))
    a, b = tmp_path / 'lr-continuous', tmp_path / 'lr-resumed'
    run_hybrid(cfg, a, fork_from=parent, online_rounds=2)
    run_hybrid(cfg, b, fork_from=parent, online_rounds=2, max_steps_this_session=1)
    initial = load_alphazero_checkpoint(b / 'rounds/round-000/checkpoints/initial.pt')
    assert initial.learner_config['learning_rate'] == 2.5e-5
    assert all(g['lr'] == 2.5e-5 for g in initial.optimizer_state_dict['param_groups'])
    assert initial.sampling_rng_state == before.sampling_rng_state
    assert torch.equal(initial.torch_cpu_rng_state, before.torch_cpu_rng_state)
    for key in before.model_state_dict:
        torch.testing.assert_close(before.model_state_dict[key], initial.model_state_dict[key], rtol=0, atol=0)
    for key, params in before.optimizer_state_dict['state'].items():
        for name in params:
            torch.testing.assert_close(params[name], initial.optimizer_state_dict['state'][key][name], rtol=0, atol=0)
    run_hybrid(cfg, b, resume=True, online_rounds=2)
    x = load_alphazero_checkpoint(a / 'rounds/round-001/checkpoints/final.pt')
    y = load_alphazero_checkpoint(b / 'rounds/round-001/checkpoints/final.pt')
    assert x.learner_config == y.learner_config and x.sampling_rng_state == y.sampling_rng_state
    for key in x.model_state_dict:
        torch.testing.assert_close(x.model_state_dict[key], y.model_state_dict[key], rtol=0, atol=0)
    for key, params in x.optimizer_state_dict['state'].items():
        for name in params:
            torch.testing.assert_close(params[name], y.optimizer_state_dict['state'][key][name], rtol=0, atol=0)
    with pytest.raises(ValueError, match='identical frozen'):
        run_hybrid(replace(cfg, learning_rate_override=5e-5), b, resume=True, online_rounds=3)


def test_suite_real_cpu_pipeline_resumes_missing_completion_marker(tmp_path, source, monkeypatch):
    from types import SimpleNamespace
    from splendor import iteration_suite as suite
    parent_cfg = replace(small_config(source), mix_schedule=(), replay_max_games=32, updates_per_round=1)
    parent = tmp_path / 'suite-parent'
    run_hybrid(parent_cfg, parent, online_rounds=3)
    start = parent / 'rounds/round-002/checkpoints/final.pt'
    seed = next(base for base in range(1000, 2000) if all(
        0 < sum(derive_seed(LoopSeeds(0, base, 0).split_for(r), r, gid, 'split') < (1 << 63)
                for gid in range(4)) < 4 for r in (3, 4)))
    template = read_json(Path('configs/entity_attention_v1/hybrid_split_policy_value.json'))['evaluation']
    template['device'] = 'cpu'
    template['execution'].update(inference_mode='eager', active_games=2, max_batch_size=2)
    for spec in (template['policy_a'], template['policy_b']):
        spec['checkpoint'] = str(start)
        spec['inference'].update(mode='eager', actor_projection=False)
        spec['search_config']['simulations'] = 2
    cfg = replace(parent_cfg, new_games_per_round=4, updates_per_round=2,
        selfplay=replace(parent_cfg.selfplay, validation_fraction=.5), evaluation=template,
        split_training=SplitTrainingConfig(recent_rounds=2, value_recent_positions=2))
    base = tmp_path / 'base-config.json'
    write_json(base, cfg.to_dict())
    plan = suite.read(suite.ROOT / 'configs/entity_attention_v1/iteration_trend_20261008.json')
    plan.update(base_config=str(base), parent_run=str(parent), fork_round=3,
        start_checkpoint=str(start), start_sha256=sha256(start), rounds=2,
        artifact_root=str(tmp_path / 'artifacts'), runtime_root=str(tmp_path / 'runtime'),
        run_root=str(tmp_path / 'runs'), replay_cache_mib=16,
        arms=[{'id': 'steps50', 'updates_per_round': 1},
              {'id': 'lr_low', 'updates_per_round': 2, 'learning_rate_override': 2.5e-5},
              {'id': 'fresh50', 'updates_per_round': 1, 'policy_recent_rounds': 1}],
        collection_execution={},
        evaluation_execution={'backend': 'native_batched', 'active_games': 2, 'max_batch_size': 2,
                              'inference_mode': 'eager', 'max_batch_wait_ms': 0.},
        evaluation={'rounds': [1, 2], 'pairs': 1, 'bootstrap_resamples': 100})
    plan['replicates'] = [{**plan['replicates'][0], 'split_seed': seed}]
    cg = tmp_path / 'cg'
    cg.mkdir()
    for name, value in [('memory.max', str(16*suite.GIB)), ('memory.swap.max', '0'), ('memory.peak', '1024')]:
        (cg / name).write_text(value)
    monkeypatch.setattr(suite, 'cgroup_path', lambda: cg)
    calls = []
    interrupt = 'train'
    def stage(plan, task, key, *, round_number=None):
        nonlocal interrupt
        calls.append((task, key, round_number))
        if task == 'prepare':
            return
        args = SimpleNamespace(worker=task, key=key, round=round_number, memory_max=16*suite.GIB,
            manifest=Path(plan['artifact_root']) / 'manifest.json', process_identity=tmp_path / 'process.json')
        suite.worker(args)
        if interrupt == 'train' and task == 'train':
            # Reproduce interruption between committing the round and the suite marker.
            suite.completion_path(plan, task, key, round_number).unlink()
            interrupt = 'evaluate'
            raise KeyboardInterrupt
        if interrupt == 'evaluate' and task == 'evaluate' and key == 'seed1_steps50_r002':
            # The completed panel must refresh stale analysis even when the parent
            # is interrupted before it marks analysis dirty. No panel is replayed.
            interrupt = None
            raise KeyboardInterrupt
    monkeypatch.setattr(suite, 'run_stage', stage)
    with pytest.raises(KeyboardInterrupt):
        suite.run_suite(plan)
    final = Path(plan['run_root']) / 'seed1_steps50/rounds/round-000/checkpoints/final.pt'
    committed = sha256(final)
    with pytest.raises(KeyboardInterrupt):
        suite.run_suite(plan)
    assert suite.completion_path(plan, 'evaluate', 'seed1_steps50_r002').exists()
    suite.run_suite(plan)
    assert sha256(final) == committed
    analysis = suite.read(Path(plan['artifact_root']) / 'analysis.json')
    assert analysis['status'] == 'complete' and len(analysis['panels']) == 6
    assert len(analysis['training_diagnostics']) == 6
    for first, second in [('steps50', 'lr_low'), ('lr_low', 'fresh50')]:
        boundary = calls.index(('train', f'seed1_{second}', 1))
        assert calls.index(('evaluate', f'seed1_{first}_r002', None)) < boundary
    assert len(analysis['trends']['fresh50']) == 2
    assert analysis['trends']['fresh50'][1]['change_from_previous']['previous_round'] == 1
    completed_calls = len(calls)
    suite.run_suite(plan)
    assert len(calls) == completed_calls
    changed = dict(plan, analysis_seed=plan['analysis_seed'] + 1)
    with pytest.raises(ValueError, match='frozen suite'):
        suite.run_suite(changed)
