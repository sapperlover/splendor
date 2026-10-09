from dataclasses import asdict, replace
from pathlib import Path
import weakref

import numpy as np
import pytest
import torch

from splendor import policy_averaging as averaging
from splendor import iteration_suite as suite
from splendor.alphazero_recording import read_json, sha256
from splendor.alphazero_replay import ReplayRun, build_replay_index
from splendor.alphazero_train import AlphazeroLearner, LearnerConfig
from splendor.entity_attention import EntityAttention
from splendor.hybrid_train import HybridConfig
from splendor.mcts_policy import MCTSConfig
from splendor.rng import DeterministicRng
from splendor.selfplay_data import load_game
from splendor.selfplay_fast import ExecutionConfig, FastSelfPlayConfig, generate
from splendor.multiworker_gpu import FixedPoolOwner


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    root = tmp_path_factory.mktemp("averaged-policy")
    torch.set_num_threads(1)
    torch.manual_seed(91)
    checkpoint = root / "teacher.pt"
    AlphazeroLearner(EntityAttention(), LearnerConfig(device="cpu"), sampling_rng=DeterministicRng(0)).save(checkpoint)
    config = FastSelfPlayConfig(protocol="mcts-selfplay-v2", run_id="m4-test", round_index=3,
        checkpoint=str(checkpoint), num_games=2, seed=1717, search_config=MCTSConfig(simulations=5, determinizations=1),
        execution=ExecutionConfig(backend="native_cpu", engine_mode="fast", batch_interface="native", max_batch_size=4),
        validation_fraction=.5, split_seed=123)
    collection = root / "round-000/selfplay"
    generate(config, collection)
    return root, collection, config


def test_behavior_plus_three_preserves_archives_and_targets(source, monkeypatch):
    root, collection, config = source
    hashes = {p.name: sha256(p) for p in (collection / "games").glob("*.npz")}
    output = root / "round-000/policy_labels"
    protocol = averaging.PolicyAveragingConfig(active_roots=4, games_per_block=2)
    last_jobs = []
    original_runner = averaging.NativeBatchRunner
    class ObservedRunner(original_runner):
        def run(self, jobs, **kwargs):
            last_jobs[:] = [weakref.ref(job) for job in jobs]
            return super().run(jobs, **kwargs)
    monkeypatch.setattr(averaging, "NativeBatchRunner", ObservedRunner)
    def confirm_released(*_):
        assert all(ref() is None for ref in last_jobs)
    manifest = averaging.generate_labels(collection, output, protocol, device="cpu",
                                         progress_callback=confirm_released)
    assert manifest["samples"] == 4 and manifest["extra_searches"] == 3
    assert manifest["behavior_search_reused"]
    index = build_replay_index([ReplayRun(3, config.run_id, collection)], max_games=2)
    store = averaging.AveragedTargetStore(index.games, required_run_ids={config.run_id})
    expected_simulations = 0
    for game in index.games:
        arrays, _ = load_game(game.path)
        with np.load(output / "games" / game.path.name) as z:
            np.testing.assert_array_equal(z["targets"][0].astype(np.float32), arrays["policy_target"])
            np.testing.assert_array_equal(z["target_mean"], z["targets"].mean(axis=0).astype(np.float32))
            for i in np.flatnonzero(arrays["searchable"]):
                a,b = arrays["candidate_offsets"][i:i+2]
                np.testing.assert_array_equal(z["extra_root_visit_counts"][:,a:b].sum(axis=1), [5,5,5])
                assert len(set(z["extra_world_seeds"][:,i])) == len(set(z["extra_noise_seeds"][:,i])) == 3
            sample = index.cache.sample(game, 0)
            transformed = store.transform(sample)
            assert transformed is not sample
            np.testing.assert_array_equal(transformed.arrays["policy_target"], z["target_mean"])
            assert all(transformed.arrays[k] is sample.arrays[k] for k in sample.arrays if k != "policy_target")
        expected_simulations += 3*5*int(arrays["searchable"].sum())
    assert manifest["extra_simulations"] == expected_simulations
    assert hashes == {p.name: sha256(p) for p in (collection / "games").glob("*.npz")}
    assert averaging.generate_labels(collection, output, protocol, device="cpu") == manifest
    with pytest.raises(ValueError, match="identity changed"):
        averaging.generate_labels(collection, output, replace(protocol, noise_seed=99), device="cpu")


def test_relabel_interrupt_reuses_committed_search_chunks(source, tmp_path, monkeypatch):
    _, collection, _ = source
    protocol = averaging.PolicyAveragingConfig(active_roots=4, games_per_block=2)
    called = []
    def stop(*values):
        called.append(values)
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        averaging.generate_labels(collection, tmp_path, protocol, device="cpu", progress_callback=stop)
    part = next((tmp_path / "work.games").glob("*.npz"))
    checksum, mtime = sha256(part), part.stat().st_mtime_ns
    assert not (tmp_path / "manifest.json").exists()
    manifest = averaging.generate_labels(collection, tmp_path, protocol, device="cpu")
    assert manifest["status"] == "complete"
    assert sha256(part) == checksum and part.stat().st_mtime_ns == mtime


def test_missing_m4_supervision_is_an_error_for_new_teachers(source, tmp_path):
    _, collection, config = source
    index = build_replay_index([ReplayRun(3,config.run_id,collection)], max_games=2)
    game = replace(index.games[0], path=tmp_path / "selfplay/games/game-000000.npz")
    with pytest.raises(ValueError, match="no completed"):
        averaging.AveragedTargetStore([game], required_run_ids={config.run_id})
    store = averaging.AveragedTargetStore([game])
    sample = index.cache.sample(index.games[0],0)
    assert store.transform(sample) is sample  # Explicitly allowed inherited single-target teacher.


def test_single_m4_plan_has_five_paired_battles_and_per_round_probe():
    plan = suite.read(suite.ROOT / "configs/entity_attention_v1/iteration_avg4_behavior_20261009.json")
    suite.validate_plan(plan)
    configs = suite.arm_configs(plan)
    assert len(configs) == 1
    config = HybridConfig.from_dict(configs["seed1_avg4"])
    assert config.to_dict() == configs["seed1_avg4"]
    assert config.new_games_per_round == 200 and config.updates_per_round == 100
    assert config.offline.learner.learning_rate == 1e-4 and config.replay_max_games == 3000
    assert config.split_training.recent_rounds == 4 and config.split_training.value_recent_positions == 128
    assert config.policy_averaging.samples == 4
    assert config.policy_averaging.search_workers == 8
    assert config.policy_averaging.active_roots == 256
    jobs = list(suite.evaluation_jobs(plan))
    assert len(jobs) == 5 and sum(j["pairs"] for j in jobs) == 500
    assert [j["round"] for j in jobs] == [2,4,8,12,16]
    schedule = list(suite.stage_schedule(plan,jobs))
    assert sum(t == "probe" for t,_,_ in schedule) == 16
    for n in range(1,17):
        assert schedule.index(("train","seed1_avg4",n)) < schedule.index(("probe","seed1_avg4",n))
    for job in jobs:
        assert schedule.index(("probe","seed1_avg4",job["round"])) < schedule.index(("evaluate",job["id"],None))
    assert plan["resources"]["memory_max_gib"] is None


def test_fixed_root_pool_matches_single_process_and_has_cpu_only_workers(source, tmp_path, monkeypatch):
    _, collection, _ = source
    monkeypatch.setattr(FixedPoolOwner, "model", None, raising=False)
    monkeypatch.setattr(averaging, "FrozenSearchInference", lambda *a, **k: FixedPoolOwner())
    serial = averaging.PolicyAveragingConfig(active_roots=4, games_per_block=2)
    pooled = replace(serial, search_workers=2, max_batch_size=3, local_microbatch=2)
    averaging.generate_labels(collection, tmp_path / "serial", serial, device="cpu")
    manifest = averaging.generate_labels(collection, tmp_path / "pool", pooled, device="cpu")
    assert manifest["execution"]["search_workers"] == 2
    for p in (tmp_path / "serial/games").glob("*.npz"):
        with np.load(p) as expected, np.load(tmp_path / "pool/games" / p.name) as actual:
            for key in expected.files:
                np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
    stats = read_json(tmp_path / "pool" / manifest["execution"]["statistics"][0])["owner"]
    assert stats["actual_search_workers"] == 2
    assert stats["actual_active_limit"] == 4
    assert len({w["pid"] for w in stats["workers"]}) == 2
    assert all(not w["cuda_initialized"] for w in stats["workers"])


def test_fixed_root_pool_interrupt_reuses_committed_roots(source, tmp_path, monkeypatch):
    _, collection, _ = source
    monkeypatch.setattr(FixedPoolOwner, "model", None, raising=False)
    monkeypatch.setattr(averaging, "FrozenSearchInference", lambda *a, **k: FixedPoolOwner())
    protocol = averaging.PolicyAveragingConfig(active_roots=4, search_workers=2,
        max_batch_size=3, local_microbatch=2, games_per_block=2)
    def stop(block, repeat, done, total):
        if block == -1:
            raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        averaging.generate_labels(collection, tmp_path, protocol, device="cpu", progress_callback=stop)
    committed = {p: (sha256(p), p.stat().st_mtime_ns) for p in (tmp_path / "roots.games").glob("*.npz")}
    assert committed and not (tmp_path / "manifest.json").exists()
    averaging.generate_labels(collection, tmp_path, protocol, device="cpu")
    assert all((sha256(p), p.stat().st_mtime_ns) == previous for p, previous in committed.items())
