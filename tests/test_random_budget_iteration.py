from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
import json
import weakref

import numpy as np
import pytest
import torch

from splendor import random_budget_iteration as iteration
from splendor.alphazero_checkpoint import load_alphazero_checkpoint
from splendor.alphazero_loop import LoopSeeds, SelfPlaySection
from splendor.alphazero_recording import sha256
from splendor.alphazero_replay import PositionPool, ReplayGame, ReplayIndex
from splendor.alphazero_train import AlphazeroLearner, LearnerConfig
from splendor.compact_game_cache import CompactGameCache
from splendor.engine import GameEngine
from splendor.entity_attention import EntityAttention
from splendor.features import encode_decision
from splendor.features import FEATURE_SCHEMA_VERSION
from splendor.fixed_holdout import FixedHoldout, PROTOCOL as HOLDOUT_PROTOCOL, check_disjoint
from splendor.six_round_holdout import HoldoutSchedule, check_settings
from splendor.offline_train import _digest
from splendor.offline_train import make_index
from splendor.rng import DeterministicRng
from splendor.selfplay_data import STATE_FIELDS, collate_selfplay_samples


def test_policy_quotas_are_exact_cumulative_sixths():
    for numerator in range(1, 7):
        total = 0
        for step in range(1, 1001):
            total += iteration.new_policy_quota(step, 256, numerator * 4000, 24000)
            assert total == step * 256 * numerator // 6


def test_random_old_eviction_then_online_fifo_and_no_reentry():
    cfg = SimpleNamespace(replay_max_games=24, retention_seed=39)
    original = [{"round_index": r, "game_id": i} for r in range(5) for i in range(4)]
    keys = {iteration._key(g) for g in original}
    active, retired = original, set()
    for r in range(5, 13):
        new = [{"round_index": r, "game_id": i} for i in range(4)]
        selected, removed = iteration.retain_games(active, new, keys, cfg, r)
        assert (selected, removed) == iteration.retain_games(active, new, keys, cfg, r)
        assert len(selected) == 24
        assert not retired & {iteration._key(g) for g in selected}
        retired.update(iteration._key(g) for g in removed)
        if r == 5:
            assert not removed
        elif r <= 10:
            assert len(removed) == 4 and all(iteration._key(g) in keys for g in removed)
        else:
            assert {g["round_index"] for g in removed} == {r-6}
        active = selected
    assert not keys & {iteration._key(g) for g in active}


def archive(root, round_index, game_id, *, mixed=False):
    """Synthetic training arrays, used with a mocked collector (no games played)."""
    path = root / f"game-{game_id:06}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = encode_decision(GameEngine.new(42).decision())
    width = len(encoded.candidate_features)
    target = np.zeros(width * 4, dtype=np.float32)
    target[::width] = 1
    weights = np.array([1, 0, 0, 0] if mixed else [1, 1, 1, 1], dtype=np.float32)
    metadata = {"run_id": f"test-{round_index}", "game_id": game_id, "round_index": round_index,
                "split": "train" if game_id == 0 else "validation", "terminal_result": {"winner_ids": [0]},
                "feature_schema_version": FEATURE_SCHEMA_VERSION}
    arrays = {name: np.stack([getattr(encoded, name)] * 4) for name in STATE_FIELDS}
    arrays.update(candidate_features=np.concatenate([encoded.candidate_features] * 4),
        candidate_offsets=np.arange(5, dtype=np.int64)*width, policy_target=target, behavior_policy=target,
        root_network_prior=np.tile(np.full(width, 1/width, dtype=np.float32), 4),
        value_target=np.ones(4, dtype=np.int8), policy_weight=weights, value_weight=np.ones(4, dtype=np.float32),
        decision_index=np.arange(4, dtype=np.int32), player_id=np.zeros(4, dtype=np.int8),
        completed_player_turns=np.arange(4, dtype=np.int32),
        metadata_json=np.frombuffer(json.dumps(metadata).encode(), dtype=np.uint8))
    if mixed:
        arrays["low_budget"] = weights == 0
    np.savez_compressed(path, **arrays)
    stat = path.stat()
    return {**metadata, "decision_count": 4, "policy_samples": int(weights.sum()),
            "path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def test_fixed_policy_subset_sampling_and_unmodified_value_rng(tmp_path):
    old = archive(tmp_path / "old", 0, 0)
    new = archive(tmp_path / "new", 1, 0, mixed=True)
    cache = CompactGameCache(1024**2)
    index = make_index({"games": [old, new]}, 1024**2)
    index.cache = cache
    p_old = iteration.PolicyPool(index.games[:1], cache, fraction=0.25, seed=9)
    p_new = iteration.PolicyPool(index.games[1:], cache)
    again = iteration.PolicyPool(index.games[:1], cache, fraction=0.25, seed=9)
    assert len(p_old) == len(p_new) == 1
    assert p_old.subset_digest == again.subset_digest
    value = PositionPool(index.games, cache, "all")
    rng = DeterministicRng(11)
    reference_rng = DeterministicRng.restore(rng.snapshot())
    batch = iteration.draw_batch(p_old, p_new, value, rng, 256, 1, 4000, 24000, {(0, 0)})
    assert batch.source_positions["policy_new"] == 42
    assert all(s.arrays["policy_weight"][s.index] == 1 for s in batch.samples)
    assert any(s.arrays["policy_weight"][s.index] == 0 for s in batch.value_samples)
    expected = [value.sample(reference_rng)[1] for _ in range(256)]
    assert list(batch.references[256:]) == expected
    assert rng.snapshot() == reference_rng.snapshot()


def driver_case(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    original = [archive(tmp_path / "old", 0, i) for i in range(2)]
    manifest = tmp_path / "history.jsonl"
    manifest.write_text("".join(json.dumps(g)+'\n' for g in original))
    cfg0 = LearnerConfig(device="cpu", batch_size=4, history_batch_size=2, checkpoint_every_steps=1)
    torch.manual_seed(91)
    parent = AlphazeroLearner(EntityAttention(), cfg0, sampling_rng=DeterministicRng(39))
    index = make_index({"games": original}, 1024**2)
    sample = PositionPool(index.split_games("train"), index.cache, "old").sample(DeterministicRng(3))[0]
    parent.train_step(collate_selfplay_samples([sample]), references=())
    checkpoint = tmp_path / "parent.pt"
    parent.save(checkpoint)
    cfg = iteration.Config(run_id="tiny", initial_checkpoint={"path":str(checkpoint),"sha256":sha256(checkpoint)},
        history_manifest={"path":str(manifest),"sha256":sha256(manifest)}, first_round=1,last_round=1,
        expected_games_per_round=2,new_games_per_round=2,replay_max_games=4,updates_per_round=2,
        historical_policy_seed=8,retention_seed=9,validation_positions=4,validation_steps=(0,1,2),cache_bytes=1024**2,
        selfplay=SelfPlaySection(search_config={"simulations":16,"determinizations":1},execution={"backend":"native_cpu"},
            add_root_noise=True,temperature_schedule={},validation_fraction=.5,max_completed_turns=400,
            budget_randomization={"low_simulations":4,"low_probability":.75}),
        learner=replace(cfg0,learning_rate=2e-5),seeds=LoopSeeds(11,12,13))
    generated = {}
    def collect(config, root, **kwargs):
        if str(root) in generated:
            assert json.loads((root/'manifest.json').read_text())['config'] == json.loads(json.dumps(config.to_dict()))
            return {"completed_games":2,"missing_games":0}
        root.mkdir(parents=True, exist_ok=True)
        (root/'manifest.json').write_text(json.dumps({'config':config.to_dict()}))
        records = [archive(root/'games',config.round_index,i,mixed=True) for i in range(2)]
        generated[str(root)] = records
        return {"completed_games":2,"missing_games":0}
    def build(runs, **kwargs):
        records = generated[str(runs[0].root)]
        kwargs["state_path"].write_text(json.dumps({"games": records}))
        return make_index({"games": records}, 1024**2)
    monkeypatch.setattr(iteration,'generate',collect)
    monkeypatch.setattr(iteration,'build_replay_index',build)
    return cfg


def holdout_settings(tmp_path, records):
    manifest = {"protocol": HOLDOUT_PROTOCOL,
        "games": [{**g, "archive_sha256": sha256(Path(g["path"]))} for g in records]}
    manifest["digest"] = _digest(manifest)
    path = tmp_path / "holdout.json"
    path.write_text(json.dumps(manifest))
    return {"manifest": {"path": str(path), "sha256": sha256(path)}, "batch_size": 3}


def test_driver_releases_training_arrays_before_evaluation_and_next_collection(tmp_path, monkeypatch):
    cfg = driver_case(tmp_path, monkeypatch)
    caches, arrays, boundaries = [], [], []

    class TrackedCache(CompactGameCache):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            caches.append(self)

        def sample(self, *args, **kwargs):
            sample = super().sample(*args, **kwargs)
            arrays.append(weakref.ref(sample.arrays["value_target"]))
            return sample

    monkeypatch.setattr(iteration, "CompactGameCache", TrackedCache)
    template = {
        "policy_a": {"kind": "greedy_checkpoint", "name": "candidate", "checkpoint": "generated"},
        "policy_b": {"kind": "greedy_checkpoint", "name": "anchor", "checkpoint": cfg.initial_checkpoint["path"]},
        "pairs": 1, "seed_start": 1, "device": "cpu", "workers": 1,
    }
    cfg = replace(cfg, evaluation_anchor=cfg.initial_checkpoint,
                  greedy_evaluation=template, search_evaluation=template, search_evaluation_every_rounds=1)

    def assert_released():
        assert caches[-1].bytes == caches[-1].entries == 0
        assert arrays and all(ref() is None for ref in arrays)

    def panel(*args, **kwargs):
        assert_released()
        boundaries.append("evaluation")
        return SimpleNamespace(summary={"passed": True})

    collect = iteration.generate

    def generate(*args, **kwargs):
        if arrays:
            assert_released()
            boundaries.append("collection")
        return collect(*args, **kwargs)

    monkeypatch.setattr(iteration, "panel", panel)
    monkeypatch.setattr(iteration, "generate", generate)
    state = iteration.run(cfg, tmp_path / "released", rounds=2)
    assert boundaries == ["evaluation", "evaluation", "collection", "evaluation", "evaluation"]
    assert_released()
    final = load_alphazero_checkpoint(state["latest_checkpoint"]["path"])

    # The previous cache-retaining behavior is an exact training reference.
    retain = CompactGameCache.retain

    def retain_warm(self, keys):
        keys = tuple(keys)
        return retain(self, keys) if keys else 0

    monkeypatch.setattr(TrackedCache, "retain", retain_warm)
    monkeypatch.setattr(iteration, "generate", collect)
    monkeypatch.setattr(iteration, "panel", lambda *args, **kwargs: SimpleNamespace(summary={"passed": True}))
    reference = iteration.run(cfg, tmp_path / "warm", rounds=2)
    other = load_alphazero_checkpoint(reference["latest_checkpoint"]["path"])
    assert final.sampling_rng_state == other.sampling_rng_state
    torch.testing.assert_close(final.torch_cpu_rng_state, other.torch_cpu_rng_state, rtol=0, atol=0)
    for key, tensor in final.model_state_dict.items():
        torch.testing.assert_close(tensor, other.model_state_dict[key], rtol=0, atol=0)
    for pid, parameters in final.optimizer_state_dict["state"].items():
        for key, tensor in parameters.items():
            torch.testing.assert_close(tensor, other.optimizer_state_dict["state"][pid][key], rtol=0, atol=0)


def test_full_holdout_matches_joint_loss_and_preserves_model_rng(tmp_path):
    torch.set_num_threads(1)
    records = [archive(tmp_path/f"g{i}", 0, i, mixed=True) for i in (1, 2)]
    settings = holdout_settings(tmp_path, records)
    observer = FixedHoldout(settings, tmp_path/"run", "cpu")
    learner = AlphazeroLearner(EntityAttention(), LearnerConfig(device="cpu"), sampling_rng=DeterministicRng(3))
    learner.model.train()
    rng = torch.get_rng_state().clone()
    weights = {k: v.clone() for k, v in learner.model.state_dict().items()}
    result, rows = observer.evaluate(learner.model)
    assert result["metrics"]["positions"] == 8
    assert result["metrics"]["policy_weight"] == 2  # low-budget rows affect value only
    assert rows.shape == (2, 13)
    assert learner.model.training
    assert torch.equal(rng, torch.get_rng_state())
    for key, value in learner.model.state_dict().items():
        assert torch.equal(value, weights[key])
    assert max(map(abs, result["reference_verification"]["differences"].values())) < 5e-5
    with pytest.raises(ValueError, match="leaked"):
        check_disjoint(observer.manifest, [{**records[0], "split": "train"}])
    Path(records[0]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="archive changed"):
        FixedHoldout(settings, tmp_path/"other", "cpu").evaluate(learner.model)


def test_driver_holdout_backfill_is_complete_idempotent_and_training_neutral(tmp_path, monkeypatch):
    cfg = driver_case(tmp_path, monkeypatch)
    original = [json.loads(line) for line in Path(cfg.history_manifest["path"]).read_text().splitlines()]
    settings = holdout_settings(tmp_path, [g for g in original if g["split"] == "validation"])
    enabled = replace(cfg, fixed_holdout=settings)
    assert iteration.Config.from_dict(enabled.to_dict()) == enabled
    root = tmp_path/"run"
    # Start with the legacy driver contract, then enable and recover old nodes.
    iteration.run(cfg, root, rounds=1)
    iteration.run(enabled, root, rounds=2, max_steps_this_session=1)
    state = iteration.run(enabled, root, rounds=2)
    for r in range(2):
        assert len(list((root/f"rounds/round-{r:03d}/fixed_holdout").glob("*.json"))) == 3
    continuous = iteration.run(cfg, tmp_path/"plain", rounds=2)
    final = load_alphazero_checkpoint(state["latest_checkpoint"]["path"])
    other = load_alphazero_checkpoint(continuous["latest_checkpoint"]["path"])
    assert final.sampling_rng_state == other.sampling_rng_state
    for key, tensor in final.model_state_dict.items():
        torch.testing.assert_close(tensor, other.model_state_dict[key], rtol=0, atol=0)
    monkeypatch.setattr(FixedHoldout, "evaluate", lambda *args: pytest.fail("completed holdout rerun"))
    iteration.run(enabled, root, rounds=2)
    changed = json.loads(Path(settings["manifest"]["path"]).read_text())
    changed["description"] = "new panel"
    changed["digest"] = _digest({k: v for k, v in changed.items() if k != "digest"})
    replacement = tmp_path/"replacement.json"
    replacement.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="cannot change"):
        iteration.run(replace(enabled, fixed_holdout={**settings,
            "manifest": {"path": str(replacement), "sha256": sha256(replacement)}}), root, rounds=2)


def test_six_round_boundaries_freeze_all_original_validation_not_active_replay(tmp_path):
    original = archive(tmp_path/"old", 0, 1, mixed=True)
    settings = {**holdout_settings(tmp_path, [original]), "block_rounds": 6}
    output = tmp_path/"run"
    for r in range(1, 13):
        stage = output/f"rounds/round-{r-1:03d}"
        games = [archive(stage/"games", r, i, mixed=True) for i in range(2)]
        (stage/"collection_index.json").write_text(json.dumps({"games": games}))
    schedule = HoldoutSchedule(settings, output, "cpu", six_rounds=True)
    initial = schedule.current.manifest["digest"]
    for r in (1, 6):
        assert schedule.for_iteration(r).manifest["digest"] == initial
    second = schedule.for_iteration(7)
    assert second.manifest["source_iterations"] == list(range(1, 7))
    assert second.manifest["evaluated_iterations"] == list(range(7, 13))
    assert len(second.manifest["games"]) == 6
    assert {g["round_index"] for g in second.manifest["games"]} == set(range(1, 7))
    assert all(g["split"] == "validation" for g in second.manifest["games"])
    assert schedule.for_iteration(12) is second
    third = schedule.for_iteration(13)
    assert {g["round_index"] for g in third.manifest["games"]} == set(range(7, 13))
    resumed = HoldoutSchedule(settings, output, "cpu", six_rounds=True)
    assert resumed.for_iteration(7).manifest == second.manifest
    (output/"rounds/round-000/collection_index.json").write_text("changed")
    with pytest.raises(ValueError, match="collection index changed"):
        HoldoutSchedule(settings, output, "cpu", six_rounds=True).for_iteration(7)
    with pytest.raises(ValueError, match="block_rounds=6"):
        check_settings({**settings, "block_rounds": 5})


def test_online_driver_crosses_six_round_boundary_and_resumes_without_changing_training(tmp_path, monkeypatch):
    cfg = driver_case(tmp_path, monkeypatch)
    original = [json.loads(line) for line in Path(cfg.history_manifest["path"]).read_text().splitlines()]
    settings = {**holdout_settings(tmp_path, [g for g in original if g["split"] == "validation"]), "block_rounds": 6}
    enabled = replace(cfg, six_round_holdout=settings)
    assert iteration.Config.from_dict(enabled.to_dict()) == enabled
    with pytest.raises(ValueError, match="choose"):
        replace(enabled, fixed_holdout=settings)
    root = tmp_path/"run"
    iteration.run(enabled, root, rounds=6)
    iteration.run(enabled, root, rounds=7, max_steps_this_session=1)
    state = iteration.run(enabled, root, rounds=7)
    old = json.loads((root/"rounds/round-005/fixed_holdout/step-000000002.json").read_text())
    new_path = root/"fixed_holdout/blocks/iterations-007-012/rounds/round-006/fixed_holdout/step-000000002.json"
    new = json.loads(new_path.read_text())
    assert old["panel_digest"] != new["panel_digest"]
    assert len(list(new_path.parent.glob("*.json"))) == 3
    assert state["completed_rounds"][-1]["evaluations"]["fixed_holdout"]["path"] == str(new_path)
    plain = iteration.run(cfg, tmp_path/"plain", rounds=7)
    a = load_alphazero_checkpoint(state["latest_checkpoint"]["path"])
    b = load_alphazero_checkpoint(plain["latest_checkpoint"]["path"])
    assert a.sampling_rng_state == b.sampling_rng_state
    for key, value in a.model_state_dict.items():
        torch.testing.assert_close(value, b.model_state_dict[key], rtol=0, atol=0)
    monkeypatch.setattr(FixedHoldout, "evaluate", lambda *args: pytest.fail("completed block rerun"))
    iteration.run(enabled, root, rounds=7)


def test_driver_resume_and_optimizer_inheritance_without_launching_collection(tmp_path, monkeypatch):
    cfg = driver_case(tmp_path, monkeypatch)
    root = tmp_path/'segmented'
    state = iteration.run(cfg,root,rounds=2,max_steps_this_session=1)
    assert not state['completed_rounds']
    state = iteration.run(cfg,root,rounds=2)
    final = load_alphazero_checkpoint(state['latest_checkpoint']['path'])
    assert final.total_train_step == 5
    assert {int(v['step']) for v in final.optimizer_state_dict['state'].values()} == {5}
    assert final.learner_config['learning_rate'] == 2e-5
    assert len(state['completed_rounds']) == 2
    snapshot = json.loads((root/'rounds/round-001/replay_snapshot.json').read_text())
    assert snapshot['historical_games'] == 0 and snapshot['policy_new_fraction'] == 1
    continuous = iteration.run(cfg,tmp_path/'continuous',rounds=2)
    other = load_alphazero_checkpoint(continuous['latest_checkpoint']['path'])
    assert final.sampling_rng_state == other.sampling_rng_state
    for key,tensor in final.model_state_dict.items():
        torch.testing.assert_close(tensor,other.model_state_dict[key],rtol=0,atol=0)
    before = sha256(Path(state['latest_checkpoint']['path']))
    iteration.run(cfg,root,rounds=2)
    assert sha256(Path(state['latest_checkpoint']['path'])) == before
    with pytest.raises(ValueError,match='identical frozen algorithm'):
        iteration.run(replace(cfg,historical_policy_seed=10),root,rounds=2)


def test_sparse_checkpoint_union_and_midround_cadence_change_resume_exactly(tmp_path, monkeypatch):
    cfg = driver_case(tmp_path, monkeypatch)
    original = [json.loads(line) for line in Path(cfg.history_manifest["path"]).read_text().splitlines()]
    cfg = replace(cfg, updates_per_round=12, validation_steps=(0, 1, 3, 4, 8, 10, 12),
                  fixed_holdout=holdout_settings(tmp_path, [g for g in original if g["split"] == "validation"]))
    sparse = replace(cfg, learner=replace(cfg.learner, checkpoint_every_steps=2))
    interrupted = tmp_path / "interrupted"
    iteration.run(cfg, interrupted, rounds=1, max_steps_this_session=5)
    resumed = iteration.run(sparse, interrupted, rounds=1)
    continuous = tmp_path / "continuous"
    state = iteration.run(sparse, continuous, rounds=1)
    stage = continuous / "rounds/round-000"
    assert {p.name for p in (stage / "checkpoints").glob("*.pt")} == {
        "initial.pt", "final.pt", *[f"step-{s:09d}.pt" for s in (1, 2, 3, 4, 6, 8, 10, 12)]}
    assert {int(p.stem.split("-")[-1]) for p in (stage / "validation").glob("*.json")} == set(cfg.validation_steps)
    assert len(list((stage / "fixed_holdout").glob("*.json"))) == 7
    for root in (interrupted, continuous):
        assert not list(root.rglob("*.bin"))
        assert not list(root.rglob("*.steps.jsonl"))
        assert not list((root / "rounds/round-000/usage").rglob("*.metrics.jsonl"))
    a, b = [load_alphazero_checkpoint(s["latest_checkpoint"]["path"]) for s in (state, resumed)]
    assert a.learner_config["checkpoint_every_steps"] == b.learner_config["checkpoint_every_steps"] == 2
    assert a.sampling_rng_state == b.sampling_rng_state
    assert torch.equal(a.torch_cpu_rng_state, b.torch_cpu_rng_state)
    for key, tensor in a.model_state_dict.items():
        torch.testing.assert_close(tensor, b.model_state_dict[key], rtol=0, atol=0)
    for pid, saved in a.optimizer_state_dict["state"].items():
        for name, value in saved.items():
            torch.testing.assert_close(value, b.optimizer_state_dict["state"][pid][name], rtol=0, atol=0)
    with pytest.raises(ValueError, match="learner configuration"):
        AlphazeroLearner.resume(state["latest_checkpoint"]["path"], replace(sparse.learner, learning_rate=1e-4),
            replay_identity=a.replay_identity, round_index=a.round_index, round_steps_target=12)


def test_continuation_inherits_pool_optimizer_rng_and_absolute_rounds(tmp_path, monkeypatch):
    cfg = driver_case(tmp_path, monkeypatch)
    source = tmp_path/'parent'
    parent_state = iteration.run(cfg, source, rounds=2)
    state_path = source/'state.json'
    branch = replace(cfg, run_id='fork', initial_checkpoint=parent_state['latest_checkpoint'],
        continuation_state={'path':str(state_path), 'sha256':sha256(state_path)},
        selfplay=replace(cfg.selfplay, search_config={'simulations':16, 'determinizations':4},
            budget_randomization={'low_simulations':4, 'low_probability':.75, 'low_determinizations':1}))
    assert iteration.Config.from_dict(branch.to_dict()) == branch
    assert 'continuation_state' not in cfg.to_dict()
    assert iteration.check(branch)[1]['inherited_completed_rounds'] == 2
    fork_root = tmp_path/'branch'
    fork_state = iteration.run(branch, fork_root, rounds=3, max_steps_this_session=1)
    assert len(fork_state['completed_rounds']) == 2
    initial = load_alphazero_checkpoint(fork_root/'rounds/round-002/checkpoints/initial.pt')
    parent = load_alphazero_checkpoint(parent_state['latest_checkpoint']['path'])
    assert initial.total_train_step == parent.total_train_step
    assert initial.sampling_rng_state == parent.sampling_rng_state
    for key, values in initial.optimizer_state_dict['state'].items():
        for name, tensor in values.items():
            torch.testing.assert_close(tensor, parent.optimizer_state_dict['state'][key][name],rtol=0,atol=0)
    snapshot = json.loads((fork_root/'rounds/round-002/replay_snapshot.json').read_text())
    assert {g['round_index'] for g in snapshot['games']} == {2,3}
    assert {g['round_index'] for g in snapshot['evicted_games']} == {1}
    fork_state = iteration.run(branch, fork_root, rounds=3)
    assert len(fork_state['completed_rounds']) == 3
    assert load_alphazero_checkpoint(fork_state['latest_checkpoint']['path']).total_train_step == 7
    state_path.write_text(state_path.read_text()+'\n')
    with pytest.raises(ValueError, match='continuation state changed'):
        iteration.check(branch)


def tuning_case(cfg):
    execution = {"backend":"native_cuda_pool", "batch_interface":"native",
        "inference_mode":"compiled", "compile_mode":"native-cudagraph",
        "max_batch_size":500, "batch_buckets":[16,32,64,128,192,256,384,500],
        "pipeline_slots":1, "search_workers":12, "total_active_games":1000}
    old = replace(cfg, selfplay=replace(cfg.selfplay, execution=execution))
    updated = {**execution, "max_batch_size":768,
        "batch_buckets":[*execution["batch_buckets"],768], "pipeline_slots":2}
    new = replace(old, selfplay=replace(old.selfplay, execution=updated))
    return old, new


def test_execution_update_resumes_legacy_round_then_uses_new_settings(tmp_path, monkeypatch):
    old, new = tuning_case(driver_case(tmp_path, monkeypatch))
    root = tmp_path/'segmented'
    iteration.run(old, root, rounds=2, max_steps_this_session=1)
    # Simulate a round created by the old driver: manifest exists without the sidecar.
    stage = root/'rounds/round-000'
    (stage/'collection_config.json').unlink()
    manifest_before = (stage/'selfplay/manifest.json').read_bytes()
    frozen_before = (root/'config.json').read_bytes()
    state = iteration.run(new, root, rounds=2)
    first = json.loads((stage/'collection_config.json').read_text())
    second = json.loads((root/'rounds/round-001/collection_config.json').read_text())
    assert first['config']['execution']['max_batch_size'] == 500
    assert first['config']['execution']['pipeline_slots'] == 1
    assert first['requested_execution']['max_batch_size'] == 768
    assert second['config']['execution']['max_batch_size'] == 768
    assert second['config']['execution']['pipeline_slots'] == 2
    assert (stage/'selfplay/manifest.json').read_bytes() == manifest_before
    assert (root/'config.json').read_bytes() == frozen_before
    final = load_alphazero_checkpoint(state['latest_checkpoint']['path'])
    continuous = iteration.run(old, tmp_path/'continuous', rounds=2)
    other = load_alphazero_checkpoint(continuous['latest_checkpoint']['path'])
    assert final.total_train_step == other.total_train_step == 5
    assert final.sampling_rng_state == other.sampling_rng_state
    for key, tensor in final.model_state_dict.items():
        torch.testing.assert_close(tensor, other.model_state_dict[key], rtol=0, atol=0)
    for key, values in final.optimizer_state_dict['state'].items():
        for name, tensor in values.items():
            torch.testing.assert_close(tensor, other.optimizer_state_dict['state'][key][name], rtol=0, atol=0)
    invalid = replace(new, selfplay=replace(new.selfplay,
        execution={**new.selfplay.execution, 'search_workers':13}))
    with pytest.raises(ValueError, match='identical frozen algorithm'):
        iteration.run(invalid, root, rounds=2)


def test_running_driver_reloads_tuning_at_next_round_boundary(tmp_path, monkeypatch):
    old, new = tuning_case(driver_case(tmp_path, monkeypatch))
    path = tmp_path/'requested.json'
    path.write_text(json.dumps(old.to_dict()))
    collect = iteration.generate
    seen = []
    def update_during_collection(config, root, **kwargs):
        seen.append((config.execution.max_batch_size, config.execution.pipeline_slots))
        result = collect(config, root, **kwargs)
        if config.round_index == old.last_round:
            path.write_text(json.dumps(new.to_dict()))
        return result
    monkeypatch.setattr(iteration, 'generate', update_during_collection)
    root = tmp_path/'live'
    state = iteration.run(old, root, rounds=2, config_path=path)
    assert len(state['completed_rounds']) == 2
    assert seen == [(500,1),(768,2)]
    revisions = sorted((root/'source').glob('*/config-iteration-*.json'))
    assert [json.loads(p.read_text())['selfplay']['execution']['max_batch_size'] for p in revisions] == [500,768]


def test_collection_freeze_rejects_search_budget_changes(tmp_path):
    checkpoint = tmp_path/'checkpoint.pt'
    checkpoint.touch()
    raw = {'protocol':'mcts-selfplay-v2', 'run_id':'test', 'round_index':0,
        'checkpoint':str(checkpoint), 'num_games':2, 'seed':1,
        'search_config':{'simulations':16}, 'execution':{'backend':'native_cpu'}}
    iteration._collection_config(raw, tmp_path/'round')
    with pytest.raises(ValueError, match='identical frozen algorithm'):
        iteration._collection_config({**raw, 'search_config':{'simulations':32}}, tmp_path/'round')


def test_search_panel_size_execution_update_preserves_training_identity():
    import copy
    path = Path('configs/entity_attention_v1/random_budget_37423_p6k_v24k_20261019.json')
    new = json.loads(path.read_text())
    old = copy.deepcopy(new)
    old['search_evaluation']['pairs'] = 200
    old['search_evaluation']['execution'] = {'backend': 'native_batched', 'active_games': 100}
    for name in ('policy_a', 'policy_b'):
        old['search_evaluation'][name]['inference'] = {'mode': 'compiled'}
    assert iteration._algorithm_config(old) == iteration._algorithm_config(new)
    invalid = copy.deepcopy(new)
    invalid['search_evaluation']['policy_a']['search_config']['simulations'] = 128
    assert iteration._algorithm_config(old) != iteration._algorithm_config(invalid)
    assert new['search_evaluation']['pairs'] == 500
    assert new['search_evaluation']['execution'] == {**new['selfplay']['execution'], 'pipeline_slots': 1}
    assert new['selfplay']['execution']['pipeline_slots'] == 2
