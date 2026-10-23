from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json

import pytest
import torch

from test_random_budget_iteration import archive, driver_case
from splendor import random_budget_iteration as iteration
from splendor.alphazero_checkpoint import load_alphazero_checkpoint
from splendor.alphazero_recording import read_json, sha256
from splendor.offline_train import make_index


def warmup_case(tmp_path, monkeypatch):
    cfg = driver_case(tmp_path, monkeypatch)
    manifest = Path(cfg.history_manifest["path"])
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    rows.extend(archive(tmp_path / "more_history", 1, game_id) for game_id in range(2))
    manifest.write_text("".join(json.dumps(row)+"\n" for row in rows))
    cfg = replace(cfg, history_manifest={"path": str(manifest), "sha256": sha256(manifest)},
        last_round=2, new_games_per_round=1, replay_max_games=6)
    generated = {}

    def collect(config, root, **kwargs):
        if str(root) in generated:
            assert read_json(root / "manifest.json")["config"] == json.loads(json.dumps(config.to_dict()))
        else:
            root.mkdir(parents=True, exist_ok=True)
            (root / "manifest.json").write_text(json.dumps({"config": config.to_dict()}))
            generated[str(root)] = [archive(root / "games", config.round_index, config.round_index % 2, mixed=True)]
        return {"completed_games": 1, "missing_games": 0}

    monkeypatch.setattr(iteration, "generate", collect)
    monkeypatch.setattr(iteration, "build_replay_index", lambda runs, **_: make_index(
        {"games": generated[str(runs[0].root)]}, 1024**2))
    return cfg


def test_preflight_accepts_initial_gap_and_uses_actual_first_pool_fraction(tmp_path, monkeypatch):
    cfg = warmup_case(tmp_path, monkeypatch)
    records, preflight = iteration.check(cfg)
    assert len(records) == 4 and preflight["first_pool_games"] == 5
    assert preflight["pool_capacity"] == 6 and preflight["first_new_policy_fraction"] == .2
    assert iteration.Config.from_dict(cfg.to_dict()) == cfg
    with pytest.raises(ValueError, match="exceeds replay capacity"):
        iteration.check(replace(cfg, replay_max_games=4))


def test_first_two_rounds_grow_to_capacity_and_resume_exactly(tmp_path, monkeypatch):
    cfg = warmup_case(tmp_path, monkeypatch)
    root = tmp_path / "segmented"
    initial_state = iteration.run(cfg, root, rounds=2, max_steps_this_session=1)
    assert not initial_state["completed_rounds"]
    first = read_json(root / "rounds/round-000/replay_snapshot.json")
    assert len(first["games"]) == 5 and not first["evicted_games"]
    assert first["historical_games"] == 4 and first["new_games"] == 1
    assert first["policy_new_fraction"] == .2
    parent = load_alphazero_checkpoint(cfg.initial_checkpoint["path"])
    initial = load_alphazero_checkpoint(root / "rounds/round-000/checkpoints/initial.pt")
    assert initial.total_train_step == parent.total_train_step
    assert initial.sampling_rng_state == parent.sampling_rng_state
    torch.testing.assert_close(initial.torch_cpu_rng_state, parent.torch_cpu_rng_state, rtol=0, atol=0)
    for pid, state in parent.optimizer_state_dict["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(value, initial.optimizer_state_dict["state"][pid][name], rtol=0, atol=0)
    state = iteration.run(cfg, root, rounds=2)
    second = read_json(root / "rounds/round-001/replay_snapshot.json")
    assert len(second["games"]) == 6 and not second["evicted_games"]
    assert second["historical_games"] == 4 and second["new_games"] == 2
    assert second["policy_new_fraction"] == pytest.approx(1/3)
    continuous = iteration.run(cfg, tmp_path / "continuous", rounds=2)
    a, b = [load_alphazero_checkpoint(s["latest_checkpoint"]["path"]) for s in (state, continuous)]
    assert a.total_train_step == b.total_train_step == parent.total_train_step+4
    assert a.sampling_rng_state == b.sampling_rng_state
    for key, value in a.model_state_dict.items():
        torch.testing.assert_close(value, b.model_state_dict[key], rtol=0, atol=0)
    for pid, saved in a.optimizer_state_dict["state"].items():
        for name, value in saved.items():
            torch.testing.assert_close(value, b.optimizer_state_dict["state"][pid][name], rtol=0, atol=0)


def test_continuation_from_unfilled_pool_keeps_history_until_overflow(tmp_path, monkeypatch):
    cfg = warmup_case(tmp_path, monkeypatch)
    source = tmp_path / "parent"
    parent = iteration.run(cfg, source, rounds=1)
    state_path = source / "state.json"
    fork = replace(cfg, run_id="warm-fork", initial_checkpoint=parent["latest_checkpoint"],
        continuation_state={"path": str(state_path), "sha256": sha256(state_path)})
    preflight = iteration.check(fork)[1]
    assert preflight["inherited_completed_rounds"] == 1
    assert preflight["first_pool_games"] == 6
    assert preflight["first_new_policy_fraction"] == pytest.approx(1/3)
    iteration.run(fork, tmp_path / "fork", rounds=2)
    snapshot = read_json(tmp_path / "fork/rounds/round-001/replay_snapshot.json")
    assert len(snapshot["games"]) == 6 and not snapshot["evicted_games"]
    assert snapshot["historical_games"] == 4 and snapshot["policy_new_fraction"] == pytest.approx(1/3)


def test_pool_warmup_then_random_history_eviction_and_online_fifo():
    cfg = SimpleNamespace(replay_max_games=6, retention_seed=39)
    original = [{"round_index": r, "game_id": g} for r in range(2) for g in range(2)]
    keys = {iteration._key(g) for g in original}
    active, retired = original, set()
    for local_round, source_round in enumerate(range(2, 10), start=1):
        incoming = [{"round_index": source_round, "game_id": 0}]
        active, removed = iteration.retain_games(active, incoming, keys, cfg, source_round)
        assert len(active) == min(6, 4+local_round)
        assert sum(iteration._key(g) in keys for g in active) == max(0, 4-max(0, local_round-2))
        if local_round <= 2:
            assert not removed
        elif local_round <= 6:
            assert len(removed) == 1 and iteration._key(removed[0]) in keys
        else:
            assert [g["round_index"] for g in removed] == [source_round-6]
        assert not retired & {iteration._key(g) for g in active}
        retired.update(iteration._key(g) for g in removed)


@pytest.mark.parametrize("new,total", [(10000, 50000), (20000, 60000), (30000, 60000),
                                        (40000, 60000), (50000, 60000), (60000, 60000)])
def test_3000_step_policy_quotas_use_actual_pool_size(new, total):
    drawn = 0
    for step in range(1, 3001):
        drawn += iteration.new_policy_quota(step, 256, new, total)
        assert drawn == step*256*new//total


def test_formal_r31_configuration_preserves_the_iteration_protocol():
    root = Path(__file__).resolve().parents[1]
    base = read_json(root / "configs/entity_attention_v1/random_budget_r20_s256n4_noble12_20261022.json")
    config = read_json(root / "configs/entity_attention_v1/random_budget_r31_10k_p60k_20261023.json")
    assert config["initial_checkpoint"]["sha256"] == "c57a2a8a7721fe65de90dbc1336483f5d0358d10c4ff4d5c893986e88e588f6c"
    assert config["initial_checkpoint"]["path"].endswith("round-030/checkpoints/final.pt")
    assert (config["first_round"], config["last_round"], config["expected_games_per_round"]) == (65, 74, 4000)
    assert (config["new_games_per_round"], config["replay_max_games"], config["updates_per_round"]) == (10000, 60000, 3000)
    assert config["validation_steps"] == [0, 100, 250, 500, 1000, 2000, 3000]
    assert config["learner"]["learning_rate"] == 2e-5
    assert config["cache_bytes"] == 16 * 1024**3
    latest = read_json(root / "artifacts/collection_tuning_20261022/configs/full_attention_best_4000.json")["execution"]
    assert config["selfplay"]["execution"] == latest
    assert config["search_evaluation"]["execution"] == dict(latest, pipeline_slots=1)
    config["selfplay"]["execution"] = base["selfplay"]["execution"]
    config["search_evaluation"]["execution"] = base["search_evaluation"]["execution"]
    base.pop("continuation_state")
    for key in ("run_id", "initial_checkpoint", "history_manifest", "first_round", "last_round",
                "expected_games_per_round", "new_games_per_round", "replay_max_games", "updates_per_round",
                "validation_steps", "cache_bytes"):
        base.pop(key)
        config.pop(key)
    assert config == base
