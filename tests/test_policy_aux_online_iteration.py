from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from test_random_budget_iteration import archive, driver_case, holdout_settings
from splendor import random_budget_iteration as iteration
from splendor.actor_critic import build_actor_critic
from splendor.alphazero_checkpoint import load_alphazero_checkpoint
from splendor.alphazero_recording import read_json, sha256, write_json
from splendor.alphazero_replay import PositionPool
from splendor.alphazero_train import AlphazeroLearner
from splendor.compact_game_cache import CompactGameCache
from splendor.continuation_auxiliary import ContinuationAuxiliaryConfig, continuation_targets, continuation_config_identity
from splendor.offline_train import make_index
from splendor.rng import DeterministicRng
from splendor.selfplay_data import collate_selfplay_samples

AUXILIARY = {"shortterm_value_weight": 0., "opponent_policy_weight": .15}
ROOT = Path(__file__).resolve().parents[1]


def policy_archive(root, round_index, game_id):
    row = archive(root, round_index, game_id, mixed=True)
    path = Path(row["path"])
    with np.load(path) as stored:
        arrays = dict(stored)
    metadata = json.loads(arrays["metadata_json"].tobytes())
    metadata.update(status="natural_terminal", terminal_result={"winner_ids": [0], "completed_turns": [2, 2]})
    # Four positive decisions keep the protocol's floor(25% per-game) subset
    # nonempty in this four-decision synthetic fixture.
    arrays.update(player_id=np.array([0, 1, 0, 1]), value_target=np.array([1, -1, 1, -1], np.int8),
        policy_weight=np.ones(4, np.float32), low_budget=np.zeros(4, bool),
        metadata_json=np.frombuffer(json.dumps(metadata).encode(), dtype=np.uint8))
    np.savez_compressed(path, **arrays)
    stat = path.stat()
    row.update(size=stat.st_size, mtime_ns=stat.st_mtime_ns, policy_samples=4)
    return row


@pytest.fixture
def policy_case(tmp_path, monkeypatch):
    cfg = driver_case(tmp_path, monkeypatch)
    # Two different generation sizes reproduce R31(4k)+R32-R37(10k), in miniature.
    rows = [policy_archive(tmp_path / f"history-{r}", r, g) for r, n in [(0, 2), (1, 4)] for g in range(n)]
    manifest = Path(cfg.history_manifest["path"])
    manifest.write_text("".join(json.dumps(g)+"\n" for g in rows))
    base = build_actor_critic({"architecture": "entity_attention_v1", "hidden": 16, "heads": 2,
                              "layers": 1, "ffn_hidden": 32})
    from splendor.continuation_auxiliary import add_continuation_heads
    model = add_continuation_heads(base, predict_opponent_policy=True, predict_shortterm_value=False)
    teacher = AlphazeroLearner(model, replace(cfg.learner, learning_rate=1e-4), sampling_rng=DeterministicRng(93))
    auxiliary = ContinuationAuxiliaryConfig(**AUXILIARY)
    index = make_index({"games": rows}, 2**20)
    index.cache = CompactGameCache(2**20, continuation_auxiliary=auxiliary)
    value = PositionPool(index.split_games("train"), index.cache, "train")
    samples = [value.sample(DeterministicRng(3))[0]]
    teacher.train_step(collate_selfplay_samples(samples), references=(), value_batch=collate_selfplay_samples(samples),
        continuation_auxiliary=auxiliary, continuation_labels=continuation_targets(samples, auxiliary))
    path = tmp_path / "policy_teacher.pt"
    teacher.save(path)
    cfg = replace(cfg, initial_checkpoint={"path": str(path), "sha256": sha256(path)},
        history_manifest={"path": str(manifest), "sha256": sha256(manifest)}, last_round=2,
        expected_games_per_round={"1": 2, "2": 4}, replay_max_games=6,
        initial_history_eviction=True, continuation_auxiliary=AUXILIARY, encoder_grad_norm_every_steps=1)
    generated = {}
    def collect(config, output, **_):
        if str(output) not in generated:
            output.mkdir(parents=True, exist_ok=True)
            generated[str(output)] = [policy_archive(output / "games", config.round_index, g) for g in range(2)]
            write_json(output / "manifest.json", {"config": config.to_dict()})
        return {"completed_games": 2, "missing_games": 0}
    def build(runs, **kwargs):
        games = generated[str(runs[0].root)]
        write_json(kwargs["state_path"], {"games": games})
        return make_index({"games": games}, 2**20)
    monkeypatch.setattr(iteration, "generate", collect)
    monkeypatch.setattr(iteration, "build_replay_index", build)
    return cfg


def test_policy_initial_overflow_preflight_and_legacy_serialization(policy_case):
    cfg = policy_case
    assert iteration.Config.from_dict(cfg.to_dict()) == cfg
    rows, check = iteration.check(cfg)
    assert len(rows) == 6 and check["first_pool_games"] == 6
    assert check["first_evicted_history_games"] == 2 and check["first_new_policy_fraction"] == pytest.approx(1/3)
    with pytest.raises(ValueError, match="exceeds replay capacity"):
        iteration.check(replace(cfg, initial_history_eviction=False))
    with pytest.raises(ValueError, match="heads differ"):
        iteration.check(replace(cfg, continuation_auxiliary={"shortterm_value_weight": .25, "opponent_policy_weight": .15}))
    with pytest.raises(ValueError, match="count mismatch"):
        iteration.check(replace(cfg, expected_games_per_round={"1": 3, "2": 3}))
    plain = read_json(ROOT / "configs/entity_attention_v1/random_budget_r31_10k_p60k_20261023.json")
    assert iteration.Config.from_dict(plain).to_dict() == plain


def test_policy_online_rounds_preserve_optimizer_rng_auxiliary_and_exact_resume(policy_case, tmp_path):
    cfg = policy_case
    root = tmp_path / "segmented"
    iteration.run(cfg, root, rounds=2, max_steps_this_session=1)
    initial = load_alphazero_checkpoint(root / "rounds/round-000/checkpoints/initial.pt")
    teacher = load_alphazero_checkpoint(cfg.initial_checkpoint["path"])
    assert initial.total_train_step == teacher.total_train_step
    assert initial.sampling_rng_state == teacher.sampling_rng_state
    torch.testing.assert_close(initial.torch_cpu_rng_state, teacher.torch_cpu_rng_state, rtol=0, atol=0)
    for key, value in teacher.model_state_dict.items():
        torch.testing.assert_close(value, initial.model_state_dict[key], rtol=0, atol=0)
    for pid, state in teacher.optimizer_state_dict["state"].items():
        for key, value in state.items():
            torch.testing.assert_close(value, initial.optimizer_state_dict["state"][pid][key], rtol=0, atol=0)
    assert initial.learner_config["learning_rate"] == 2e-5
    first = read_json(root / "rounds/round-000/replay_snapshot.json")
    assert len(first["games"]) == 6 and len(first["evicted_games"]) == 2
    assert first["historical_games"] == 4 and first["new_games"] == 2
    state = iteration.run(cfg, root, rounds=2)
    continuous = iteration.run(cfg, tmp_path / "continuous", rounds=2)
    a, b = [load_alphazero_checkpoint(s["latest_checkpoint"]["path"]) for s in (state, continuous)]
    assert a.total_train_step == b.total_train_step == teacher.total_train_step+4
    assert a.sampling_rng_state == b.sampling_rng_state
    for key, value in a.model_state_dict.items():
        torch.testing.assert_close(value, b.model_state_dict[key], rtol=0, atol=0)
    for pid, params in a.optimizer_state_dict["state"].items():
        for key, value in params.items():
            torch.testing.assert_close(value, b.optimizer_state_dict["state"][pid][key], rtol=0, atol=0)
    for path in root.glob("rounds/*/training_metrics.jsonl"):
        updates = [row for line in path.read_text().splitlines()
                   if (row := json.loads(line))["event"] == "train_step"]
        assert updates and sum(row["opponent_positions"] for row in updates) > 0
        assert all("opponent_policy_loss" in row and "shortterm_loss" not in row for row in updates)
        assert all("encoder_grad_norm_opponent_policy" in row for row in updates)
    assert not list(root.rglob("*.bin")) and not list(root.rglob("*.steps.jsonl"))
    assert a.replay_identity["continuation_auxiliary"] == continuation_config_identity(ContinuationAuxiliaryConfig(**AUXILIARY))
    with pytest.raises(ValueError, match="identical frozen"):
        iteration.run(replace(cfg, continuation_auxiliary={**AUXILIARY, "opponent_policy_weight": .2}), root, rounds=2)


def test_policy_auxiliary_six_round_holdout_and_boundary_recovery(policy_case, tmp_path):
    cfg = policy_case
    records = [json.loads(s) for s in Path(cfg.history_manifest["path"]).read_text().splitlines()]
    settings = {**holdout_settings(tmp_path, [g for g in records if g["split"] == "validation"]), "block_rounds": 6}
    enabled = replace(cfg, six_round_holdout=settings)
    state = iteration.run(enabled, tmp_path / "holdout", rounds=1)
    final = read_json(tmp_path / "holdout/rounds/round-000/fixed_holdout/step-000000002.json")
    assert final["metrics"]["positions"] > 0
    assert len(state["completed_rounds"]) == 1
    same = iteration.run(enabled, tmp_path / "holdout", rounds=1)
    assert same == state


def test_formal_policy_online_configuration_records_selected_start_history_and_r37_anchor():
    cfg = read_json(ROOT / "configs/entity_attention_v1/random_budget_policy140221_r31_r37_p60k_20261025.json")
    base = read_json(ROOT / "configs/entity_attention_v1/random_budget_r31_10k_p60k_20261023.json")
    assert cfg["initial_checkpoint"]["sha256"] == "e4edeb20e8caa3c1abec0146d3757594dd6e0428bd833badd0cb3915c2299a9f"
    assert cfg["evaluation_anchor"]["sha256"] == "611f4f4ed18f397b03c79bb3a2beae51e7a77a747883680d9189eed3052a1045"
    assert cfg["expected_games_per_round"] == {str(r): 4000 if r == 74 else 10000 for r in range(74, 81)}
    assert cfg["selfplay"] == base["selfplay"]
    assert cfg["learner"] == base["learner"]
    assert cfg["continuation_auxiliary"] == continuation_config_identity(ContinuationAuxiliaryConfig(**AUXILIARY))
    assert cfg["initial_history_eviction"] is True
    assert cfg["search_evaluation"]["execution"] == base["search_evaluation"]["execution"]
    assert cfg["search_evaluation"]["pairs"] == 500
    assert cfg["greedy_evaluation"]["pairs"] == 1000
