from dataclasses import replace
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from test_random_budget_iteration import archive
from splendor.alphazero_checkpoint import load_alphazero_checkpoint
from splendor.alphazero_recording import read_json, sha256, write_json
from splendor.alphazero_train import LearnerConfig
from splendor.offline_mixed_budget import (MixedBudgetOfflineConfig, make_batch_drawer,
    prepare_inputs, run, select_checkpoint)
from splendor.offline_scratch import initialize_learner
from splendor.rng import DeterministicRng

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True, scope="module")
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def mixed(tmp_path, monkeypatch):
    import splendor.offline_objectives as entry
    monkeypatch.setattr(entry, "source_identity", lambda *_: {"test": True})
    rows = [archive(tmp_path / f"round-{r}", r, g, mixed=True) for r in range(2) for g in range(2)]
    manifest = tmp_path / "pool.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return MixedBudgetOfflineConfig(manifest=str(manifest), manifest_sha256=sha256(manifest),
        first_round=1, last_round=2, expected_games_per_round=2, draw_exposures=2.5,
        cache_mib=8, validation_positions=8, validation_percent_step=25,
        model_config={"architecture": "entity_set_mlp_v1", "card_hidden": 8, "noble_hidden": 4,
            "state_hidden": 16, "residual_blocks": 2, "actor_hidden": 8, "critic_hidden": 8,
            "market_relations": True, "action_bottleneck": 4, "multiplicative_interaction": True,
            "action_cross_hidden": 8, "action_cross_heads": 2},
        learner=LearnerConfig(device="cpu", batch_size=4, history_batch_size=2,
            learning_rate=1e-4, validation_batch_size=8, checkpoint_every_steps=2))


def test_prepare_and_sampling_keep_complete_policy_batch_and_all_value_steps(mixed, tmp_path):
    index, value, _, _, plan = prepare_inputs(mixed, tmp_path / "fit")
    assert plan["steps_target"] == 5 and plan["validation_steps"] == [2, 3, 4, 5]
    assert plan["splits"]["train"] == {"games": 2, "positions": 8, "policy_positions": 2}
    assert plan["sampling"]["value_draw_exposure"] == 2.5
    assert plan["sampling"]["policy_draw_exposure"] == 10
    draw = make_batch_drawer(index, value, 128)(DeterministicRng(93))
    assert len(draw.samples) == len(draw.value_samples) == 128 and len(draw.references) == 256
    assert all(s.arrays["policy_weight"][s.index] == 1 for s in draw.samples)
    assert any(s.arrays["policy_weight"][s.index] == 0 for s in draw.value_samples)
    assert all(ref.game_id == 0 for ref in draw.references)  # no held-out games
    assert all(ref.decision_index == 0 for ref in draw.references[:128])
    assert any(ref.decision_index != 0 for ref in draw.references[128:])


def test_joint_training_exact_resume_separate_ledgers_and_percentage_selection(mixed, tmp_path):
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    run(mixed, full)
    run(mixed, resumed, max_steps_this_session=1)
    assert [json.loads(x)["step"] for x in (resumed / "validation_curve.jsonl").read_text().splitlines()] == [0]
    run(mixed, resumed)
    a = load_alphazero_checkpoint(full / "checkpoints/final.pt")
    b = load_alphazero_checkpoint(resumed / "checkpoints/final.pt")
    assert a.total_train_step == b.total_train_step == 5
    assert a.sampling_rng_state == b.sampling_rng_state
    for name, value in a.model_state_dict.items():
        torch.testing.assert_close(value, b.model_state_dict[name], rtol=0, atol=0)
    for pid, state in a.optimizer_state_dict["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(value, b.optimizer_state_dict["state"][pid][name], rtol=0, atol=0)
    metrics = [json.loads(line) for line in (resumed / "training_metrics.jsonl").read_text().splitlines()]
    updates = [m for m in metrics if m["event"] == "train_step"]
    assert len(updates) == 5
    assert all(m["policy_weight_sum"] == m["value_weight_sum"] == 4 for m in updates)
    assert all(m["training_mode"] == "joint_separate_batches" and m["batch_size"] == 8 for m in updates)
    for ledger in (resumed / "references/policy").glob("*.bin"):
        assert np.all(np.fromfile(ledger, dtype="<i8").reshape(-1, 4)[:, 3] == 0)
    selected = select_checkpoint(mixed, resumed)
    assert selected["selected_step"] in [2, 3, 4, 5]
    assert read_json(resumed / "plan.json")["sampling"]["protocol"] == mixed.sampling_protocol
    for step in [2, 3, 4, 5]:
        point = read_json(resumed / f"full_validation/step-{step:09d}.json")
        assert point["metrics"]["positions"] == 8
        assert point["metrics"]["policy_weight_sum"] == 2  # mask low-budget labels during validation too
    with pytest.raises(ValueError, match="identical frozen"):
        run(replace(mixed, sampling_seed=93), resumed)


def test_actual_current_preset_gets_fresh_optimizer_and_updates_new_action_parameters(mixed, tmp_path):
    config = replace(mixed, model_config=read_json(ROOT / "configs/entity_set_mlp_v1/model_256x4.json"),
        draw_exposures=1., validation_percent_step=50)
    initial = initialize_learner(config, {"test": True}, 2)
    assert initial.model.trainable_parameter_count() == 906434 and not initial.optimizer.state
    initial_state = {k: v.clone() for k, v in initial.model.state_dict().items()}
    run(config, tmp_path / "actual")
    final = load_alphazero_checkpoint(tmp_path / "actual/checkpoints/final.pt")
    assert final.total_train_step == 2
    for name in ("action_nonlinear.0.weight", "action_nonlinear.2.weight", "interaction_scale",
                 "action_cross.query_state.weight", "action_cross.query_action.weight", "action_cross.key.weight",
                 "action_cross.value.weight", "action_cross.output.weight"):
        assert not torch.equal(initial_state[name], final.model_state_dict[name])


def test_attention128_preset_same_sampler_joint_updates_and_exact_resume(mixed, tmp_path):
    config = replace(mixed,
        model_config=read_json(ROOT / "configs/entity_attention_v1/model_relations_128x2.json"),
        draw_exposures=1., validation_percent_step=50)
    initial = initialize_learner(config, {"test": True}, 2)
    assert initial.model.trainable_parameter_count() == 583682 and not initial.optimizer.state
    initial_state = {k: v.clone() for k, v in initial.model.state_dict().items()}
    full, resumed = tmp_path / "attention-full", tmp_path / "attention-resumed"
    run(config, full)
    run(config, resumed, max_steps_this_session=1)
    run(config, resumed)
    a = load_alphazero_checkpoint(full / "checkpoints/final.pt")
    b = load_alphazero_checkpoint(resumed / "checkpoints/final.pt")
    assert a.round_train_step == b.round_train_step == 2
    assert a.sampling_rng_state == b.sampling_rng_state
    assert any(not torch.equal(initial_state[k], v) for k, v in a.model_state_dict.items())
    for key, value in a.model_state_dict.items():
        torch.testing.assert_close(value, b.model_state_dict[key], rtol=0, atol=0)
    for pid, state in a.optimizer_state_dict["state"].items():
        for key, value in state.items():
            torch.testing.assert_close(value, b.optimizer_state_dict["state"][pid][key], rtol=0, atol=0)
    point = read_json(full / "full_validation/step-000000002.json")["metrics"]
    assert point["positions"] == 8 and point["policy_positions"] == 2
    assert select_checkpoint(config, full)["selected_step"] in (1, 2)


def test_attention80k_formal_config_matches_mlp_except_model_and_manifest_path():
    baseline = read_json(ROOT / "configs/entity_set_mlp_v1/offline_80k_lr1e4_20261021.json")
    attention = read_json(ROOT / "configs/entity_attention_v1/offline_128x2_no_aux_80k_lr1e4_20261021.json")
    assert attention.pop("model_config") == read_json(ROOT / "configs/entity_attention_v1/model_relations_128x2.json")
    baseline.pop("model_config")
    assert attention.pop("manifest") != baseline.pop("manifest")
    assert attention == baseline


def test_attention_pipeline_default_trains_selects_and_has_no_arena(tmp_path, monkeypatch):
    path = ROOT / "artifacts/entity_attention_128x2_offline_80k_20261021/run.py"
    spec = importlib.util.spec_from_file_location("attention80k_pipeline", path)
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    art, output, runtime, source = [tmp_path / p for p in ("report", "fit", "runtime", "source")]
    write_json(art / "source.json", {"test": True})
    for name, value in (("ART", art), ("OUTPUT", output), ("RUNTIME", runtime)):
        monkeypatch.setattr(entry, name, value)
    monkeypatch.setattr(entry, "prepare", lambda: {})
    monkeypatch.setattr(entry, "frozen_source", lambda: source)
    monkeypatch.setattr(entry, "report", lambda: {})
    monkeypatch.setattr(entry, "environment_identity", lambda: {"test": True})
    monkeypatch.setattr(entry, "identity", lambda p: {"path": str(p), "sha256": sha256(p)})
    phases = []

    def fake_child(command, phase, frozen):
        assert frozen == source
        assert command[command.index("-m") + 1] == "splendor.offline_mixed_budget"
        assert "--select-only" in command if phase == "select" else "--select-only" not in command
        phases.append(phase)
        if phase == "train":
            write_json(output / "progress.json", {"status": "complete"})

    monkeypatch.setattr(entry, "run_child", fake_child)
    monkeypatch.setattr(entry.sys, "argv", [str(path)])
    assert entry.main() == 0
    assert phases == ["train", "select"]
    assert read_json(art / "execution.json")["sessions"][0]["status"] == "complete"


def test_mixed_config_rejects_auxiliary_and_unknown_sampling(mixed):
    with pytest.raises(ValueError, match="auxiliary"):
        replace(mixed, value_auxiliary={})
    with pytest.raises(ValueError, match="sampling"):
        replace(mixed, sampling_protocol="other")


def pipeline():
    path = ROOT / "artifacts/entity_set_mlp_256x4_offline_80k_20261021/run.py"
    spec = importlib.util.spec_from_file_location("set_mlp_offline_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_arena_configs_use_selected_mlp_and_r20_equal_budgets_and_collection_execution(mixed, tmp_path, monkeypatch):
    entry = pipeline()
    learner = initialize_learner(mixed, {"test": True}, 5)
    candidate = tmp_path / "candidate.pt"
    learner.save(candidate)
    opponent = tmp_path / "opponent.pt"
    learner.save(opponent)
    monkeypatch.setattr(entry, "OPPONENT", opponent)
    selection = {"checkpoint": str(candidate), "checkpoint_sha256": sha256(candidate), "selected_step": 3}
    plan = {"opponent": {"sha256": sha256(opponent)}, "selection": "full KL+MSE",
        "catalog_fingerprint": entry.load_default_catalog().data_fingerprint}
    for panel in entry.PANELS:
        raw = entry.evaluation_config(plan, selection, panel)
        assert len(raw["game_seeds"]) == panel["pairs"]
        assert raw["policy_a"]["checkpoint"] == str(candidate)
        assert raw["policy_b"]["checkpoint"] == str(opponent)
        if panel["kind"] == "greedy":
            assert raw["device"] == "cpu" and raw["workers"] == 8 and "execution" not in raw
        else:
            execution = raw["execution"]
            assert panel["pairs"] == 500 and raw["device"] == "cuda"
            assert execution["search_workers"] == 12 and execution["total_active_games"] == 1000
            assert execution["max_batch_size"] == 500 and execution["pipeline_slots"] == 1
            for side in ("policy_a", "policy_b"):
                search = raw[side]["search_config"]
                assert search["simulations"] == 1024 and search["determinizations"] == 1
                assert search["root_noise_fraction"] == search["forced_playouts"] == 0
                assert not search["policy_target_pruning"]


def test_prepared_report_does_not_claim_training_or_arena_completion(tmp_path, monkeypatch):
    entry = pipeline()
    monkeypatch.setattr(entry, "OUTPUT", tmp_path / "fit")
    monkeypatch.setattr(entry, "ART", tmp_path / "report")
    (tmp_path / "report").mkdir()
    result = entry.report({"steps_target": 100, "panels": list(entry.PANELS)})
    assert result["status"] == "prepared_not_started"
    assert all(p["status"] == "pending" for p in result["panels"])
    assert result["selection"] is None


def test_pipeline_runs_train_select_then_both_arenas_serially_without_launching_work(mixed, tmp_path, monkeypatch):
    entry = pipeline()
    art, output, runtime, source = [tmp_path / p for p in ("report", "fit", "runtime", "source")]
    art.mkdir()
    write_json(art / "source.json", {"test": True})
    learner = initialize_learner(mixed, {"test": True}, 5)
    candidate = tmp_path / "candidate.pt"
    learner.save(candidate)
    plan = {"steps_target": 5, "panels": list(entry.PANELS), "opponent": {"sha256": sha256(candidate)},
        "selection": "full KL+MSE", "catalog_fingerprint": entry.load_default_catalog().data_fingerprint}
    for name, value in (("ART", art), ("OUTPUT", output), ("RUNTIME", runtime), ("OPPONENT", candidate)):
        monkeypatch.setattr(entry, name, value)
    monkeypatch.setattr(entry, "prepare", lambda: plan)
    monkeypatch.setattr(entry, "frozen_source", lambda: source)
    monkeypatch.setattr(entry, "environment_identity", lambda: {"test": True})
    monkeypatch.setattr(entry.signal, "signal", lambda *_: None)
    monkeypatch.setattr(entry, "identity", lambda path: {"path": str(path), "sha256": sha256(path)})
    monkeypatch.setattr(entry, "report", lambda _: None)
    phases = []
    monkeypatch.setattr(entry, "summarize_panel", lambda p: {"status": "complete" if p["id"] in phases else "pending"})

    def fake_child(command, phase, frozen):
        assert frozen == source
        if phase in ("train", "select"):
            assert command[command.index("-m")+1] == "splendor.offline_mixed_budget"
            assert command[command.index("--config")+1].startswith(str(source))
        if phase == "train":
            write_json(output / "progress.json", {"status": "complete", "step": 5})
        elif phase == "select":
            write_json(output / "selection.json", {"checkpoint": str(candidate), "checkpoint_sha256": sha256(candidate), "selected_step": 3})
        phases.append(phase)

    monkeypatch.setattr(entry, "run_child", fake_child)
    monkeypatch.setattr(entry.sys, "argv", ["run.py", "run"])
    assert entry.main() == 0
    assert phases == ["train", "select", "greedy_vs_R20", "search_S1024N1_vs_R20"]
    assert read_json(art / "execution.json")["sessions"][0]["status"] == "complete"
