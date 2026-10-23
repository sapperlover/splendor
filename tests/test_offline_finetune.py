from dataclasses import asdict, replace
from pathlib import Path
import sys

import pytest
import torch

from test_offline_mixed_budget import mixed, one_thread
from splendor.alphazero_checkpoint import load_alphazero_checkpoint
from splendor.alphazero_recording import read_json, sha256, write_json
from splendor.offline_finetune import (FinetuneOfflineConfig, initialize_learner,
    prepare_inputs, run, select_checkpoint)
from splendor.offline_mixed_budget import run as run_scratch

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def continuation(mixed, tmp_path):
    mixed = replace(mixed, model_config={"architecture": "entity_attention_v1", "hidden": 16,
        "heads": 2, "layers": 1, "ffn_hidden": 32})
    parent = tmp_path / "parent"
    run_scratch(mixed, parent, max_steps_this_session=3)
    checkpoint = parent / "checkpoints/step-000000003.pt"
    raw = asdict(mixed)
    for key in ("initialization_seed", "model_config", "validation_percent_step"):
        raw.pop(key)
    raw.update(initial_checkpoint=str(checkpoint), initial_checkpoint_sha256=sha256(checkpoint),
        draw_exposures=10., validation_interval_steps=2, early_stopping_increases=3)
    raw["learner"]["learning_rate"] = 2e-5
    return FinetuneOfflineConfig.from_dict(raw), parent


def test_inherits_weights_adamw_and_all_rng_while_only_changing_lr(continuation, tmp_path):
    config, parent = continuation
    _, _, identity, fixed, plan = prepare_inputs(config, tmp_path / "fit")
    assert fixed == read_json(parent / "fixed_validation.json")
    assert read_json(tmp_path / "fit/dataset.json") == read_json(parent / "dataset.json")
    saved = load_alphazero_checkpoint(config.initial_checkpoint)
    learner = initialize_learner(config, identity, plan["steps_target"])
    assert learner.total_train_step == 3 and learner.round_train_step == 0
    assert learner.round_steps_target == 20 and plan["validation_steps"] == list(range(2, 21, 2))
    assert learner.sampling_rng.snapshot() == saved.sampling_rng_state
    torch.testing.assert_close(torch.get_rng_state(), saved.torch_cpu_rng_state, rtol=0, atol=0)
    for key, value in learner.model.state_dict().items():
        torch.testing.assert_close(value, saved.model_state_dict[key], rtol=0, atol=0)
    optimizer = learner.optimizer.state_dict()
    for group, old in zip(optimizer["param_groups"], saved.optimizer_state_dict["param_groups"]):
        assert group["lr"] == 2e-5 and old["lr"] == 1e-4
        assert {k:v for k,v in group.items() if k != "lr"} == {k:v for k,v in old.items() if k != "lr"}
    for pid, state in saved.optimizer_state_dict["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(value, optimizer["state"][pid][name], rtol=0, atol=0)
    with pytest.raises(ValueError, match="preserve parent"):
        initialize_learner(replace(config, learner=replace(config.learner, max_grad_norm=2.)), identity, 20)


def test_interval_early_stop_exact_resume_and_selection(continuation, tmp_path, monkeypatch):
    import splendor.offline_objectives as entry
    original = entry.evaluate

    def evaluate(learner, index, fixed, auxiliary):
        row = original(learner, index, fixed, auxiliary)
        row.update(policy_kl=1.+learner.round_train_step*.1, value_loss=.2)
        return row

    monkeypatch.setattr(entry, "evaluate", evaluate)
    config, _ = continuation
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    run(config, full)
    run(config, resumed, max_steps_this_session=3)
    assert sorted(p.name for p in (resumed / "full_validation").glob("*.json")) == ["step-000000002.json"]
    run(config, resumed)
    a, b = [load_alphazero_checkpoint(p / "checkpoints/final.pt") for p in (full, resumed)]
    assert a.round_train_step == b.round_train_step == 8
    assert a.total_train_step == b.total_train_step == 11
    assert a.sampling_rng_state == b.sampling_rng_state
    torch.testing.assert_close(a.torch_cpu_rng_state, b.torch_cpu_rng_state, rtol=0, atol=0)
    for key, value in a.model_state_dict.items():
        torch.testing.assert_close(value, b.model_state_dict[key], rtol=0, atol=0)
    for pid, state in a.optimizer_state_dict["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(value, b.optimizer_state_dict["state"][pid][name], rtol=0, atol=0)
    progress = read_json(resumed / "progress.json")
    assert progress["status"] == "complete" and progress["stop_reason"] == "consecutive_validation_increases"
    selected = select_checkpoint(config, resumed)
    assert selected["selected_step"] == 2 and selected["validation_interval_steps"] == 2
    assert "validation_percent_step" not in selected
    assert [r["step"] for r in selected["candidates"]] == [2, 4, 6, 8]


def test_nonmultiple_endpoint_is_evaluated_without_waiting_for_next_interval(continuation, tmp_path):
    config, _ = continuation
    config = replace(config, draw_exposures=2.5)
    run(config, tmp_path / "fit")
    progress = read_json(tmp_path / "fit/progress.json")
    assert progress["step"] == 5 and progress["stop_reason"] == "exposure_budget"
    selected = select_checkpoint(config, tmp_path / "fit")
    assert [row["step"] for row in selected["candidates"]] == [2, 4, 5]


def test_cli_only_trains_and_selects(continuation, tmp_path, monkeypatch):
    import splendor.offline_finetune as entry
    config, _ = continuation
    path, output = tmp_path / "config.json", tmp_path / "fit"
    write_json(path, asdict(config))
    calls = []

    def fake_run(*args, **kwargs):
        calls.append("train")
        write_json(output / "progress.json", {"status": "complete"})

    monkeypatch.setattr(entry, "run", fake_run)
    monkeypatch.setattr(entry, "select_checkpoint", lambda *_: calls.append("select") or {
        "selected_step": 2, "selection_objective": 1.})
    monkeypatch.setattr(sys, "argv", ["offline_finetune", "--config", str(path), "--output", str(output)])
    entry.main()
    assert calls == ["train", "select"]


def test_interval_and_stopping_reject_nonpositive_or_noninteger_values(continuation):
    config, _ = continuation
    for name in ("validation_interval_steps", "early_stopping_increases"):
        for value in (0, -1, True, 1.5):
            with pytest.raises(ValueError, match="positive integer"):
                replace(config, **{name:value})


def test_formal_config_preserves_parent_settings_except_requested_continuation_changes():
    parent = read_json(ROOT / "configs/entity_attention_v1/offline_128x2_r21_r31_early_stop_20261022.json")
    current = read_json(ROOT / "configs/entity_attention_v1/offline_128x2_r21_r31_lr2e5_20261022.json")
    for name in ("model_config", "initialization_seed", "validation_percent_step"):
        parent.pop(name)
    assert current.pop("validation_interval_steps") == 2000
    assert current.pop("initial_checkpoint").endswith("step-000059453.pt")
    assert current.pop("initial_checkpoint_sha256") == "3f43ec717fe3b2a548f5eb3e5e9878d3e402b15c9d848871e9934543b4862f5a"
    assert current["learner"]["learning_rate"] == 2e-5
    parent["learner"]["learning_rate"] = 2e-5
    assert current == parent
