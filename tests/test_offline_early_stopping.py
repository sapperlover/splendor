from dataclasses import asdict, replace
import importlib.util
from pathlib import Path

import pytest
import torch

from test_offline_mixed_budget import mixed, one_thread
from splendor.alphazero_checkpoint import load_alphazero_checkpoint
from splendor.alphazero_recording import read_json, sha256, write_json
from splendor.offline_early_stopping import EarlyStoppingOfflineConfig, run
from splendor.offline_mixed_budget import prepare_inputs, select_checkpoint
from splendor.offline_scratch import initialize_learner

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def early(mixed):
    return EarlyStoppingOfflineConfig.from_dict({**asdict(mixed), "draw_exposures": 10.,
        "validation_percent_step": 5, "early_stopping_increases": 3,
        "model_config": {"architecture": "entity_attention_v1", "hidden": 16, "heads": 2,
                         "layers": 1, "ffn_hidden": 32}})


def scripted_validation(monkeypatch, objectives, *, interrupt_at=None):
    import splendor.offline_objectives as entry
    original = entry.evaluate
    interrupted = False

    def evaluate(learner, index, fixed, auxiliary):
        nonlocal interrupted
        if fixed is None and learner.round_train_step == interrupt_at and not interrupted:
            interrupted = True
            raise KeyboardInterrupt
        metrics = original(learner, index, fixed, auxiliary)
        # Subset increases at every step; stopping must use only the full holdout.
        metrics["policy_kl"] = (100.+learner.round_train_step if fixed is not None
                                else objectives[learner.round_train_step-1])
        metrics["value_loss"] = .25
        return metrics

    monkeypatch.setattr(entry, "evaluate", evaluate)


@pytest.mark.parametrize("objectives,stop,best", [
    ([1., 1.1, 1.2, 1.3], 4, 1),
    ([1., 1.1, 1.2, 1.2, 1.3, 1.4, 1.3, 1.4, 1.5, 1.6], 10, 1),
    ([1., .8, .8, .9, 1., 1.1], 6, 2),
    ([1.-i*.01 for i in range(20)], 20, 20),
])
def test_full_percentage_stopping_resets_and_selects_best(early, tmp_path, monkeypatch, objectives, stop, best):
    scripted_validation(monkeypatch, objectives)
    output = tmp_path / "fit"
    _, _, _, _, plan = prepare_inputs(early, output)
    assert plan["steps_target"] == 20 and plan["validation_steps"] == list(range(1, 21))
    run(early, output)
    progress = read_json(output / "progress.json")
    assert progress["status"] == "complete" and progress["step"] == stop
    assert progress["stop_reason"] == ("exposure_budget" if stop == 20 else "consecutive_validation_increases")
    assert load_alphazero_checkpoint(output / "checkpoints/final.pt").round_train_step == stop
    assert len(list((output / "full_validation").glob("step-*.json"))) == stop
    selected = select_checkpoint(early, output)
    assert selected["selected_step"] == best and len(selected["candidates"]) == stop
    assert read_json(output / "early_stopping.json")["triggered"] == (stop != 20)


@pytest.mark.parametrize("interrupt_validation", [False, True])
def test_early_stop_exact_resume_and_no_extra_updates(early, tmp_path, monkeypatch, interrupt_validation):
    objectives = [1., 1.1, 1.2, 1.3]
    scripted_validation(monkeypatch, objectives)
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    run(early, full)
    # Restore the real evaluator before wrapping it again.
    monkeypatch.undo()
    import splendor.offline_objectives as entry
    monkeypatch.setattr(entry, "source_identity", lambda *_: {"test": True})
    scripted_validation(monkeypatch, objectives, interrupt_at=4 if interrupt_validation else None)
    if interrupt_validation:
        with pytest.raises(KeyboardInterrupt):
            run(early, resumed)
        assert read_json(resumed / "progress.json")["step"] == 4
    else:
        run(early, resumed, max_steps_this_session=3)
        assert read_json(resumed / "progress.json")["status"] == "incomplete"
    run(early, resumed)
    a, b = [load_alphazero_checkpoint(p / "checkpoints/final.pt") for p in (full, resumed)]
    assert a.round_train_step == b.round_train_step == 4
    assert a.sampling_rng_state == b.sampling_rng_state
    torch.testing.assert_close(a.torch_cpu_rng_state, b.torch_cpu_rng_state, rtol=0, atol=0)
    for key, value in a.model_state_dict.items():
        torch.testing.assert_close(value, b.model_state_dict[key], rtol=0, atol=0)
    for pid, state in a.optimizer_state_dict["state"].items():
        for key, value in state.items():
            torch.testing.assert_close(value, b.optimizer_state_dict["state"][pid][key], rtol=0, atol=0)
    run(early, resumed)
    assert load_alphazero_checkpoint(resumed / "checkpoints/final.pt").round_train_step == 4
    assert select_checkpoint(early, resumed)["selected_step"] == 1
    with pytest.raises(ValueError, match="identical frozen"):
        run(replace(early, early_stopping_increases=2), resumed)


def pipeline():
    path = ROOT / "artifacts/entity_attention_128x2_offline_r21_r31_20261022/run.py"
    spec = importlib.util.spec_from_file_location("attention44k_pipeline", path)
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    return entry


def test_four_panel_configs_order_opponents_and_equal_search_budget(early, tmp_path, monkeypatch):
    entry = pipeline()
    candidate = tmp_path / "candidate.pt"
    initialize_learner(early, {"test": True}, 20).save(candidate)
    art = tmp_path / "report"
    monkeypatch.setattr(entry, "ART", art)
    raw = read_json(entry.SOURCE_CONFIG)
    write_json(art / "evaluation_templates.json", {kind: raw[f"{kind}_evaluation"] for kind in ("greedy", "search")})
    selection = {"checkpoint": str(candidate), "checkpoint_sha256": sha256(candidate), "selected_step": 4}
    opponents = {}
    for n in (20, 31):
        path = tmp_path / f"r{n}.pt"
        path.write_bytes(candidate.read_bytes())
        opponents[f"R{n}"] = {"path": str(path), "sha256": sha256(path)}
    plan = {"opponents": opponents, "catalog_fingerprint": "test", "selection": "full KL+MSE"}
    assert [(p["kind"], p["opponent"], p["pairs"]) for p in entry.PANELS] == [
        ("greedy", "R20", 1000), ("greedy", "R31", 1000), ("search", "R20", 500), ("search", "R31", 500)]
    for panel in entry.PANELS:
        config = entry.evaluation_config(plan, selection, panel)
        assert len(config["game_seeds"]) == panel["pairs"]
        assert config["policy_a"]["checkpoint"] == str(candidate)
        assert config["policy_b"]["checkpoint"] == opponents[panel["opponent"]]["path"]
        if panel["kind"] == "greedy":
            assert config["device"] == "cpu" and config["workers"] == 8 and "execution" not in config
        else:
            assert config["execution"]["pipeline_slots"] == 1
            assert config["execution"]["search_workers"] == 12
            for side in ("policy_a", "policy_b"):
                search = config[side]["search_config"]
                assert (search["simulations"], search["determinizations"]) == (1024, 1)
                assert search["root_noise_fraction"] == search["forced_playouts"] == 0
                assert not search["policy_target_pruning"]


def test_config_patience_validation(early):
    for value in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            replace(early, early_stopping_increases=value)


def test_formal_config_changes_only_data_range_and_stopping():
    entry = pipeline()
    baseline = read_json(entry.BASE_CONFIG)
    current = read_json(entry.CONFIG)
    assert current.pop("early_stopping_increases") == 3
    for key, expected in (("first_round", 64), ("last_round", 74)):
        assert current[key] == expected
        current[key] = baseline[key]
    for key in ("manifest", "manifest_sha256"):
        assert current[key] != baseline[key]
        current[key] = baseline[key]
    assert current == baseline


def test_pipeline_runs_four_panels_after_one_frozen_selection(tmp_path, monkeypatch):
    entry = pipeline()
    art, output, runtime = [tmp_path / p for p in ("art", "output", "runtime")]
    for key, value in (("ART", art), ("OUTPUT", output), ("RUNTIME", runtime)):
        monkeypatch.setattr(entry, key, value)
    write_json(art / "source.json", {"test": True})
    selection = {"selected_step": 4, "checkpoint": "selected", "checkpoint_sha256": "frozen"}
    monkeypatch.setattr(entry, "prepare", lambda: {})
    monkeypatch.setattr(entry, "frozen_source", lambda: tmp_path / "source")
    monkeypatch.setattr(entry, "report", lambda plan: {})
    monkeypatch.setattr(entry, "environment_identity", lambda: {"test": True})
    monkeypatch.setattr(entry, "identity", lambda path: {"test": True})
    monkeypatch.setattr(entry, "evaluation_config", lambda plan, selected, panel: {"candidate": selected, "panel": panel})
    done, phases = set(), []
    monkeypatch.setattr(entry, "summarize_panel", lambda panel: {"status": "complete" if panel["id"] in done else "pending"})

    def child(command, phase, source):
        phases.append(phase)
        if phase == "train":
            write_json(output / "progress.json", {"status": "complete", "stop_reason": "consecutive_validation_increases"})
        elif phase == "select":
            write_json(output / "selection.json", selection)
        else:
            assert read_json(art / "selected_candidate.json") == selection
            done.add(phase)

    monkeypatch.setattr(entry, "run_child", child)
    monkeypatch.setattr(entry.sys, "argv", ["run.py", "run"])
    assert entry.main() == 0
    assert phases == ["train", "select", *[p["id"] for p in entry.PANELS]]
    phases.clear()
    assert entry.main() == 0
    assert phases == ["select"]  # Completed training and arenas are skipped.
