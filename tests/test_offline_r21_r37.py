from dataclasses import replace
import importlib.util
import json
from pathlib import Path

import pytest

from test_offline_mixed_budget import mixed, one_thread
from test_random_budget_iteration import archive
from splendor.alphazero_recording import read_json, sha256, write_json
from splendor.offline_mixed_budget import prepare_inputs
from splendor.offline_objectives import expected_round_game_counts


def test_heterogeneous_round_counts_preserve_original_games_and_resume(mixed, tmp_path):
    manifest = Path(mixed.manifest)
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    rows.append(archive(tmp_path / "extra", 1, 2, mixed=True))
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    cfg = replace(mixed, manifest_sha256=sha256(manifest), expected_games_per_round={"1": 2, "2": 3})
    output = tmp_path / "fit"
    first = prepare_inputs(cfg, output)
    assert read_json(output / "dataset.json")["games"] == rows
    assert first[-1]["splits"]["train"]["games"] == 2
    assert first[-1]["splits"]["validation"]["games"] == 3
    assert prepare_inputs(cfg, output)[2] == first[2]
    with pytest.raises(ValueError, match="identical frozen"):
        prepare_inputs(replace(cfg, expected_games_per_round={"1": 3, "2": 2}), output)
    # An equal total must not hide a missing game in one generation.
    with pytest.raises(ValueError, match="game count"):
        prepare_inputs(replace(cfg, expected_games_per_round={"1": 3, "2": 2}), tmp_path / "wrong")


@pytest.mark.parametrize("expected", [0, True, 1.5, {"1": 2}, {"1": 2, "2": 0},
                                     {"1": 2, "2": True}, {"0": 2, "1": 3}])
def test_expected_counts_reject_missing_extra_or_invalid_rounds(expected):
    with pytest.raises(ValueError, match="expected_games_per_round"):
        expected_round_game_counts(1, 2, expected)


def test_integer_counts_keep_legacy_zero_based_generation_mapping():
    assert expected_round_game_counts(64, 65, 4000) == {63: 4000, 64: 4000}
    assert expected_round_game_counts(74, 75, {"74": 4000, "75": 10000}) == {73: 4000, 74: 10000}


def pipeline():
    root = Path(__file__).resolve().parents[1]
    path = root / "artifacts/entity_attention_128x2_shortterm_value_r21_r37_20261023/run.py"
    spec = importlib.util.spec_from_file_location("attention104k_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_r37_panels_use_latest_a_and_greedy_preceding_search(tmp_path, monkeypatch):
    entry = pipeline()
    monkeypatch.setattr(entry, "ART", tmp_path)
    templates = {kind: read_json(path) for kind, path in entry.EVALUATION_CONFIGS.items()}
    write_json(tmp_path / "evaluation_templates.json", templates)
    candidate, opponent = tmp_path / "candidate.pt", tmp_path / "r37.pt"
    candidate.write_bytes(b"candidate")
    opponent.write_bytes(b"r37")
    selection = {"checkpoint": str(candidate), "checkpoint_sha256": sha256(candidate), "selected_step": 42}
    plan = {"opponents": {"R37": {"path": str(opponent), "sha256": sha256(opponent)}},
            "catalog_fingerprint": "test", "selection": "main full KL+MSE"}
    assert [(p["kind"], p["opponent"], p["pairs"]) for p in entry.PANELS] == [
        ("greedy", "R37", 1000), ("search", "R37", 500)]
    for panel in entry.PANELS:
        cfg = entry.evaluation_config(plan, selection, panel)
        assert cfg["policy_b"]["checkpoint"] == str(opponent)
        assert len(cfg["game_seeds"]) == panel["pairs"]
        if panel["kind"] == "search":
            assert cfg["execution"] == templates["search"]["execution"]
            assert cfg["execution"]["pipeline_slots"] == 2
            for side in ("policy_a", "policy_b"):
                assert cfg[side]["search_config"]["simulations"] == 1024
                assert cfg[side]["search_config"]["determinizations"] == 1
                assert cfg[side]["search_config"]["root_noise_fraction"] == 0
        else:
            assert cfg["workers"] == 8 and cfg["device"] == "cpu"
            assert all(cfg[side]["kind"] == "greedy_checkpoint" for side in ("policy_a", "policy_b"))


def test_formal_shortterm_config_keeps_previous_training_and_new_counts():
    entry = pipeline()
    config, base = read_json(entry.CONFIG), read_json(entry.BASE_CONFIG)
    assert config["expected_games_per_round"] == {
        str(r + 43): 4000 if r <= 31 else 10000 for r in range(21, 38)}
    assert (config["first_round"], config["last_round"]) == (64, 80)
    assert config["continuation_auxiliary"]["shortterm_value_weight"] == .25
    assert config["continuation_auxiliary"]["opponent_policy_weight"] == 0
    for key in ("manifest", "manifest_sha256", "last_round", "expected_games_per_round"):
        config.pop(key)
        base.pop(key)
    assert config == base
