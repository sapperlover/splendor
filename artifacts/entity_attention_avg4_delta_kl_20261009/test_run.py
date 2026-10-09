"""Finite regression checks for resampling, completed sets and unseen selection."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

MODULE = importlib.util.spec_from_file_location("delta_kl_probe", Path(__file__).with_name("run.py"))
probe = importlib.util.module_from_spec(MODULE)
sys.modules[MODULE.name] = probe
MODULE.loader.exec_module(probe)


def test_cluster_bootstrap_counts_duplicate_draws():
    groups = [[-4., -2.], [7.], [1., 3., 6.]]
    rng = np.random.default_rng(19)
    draws = rng.integers(0, 3, size=(31, 3))
    expected = [np.mean([value for group in draw for value in groups[group]]) for draw in draws]
    result = probe.bootstrap_mean([-4., -2., 7., 1., 3., 6.],
                                  [0, 0, 1, 2, 2, 2], 31, 19)
    assert np.allclose([result["lower"], result["upper"]], np.percentile(expected, [2.5, 97.5]))
    assert result["estimate"] == np.mean([-4., -2., 7., 1., 3., 6.])
    assert result["games"] == 3


def test_round_strata_keep_original_position_weights():
    result = probe.bootstrap_mean([0., 0., 100.], [1, 2, 1], 200, 9,
                                  strata=["r004", "r004", "r008"])
    assert np.allclose([result[k] for k in ("estimate", "lower", "upper")], 100. / 3.)
    assert result["games"] == 3


def test_analysis_loads_all_completed_sets(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "RUNTIME", tmp_path)
    stage = tmp_path / "r004"
    stage.mkdir()
    for filename, name in [("positions.jsonl", "train"), ("positions_ledger.jsonl", "train_ledger"),
                           ("positions_unseen.jsonl", "train_unseen")]:
        (stage / filename).write_text(json.dumps({"set": name}) + "\n")
    result = probe.load_result_rows(["r004"])
    assert [r["set"] for r in result["r004"]] == ["train", "train_ledger", "train_unseen"]


def test_unseen_roots_exclude_current_policy_draws_and_validation(tmp_path, monkeypatch):
    usage = tmp_path / "usage/policy_recent"
    usage.mkdir(parents=True)
    np.asarray([[0, 90, 10, 1], [1, 90, 10, 1], [1, 89, 10, 2]], dtype="<i8").tofile(usage / "draws.bin")
    games = []
    for game, split, roots in [(10, "train", [1, 2]), (11, "validation", [1]), (12, "train", [3])]:
        path = tmp_path / f"game-{game:06d}.npz"
        path.write_bytes(bytes([game]))
        games.append({"path": path, "meta": {"game_id": game, "split": split},
                      "roots": dict.fromkeys(roots)})
    monkeypatch.setattr(probe, "round_stage", lambda _: tmp_path)
    monkeypatch.setattr(probe, "load_round_roots", lambda _: games)
    stats, refs = probe.unseen_refs("r012", {"round_index": 90}, count=2)
    assert {(r["game_id"], r["decision_index"]) for r in refs} == {(10, 2), (12, 3)}
    assert all(r["split"] == "train_unseen" for r in refs)
    assert stats["draws_current_round"] == 2
    assert stats["seen_searchable"] == 1
    assert stats["unseen_searchable"] == 2
