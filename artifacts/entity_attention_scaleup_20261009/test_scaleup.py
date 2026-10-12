"""Focused CPU checks for the 2026-10-09 scale-up artifact.

Run: conda run -n splendor env PYTHONPATH=src python -m pytest -q \
        artifacts/entity_attention_scaleup_20261009/test_scaleup.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import common

import battle_extend
import expand_roots
import analyze

DELTA = expand_roots.DELTA


def test_stored_bootstrap_keeps_same_id_games_in_different_rounds_separate(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "KEYS", ("r004",))
    monkeypatch.setattr(analyze.PH, "RUNTIME", tmp_path / "published")
    monkeypatch.setattr(common, "LR_RUNTIME", tmp_path / "lr")
    base = [{"round": r, "game_id": 7, "policy_ce_sum": 1.0,
             "target_entropy_sum": 0.0, "policy_weight_sum": 1.0} for r in (79, 80)]
    after = [{**row, "policy_ce_sum": value} for row, value in zip(base, (0.0, 2.0))]
    for path, rows in ((analyze.PH.RUNTIME / "r004/A/holdout_games_000500.json", base),
                       (common.LR_RUNTIME / "r004/holdout_games_000500.json", after)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows))
    result = analyze.paired_vs_A(500)
    assert result["games"] == 2
    assert result["scope"] == "recent_four_holdout"
    assert result["estimate"] == 0.0
    # Merging these two games would incorrectly produce a zero-width interval.
    assert result["lower"] == -1.0 and result["upper"] == 1.0


def test_fresh_net_change_is_against_start_and_excludes_train_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "LR_RUNTIME", tmp_path)
    rows = []
    for split in ("train", "validation"):
        for game in (0, 1):
            for arm, value in (("start", 1.0), ("A", 2.0), ("lr2.5e-5", 1.5)):
                rows.append({"key": "r004", "tag": "train", "set": split,
                             "game_id": game, "decision_index": 0, "arm": arm,
                             "step": 0 if arm == "start" else 500,
                             "kl_fresh": value if split == "validation" else 100.0})
    (tmp_path / "fresh_rows.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows))
    result = analyze.fresh_validation_vs_start()["arms"]
    assert result["A"][500]["estimate"] == 1.0
    # Better than A still means worse than the shared starting checkpoint.
    assert result["lr2.5e-5"][500]["estimate"] == 0.5
    assert result["lr2.5e-5"][500]["positions"] == 2


def test_expansion_spec_matches_recomputed_pools():
    spec = common.read_json(common.ART / "expand_specification.json")
    rounds = DELTA.read_json(DELTA.ART / "specification.json")["rounds"]
    for name, entry in spec["rounds"].items():
        identity = rounds[name]
        for kind, info in entry["sets"].items():
            pool, stats = expand_roots.pool_for(name, identity, kind)
            assert len(pool) == info["pool_size"]
            assert stats["draws_current_round"] == entry["draws_current_round"]
            assert stats["ledger"] == entry["ledger"]
            pool_keys = {(ref["game_id"], ref["decision_index"]) for ref in pool}
            published = expand_roots.existing_rows(name, expand_roots.KINDS[kind]["tag"])
            published_keys = {(int(row["game_id"]), int(row["decision_index"]))
                              for row in published}
            new_keys = {(ref["game_id"], ref["decision_index"]) for ref in info["refs"]}
            assert len(new_keys) == info["new_roots"] == 768
            assert not (new_keys & published_keys)
            assert new_keys <= pool_keys
            assert info["available_after_published"] == len(pool_keys - published_keys)


def test_expanded_summary_reproduces_published_estimates():
    summary = common.read_json(common.ART / "expanded_summary.json")
    for kind, entry in summary["reproduction_of_published"].items():
        assert entry["positions"] == 768
        assert entry["max_abs_diff"] <= 1e-12
        assert abs(entry["estimate"] - entry["published_estimate"]) <= 1e-12


def test_battle_pair_score_reproduces_published_panels():
    for key in ("r002", "r004", "r008", "r012"):
        path = common.BEHAVIOR_DIR / f"evaluations/seed1_avg4_{key}.json"
        data = common.read_json(path)
        if not data.get("passed") or data["completed_pairs"] != data["total_pairs"]:
            pytest.skip(f"{key} panel incomplete")
        scores = battle_extend.panel_scores(path)
        assert len(scores) == data["total_pairs"]
        values = list(scores.values())
        assert abs(sum(values) / len(values) - data["win_rate_score"]) < 1e-9


def test_extension_seed_domains_are_disjoint():
    published = common.read_json(common.BEHAVIOR_DIR /
                                 "evaluation_configs/seed1_avg4_r004.json")
    extension = common.read_json(common.ART / "battles/configs/seed1_avg4_r004_ext200.json")
    assert extension["pairs"] == 200
    assert extension["seed_start"] == published["seed_start"] + published["pairs"]
    assert extension["policy_seed_base"] == published["policy_seed_base"] + 2 * published["pairs"]
    published_seeds = set(range(published["seed_start"],
                                published["seed_start"] + published["pairs"]))
    extension_seeds = set(range(extension["seed_start"],
                                extension["seed_start"] + extension["pairs"]))
    assert not (published_seeds & extension_seeds)


def test_fresh_rows_match_target_identity():
    rows_path = common.LR_RUNTIME / "fresh_rows.jsonl"
    if not rows_path.exists():
        pytest.skip("fresh rows not evaluated in this environment")
    by_key: dict[str, list[tuple[int, int]]] = {}
    with rows_path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["tag"] == "train" and len(by_key.setdefault(row["key"], [])) < 5:
                by_key[row["key"]].append((row["game_id"], row["decision_index"]))
            if len(by_key) >= 3 and all(len(v) >= 5 for v in by_key.values()):
                break
    assert by_key, "no train rows found"
    for key, wanted in by_key.items():
        with np.load(DELTA.RUNTIME / key / "fresh_targets.npz", allow_pickle=False) as archive:
            identity = {(int(g), int(d)) for g, d in
                        zip(archive["game_id"], archive["decision_index"])}
        assert set(wanted) <= identity
