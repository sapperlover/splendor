"""Recompute the 100-pair result from durable game logs; never runs games or uses CUDA."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from splendor.evaluation import FixedEvaluationConfig, _evaluation_summary_from_records

ART = Path(__file__).resolve().parent
ROOT = ART.parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze():
    manifest = read(ART / "manifest.json")
    saved = read(ART / "evaluation.json")
    config = FixedEvaluationConfig.from_dict(read(ART / "evaluation_config.json"))
    assert saved["passed"] and saved["completed_pairs"] == 100 and saved["valid_games"] == 200
    assert config.policy_a.checkpoint == config.policy_b.checkpoint
    assert saved["policy_b_stats"] == {}
    assert saved["execution_stats"]["greedy_decisions"] > 0
    assert config.policy_a.search_backend == "native_chance" and config.policy_b.kind == "greedy_checkpoint"
    assert config.policy_b.search_config is None and config.policy_b.search_backend == "python"
    assert config.policy_a.chance_widening == {"k":0.25,"alpha":0.5} and config.policy_b.chance_widening is None
    assert config.policy_a.search_config["simulations"] == 1024
    assert config.policy_a.search_config["determinizations"] == 1
    assert config.policy_a.search_config["root_noise_fraction"] == 0
    assert len(config.game_seeds) == len(set(config.game_seeds)) == 100
    for item in [manifest["checkpoint"], *manifest["inputs"], *manifest.get("outputs", [])]:
        path = Path(item["path"])
        assert sha(path) == item["sha256"] and path.stat().st_size == item["bytes"], path
    source = manifest["source"]
    snapshot = ROOT / "runtime/entity_attention_chance_pw_greedy_20261007/source"
    for path, expected in {**source["source_files"], **source["additional_files"]}.items():
        assert sha(snapshot / path) == expected, path
    assert sha(snapshot / "dirty.patch") == source["dirty_diff_sha256"]
    assert {item["sha256"] for item in manifest["native_snapshots"]} == set(source["native_extensions"].values())
    for item in manifest["native_snapshots"]:
        assert sha(item["path"]) == item["sha256"]
    for name in ("policy_a", "policy_b"):
        assert saved["checkpoint_identity"][name]["sha256"] == manifest["checkpoint"]["sha256"]
    logs = Path(saved["game_log_root"])
    paths = sorted(logs.glob("pair-*.jsonl"))
    assert len(paths) == 100
    records = {}
    pair_score_counts = Counter()
    statuses = Counter()
    score = {"a_win": 1.0, "draw": 0.5, "a_loss": 0.0}
    for path in paths:
        pair_id = int(path.stem.split("-")[1])
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows) == 2
        rows.sort(key=lambda row: row["game_in_pair"])
        assert [row["game_in_pair"] for row in rows] == [0, 1]
        for seat, row in enumerate(rows):
            assert row["pair_id"] == pair_id
            assert row["game_seed"] == config.game_seeds[pair_id]
            assert row["first_player"] == seat
            assert row["policy_names"] == [config.policy_a.name, config.policy_b.name]
            assert row["policy_seeds"] == [config.policy_seed_base + pair_id * 2,
                                            config.policy_seed_base + pair_id * 2 + 1]
            assert row["status"] in ("natural_terminal", "policy_cycle_draw")
            assert row["error_type"] is None and row["error_detail"] is None
            assert row["policy_b_stats"] == {}
            for key in ("policy_a_stats",):
                stats = row[key]
                assert stats["simulations"] == 1024 * stats["searchable_decisions"]
                assert stats["determinizations"] == stats["searchable_decisions"]
                assert stats["transition_cap_hits"] == 0
            statuses[row["status"]] += 1
        pair_score_counts[str(sum(score[row["outcome_for_a"]] for row in rows) / 2)] += 1
        records[pair_id] = rows
    recomputed = _evaluation_summary_from_records(
        config, records, output_path=ART / "evaluation.json", game_log_root=logs,
        elapsed_seconds=saved["elapsed_seconds_this_invocation"],
    )
    for key, value in recomputed.items():
        if key != "updated_at":
            assert saved[key] == value, key
    interval = saved["paired_bootstrap"]
    return {
        "protocol": "attention-chance-pw-greedy-analysis-v1", "passed": True,
        "pairs": 100, "valid_games": 200, "new_search_wins": saved["a_wins"],
        "draws": saved["draws"], "new_search_losses": saved["a_losses"],
        "new_search_score": saved["win_rate_score"], "paired_bootstrap": interval,
        "new_search_a_first": saved["a_first"], "new_search_a_second": saved["a_second"],
        "pair_mean_score_distribution": dict(sorted(pair_score_counts.items())),
        "statuses": dict(sorted(statuses.items())),
        "elapsed_seconds": saved["elapsed_seconds_this_invocation"],
        "null_half_score_in_interval": interval["lower"] <= 0.5 <= interval["upper"],
        "raw_game_logs_recomputed_and_seed_budget_validated": True,
        "source_snapshot_files_verified": len(source["source_files"]) + len(source["additional_files"]),
        "greedy_no_search_verified": True,
        "greedy_owner_decisions": saved["execution_stats"]["greedy_decisions"],
        "game_log_hashes": {str(path.relative_to(ROOT)): sha(path) for path in paths},
        "scope": "Frozen16500-step checkpoint, CPW k0.25/alpha0.5 S1024 vs same-weight greedy argmax with zero simulations,100 paired game seeds; no training or equal-wall-time comparison.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Write the initial analysis; otherwise verify read-only")
    args = parser.parse_args()
    result = analyze()
    if args.write:
        path = ART / "analysis.json"
        assert not path.exists()
        path.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    else:
        assert result == read(ART / "analysis.json")
    print(json.dumps({key: value for key, value in result.items() if key != "game_log_hashes"}, ensure_ascii=False))
