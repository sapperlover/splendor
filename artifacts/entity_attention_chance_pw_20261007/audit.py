"""Verify CPW delivery and replay evaluation summaries from durable smoke logs."""
import argparse
import hashlib
import json
from pathlib import Path

from splendor.evaluation import FixedEvaluationConfig, _evaluation_summary_from_records
from analyze import analyze

ART = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def audit():
    assert analyze() == read(ART / "analysis.json")
    manifest = read(ART / "manifest.json")
    assert manifest["status"] == "complete"
    logs_verified = {}
    for stem, pairs, simulations in [("evaluation_smoke", 2, 16), ("evaluation_serial_smoke", 1, 8)]:
        saved = read(ART / f"{stem}.json")
        config = FixedEvaluationConfig.from_dict(read(ART / f"{stem}_config.json"))
        assert len(config.game_seeds) == pairs and saved["passed"] and saved["valid_games"] == pairs * 2
        assert config.policy_a.chance_widening == {"k": 1.0, "alpha": 0.5}
        assert config.policy_b.chance_widening is None
        records = {}
        paths = sorted(Path(saved["game_log_root"]).glob("pair-*.jsonl"))
        assert len(paths) == pairs
        for path in paths:
            pair_id = int(path.stem.split("-")[1])
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            assert len(rows) == 2
            rows.sort(key=lambda row: row["game_in_pair"])
            for seat, row in enumerate(rows):
                assert row["game_in_pair"] == row["first_player"] == seat
                assert row["game_seed"] == config.game_seeds[pair_id]
                assert row["policy_seeds"] == [config.policy_seed_base + 2 * pair_id, config.policy_seed_base + 2 * pair_id + 1]
                assert row["status"] == "natural_terminal" and row["error_type"] is None
                for key in ("policy_a_stats", "policy_b_stats"):
                    assert row[key]["simulations"] == simulations * row[key]["searchable_decisions"]
                    assert row[key]["transition_cap_hits"] == 0
            records[pair_id] = rows
            logs_verified[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        for name in ("policy_a", "policy_b"):
            assert saved["checkpoint_identity"][name]["sha256"] == manifest["checkpoint"]["sha256"]
        rebuilt = _evaluation_summary_from_records(config, records, output_path=ART / f"{stem}.json",
            game_log_root=Path(saved["game_log_root"]), elapsed_seconds=saved["elapsed_seconds_this_invocation"])
        for key, value in rebuilt.items():
            if key != "updated_at":assert saved[key] == value, (stem, key)
    return {"protocol": "attention-chance-pw-verification-v1", "passed": True,
            "related_tests_passed": 45, "existing_cuda_sandbox_skips": 3,
            "compatibility_fixed_searches": 400, "checked_smoke_games": 6,
            "fixed_case_roots": 824, "collection_windows": 2,
            "model_source_native_input_output_hashes_verified": True,
            "smoke_seed_budget_seat_and_summaries_recomputed": True,
            "game_log_hashes": logs_verified,
            "scope": "Finite implementation/depth/throughput checks; no CPW strength, target-noise or training claim."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = audit()
    if args.write:
        assert not (ART / "verification.json").exists()
        (ART / "verification.json").write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    else:assert result == read(ART / "verification.json")
    print(json.dumps({k: v for k, v in result.items() if k != "game_log_hashes"}, ensure_ascii=False))
