"""Read-only audit of the completed screening; no searches or training."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

ART = Path(__file__).resolve().parent
ROOT = ART.parents[1]
CASES = ((512, 1), (1024, 1), (512, 2), (1024, 2))


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_file(record):
    path = Path(record["path"])
    assert path.is_file(), path
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"], path
    if "bytes" in record:
        assert path.stat().st_size == record["bytes"], path


def check_compile(stats):
    assert not stats["compile_failures"] and not stats["disabled_graphs"]
    assert stats["compile_failure_fallbacks"] == stats["eager_fallbacks"] == 0
    assert stats["effective_mode"] and set(stats["effective_mode"].values()) == {"compiled"}


def main():
    manifest = read(ART / "manifest.json")
    selection = read(ART / "selected_candidates.json")
    assert selection["selection_count"] == len(selection["selected"]) == 2
    assert {row["case"] for row in selection["selected"]} == {"s512_n2", "s1024_n2"}
    assert not selection["next_training_proposal"]["launched"]
    for record in [manifest["checkpoint"], manifest["opponent"], *manifest["inputs"], *manifest["outputs"]]:
        check_file(record)
    for filename, digest in manifest["source"]["source_files"].items():
        check_file({"path": str(ROOT / filename), "sha256": digest})
    for filename, digest in manifest["source"]["native_extensions"].items():
        check_file({"path": filename, "sha256": digest})

    baseline_cfg = read(ART / "configs/eval_s1024_n1.json")
    windows = []
    for s, n in CASES:
        case = f"s{s}_n{n}"
        cfg = read(ART / f"configs/eval_{case}.json")
        comparison = copy.deepcopy(cfg)
        comparison["policy_a"]["search_config"].update(simulations=1024, determinizations=1)
        assert comparison == baseline_cfg, case
        panel = read(ART / f"eval_{case}.json")
        assert panel["config"]["policy_a"]["search_config"]["simulations"] == s
        assert panel["config"]["policy_a"]["search_config"]["determinizations"] == n
        assert panel["config"]["policy_b"] == cfg["policy_b"]
        assert panel["passed"] and panel["completed_pairs"] == 50
        assert panel["valid_games"] == panel["completed_games"] == panel["total_games"] == 100
        for key in ("exceptions", "engineering_truncations", "forced_skips", "illegal_actions", "invariant_failures"):
            assert panel[key] == 0, (case, key)
        for actor in ("policy_a", "policy_b"):
            identity = manifest["checkpoint"] if actor == "policy_a" else manifest["opponent"]
            assert panel["checkpoint_identity"][actor]["sha256"] == identity["sha256"]
            assert panel[f"{actor}_stats"]["transition_cap_hits"] == 0
        stats = panel["policy_a_stats"]
        assert stats["simulations"] == stats["searchable_decisions"] * s
        assert stats["determinizations"] == stats["searchable_decisions"] * n
        assert abs(panel["win_rate_score"] - (panel["a_wins"] + panel["draws"] / 2) / 100) < 1e-12
        for model in panel["execution_stats"]["models"].values():
            check_compile(model["inference"]["compile"])
        for rep in (1, 2):
            window = read(ART / f"window_{case}_rep{rep}.json")
            assert (window["simulations"], window["determinizations"]) == (s, n)
            assert window["checkpoint"] == manifest["checkpoint"]["path"]
            execution = window["execution"]
            assert execution["total_active_games"] == 100
            assert execution["shared_slots"] == 2 and execution["pipeline_slots"] == 1
            assert window["clock"]["owner_measured_seconds"] >= 35
            assert len(window["worker_compile"]) == 1
            for stats in window["worker_compile"]:
                check_compile(stats)
                assert len(stats["effective_mode"]) == 20
            inference = window["gpu_owner"]["inference"]
            assert inference["inference"]["compile_mode"] == "native-cudagraph"
            assert inference["native_graph_captures"] > 0 and inference["native_graph_replays"] > 0
            windows.append({"case": case, "repeat": rep,
                            "graph_captures": inference["native_graph_captures"],
                            "graph_replays": inference["native_graph_replays"]})

    identities = list((ART / "recording").glob("*/identities/*.json"))
    sessions = list((ART / "recording").glob("*/sessions/*.json"))
    assert len(identities) == len(sessions) == 11
    for path in identities:
        assert read(path)["source_sha256"] == manifest["source"]["source_sha256"], path
    for path in sessions:
        session = read(path)
        assert session["status"] == "complete" and session["error"] is None, path

    roots = read(ART / "root_results.json")
    bank = read(ART / "root_bank.json")
    assert len(bank["positions"]) == 24 and len(roots["rows"]) == 576
    for record in bank["inputs"]:
        check_file(record)
    seen = set()
    for row in roots["rows"]:
        identity = (row["case"], row["seed"], row["root_noise"], row["root_id"])
        assert identity not in seen
        seen.add(identity)
        s, n = next((s, n) for s, n in CASES if row["case"] == f"s{s}_n{n}")
        assert row["simulations"] == sum(row["simulation_budgets"]) == s
        assert len(row["simulation_budgets"]) == n and row["transition_cap_hits"] == 0
        target = row["policy_target"]
        assert abs(sum(target) - 1) < 1e-7 and min(target) >= 0
        for j, value in enumerate(target):
            average = sum(visits[j] / budget for visits, budget in
                          zip(row["per_world_root_visit_counts"], row["simulation_budgets"])) / n
            assert abs(value - average) < 1e-7

    print(json.dumps({"passed": True, "evaluation_panels": 4, "valid_games": 400,
                      "new_gpu_sessions": 11, "fixed_root_searches": 576,
                      "cuda_graph_windows": windows, "artifact_hashes": len(manifest["outputs"]),
                      "legacy_baseline_source_identity": "not recorded; not backfilled"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
