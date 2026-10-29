"""Validate the completed matrix and preserve a compact verification receipt."""
import hashlib
import importlib.util
import json
from pathlib import Path

ART = Path(__file__).resolve().parent
ROOT = ART.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    read = lambda path: json.loads(path.read_text())
    plan, live, summary = [read(ART / name) for name in ("plan.json", "live.json", "summary.json")]
    assert summary["complete"] and live["stage"] == "complete"
    assert summary["completed_searches"] == summary["expected_searches"] == 1824
    assert plan["round"] == 12
    spec = importlib.util.spec_from_file_location("local12_verify", ART / "analyze.py")
    analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analysis)
    analysis.identity_check(plan)
    expected = {r["decision_index"] for r in analysis.records() if len(r["request"]["legal_actions"]) > 1}
    paths, simulations, roots = [], 0, 0
    for panel in plan["panels"]:
        for seed in plan["seeds"]:
            path = ART / "search" / panel["name"] / f"{seed}.json"
            doc = read(path)
            assert doc["panel"] == panel and doc["seed"] == seed
            assert len(doc["rows"]) == len(expected) == 76
            assert {row["decision_index"] for row in doc["rows"]} == expected
            for row in doc["rows"]:
                assert sum(row["visits"]) == sum(row["simulation_budgets"]) == panel["simulations"]
                assert len(row["simulation_budgets"]) == panel["determinizations"]
                simulations += sum(row["visits"])
                roots += 1
            paths.append(path)
    assert roots == 1824 and simulations == 15876096
    compile_stats = live["inference"]["compile"]
    assert not compile_stats["compile_failures"]
    assert compile_stats["compile_failure_fallbacks"] == compile_stats["eager_fallbacks"] == 0
    guard = read(ART / "original_guard.json")
    for relative, digest in guard.items():
        assert sha(ROOT / relative) == digest, relative
    for relative, digest in read(ART / "comparison.json")["source_sha256"].items():
        assert sha(ROOT / relative) == digest, relative
    receipt = {"complete": True, "root_searches": roots, "simulations": simulations,
        "elapsed_search_seconds": live["elapsed_this_session"], "compile_failures": 0,
        "fallbacks": 0, "old_files_unchanged": len(guard),
        "baseline_16_roots": read(ART / "baseline_check.json"),
        "formal_raw_sha256": {str(p.relative_to(ROOT)): sha(p) for p in paths},
        "producer_sha256": sha(Path(__file__))}
    (ART / "verification.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({k: v for k, v in receipt.items() if k not in ("baseline_16_roots", "formal_raw_sha256")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
