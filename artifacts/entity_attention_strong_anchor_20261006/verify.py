"""Bounded integrity checks for completed evaluations and the design artifact."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    specification = read(ART / "specification.json")
    analysis = read(ART / "analysis.json")
    plan = read(ART / "training_plan.json")
    checks = {}
    for item in specification["inputs"]:
        assert digest(ROOT / item["path"]) == item["sha256"], item["path"]
    checks["frozen_inputs_and_formal_run_state_unchanged"] = len(specification["inputs"])
    for item in analysis["inputs"]:
        assert digest(ROOT / item["path"]) == item["sha256"], item["path"]
    assert digest(ART / "analyze.py") == analysis["analysis_script_sha256"]
    checks["analysis_inputs_and_script_unchanged"] = len(analysis["inputs"])

    recordings = [read(path) for path in sorted((ART / "recording").glob("*.json"))]
    assert len(recordings) == 2
    by_case = {record["case"]: record for record in recordings}
    assert set(by_case) == {"round9", "round50"}
    source = by_case["round9"]["source"]
    assert by_case["round50"]["source"] == source
    for rel, hashed in source["source_files"].items():
        assert digest(ROOT / rel) == hashed, rel
    for absolute, hashed in source["native_extensions"].items():
        assert digest(Path(absolute)) == hashed, absolute
    for record in recordings:
        assert record["passed"] and record["inputs_unchanged"]
        assert record["runner_sha256"] == digest(ART / "run_evaluation.py")
        assert record["native_source_sha256"] == digest(ROOT / "src/splendor/native_search.cpp")
        case = specification["cases"][record["case"]]
        panel = read(ART / case["output"])
        assert digest(ART / case["output"]) == record["output_sha256"]
        assert panel["passed"] and panel["valid_games"] == 200 and panel["completed_pairs"] == 100
        assert all(panel[key] == 0 for key in (
            "illegal_actions", "exceptions", "invariant_failures", "engineering_truncations"))
        assert panel["checkpoint_identity"]["policy_a"]["sha256"] == case["sha256"]
    checks["new_panels_valid_games"] = 400
    checks["new_panels_same_implementation"] = True
    configs = [read(ART / specification["cases"][case]["config"]) for case in ("round9", "round50")]
    for config in configs:
        config["policy_a"].pop("checkpoint")
        config["policy_a"].pop("name")
    assert configs[0] == configs[1]
    checks["panel_configs_differ_only_in_candidate_identity"] = True

    identity = read(ART / "anchor_identity.json")
    assert identity["identical_model_tensors"] and identity["max_abs_difference"] == 0
    assert identity["tensor_count"] == 69
    for item in identity["checkpoints"]:
        assert digest(ROOT / item["path"]) == item["sha256"]
    checks["theta5000_weight_identity_record_and_checkpoint_files"] = True

    assert plan["status"] == "designed_not_implemented"
    assert not plan["training_started"] and not plan["executable_hybrid_config"]
    assert not (ROOT / plan["branch"]["output"]).exists()
    for item in (plan["base_config_reference"], plan["starting_point"]["inherited_snapshot"],
                 plan["static_replay_first"]["checkpoint"], plan["static_replay_first"]["snapshot"],
                 plan["observations"]["primary_evaluation"]["anchor"]):
        assert digest(ROOT / item["path"]) == item["sha256"]
    assert plan["starting_point"]["checkpoint"]["sha256"] == specification["cases"]["round50"]["sha256"]
    assert plan["starting_point"]["checkpoint"]["step"] == 14500
    assert plan["sampling"]["offline_quota"] == 0 and plan["sampling"]["online_fraction"] == 1.0
    assert plan["sampling"]["pools_disjoint"] and plan["sampling"]["do_not_reset_old_offline_mix_schedule"]
    assert plan["branch"]["first_phase_rounds"] * plan["branch"]["new_games_per_round"] == 3000
    assert plan["branch"]["first_phase_rounds"] * plan["branch"]["updates_per_round"] == 1500
    assert 14500 + 1500 == plan["branch"]["first_phase_final_step"]
    schedule = plan["sampling"]["schedule"]
    assert [(r["first_round"], r["last_round"], r["latest_quota"], r["history_quota"])
            for r in schedule] == [(1, 2, 32, 224), (3, 5, 48, 208), (6, None, 64, 192)]
    for row in schedule:
        assert row["latest_quota"] + row["history_quota"] == 256
        assert row["latest_quota"] / 256 == row["latest_fraction"]
    latest = sum(next(row["latest_quota"] for row in schedule
                      if row["first_round"] <= r and (row["last_round"] is None or r <= row["last_round"]))
                 * 100 for r in range(1, 16))
    assert latest == plan["sampling"]["first_phase_latest_draws"] == 84800
    assert latest + plan["sampling"]["first_phase_history_draws"] == 1500 * 256
    teachers = plan["starting_point"]["inherited_teachers"]
    assert len(teachers) == 15 and sum(t["games"] for t in teachers) == 3000
    assert [t["parent_human_round"] for t in teachers] == list(range(36, 51))
    for teacher in teachers:
        for item in (teacher["manifest"], teacher["checkpoint_identity"]):
            path = Path(item["path"])
            assert digest(path if path.is_absolute() else ROOT / path) == item["sha256"]
    checks["design_starting_point_quotas_sources_and_budget"] = True
    checks["new_sampling_training_not_started"] = True

    documents = [ART / "README.md", ROOT / "docs/archive/entity_attention_fresh_share_zh-CN.md",
                 ROOT / "docs/guides/entity_attention_hybrid_s1024_n1_zh-CN.md",
                 ROOT / "docs/guides/entity_attention_offline_zh-CN.md", ROOT / "README.md",
                 ROOT / "docs/index.md", ROOT / "docs/current_project_state_zh-CN.md",
                 ROOT / "configs/entity_attention_v1/README.md"]
    links = []
    for path in documents:
        for raw in re.findall(r"\[[^\]]*\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            if "://" in raw or raw.startswith("#"):
                continue
            target = raw.split("#", 1)[0]
            if target == "verification.json" and path == ART / "README.md":
                continue  # This invocation writes the linked report below.
            assert (path.parent / target).exists(), (path, target)
            links.append({"document": str(path.relative_to(ROOT)), "target": target})
    checks["scoped_document_local_links"] = len(links)
    for path in (ART / "run_evaluation.py", ART / "analyze.py", Path(__file__)):
        compile(path.read_text(), str(path), "exec")
    checks["artifact_python_syntax"] = 3
    evidence = [ART / "specification.json", ART / "analysis.json", ART / "anchor_identity.json",
                ART / "training_plan.json", ART / "README.md", *documents[1:],
                ART / "run_evaluation.py", ART / "analyze.py", Path(__file__),
                *sorted((ART / "configs").glob("*.json")),
                *sorted((ART / "evaluations").glob("*.json")),
                *sorted((ART / "recording").glob("*.json"))]
    result = {"protocol": "entity-attention-strong-anchor-verification-v1", "passed": True,
              "scope": "Completed 400 evaluation games and design integrity; no new sampler implementation or training.",
              "checks": checks,
              "evidence_sha256": {str(path.relative_to(ROOT)): digest(path) for path in evidence}}
    (ART / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"passed": True, "checks": checks}, ensure_ascii=False))


if __name__ == "__main__":
    main()
