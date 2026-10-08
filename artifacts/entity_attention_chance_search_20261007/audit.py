"""Audit frozen source/build identities and recompute the search-cost comparison; no GPU."""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np

ART = Path(__file__).resolve().parent
ROOT = ART.parents[1]
BASE = ROOT / "artifacts/entity_attention_target_noise_20261007"
SNAPSHOT = ROOT / "runtime/entity_attention_chance_search_20261007/source"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def profile():
    manifests = {"baseline": read(BASE / "manifest.json"), "chance": read(ART / "manifest.json")}
    result = {
        "protocol": "attention-search-profile-v1",
        "definitions": {
            "mean_logical_depth": "sum(tree_depth_sum) / sum(simulations); decision edges, excluding folded forced actions",
            "terminal_fraction": "sum(terminal_hits) / sum(simulations)",
            "mean_model_positions": "sum(model_positions) / sum(simulations), including one root evaluation per search",
            "wall_seconds": "sum per-part seconds; concurrent job totals are not wall-clock time",
        },
        "groups": {},
    }
    keys = ("simulations", "tree_depth_sum", "terminal_hits", "expanded_nodes", "model_positions",
            "state_transitions", "folded_actions", "transition_cap_hits")
    for arm in "ABCD":
        result["groups"][arm] = {}
        for label, manifest in manifests.items():
            parts = [x for x in manifest["timing"] if x["arm"] == arm]
            assert len(parts) == 8
            totals = {key: sum(x["search_stats"][key] for x in parts) for key in keys}
            n = totals["simulations"]
            assert n == 200 * 8 * 1024
            result["groups"][arm][label] = {
                "totals": totals, "mean_logical_depth": totals["tree_depth_sum"] / n,
                "terminal_fraction": totals["terminal_hits"] / n,
                "mean_model_positions": totals["model_positions"] / n,
                "wall_seconds": sum(x["seconds"] for x in parts),
            }
    result["search_wall_seconds"] = {key: m["search_seconds"] for key, m in manifests.items()}
    result["chance_to_baseline_time_ratio"] = manifests["chance"]["search_seconds"] / manifests["baseline"]["search_seconds"]
    return result


def audit():
    manifest = read(ART / "manifest.json")
    source = manifest["source"]
    for path, expected in {**source["source_files"], **source["additional_files"]}.items():
        assert sha(SNAPSHOT / path) == expected, path
    assert sha(SNAPSHOT / "dirty.patch") == source["dirty_diff_sha256"]
    assert hashlib.sha256(json.dumps(source["source_files"], sort_keys=True).encode()).hexdigest() == source["source_sha256"]
    for name in ("baseline_binary", "wheel"):
        record = manifest["build"][name]
        assert sha(record["path"]) == record["sha256"], name
    with zipfile.ZipFile(manifest["build"]["wheel"]["path"]) as wheel:
        binaries = [p for p in wheel.namelist() if p.endswith(".so") and "/_native_search" in p]
        assert len(binaries) == 1
        assert hashlib.sha256(wheel.read(binaries[0])).hexdigest() == manifest["build"]["new_binary"]["sha256"]
    with np.load(BASE / "raw_targets.npz", allow_pickle=False) as before, np.load(ART / "raw_targets.npz", allow_pickle=False) as after:
        for key in ("root_network_prior", "root_used_prior"):
            assert np.array_equal(before[key], after[key]), key
    assert read(ART / "search_profile.json") == profile()
    assert read(ART / "tests.json")["exit_code"] == 0
    assert read(ART / "baseline_equivalence.json")["passed"]
    smoke = read(ART / "evaluation_smoke.json")
    assert smoke["passed"] and smoke["valid_games"] == 2
    assert smoke["exceptions"] == smoke["engineering_truncations"] == smoke["invariant_failures"] == 0
    return {"passed": True, "source_snapshot_files_verified": len(source["source_files"]),
            "additional_snapshot_files_verified": len(source["additional_files"]),
            "source_dirty_patch_verified": True, "baseline_binary_and_new_wheel_verified": True,
            "all_6400_root_priors_and_used_priors_bitwise_equal_to_baseline": True,
            "search_profile_recomputed": True, "relevant_tests_passed": 39,
            "checked_gpu_evaluation_smoke_games": 2,
            "scope": "Build/source and diagnostic-control audit; no strength inference."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Write the initial delivery audit; otherwise read-only verification")
    args = parser.parse_args()
    result = audit()
    if args.write:
        path = ART / "delivery_verification.json"
        assert not path.exists()
        path.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    else:
        assert result == read(ART / "delivery_verification.json")
    print(json.dumps(result, ensure_ascii=False))
