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
CHANCE = ROOT / "artifacts/entity_attention_chance_search_20261007"
SNAPSHOT = ROOT / "runtime/entity_attention_chance_pw_noise_20261007/source"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def profile():
    manifests = {"baseline": read(BASE / "manifest.json"), "unlimited_chance": read(CHANCE / "manifest.json"), "cpw": read(ART / "manifest.json")}
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
    result["cpw_to_baseline_time_ratio"] = manifests["cpw"]["search_seconds"] / manifests["baseline"]["search_seconds"]
    result["cpw_to_unlimited_chance_time_ratio"] = manifests["cpw"]["search_seconds"] / manifests["unlimited_chance"]["search_seconds"]
    return result


def audit():
    manifest = read(ART / "manifest.json")
    source = manifest["source"]
    previous = read(ROOT / "artifacts/entity_attention_chance_pw_20261007/manifest.json")["source"]
    assert source["source_files"] == previous["source_files"]
    assert source["additional_files"]["src/splendor/native_search.cpp"] == previous["additional_files"]["src/splendor/native_search.cpp"]
    for path, expected in {**source["source_files"], **source["additional_files"]}.items():
        assert sha(SNAPSHOT / path) == expected, path
    assert sha(SNAPSHOT / "dirty.patch") == source["dirty_diff_sha256"]
    assert hashlib.sha256(json.dumps(source["source_files"], sort_keys=True).encode()).hexdigest() == source["source_sha256"]
    for name in ("baseline_extension", "isolated_wheel", "native_extension"):
        record = manifest["build"][name]
        assert sha(record["path"]) == record["sha256"], name
    with zipfile.ZipFile(manifest["build"]["isolated_wheel"]["path"]) as wheel:
        binaries = [p for p in wheel.namelist() if p.endswith(".so") and "/_native_search" in p]
        assert len(binaries) == 1
        assert hashlib.sha256(wheel.read(binaries[0])).hexdigest() == manifest["build"]["native_extension"]["sha256"]
    for path, expected in source["native_extensions"].items():
        assert sha(SNAPSHOT / Path(path).relative_to(ROOT)) == expected
    with np.load(BASE / "raw_targets.npz", allow_pickle=False) as before, np.load(ART / "raw_targets.npz", allow_pickle=False) as after:
        for key in ("root_network_prior", "root_used_prior"):
            assert np.array_equal(before[key], after[key]), key
    assert read(ART / "search_profile.json") == profile()
    assert read(ART / "baseline_equivalence.json")["passed"]
    smoke = read(ART / "smoke.json")
    assert smoke["passed"] and smoke["identical_targets"] and smoke["roots"] == 8
    with np.load(CHANCE / "raw_targets.npz", allow_pickle=False) as before, np.load(ART / "raw_targets.npz", allow_pickle=False) as after:
        for key in ("root_network_prior", "root_used_prior"):
            assert np.array_equal(before[key], after[key]), key
    assert manifest["chance_widening"] == {"k": 0.25, "alpha": 0.5}
    return {"passed": True, "source_snapshot_files_verified": len(source["source_files"]),
            "additional_snapshot_files_verified": len(source["additional_files"]),
            "source_dirty_patch_verified": True, "CPW_binary_and_wheel_verified": True,
            "all_6400_root_priors_and_used_priors_bitwise_equal_to_both_references": True,
            "search_profile_recomputed": True, "production_source_unchanged_from_prior_45_test_CPW_delivery": True,
            "A_fixed_seed_search_smoke_roots": 8,
            "scope": "Build/source and diagnostic-control audit; no strength inference."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Write the initial delivery audit; otherwise read-only verification")
    args = parser.parse_args()
    if args.write:
        (ART / "search_profile.json").write_text(json.dumps(profile(),ensure_ascii=False,sort_keys=True,indent=2)+"\n")
    result = audit()
    if args.write:
        path = ART / "delivery_verification.json"
        assert not path.exists()
        path.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    else:
        assert result == read(ART / "delivery_verification.json")
    print(json.dumps(result, ensure_ascii=False))
