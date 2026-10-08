"""Read-only reproduction of bounded CPW depth, compatibility and collection results."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ART = Path(__file__).resolve().parent
ROOT = ART.parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def statistics(rows):
    simulations = sum(x["stats"]["simulations"] for x in rows)
    return {"roots": len(rows), "mean_depth": sum(x["stats"]["tree_depth_sum"] for x in rows) / simulations,
            "terminal_fraction": sum(x["stats"]["terminal_hits"] for x in rows) / simulations,
            "mean_target_kl_to_network": float(np.mean([
                sum(p * math.log(p / max(q, 1e-12)) for p, q in zip(x["target"], x["prior"]) if p > 0)
                for x in rows]))}


def analyze():
    manifest = read(ART / "manifest.json")
    for item in [manifest["checkpoint"], manifest["bank"], manifest["probe_script"],
                 manifest["selection"], *manifest["inputs"], *manifest.get("outputs", [])]:
        path = Path(item["path"])
        assert path.stat().st_size == item["bytes"] and sha(path) == item["sha256"], path
    source = manifest["source"]
    snapshot = ROOT / "runtime/entity_attention_chance_pw_20261007/source"
    for path, expected in {**source["source_files"], **source["additional_files"]}.items():
        assert sha(snapshot / path) == expected, path
    assert sha(snapshot / "dirty.patch") == source["dirty_diff_sha256"]
    for path, expected in source["native_extensions"].items():
        assert sha(snapshot / Path(path).name) == expected
    cases = {}
    for path in sorted((ART / "cases").glob("*.json")):
        case = read(path)
        for row in case["records"]:
            assert sum(row["root_counts"]) == row["stats"]["simulations"] == 1024
            assert np.allclose(np.asarray(row["root_counts"]) / 1024, row["target"], atol=0, rtol=0)
            assert row["stats"]["transition_cap_hits"] == 0
            if case["backend"] == "native_chance":
                diagnostic = row["chance"]
                assert diagnostic["root_hidden_samples"] == 1024
                if case["chance_widening"]:
                    k, alpha = (case["chance_widening"][x] for x in ("k", "alpha"))
                    assert diagnostic["chance_pw_fresh_samples"] + diagnostic["chance_pw_reuses"] == diagnostic["chance_draws"]
                    for pools in diagnostic["root_chance_pools"]:
                        for pool in pools:
                            assert len(set(pool["outcomes"])) == len(pool["outcomes"])
                            assert len(pool["outcomes"]) <= min(pool["pool_size"], max(1, math.ceil(k * pool["visits"] ** alpha)))
        stats = statistics(case["records"])
        assert stats["mean_depth"] == case["mean_depth"]
        stats.update(wall_seconds=case["wall_seconds"], peak_rss_kib=case["peak_process_rss_kib_cumulative"],
                     subset=case["subset"], backend=case["backend"], chance_widening=case["chance_widening"])
        stats["strata"] = {"blind" if blind else "no_blind": statistics([
            x for x in case["records"] if bool(x["opponent_blind_reserved"]) == blind]) for blind in (False, True)}
        cases[path.stem] = stats
    for backend, baseline in [("native", "entity_attention_target_noise_20261007"),
                              ("native_chance", "entity_attention_chance_search_20261007")]:
        case = read(ART / "cases" / f"{backend}_s1024_k0_a0.5_confirmation_b200.json")
        with np.load(ROOT / "artifacts" / baseline / "raw_targets.npz", allow_pickle=False) as raw:
            for row in case["records"]:
                i = row["bank_index"]; size = len(row["target"])
                for field, saved in [("target", "policy_target"), ("prior", "root_network_prior"), ("root_counts", "root_visit_counts")]:
                    assert np.array_equal(row[field], raw[saved][0, 0, i, :size])
                assert row["chosen_candidate_index"] == raw["chosen_candidate_index"][0, 0, i]
    windows = {}
    for name in ("old", "pw"):
        window = read(ART / "collection" / f"{name}_window.json")
        collection_manifest = read(ART / "collection" / f"{name}_window.manifest.json")
        assert collection_manifest["checkpoint_identity"]["sha256"] == manifest["checkpoint"]["sha256"]
        assert collection_manifest["versions"]["search"] == (
            "determinized-puct-native-v1" if name == "old" else "sampled-chance-pw-puct-native-v1")
        assert window["execution"]["search_workers"] == 8
        assert window["execution"]["shared_slots"] == 2 and window["execution"]["pipeline_slots"] == 1
        assert window["execution"]["total_active_games"] == 200
        assert window["simulations"] == 1024 and window["determinizations"] == 1
        assert window["chance_widening"] == (None if name == "old" else {"k": 0.25, "alpha": 0.5})
        clock = window["clock"];rate = clock["rates"]["simulations_per_second"]
        windows[name] = {"rates": clock["rates"], "clock_totals": clock["totals"],
                         "common_seconds": clock["common_seconds"], "process_seconds": window["process_seconds"],
                         "compile_warmup_seconds": window["gpu_owner"]["benchmark"]["compile_warmup_seconds"],
                         "worker_cpu_seconds": clock["worker_cpu_seconds"],
                         "owner_cpu_seconds": clock["owner_cpu_seconds"],
                         "owner_timings": clock["owner_timings"],
                         "estimated_200_game_seconds_at_65_searches_per_game": 200 * 65 * 1024 / rate}
    smoke = read(ART / "evaluation_smoke.json")
    assert smoke["passed"] and smoke["completed_pairs"] == 2 and smoke["valid_games"] == 4
    assert all(row["status"] == "natural_terminal" for pair in smoke["pairs"] for row in pair["games"])
    assert smoke["exceptions"] == smoke["engineering_truncations"] == smoke["invariant_failures"] == 0
    old = cases["native_s1024_k0_a0.5_confirmation_b200"]
    unlimited = cases["native_chance_s1024_k0_a0.5_confirmation_b200"]
    pw = cases["native_chance_s1024_k0.25_a0.5_confirmation_b200"]
    return {"protocol": "attention-chance-pw-analysis-v1", "passed": True, "cases": cases,
            "legacy_and_unlimited_400_searches_bitwise_match_frozen_targets_priors_counts_choices": True,
            "depth_gain_pw_vs_unlimited_fraction": pw["mean_depth"] / unlimited["mean_depth"] - 1,
            "depth_pw_vs_old_fraction": pw["mean_depth"] / old["mean_depth"],
            "fixed_root_wall_ratio_pw_vs_unlimited": pw["wall_seconds"] / unlimited["wall_seconds"],
            "collection_windows": windows,
            "estimated_collection_work_time_ratio_pw_vs_old": windows["old"]["rates"]["simulations_per_second"] / windows["pw"]["rates"]["simulations_per_second"],
            "checked_cuda_graph_smoke_valid_games": 4,
            "scope": "Frozen16500-step checkpoint,200 fixed roots,S1024; finite-support draw CPW only. Real8-worker collection60-second windows; 200-game duration is a workload estimate at65 searched roots/game, not a completed-round measurement. FourS16 checked games are API smoke only; no strength/training/target-noise conclusion."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = analyze()
    if args.write:
        assert not (ART / "analysis.json").exists()
        (ART / "analysis.json").write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    else:
        assert result == read(ART / "analysis.json")
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}, ensure_ascii=False))
