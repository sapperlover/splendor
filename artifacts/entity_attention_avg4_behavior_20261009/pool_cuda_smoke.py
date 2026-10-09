"""Bounded real-budget validation on two copied, already completed games.

Never resumes the formal experiment. Compare serial/pool labels and exercise
the actual CUDA pool's interrupt/resume; keep all heavy files under runtime.
"""
from __future__ import annotations

import gc
import argparse
import json
import multiprocessing as mp
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from splendor.alphazero_recording import read_json, sha256, write_json
from splendor.policy_averaging import PolicyAveragingConfig, generate_labels
from splendor.selfplay_data import load_game


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("runtime/entity_attention_avg4_behavior_20261009/pool_cuda_validation_v2"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = args.output
    source = Path("runs/entity_attention_avg4_behavior_20261009/seed1_avg4/rounds/round-000/selfplay")
    selected = sorted((source / "games").glob("game-*.npz"))[:2]
    assert len(selected) == 2
    hashes = {str(p): sha256(p) for p in selected}
    collection = root / "source"
    (collection / "games").mkdir(parents=True, exist_ok=True)
    for path in selected:
        shutil.copy2(path, collection / "games" / path.name)
    manifest = read_json(source / "manifest.json")
    manifest["config"]["num_games"] = 2
    manifest["validation_subset"] = {"source": str(source), "source_sha256": sha256(source / "manifest.json"),
                                     "archives": hashes, "scope": "two completed natural games copied for validation"}
    write_json(collection / "manifest.json", manifest)
    plan = read_json(Path("configs/entity_attention_v1/iteration_avg4_behavior_20261009.json"))
    pooled = PolicyAveragingConfig(**plan["arms"][0]["policy_averaging"])
    serial = PolicyAveragingConfig(active_roots=64)
    timing, results = {}, {}
    for name, config in (("serial", serial), ("pool", pooled)):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        results[name] = generate_labels(collection, root / name, config, device="cuda")
        timing[name] = time.perf_counter() - start
    max_diff, changed, repeats = 0., 0, 0
    for path in selected:
        arrays, _ = load_game(path)
        with np.load(root / "serial/games" / path.name) as a, np.load(root / "pool/games" / path.name) as b:
            np.testing.assert_array_equal(a["extra_world_seeds"], b["extra_world_seeds"])
            np.testing.assert_array_equal(a["extra_noise_seeds"], b["extra_noise_seeds"])
            np.testing.assert_array_equal(b["targets"][0].astype(np.float32), arrays["policy_target"])
            np.testing.assert_array_equal(b["target_mean"], b["targets"].mean(axis=0).astype(np.float32))
            for decision in np.flatnonzero(arrays["searchable"]):
                lo, hi = arrays["candidate_offsets"][decision:decision + 2]
                np.testing.assert_array_equal(b["extra_root_visit_counts"][:, lo:hi].sum(axis=1), [1024] * 3)
                max_diff = max(max_diff, float(np.abs(a["targets"][1:, lo:hi] - b["targets"][1:, lo:hi]).max()))
                changed += int((a["targets"][1:, lo:hi].argmax(axis=1) != b["targets"][1:, lo:hi].argmax(axis=1)).sum())
                repeats += 3
    interrupted = root / "interrupted"
    before_shared = set(Path("/dev/shm").glob("spgpu_*"))
    def stop(block, repeat, done, total):
        if block < 0 and done >= 8:
            raise KeyboardInterrupt
    try:
        generate_labels(collection, interrupted, pooled, device="cuda", progress_callback=stop)
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError("pool did not reach the interrupt boundary")
    committed = {p: (sha256(p), p.stat().st_mtime_ns) for p in (interrupted / "roots.games").glob("*.npz")}
    assert committed and not (interrupted / "manifest.json").exists()
    assert set(Path("/dev/shm").glob("spgpu_*")) == before_shared
    assert not any(p.name.startswith("native-search-") for p in mp.active_children())
    resumed = generate_labels(collection, interrupted, pooled, device="cuda")
    assert all((sha256(p), p.stat().st_mtime_ns) == old for p, old in committed.items())
    assert all(sha256(Path(p)) == checksum for p, checksum in hashes.items())
    owner = read_json(root / "pool" / results["pool"]["execution"]["statistics"][0])["owner"]
    assert owner["actual_search_workers"] == 8
    assert len({w["pid"] for w in owner["workers"]}) == 8
    assert all(not w["cuda_initialized"] for w in owner["workers"])
    report = {"status": "passed", "scope": "two copied real S1024/N1 games, not a full production round or strength panel",
              "timing_scope": "one tiny two-game subset; includes worker startup and short tail, not full-round speed evidence",
              "source_archives": hashes, "config": {"serial": asdict(serial), "pool": asdict(pooled)},
              "seconds": timing, "speed_ratio_serial_over_pool": timing["serial"] / timing["pool"],
              "searchable_roots": repeats // 3, "extra_searches": repeats,
              "extra_simulations_per_variant": results["pool"]["extra_simulations"],
              "seed_arrays_equal": True, "behavior_is_sample0": True, "mean_of_four_verified": True,
              "raw_archives_unchanged": True, "extra_budget_each_1024": True,
              "serial_pool_max_target_abs_difference": max_diff, "serial_pool_top1_changes": changed,
              "interrupted_committed_roots_reused": len(committed), "resume_complete": resumed["status"],
              "interrupt_workers_and_shared_segments_cleaned": True, "owner": owner,
              "raw_output": str(root), "pool_manifest_sha256": sha256(root / "pool/manifest.json")}
    write_json(Path("artifacts/entity_attention_avg4_behavior_20261009/pool_cuda_smoke.json"), report)
    print(json.dumps({k: report[k] for k in ("status", "seconds", "speed_ratio_serial_over_pool", "extra_simulations_per_variant",
                      "serial_pool_max_target_abs_difference", "interrupted_committed_roots_reused")}), flush=True)


if __name__ == "__main__":
    main()
