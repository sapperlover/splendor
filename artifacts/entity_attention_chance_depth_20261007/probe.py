"""Frozen-root search-budget/depth calibration, retaining per-root statistics and targets."""
from __future__ import annotations

import argparse
import gc
import json
import resource
import time
from pathlib import Path

import torch

from splendor.alphazero_recording import environment_identity, read_json, sha256, write_json
from splendor.compact_trace import decode_request
from splendor.mcts_policy import MCTSConfig
from splendor.native_policy import NativePUCTSearchPolicy, NativeChancePUCTSearchPolicy, NativeBatchRunner
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.search_inference import FrozenSearchInference, InferenceConfig

ART = Path(__file__).resolve().parent
ROOT = ART.parents[1]


def run(args):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = read_json(ART / "manifest.json")
    assert sha256(Path(__file__)) == manifest["probe_script"]["sha256"]
    checkpoint = Path(manifest["checkpoint"]["path"])
    assert sha256(checkpoint) == manifest["checkpoint"]["sha256"]
    bank = read_json(Path(manifest["bank"]["path"]))["positions"]
    selection = read_json(ART / "selection.json")
    indices = selection["pilot_indices" if args.subset == "pilot" else "confirmation_indices"]
    model = load_frozen_actor_critic_checkpoint(checkpoint, device="cuda")
    inference = InferenceConfig(mode="eager", actor_projection=False, candidate_buckets=(16, 32, 48, 64))
    evaluator = FrozenSearchInference(model, device="cuda", config=inference)
    cls = NativeChancePUCTSearchPolicy if args.backend == "native_chance" else NativePUCTSearchPolicy
    catalog = cls(model, evaluator=evaluator).catalog
    runner = NativeBatchRunner(evaluator, catalog, capacity=args.batch_size, buckets=(16, 32, 48, 64), slots=1)
    for simulations in args.sims:
        name = f"{args.backend}_s{simulations}_{args.subset}_b{args.batch_size}"
        output = ART / "cases" / f"{name}.json"
        assert not output.exists(), output
        config = MCTSConfig(**dict(manifest["search_config"], simulations=simulations,
                                 seed=16710000, root_noise_fraction=0.0))
        records = []
        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        for start in range(0, len(indices), args.batch_size):
            subset = indices[start:start + args.batch_size]
            requests = [decode_request(bank[i]["request"]) for i in subset]
            policy = cls(model, evaluator=evaluator, config=config)
            tick = time.perf_counter()
            jobs = [policy.start_job(request, add_root_noise=False, temperature=0, sampling_seed=16720000)
                    for request in requests]
            while any(not job.done for job in jobs):
                assert runner.run([job for job in jobs if not job.done]) > 0
            for index, job in zip(subset, jobs):
                result = job.result
                stats = job.core.stats()
                assert result.simulations == simulations == sum(result.root_visit_counts)
                assert result.transition_cap_hits == 0
                row = {"bank_index": index, "stage": bank[index]["stage"],
                       "opponent_blind_reserved": bank[index]["opponent_blind_reserved"],
                       "stats": dict(stats), "root_counts": list(result.root_visit_counts),
                       "target": list(result.policy_target), "prior": list(result.root_network_prior),
                       "chosen_candidate_index": result.chosen_candidate_index,
                       "mean_depth": stats["tree_depth_sum"] / simulations}
                if args.backend == "native_chance":
                    row["chance"] = dict(job.core.chance_diagnostics())
                    assert row["chance"]["root_hidden_samples"] == simulations
                records.append(row)
            del job, jobs, policy
            gc.collect()
            print(json.dumps({"event": "depth_chunk", "case": name,
                              "completed_roots": len(records), "total_roots": len(indices),
                              "chunk_seconds": time.perf_counter() - tick}, ensure_ascii=False), flush=True)
        elapsed = time.perf_counter() - wall_start
        total_simulations = sum(row["stats"]["simulations"] for row in records)
        total_depth = sum(row["stats"]["tree_depth_sum"] for row in records)
        result = {"protocol": "attention-chance-depth-case-v1", "backend": args.backend,
                  "simulations_per_root": simulations, "subset": args.subset,
                  "batch_capacity": args.batch_size, "roots": len(records),
                  "total_simulations": total_simulations, "mean_depth": total_depth / total_simulations,
                  "wall_seconds": elapsed, "cpu_seconds": time.process_time() - cpu_start,
                  "peak_process_rss_kib_cumulative": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  "records": records, "inference_config": inference.to_dict(),
                  "environment": environment_identity(), "tf32_matmul": False, "tf32_cudnn": False}
        write_json(output, result)
        print(json.dumps({"event": "depth_case_done", "case": name,
                          "mean_depth": result["mean_depth"], "wall_seconds": elapsed,
                          "peak_rss_kib": result["peak_process_rss_kib_cumulative"]}, ensure_ascii=False), flush=True)
    assert sha256(checkpoint) == manifest["checkpoint"]["sha256"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("native", "native_chance"), required=True)
    parser.add_argument("--subset", choices=("pilot", "confirmation"), required=True)
    parser.add_argument("--sims", type=int, nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    args = parser.parse_args()
    assert args.batch_size > 0 and all(s > 0 for s in args.sims)
    run(args)
