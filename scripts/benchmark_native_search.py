"""Same-checkpoint fixed-position Python/native PUCT panel."""

from __future__ import annotations

import argparse
import json
import resource
import statistics
import time
from dataclasses import asdict
from pathlib import Path

import torch

from splendor.benchmark_search import fixed_requests
from splendor.mcts_policy import MCTSConfig, PUCTSearchPolicy
from splendor.native_policy import NativePUCTSearchPolicy
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint


def run(checkpoint: Path, *, device: str, simulations: int, worlds: int,
        repeats: int, warmup: int) -> dict:
    torch.set_num_threads(1)
    model = load_frozen_actor_critic_checkpoint(checkpoint, device=device)
    config = MCTSConfig(simulations=simulations, determinizations=worlds,
                        seed=9500000, root_noise_fraction=0)
    output = {"checkpoint": str(checkpoint.resolve()), "device": device,
              "simulations": simulations, "determinizations": worlds,
              "repeats": repeats, "warmup": warmup, "cases": {}}
    for name, request, _trace in fixed_requests():
        row = {}
        for backend, cls in (("python", PUCTSearchPolicy), ("native", NativePUCTSearchPolicy)):
            policy = cls(model, config=config, device=device)
            for _ in range(warmup):
                policy.search(request)
            durations = []
            samples = []
            for _ in range(repeats):
                before = policy.stats.model_positions
                started = time.perf_counter()
                result = policy.search(request)
                durations.append(time.perf_counter() - started)
                samples.append({"chosen_candidate_index": result.chosen_candidate_index,
                                "simulations": result.simulations,
                                "model_positions": policy.stats.model_positions - before,
                                "root_visit_counts": result.root_visit_counts})
            row[backend] = {"seconds": durations, "median_seconds": statistics.median(durations),
                            "samples": samples}
        output["cases"][name] = row
    output["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--simulations", type=int, default=1024)
    parser.add_argument("--determinizations", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()
    result = run(args.checkpoint, device=args.device, simulations=args.simulations,
                 worlds=args.determinizations, repeats=args.repeats, warmup=args.warmup)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({name: {backend: item["median_seconds"] for backend, item in values.items()}
                      for name, values in result["cases"].items()}, sort_keys=True))


if __name__ == "__main__":
    main()
