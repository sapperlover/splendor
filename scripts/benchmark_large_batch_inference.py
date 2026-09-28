"""CPU/GPU throughput comparison for large batches of real search requests.

Builds a realistic pool of frozen-search requests by playing seeded random
games with the fast engine, then times ``policy_value`` / ``policy_only`` /
``value_only`` at batch sizes up to ``--max-batch``.  Every batch contains
distinct requests; timings include feature encoding, collation, device
transfer and output retrieval, exactly as the search/collector paths pay them.

Example:

    conda run --no-capture-output -n splendor python -u \
      scripts/benchmark_large_batch_inference.py \
      --checkpoint runs/5c_main_seed1/checkpoints/local_005000000_lifetime_005512112.pt \
      --requests 1024 --max-batch 512 --repeats 5 \
      --output runtime/search_accel/large_batch_cpu.json --device cpu --threads 1
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import statistics
import time
from pathlib import Path
from typing import Any, Sequence

import torch

from splendor.contracts import DecisionRequest
from splendor.engine import GameEngine
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.search_inference import FrozenSearchInference


def collect_requests(count: int, *, seed: int = 20240928) -> tuple[DecisionRequest, ...]:
    """Collect distinct policy requests from seeded random fast-engine games."""

    rng = random.Random(seed)
    requests: list[DecisionRequest] = []
    game_seed = seed
    while len(requests) < count:
        engine = GameEngine.new(game_seed, first_player=game_seed % 2, mode="fast")
        for _ in range(400):
            current = engine.decision()
            if not isinstance(current, DecisionRequest):
                break
            requests.append(current)
            engine.apply(current.legal_actions[rng.randrange(len(current.legal_actions))])
            if len(requests) >= count:
                break
        game_seed += 1
    return tuple(requests[:count])


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def measure(
    evaluator: FrozenSearchInference,
    batch: Sequence[DecisionRequest],
    method: str,
    *,
    device: str,
    repeats: int,
) -> dict[str, float]:
    fn = getattr(evaluator, method)
    for _ in range(2):
        fn(batch)
    _sync(device)
    times = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn(batch)
        _sync(device)
        times.append(time.perf_counter() - started)
    median = statistics.median(times)
    return {
        "median_seconds": median,
        "min_seconds": min(times),
        "max_seconds": max(times),
        "seconds_per_position": median / len(batch),
        "positions_per_second": len(batch) / median,
    }


def run(
    checkpoint: Path,
    *,
    device: str,
    threads: int,
    request_count: int,
    max_batch: int,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    torch.set_num_threads(threads)
    model = load_frozen_actor_critic_checkpoint(checkpoint, device=device)
    evaluator = FrozenSearchInference(model, device=device)
    requests = collect_requests(request_count, seed=seed)
    phases: dict[str, int] = {}
    candidate_counts: list[int] = []
    for request in requests:
        key = request.observation.phase.value
        phases[key] = phases.get(key, 0) + 1
        candidate_counts.append(len(request.legal_actions))

    result: dict[str, Any] = {
        "checkpoint": str(checkpoint.resolve()),
        "device": device,
        "threads": threads,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "request_seed": seed,
        "request_count": len(requests),
        "phase_counts": phases,
        "candidate_count_median": statistics.median(candidate_counts),
        "candidate_count_max": max(candidate_counts),
        "repeats": repeats,
        "max_batch": max_batch,
        "batches": {},
    }
    if device == "cuda":
        result["cuda_device_name"] = torch.cuda.get_device_name(0)
        torch.cuda.reset_peak_memory_stats()
    sizes = []
    size = 1
    while size <= max_batch:
        sizes.append(size)
        size *= 2
    if max_batch not in sizes:
        sizes.append(max_batch)
    for size in sizes:
        if size > len(requests):
            break
        batch = requests[:size]
        entry: dict[str, Any] = {"batch_size": size}
        for method in ("policy_value", "policy_only", "value_only"):
            entry[method] = measure(evaluator, batch, method, device=device, repeats=repeats)
        result["batches"][str(size)] = entry
    if device == "cuda":
        result["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
        result["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / (1024 * 1024)
    return result


def summarize(result: dict[str, Any]) -> None:
    print(f"device={result['device']} threads={result['threads']} "
          f"requests={result['request_count']} phases={result['phase_counts']}")
    header = f"{'batch':>6} {'policy_value ms/pos':>20} {'policy_only ms/pos':>19} {'value_only ms/pos':>18}"
    print(header)
    for size, entry in result["batches"].items():
        print(
            f"{size:>6} "
            f"{entry['policy_value']['seconds_per_position'] * 1e3:>20.3f} "
            f"{entry['policy_only']['seconds_per_position'] * 1e3:>19.3f} "
            f"{entry['value_only']['seconds_per_position'] * 1e3:>18.3f}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--requests", type=int, default=1024)
    parser.add_argument("--max-batch", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20240928)
    args = parser.parse_args(argv)
    if min(args.threads, args.requests, args.max_batch, args.repeats) <= 0:
        raise ValueError("threads, requests, max-batch and repeats must be positive")
    result = run(
        args.checkpoint,
        device=args.device,
        threads=args.threads,
        request_count=args.requests,
        max_batch=args.max_batch,
        repeats=args.repeats,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summarize(result)
    print(f"written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
