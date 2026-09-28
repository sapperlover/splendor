"""Sequential batch=1 search versus FP32 batched search (D6 and MCTS).

Collects a pool of real decision requests from seeded random games, runs the
scalar path per request and the batched path over many requests at once, then
reports chosen-action flip rate, root-value / target differences, physical
batch statistics and wall-clock speedup.
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

from splendor.batched_search import BatchedSearchRunner
from splendor.contracts import DecisionRequest
from splendor.engine import GameEngine
from splendor.mcts_policy import MCTSConfig, PUCTSearchPolicy
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.replay import to_primitive
from splendor.search_policy import PolicyValueSearchPolicy, SearchConfig


def collect_requests(count: int, *, seed: int, stride: int = 4) -> list[DecisionRequest]:
    """Collect decision requests with at least two candidates from random games."""

    rng = random.Random(seed)
    requests: list[DecisionRequest] = []
    game_seed = seed * 1000
    while len(requests) < count:
        engine = GameEngine.new(game_seed, first_player=game_seed % 2, mode="fast")
        decision = 0
        for _ in range(400):
            current = engine.decision()
            if not isinstance(current, DecisionRequest):
                break
            if len(current.legal_actions) > 1 and decision % stride == 0:
                requests.append(current)
                if len(requests) >= count:
                    break
            engine.apply(current.legal_actions[rng.randrange(len(current.legal_actions))])
            decision += 1
        game_seed += 1
        if game_seed > seed * 1000 + 400:
            raise RuntimeError("could not collect enough decision requests")
    return requests


def _top_two_margin(values: Sequence[float]) -> float | None:
    ordered = sorted(values, reverse=True)
    if len(ordered) < 2:
        return None
    return ordered[0] - ordered[1]


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def sequential_d6(policy: PolicyValueSearchPolicy, requests, device: str):
    started = time.perf_counter()
    choices = []
    values = []
    logical = []
    seen = 0
    for request in requests:
        choices.append(policy.choose(request))
        values.append(dict(policy.last_root_values or {}))
        logical.append(policy.stats.model_positions - seen)
        seen = policy.stats.model_positions
    _sync(device)
    return choices, time.perf_counter() - started, values, logical


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_num_threads(args.threads)
    model = load_frozen_actor_critic_checkpoint(args.checkpoint, device=args.device)
    requests = collect_requests(args.positions, seed=args.seed)
    result: dict[str, Any] = {
        "algorithm": args.algorithm,
        "checkpoint": str(args.checkpoint.resolve()),
        "device": args.device,
        "threads": args.threads,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "positions": len(requests),
        "batch_size": args.batch_size,
        "repeats": args.repeats,
        "seed": args.seed,
    }
    if args.algorithm == "d6":
        config = SearchConfig(
            depth=args.depth, top_k=args.top_k, determinizations=args.determinizations,
            seed=args.search_seed, fold_forced_phases=not args.plain_phases,
        )
        policy = PolicyValueSearchPolicy(model, config=config, device=args.device,
                                         engine_mode="fast")
        baseline_runs = []
        for _ in range(args.repeats):
            choices, elapsed, values, logical = sequential_d6(policy, requests, args.device)
            baseline_runs.append({"seconds": elapsed, "choices": choices, "values": values,
                                  "logical": logical})
        runner = BatchedSearchRunner(policy, max_batch_size=args.batch_size)
        batched_runs = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            batched_choices = runner.choose_many(
                policy, requests, pool_size=args.pool_size
            )
            _sync(args.device)
            elapsed = time.perf_counter() - started
            sessions = runner.last_sessions
            stats = runner.stats
            batched_runs.append({
                "seconds": elapsed,
                "choices": batched_choices,
                "sessions": sessions,
                "stats": stats.to_dict(),
            })
            runner.stats = type(stats)()
        baseline = baseline_runs[0]
        flips = [index for index, (a, b) in enumerate(zip(baseline["choices"],
                                                          batched_runs[0]["choices"])) if a != b]
        margins = [
            margin
            for margin in (
                _top_two_margin(list(values.values())) for values in baseline["values"]
            )
            if margin is not None
        ]
        value_diffs = []
        matched_values = 0
        for index, session in enumerate(batched_runs[0]["sessions"]):
            reference = baseline["values"][index]
            current = session.policy.last_root_values or {}
            for action, value in current.items():
                if action in reference:
                    matched_values += 1
                    value_diffs.append(abs(value - reference[action]))
        logical = sum(session.policy.stats.model_positions
                      for session in batched_runs[0]["sessions"])
        result.update({
            "sequential_seconds": [run_["seconds"] for run_ in baseline_runs],
            "batched_seconds": [run_["seconds"] for run_ in batched_runs],
            "flip_count": len(flips),
            "flip_rate": len(flips) / len(requests),
            "flipped_positions": flips,
            "max_root_value_diff": max(value_diffs, default=0.0),
            "root_value_margin_median": statistics.median(margins) if margins else None,
            "root_value_margin_min": min(margins) if margins else None,
            "positions_with_margin_below_1e-6": sum(m < 1e-6 for m in margins),
            "matched_root_values": matched_values,
            "logical_positions": logical,
            "sequential_logical_positions": sum(baseline["logical"]),
            "physical_batches": batched_runs[0]["stats"]["physical_batches"],
            "mean_batch_size": batched_runs[0]["stats"]["mean_batch_size"],
            "batch_size_histogram": batched_runs[0]["stats"]["batch_size_histogram"],
            "deduplicated_positions": batched_runs[0]["stats"]["deduplicated_positions"],
            "forward_seconds": batched_runs[0]["stats"]["forward_seconds"],
        })
    else:
        config = MCTSConfig(
            simulations=args.simulations, determinizations=args.mcts_determinizations,
            seed=args.search_seed, fold_forced_phases=not args.plain_phases,
        )
        policy = PUCTSearchPolicy(model, config=config, device=args.device, engine_mode="fast")
        baseline_runs = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            seen = policy.stats.model_positions
            results = [policy.search(request) for request in requests]
            _sync(args.device)
            baseline_runs.append({"seconds": time.perf_counter() - started, "results": results,
                                  "logical": policy.stats.model_positions - seen})
        runner = BatchedSearchRunner(policy, max_batch_size=args.batch_size)
        batched_runs = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            seen = policy.stats.model_positions
            batched_results = runner.search_many(policy, requests)
            _sync(args.device)
            stats = runner.stats
            batched_runs.append({
                "seconds": time.perf_counter() - started,
                "results": batched_results,
                "stats": stats.to_dict(),
                "logical": policy.stats.model_positions - seen,
            })
            runner.stats = type(stats)()
        baseline = baseline_runs[0]
        flips = []
        target_diffs = []
        margins = []
        visit_margins = []
        for match_result in baseline["results"]:
            margin = _top_two_margin(match_result.policy_target)
            if margin is not None:
                margins.append(margin)
            ordered_visits = sorted(match_result.root_visit_counts, reverse=True)
            if len(ordered_visits) >= 2:
                visit_margins.append(ordered_visits[0] - ordered_visits[1])
        for index, (a, b) in enumerate(zip(baseline["results"], batched_runs[0]["results"])):
            if requests[index].legal_actions[a.chosen_candidate_index] != requests[index].legal_actions[b.chosen_candidate_index]:
                flips.append(index)
            target_diffs.append(
                max(abs(x - y) for x, y in zip(a.policy_target, b.policy_target))
            )
        result.update({
            "simulations": args.simulations,
            "sequential_logical_positions": baseline["logical"],
            "batched_logical_positions": batched_runs[0]["logical"],
            "determinizations": args.mcts_determinizations,
            "sequential_seconds": [run_["seconds"] for run_ in baseline_runs],
            "batched_seconds": [run_["seconds"] for run_ in batched_runs],
            "flip_count": len(flips),
            "flip_rate": len(flips) / len(requests),
            "flipped_positions": flips,
            "max_target_diff": max(target_diffs, default=0.0),
            "target_margin_median": statistics.median(margins) if margins else None,
            "visit_margin_min": min(visit_margins) if visit_margins else None,
            "positions_with_visit_margin_le_1": sum(v <= 1 for v in visit_margins),
            "mean_target_diff": statistics.mean(target_diffs) if target_diffs else 0.0,
            "physical_batches": batched_runs[0]["stats"]["physical_batches"],
            "mean_batch_size": batched_runs[0]["stats"]["mean_batch_size"],
            "batch_size_histogram": batched_runs[0]["stats"]["batch_size_histogram"],
            "forward_seconds": batched_runs[0]["stats"]["forward_seconds"],
        })
    result["sequential_median_seconds"] = statistics.median(result["sequential_seconds"])
    result["batched_median_seconds"] = statistics.median(result["batched_seconds"])
    result["speedup"] = (
        result["sequential_median_seconds"] / result["batched_median_seconds"]
        if result["batched_median_seconds"] else 0.0
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--algorithm", choices=("d6", "mcts"), default="d6")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--positions", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--pool-size", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--search-seed", type=int, default=9500000)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--determinizations", type=int, default=1)
    parser.add_argument("--simulations", type=int, default=256)
    parser.add_argument("--mcts-determinizations", type=int, default=2)
    parser.add_argument("--plain-phases", action="store_true")
    args = parser.parse_args(argv)
    if min(args.positions, args.batch_size, args.repeats) <= 0:
        raise ValueError("positions, batch size and repeats must be positive")
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n",
                           encoding="utf-8")
    summary = {key: value for key, value in result.items()
               if key not in ("sequential_seconds", "batched_seconds", "flipped_positions")}
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    print(f"written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
