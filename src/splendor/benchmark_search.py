"""Repeatable fixed-request search and frozen-inference throughput benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import torch

from .contracts import DecisionRequest
from .engine import GameEngine
from .mcts_policy import MCTSConfig, PUCTSearchPolicy
from .opponent_pool import load_frozen_actor_critic_checkpoint
from .replay import to_primitive
from .search_inference import FrozenSearchInference
from .search_policy import PolicyValueSearchPolicy, SearchConfig


def fixed_requests(seed: int = 42) -> tuple[tuple[str, DecisionRequest, list[dict[str, Any]]], ...]:
    engine = GameEngine.new(seed, first_player=0, mode="checked")
    rng = random.Random(seed ^ 0xBEEF)
    actions: list[dict[str, Any]] = []
    selected: list[tuple[str, DecisionRequest, list[dict[str, Any]]]] = []
    for index in range(100):
        current = engine.decision()
        if not isinstance(current, DecisionRequest):
            break
        if index in (0, 20, 40):
            selected.append(({0: "opening", 20: "middle", 40: "late"}[index], current, list(actions)))
        choice = rng.randrange(len(current.legal_actions))
        action = current.legal_actions[choice]
        actions.append({"candidate_index": choice, "action": to_primitive(action)})
        engine.apply(action)
    if len(selected) < 3:
        raise RuntimeError("fixed seed did not produce opening, middle and late requests")
    return tuple(selected)


def micro_requests(seed: int = 42) -> tuple[DecisionRequest, ...]:
    engine = GameEngine.new(seed, first_player=0, mode="checked")
    rng = random.Random(seed ^ 0xBEEF)
    requests = []
    while len(requests) < 64:
        current = engine.decision()
        if not isinstance(current, DecisionRequest):
            raise RuntimeError("fixed microbenchmark trajectory ended before 64 requests")
        requests.append(current)
        engine.apply(current.legal_actions[rng.randrange(len(current.legal_actions))])
    return tuple(requests)


def _case(name: str, model, request: DecisionRequest, device: str, engine_mode: str):
    if name == "ab_d6_top5":
        return PolicyValueSearchPolicy(model, config=SearchConfig(depth=6, top_k=5,
            determinizations=1, seed=9500000, fold_forced_phases=True),
            device=device, engine_mode=engine_mode)
    parameters = {
        "mcts_s1024_n1": (1024, 1),
        "mcts_s256_n2": (256, 2),
        "mcts_s1024_n2": (1024, 2),
    }
    s, n = parameters[name]
    return PUCTSearchPolicy(model, config=MCTSConfig(simulations=s, determinizations=n,
        seed=9500000), device=device, engine_mode=engine_mode)


def run(checkpoint: Path, *, device: str, repeats: int, warmup: int) -> dict[str, Any]:
    torch.set_num_threads(1)
    model = load_frozen_actor_critic_checkpoint(checkpoint, device=device)
    cases = fixed_requests()
    result: dict[str, Any] = {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "python": platform.python_version(), "torch": torch.__version__,
        "platform": platform.platform(), "processor": platform.processor(),
        "cuda_available": torch.cuda.is_available(), "device": device,
        "threads": torch.get_num_threads(), "repeats": repeats, "warmup": warmup,
        "fixed_game_seed": 42, "cases": {}, "inference_microbenchmark": {},
    }
    for position, request, prefix in cases:
        result["cases"][position] = {"phase": request.observation.phase.value,
            "action_prefix": prefix, "searches": {}}
        for name in ("ab_d6_top5_checked", "ab_d6_top5", "mcts_s1024_n1",
                     "mcts_s256_n2", "mcts_s1024_n2"):
            rows = []
            for iteration in range(warmup + repeats):
                engine_mode = "checked" if name.endswith("_checked") else "fast"
                policy = _case(name.removesuffix("_checked"), model, request, device, engine_mode)
                started = time.perf_counter()
                choice = policy.choose(request)
                if device == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                if iteration >= warmup:
                    rows.append({"wall_seconds": elapsed, "chosen_action": to_primitive(choice),
                                 "stats": policy.stats.to_dict()})
            times = [item["wall_seconds"] for item in rows]
            result["cases"][position]["searches"][name] = {
                "median_seconds": statistics.median(times), "min_seconds": min(times),
                "max_seconds": max(times), "trials": rows,
            }
    # This isolates inference from gameplay. It includes feature encoding, batch
    # collation, device transfer and synchronization, and uses distinct requests.
    requests = micro_requests()
    result["micro_phase_counts"] = {
        phase: sum(request.observation.phase.value == phase for request in requests)
        for phase in sorted({request.observation.phase.value for request in requests})
    }
    evaluator = FrozenSearchInference(model, device=device)
    for batch_size in (1, 8, 32, 64):
        batch = requests[:batch_size]
        for _ in range(warmup):
            evaluator.policy_value(batch)
        times = []
        feature_before = evaluator.stats.feature_seconds
        forward_before = evaluator.stats.forward_seconds
        for _ in range(repeats):
            started = time.perf_counter()
            evaluator.policy_value(batch)
            times.append(time.perf_counter() - started)
        result["inference_microbenchmark"][str(batch_size)] = {
            "median_wall_seconds": statistics.median(times),
            "seconds_per_position": statistics.median(times) / batch_size,
            "feature_seconds_total": evaluator.stats.feature_seconds - feature_before,
            "forward_seconds_total": evaluator.stats.forward_seconds - forward_before,
        }
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args(argv)
    if args.repeats <= 0 or args.warmup < 0:
        raise ValueError("repeats must be positive and warmup nonnegative")
    result = run(args.checkpoint, device=args.device, repeats=args.repeats, warmup=args.warmup)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "cases": list(result["cases"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
