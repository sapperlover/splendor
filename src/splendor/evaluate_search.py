"""Evaluate policy/critic search directly against its raw greedy policy."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import fields, replace
from pathlib import Path
from typing import Sequence

import torch

from .evaluation import (
    PolicySpec,
    SerializablePolicySpec,
    evaluate_policies,
    evaluate_policies_parallel,
)
from .neural_policy import GreedyActorCriticPolicy
from .opponent_pool import load_frozen_actor_critic_checkpoint
from .paired_statistics import pair_scores, paired_bootstrap_mean
from .search_policy import (
    PolicyValueSearchPolicy,
    SearchConfig,
    SearchStats,
    decision_quality_metrics,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--pairs",
        type=int,
        default=250,
        help="paired seeds; 250 pairs produce 500 games",
    )
    parser.add_argument("--seed-start", type=int, default=8_500_000)
    parser.add_argument("--policy-seed-base", type=int, default=8_600_000)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--determinizations", type=int, default=1)
    parser.add_argument("--search-seed", type=int, default=8_700_000)
    parser.add_argument(
        "--leaf-value-clip",
        type=float,
        default=1.0,
        help="critic clipping magnitude; use 0 to disable",
    )
    parser.add_argument(
        "--plain-phases",
        action="store_true",
        help=(
            "disable macro folding for diagnostics; strength curves use the "
            "default folded depth, where a ply is one real choice instead of "
            "one model decision request"
        ),
    )
    parser.add_argument("--max-completed-turns", type=int, default=400)
    parser.add_argument("--bootstrap-resamples", type=int, default=100_000)
    parser.add_argument("--bootstrap-seed", type=int, default=8_800_000)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="pair-parallel CPU workers; values above 1 force CPU inference",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=5,
        help="print one compact progress event every N paired seeds",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _sum_search_stats(policies: list[PolicyValueSearchPolicy]) -> dict[str, int | float]:
    result: dict[str, int | float] = {}
    for field in fields(SearchStats):
        result[field.name] = sum(
            getattr(policy.stats, field.name) for policy in policies
        )
    decisions = int(result["decisions"])
    total_seconds = float(result["total_seconds"])
    result["mean_seconds_per_search_decision"] = (
        total_seconds / decisions if decisions else 0.0
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.pairs <= 0:
        raise ValueError("pairs must be positive")
    if args.progress_every <= 0:
        raise ValueError("progress_every must be positive")
    if args.bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be positive")
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    if args.workers > 1 and args.device != "cpu":
        raise ValueError("parallel search evaluation requires --device cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    checkpoint = args.checkpoint.resolve()
    base_config = SearchConfig(
        top_k=args.top_k,
        depth=args.depth,
        determinizations=args.determinizations,
        seed=args.search_seed,
        leaf_value_clip=(
            None if args.leaf_value_clip == 0 else args.leaf_value_clip
        ),
        fold_forced_phases=not args.plain_phases,
    )
    search_policies: list[PolicyValueSearchPolicy] = []
    model = (
        load_frozen_actor_critic_checkpoint(checkpoint, device=args.device)
        if args.workers == 1
        else None
    )

    def search_factory(seed: int) -> PolicyValueSearchPolicy:
        if model is None:
            raise AssertionError("serial search model was not loaded")
        policy = PolicyValueSearchPolicy(
            model,
            config=replace(base_config, seed=base_config.seed ^ seed),
            device=args.device,
        )
        search_policies.append(policy)
        return policy

    def raw_factory(seed: int) -> GreedyActorCriticPolicy:
        del seed
        if model is None:
            raise AssertionError("serial greedy model was not loaded")
        return GreedyActorCriticPolicy(model, device=args.device)

    def progress(completed: int, total: int) -> None:
        if completed % args.progress_every == 0 or completed == total:
            print(
                json.dumps(
                    {
                        "event": "search_evaluation_progress",
                        "completed_pairs": completed,
                        "total_pairs": total,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    started = time.perf_counter()
    seeds = tuple(range(args.seed_start, args.seed_start + args.pairs))
    if args.workers == 1:
        result = evaluate_policies(
            PolicySpec("policy_value_search", search_factory),
            PolicySpec("raw_greedy_policy", raw_factory),
            seeds,
            policy_seed_base=args.policy_seed_base,
            max_completed_turns=args.max_completed_turns,
            failure_dir=args.output.parent / "failures" / args.output.stem,
            progress_callback=progress,
            adjudicate_deterministic_policy_cycles=True,
        )
        search_stats = _sum_search_stats(search_policies)
    else:
        parallel = evaluate_policies_parallel(
            SerializablePolicySpec.search_checkpoint(
                "policy_value_search",
                checkpoint,
                search_config={
                    "top_k": base_config.top_k,
                    "depth": base_config.depth,
                    "determinizations": base_config.determinizations,
                    "seed": base_config.seed,
                    "leaf_value_clip": base_config.leaf_value_clip,
                    "fold_forced_phases": base_config.fold_forced_phases,
                },
            ),
            SerializablePolicySpec.greedy_checkpoint(
                "raw_greedy_policy", checkpoint
            ),
            seeds,
            workers=args.workers,
            policy_seed_base=args.policy_seed_base,
            max_completed_turns=args.max_completed_turns,
            failure_dir=args.output.parent / "failures" / args.output.stem,
            progress_callback=progress,
            adjudicate_deterministic_policy_cycles=True,
        )
        result = parallel.result
        search_stats = dict(parallel.policy_a_stats)
        decisions = int(search_stats.get("decisions", 0))
        total_seconds = float(search_stats.get("total_seconds", 0.0))
        search_stats["mean_seconds_per_search_decision"] = (
            total_seconds / decisions if decisions else 0.0
        )
    elapsed = time.perf_counter() - started
    _, scores = pair_scores(result)
    bootstrap = paired_bootstrap_mean(
        scores,
        resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed,
    )
    checkpoint_payload = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    payload = {
        "protocol": "policy-value-search-v1",
        "checkpoint": str(checkpoint),
        "checkpoint_lifetime_learning_decisions": checkpoint_payload.get(
            "global_learning_decision_step"
        ),
        "device": args.device,
        "workers": args.workers,
        "search": {
            "algorithm": "alpha_beta_strict_root",
            "top_k": base_config.top_k,
            "depth": base_config.depth,
            "depth_unit": "model_decision_request",
            "backup": "root-perspective_minimax",
            "leaf": "critic",
            "leaf_encoding": "state_only",
            "leaf_value_clip": base_config.leaf_value_clip,
            "determinizations": base_config.determinizations,
            "fold_forced_phases": base_config.fold_forced_phases,
            "information_set_safe": True,
            "seed": base_config.seed,
        },
        "evaluation": {
            "pairs": args.pairs,
            "games": args.pairs * 2,
            "seed_start": args.seed_start,
            "policy_seed_base": args.policy_seed_base,
            "elapsed_seconds": elapsed,
            "paired_bootstrap": bootstrap.to_dict(),
            "result": result.to_dict(include_matches=True),
        },
        "search_stats": search_stats,
        "decision_quality": decision_quality_metrics(search_stats),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "event": "search_evaluation_completed",
                "passed": result.passed,
                "pairs": args.pairs,
                "games": args.pairs * 2,
                "score_rate": result.win_rate_score,
                "elapsed_seconds": elapsed,
                "output": str(args.output.resolve()),
                "search_stats": payload["search_stats"],
                "decision_quality": payload["decision_quality"],
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
