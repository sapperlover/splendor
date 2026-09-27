"""Deprecated compatibility CLI for one search-vs-raw fixed matchup.

Use ``python -m splendor.evaluation --config <json>`` for new evaluations.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .evaluation import (
    FixedEvaluationConfig,
    SerializablePolicySpec,
    atomic_write_json,
    run_fixed_evaluation,
)
from .search_policy import decision_quality_metrics


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=250)
    parser.add_argument("--seed-start", type=int, default=8_500_000)
    parser.add_argument("--policy-seed-base", type=int, default=8_600_000)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--determinizations", type=int, default=1)
    parser.add_argument("--search-seed", type=int, default=8_700_000)
    parser.add_argument("--leaf-value-clip", type=float, default=1.0)
    parser.add_argument("--plain-phases", action="store_true")
    parser.add_argument("--max-completed-turns", type=int, default=400)
    parser.add_argument("--bootstrap-resamples", type=int, default=100_000)
    parser.add_argument("--bootstrap-seed", type=int, default=8_800_000)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--summary-every", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.pairs <= 0:
        raise ValueError("pairs must be positive")
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    if args.progress_every <= 0 or args.summary_every <= 0:
        raise ValueError("progress_every and summary_every must be positive")
    if args.bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be positive")
    if args.workers > 1 and args.device != "cpu":
        raise ValueError("parallel search evaluation requires --device cpu")

    checkpoint = args.checkpoint.resolve()
    search_config = {
        "top_k": args.top_k,
        "depth": args.depth,
        "determinizations": args.determinizations,
        "seed": args.search_seed,
        "leaf_value_clip": None if args.leaf_value_clip == 0 else args.leaf_value_clip,
        "fold_forced_phases": not args.plain_phases,
    }
    config = FixedEvaluationConfig(
        policy_a=SerializablePolicySpec.search_checkpoint(
            "policy_value_search", checkpoint, search_config=search_config
        ),
        policy_b=SerializablePolicySpec.greedy_checkpoint(
            "raw_greedy_policy", checkpoint
        ),
        game_seeds=tuple(range(args.seed_start, args.seed_start + args.pairs)),
        policy_seed_base=args.policy_seed_base,
        workers=args.workers,
        device=args.device,
        max_completed_turns=args.max_completed_turns,
        adjudicate_deterministic_policy_cycles=True,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
        metadata={"compatibility_entrypoint": "evaluate_search"},
    )

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

    print(
        "evaluate_search.py is deprecated; use splendor.evaluation with a JSON config.",
        file=sys.stderr,
        flush=True,
    )
    try:
        run = run_fixed_evaluation(
            config,
            args.output,
            resume=args.resume,
            progress_every=args.progress_every,
            summary_every=args.summary_every,
            progress_callback=progress,
        )
    except KeyboardInterrupt:
        print(
            json.dumps(
                {"event": "search_evaluation_interrupted", "summary_written": True},
                sort_keys=True,
            ),
            flush=True,
        )
        return 130

    summary = dict(run.summary)
    search_stats = dict(summary["policy_a_stats"])
    decisions = int(search_stats.get("decisions", 0))
    total_seconds = float(search_stats.get("total_seconds", 0.0))
    search_stats["mean_seconds_per_search_decision"] = (
        total_seconds / decisions if decisions else 0.0
    )
    summary["search"] = {
        "algorithm": "alpha_beta_strict_root",
        "depth_unit": (
            "real_choice_after_forced_phase_folding"
            if search_config["fold_forced_phases"]
            else "model_decision_request"
        ),
        "backup": "root-perspective_minimax",
        "leaf": "critic",
        "leaf_encoding": "state_only",
        "information_set_safe": True,
        **search_config,
    }
    summary["search_stats"] = search_stats
    summary["decision_quality"] = decision_quality_metrics(search_stats)
    atomic_write_json(args.output, summary)
    print(
        json.dumps(
            {
                "event": "search_evaluation_completed",
                "passed": summary["passed"],
                "pairs": summary["completed_pairs"],
                "games": summary["completed_games"],
                "score_rate": summary["win_rate_score"],
                "output": str(args.output.resolve()),
                "game_log_root": str(run.game_log_root.resolve()),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
