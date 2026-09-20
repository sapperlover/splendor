"""Expanded Stage 4 checkpoint evaluation and paired bootstrap report."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .evaluation import PolicySpec, evaluate_policies, save_evaluation
from .neural_policy import GreedyActorCriticPolicy
from .opponent_pool import load_frozen_actor_critic_checkpoint
from .paired_statistics import (
    pair_scores,
    paired_bootstrap_difference,
    paired_bootstrap_mean,
    serialized_pair_scores,
)
from .train_5a import dev_eval_game_seeds


_PANEL_POLICY_SEED_OFFSET = 6_000_000_000_000_000


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _heuristic_differences(
    source_panel_path: Path,
    candidate_panel_path: Path,
    *,
    resamples: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    source = _load_json(source_panel_path)["evaluations"]
    candidate = _load_json(candidate_panel_path)["evaluations"]
    output: dict[str, object] = {}
    for index, name in enumerate(("random", "score", "development")):
        source_seeds, source_scores = serialized_pair_scores(source[name])
        candidate_seeds, candidate_scores = serialized_pair_scores(candidate[name])
        interval = paired_bootstrap_difference(
            candidate_seeds,
            candidate_scores,
            source_seeds,
            source_scores,
            resamples=resamples,
            seed=bootstrap_seed + index + 1,
        )
        output[name] = {
            "source_score_rate": float(source_scores.mean()),
            "candidate_score_rate": float(candidate_scores.mean()),
            "candidate_minus_source": interval.to_dict(),
        }
    return output


def run(args: argparse.Namespace) -> dict[str, object]:
    output_dir: Path = args.output_dir
    direct_path = output_dir / "direct_512112_vs_403183.json"
    if args.reuse_direct and direct_path.exists():
        payload = _load_json(direct_path)
        direct_seeds, direct_scores = serialized_pair_scores(payload)
        elapsed = None
        direct_summary = {
            key: payload[key]
            for key in (
                "a_wins",
                "draws",
                "a_losses",
                "win_rate_score",
                "a_first",
                "a_second",
                "total_pairs",
                "total_games",
                "valid_games",
                "engineering_truncations",
                "exceptions",
                "illegal_actions",
                "invariant_failures",
                "passed",
            )
        }
    else:
        candidate_model = load_frozen_actor_critic_checkpoint(
            args.candidate_checkpoint, device=args.device
        )
        source_model = load_frozen_actor_critic_checkpoint(
            args.source_checkpoint, device=args.device
        )
        candidate = PolicySpec(
            "ppo-512112-greedy",
            lambda seed: GreedyActorCriticPolicy(candidate_model, device=args.device),
        )
        source = PolicySpec(
            "ppo-403183-greedy",
            lambda seed: GreedyActorCriticPolicy(source_model, device=args.device),
        )
        seeds = dev_eval_game_seeds(args.panel_seed, args.pairs)
        started = time.perf_counter()
        result = evaluate_policies(
            candidate,
            source,
            seeds,
            policy_seed_base=-(
                _PANEL_POLICY_SEED_OFFSET + args.panel_seed * 10_000_000
            ),
            max_completed_turns=args.max_completed_turns,
            failure_dir=output_dir / "failures",
        )
        elapsed = time.perf_counter() - started
        if not result.passed:
            save_evaluation(direct_path, result)
            raise RuntimeError("expanded direct evaluation contains invalid games")
        save_evaluation(direct_path, result)
        direct_seeds, direct_scores = pair_scores(result)
        direct_summary = {
            "a_wins": result.a_wins,
            "draws": result.draws,
            "a_losses": result.a_losses,
            "win_rate_score": result.win_rate_score,
            "a_first": result.to_dict(include_matches=False)["a_first"],
            "a_second": result.to_dict(include_matches=False)["a_second"],
            "total_pairs": result.total_pairs,
            "total_games": result.total_games,
            "valid_games": result.valid_games,
            "engineering_truncations": result.engineering_truncations,
            "exceptions": result.exceptions,
            "illegal_actions": result.illegal_actions,
            "invariant_failures": result.invariant_failures,
            "passed": result.passed,
        }

    if len(direct_seeds) != args.pairs:
        raise ValueError(
            f"direct evaluation has {len(direct_seeds)} pairs, expected {args.pairs}"
        )
    direct_interval = paired_bootstrap_mean(
        direct_scores,
        resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed,
    )
    report: dict[str, object] = {
        "method": {
            "confidence": 0.95,
            "bootstrap": "paired-seed percentile",
            "bootstrap_resamples": args.bootstrap_resamples,
            "bootstrap_seed": args.bootstrap_seed,
            "pair_unit": "mean of the two swapped-first-player games for one game seed",
            "panel_seed": args.panel_seed,
            "policy_seed_base": -(
                _PANEL_POLICY_SEED_OFFSET + args.panel_seed * 10_000_000
            ),
        },
        "checkpoints": {
            "source_403183": str(args.source_checkpoint),
            "candidate_512112": str(args.candidate_checkpoint),
        },
        "direct_512112_vs_403183": {
            **direct_summary,
            "elapsed_seconds": elapsed,
            "score_rate_ci": direct_interval.to_dict(),
            "evaluation_path": str(direct_path),
        },
        "heuristic_paired_differences": _heuristic_differences(
            args.source_panel,
            args.candidate_panel,
            resamples=args.bootstrap_resamples,
            bootstrap_seed=args.bootstrap_seed,
        ),
    }
    _write_json(output_dir / "paired_bootstrap_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--source-panel", type=Path, required=True)
    parser.add_argument("--candidate-panel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=1_000)
    parser.add_argument("--panel-seed", type=int, default=2)
    parser.add_argument("--bootstrap-resamples", type=int, default=100_000)
    parser.add_argument("--bootstrap-seed", type=int, default=5_120_403_183)
    parser.add_argument("--max-completed-turns", type=int, default=400)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--reuse-direct", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
