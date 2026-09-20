"""Standalone quick, formal DEV, and held-out FINAL evaluation for 5C."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import torch

from .opponent_pool import load_frozen_actor_critic_checkpoint
from .paired_statistics import pair_scores, paired_bootstrap_mean
from .train_5c import (
    _FORMAL_GAME_RNG_OFFSET,
    _FORMAL_POLICY_RNG_OFFSET,
    _FINAL_POLICY_RNG_OFFSET,
    _QUICK_GAME_RNG_OFFSET,
    _QUICK_POLICY_RNG_OFFSET,
    _evaluation_game_seeds,
    _panel_record,
    _write_panel,
    build_five_c_eval_opponents,
    evaluate_five_c_panel,
    final_eval_game_seeds,
    load_five_c_config,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    checkpoint = parser.add_mutually_exclusive_group(required=True)
    checkpoint.add_argument("--checkpoint", type=Path)
    checkpoint.add_argument("--checkpoint-manifest", type=Path)
    parser.add_argument("--kind", choices=("quick", "formal", "final"), required=True)
    parser.add_argument("--pairs", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=100_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_five_c_config(args.config)
    if args.checkpoint_manifest is not None:
        manifest = json.loads(args.checkpoint_manifest.read_text(encoding="utf-8"))
        checkpoint_value = manifest.get("new_best_checkpoint", manifest.get("checkpoint"))
        if checkpoint_value is None:
            raise ValueError("checkpoint manifest has no checkpoint path")
        checkpoint = Path(checkpoint_value)
    else:
        checkpoint = args.checkpoint
    if checkpoint is None:
        raise AssertionError("checkpoint argument was not resolved")
    if args.kind == "quick":
        default_pairs = config.quick_eval_pairs
        seed_domain = "QUICK"
        game_seed_offset = _QUICK_GAME_RNG_OFFSET
        policy_seed_offset = _QUICK_POLICY_RNG_OFFSET
        panel_seed = config.quick_panel_seed
    elif args.kind == "formal":
        default_pairs = config.formal_eval_pairs
        seed_domain = "FORMAL"
        game_seed_offset = _FORMAL_GAME_RNG_OFFSET
        policy_seed_offset = _FORMAL_POLICY_RNG_OFFSET
        panel_seed = config.formal_panel_seed
    else:
        default_pairs = config.final_eval_min_pairs
        seed_domain = "FINAL"
        game_seed_offset = None
        policy_seed_offset = _FINAL_POLICY_RNG_OFFSET
        panel_seed = config.final_panel_seed
    pairs = default_pairs if args.pairs is None else args.pairs
    if pairs <= 0:
        raise ValueError("evaluation pair count must be positive")
    if args.kind == "final" and pairs < config.final_eval_min_pairs:
        raise ValueError("5C FINAL evaluation requires at least 500 paired seeds")

    model = load_frozen_actor_critic_checkpoint(checkpoint, device=args.device)
    opponents = build_five_c_eval_opponents(
        config,
        device=args.device,
        kind=args.kind,
    )
    if game_seed_offset is not None:
        seeds = _evaluation_game_seeds(game_seed_offset, panel_seed, pairs)
        policy_seed_base = -(policy_seed_offset + panel_seed * 1_000_000)
    else:
        seeds = final_eval_game_seeds(config.final_panel_seed, pairs)
        policy_seed_base = -(
            _FINAL_POLICY_RNG_OFFSET + config.final_panel_seed * 10_000_000
        )
    results, elapsed = evaluate_five_c_panel(
        model,
        opponents=opponents,
        seeds=seeds,
        policy_seed_base=policy_seed_base,
        device=args.device,
        max_completed_turns=config.max_completed_turns,
        failure_dir=args.output.parent / "failures" / args.output.stem,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    lifetime_step = int(payload["global_learning_decision_step"])
    source_lifetime = 512_112
    run_local_step = max(0, lifetime_step - source_lifetime)
    source_updates = int(
        torch.load(
            Path(config.source_checkpoint), map_location="cpu", weights_only=True
        )["update_count"]
    )
    lifetime_updates = int(payload["update_count"])
    record = _panel_record(
        results,
        kind=args.kind,
        run_local_step=run_local_step,
        lifetime_step=lifetime_step,
        run_local_update_count=max(0, lifetime_updates - source_updates),
        lifetime_update_count=lifetime_updates,
        elapsed_seconds=elapsed,
    )
    _write_panel(
        args.output,
        results=results,
        record=record,
        config=config,
        seed_domain=seed_domain,
        checkpoint_path=checkpoint,
    )
    stored = json.loads(args.output.read_text(encoding="utf-8"))
    stored["paired_bootstrap_95"] = {
        name: paired_bootstrap_mean(
            pair_scores(result)[1],
            resamples=args.bootstrap_resamples,
            seed=(
                5_100_000_000
                + (0 if seed_domain == "DEV" else 1_000_000)
                + index
            ),
        ).to_dict()
        for index, (name, result) in enumerate(results.items())
    }
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(stored, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint.resolve()),
                "kind": args.kind,
                "seed_domain": seed_domain,
                "pairs": pairs,
                "panel_summary": record,
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
