"""Folded-depth ladder: how much strength does search depth still buy?

The ladder walks rungs of adjacent depths, each played as a paired, seat-swapped
search-vs-search match.  Folding forced plies into macro transitions is the
default here, so ``depth`` counts real choices:

* up-phase: start at ``--start-depth`` and evaluate ``d+1`` vs ``d``;
* advance when the cumulative score rate is **above** ``--threshold`` (0.54);
* when the rate sits between ``--floor`` (0.50) and the threshold inclusive,
  play one more block and decide again;
* below the floor the rung fails immediately;
* when an up rung fails, the ladder turns around and probes downwards
  (``d`` vs ``d-1``, then ``d-1`` vs ``d-2``, ...).  The first downward probe
  whose rate clears the threshold stops the whole ladder; probes keep walking
  down while they stay at or below the threshold, until ``--min-depth``.

Durability is provided by ``evaluation.py``: every worker appends one fsynced
JSON line per *finished game* to
``<output>.games/stage-XXX/bYY/pair-NNNNNNN.jsonl``. Stopping at any moment
keeps completed pairs; ``--resume`` replays only incomplete pairs with their
original pair ids so policy seeds stay reproducible. This module owns the rung
schedule, threshold decisions, and stage-level statistics.

Reference run (frozen 5.5M checkpoint, top-5, folded, 8 workers):

~~~bash
python -u -m splendor.ladder_search \
  --checkpoint runs/5c_main_seed1/checkpoints/local_005000000_lifetime_005512112.pt \
  --output artifacts/work_package_5c/folded_depth_ladder_5p5m.json \
  --start-depth 4 --max-depth 6 --workers 8
~~~
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

from .evaluation import (
    EvaluationGameStore as GameStore,
    atomic_write_json,
    FixedEvaluationConfig,
    ParallelPairResult,
    SerializablePolicySpec,
    run_fixed_evaluation,
)
from .paired_statistics import is_scored_status, paired_bootstrap_mean
from .runner import RunStatus
from .search_policy import decision_quality_metrics


PROTOCOL = "folded-depth-ladder-v4"
DEFAULT_GAME_SEED_START = 9_100_000
DEFAULT_POLICY_SEED_BASE = 9_200_000
DEFAULT_SEARCH_SEED = 9_300_000
DEFAULT_BOOTSTRAP_SEED = 9_400_000

# Measured folded per-search CPU seconds and search decisions per game
# (5M checkpoint, top-5, one thread, search-vs-search smoke).  Used only for the
# up-front wall-clock projection, and deliberately pessimistic.
_MEASURED_FOLDED_SECONDS = {4: 0.43, 5: 1.56}
_MEASURED_GROWTH_PER_PLY = 3.6
_SEARCH_DECISIONS_PER_GAME_PER_SIDE = 38
_OUTCOME_SCORES = {"a_win": 1.0, "draw": 0.5, "a_loss": 0.0}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-depth", type=int, default=4)
    parser.add_argument(
        "--min-depth",
        type=int,
        default=1,
        help="lowest incumbent depth the downward probe may test",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=8,
        help="stop the up-phase after evaluating this depth against depth-1",
    )
    parser.add_argument(
        "--pairs",
        type=int,
        default=250,
        help="paired seeds per block; 250 pairs = 500 games",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.54,
        help="cumulative score rate above which a rung advances",
    )
    parser.add_argument(
        "--floor",
        type=float,
        default=0.50,
        help="below this cumulative score rate a rung stops immediately",
    )
    parser.add_argument(
        "--extra-blocks",
        type=int,
        default=1,
        help="extra blocks played when the rate sits between floor and threshold",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--determinizations", type=int, default=1)
    parser.add_argument(
        "--leaf-value-clip",
        type=float,
        default=1.0,
        help="critic clipping magnitude; use 0 to disable",
    )
    parser.add_argument("--game-seed-start", type=int, default=DEFAULT_GAME_SEED_START)
    parser.add_argument("--policy-seed-base", type=int, default=DEFAULT_POLICY_SEED_BASE)
    parser.add_argument("--search-seed", type=int, default=DEFAULT_SEARCH_SEED)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--bootstrap-resamples", type=int, default=100_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-completed-turns", type=int, default=400)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument(
        "--summary-every",
        type=int,
        default=25,
        help="rewrite the human-readable JSON summary every N finished pairs",
    )
    parser.add_argument(
        "--plain-phases",
        action="store_true",
        help="disable macro folding (only for diagnostics; curves use folding)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue from an existing game log instead of starting over",
    )
    return parser


def _search_config(args: argparse.Namespace, depth: int) -> dict[str, Any]:
    return {
        "top_k": args.top_k,
        "depth": depth,
        "determinizations": args.determinizations,
        "seed": args.search_seed,
        "leaf_value_clip": (
            None if args.leaf_value_clip == 0 else args.leaf_value_clip
        ),
        "fold_forced_phases": not args.plain_phases,
    }


def _run_header(args: argparse.Namespace, checkpoint: Path) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "checkpoint": str(checkpoint),
        "threshold": args.threshold,
        "floor": args.floor,
        "min_depth": args.min_depth,
        "max_depth": args.max_depth,
        "pairs_per_block": args.pairs,
        "extra_blocks": args.extra_blocks,
        "config": _search_config(args, args.start_depth),
        "seed_domain": {
            "game_seed_start": args.game_seed_start,
            "policy_seed_base": args.policy_seed_base,
            "search_seed": args.search_seed,
            "bootstrap_seed": args.bootstrap_seed,
        },
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def _check_resume_header(
    header: Mapping[str, Any], args: argparse.Namespace, checkpoint: Path
) -> None:
    expected = _run_header(args, checkpoint)
    for name in (
        "protocol",
        "checkpoint",
        "threshold",
        "floor",
        "pairs_per_block",
        "extra_blocks",
        "config",
        "seed_domain",
    ):
        if header.get(name) != expected[name]:
            raise ValueError(
                f"existing game log was written with a different {name}: "
                f"{header.get(name)!r} != {expected[name]!r}"
            )
    # ``min_depth`` / ``max_depth`` only bound how far the plan walks; raising
    # them extends an existing run, so they are checked separately by the
    # caller instead of invalidating the durable log.


@dataclass(frozen=True, slots=True)
class Rung:
    """One adjacent-depth comparison, in execution order."""

    index: int
    mode: str
    incumbent_depth: int
    challenger_depth: int



def _block_seeds(args: argparse.Namespace, block: int) -> tuple[int, ...]:
    start = args.game_seed_start + block * args.pairs
    return tuple(range(start, start + args.pairs))


def _folded_seconds(depth: int) -> float:
    if depth in _MEASURED_FOLDED_SECONDS:
        return _MEASURED_FOLDED_SECONDS[depth]
    if depth < min(_MEASURED_FOLDED_SECONDS):
        base_depth = min(_MEASURED_FOLDED_SECONDS)
        return _MEASURED_FOLDED_SECONDS[base_depth] / (
            _MEASURED_GROWTH_PER_PLY ** (base_depth - depth)
        )
    base_depth = max(_MEASURED_FOLDED_SECONDS)
    return _MEASURED_FOLDED_SECONDS[base_depth] * (
        _MEASURED_GROWTH_PER_PLY ** (depth - base_depth)
    )


def _project_seconds(
    challenger_depth: int, incumbent_depth: int, pairs: int, workers: int
) -> float:
    per_game = _SEARCH_DECISIONS_PER_GAME_PER_SIDE * (
        _folded_seconds(challenger_depth) + _folded_seconds(incumbent_depth)
    )
    return per_game * pairs * 2 / max(1, workers)


def _projection_rows(
    args: argparse.Namespace,
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for candidate in range(args.start_depth + 1, args.max_depth + 1):
        seconds = _project_seconds(candidate, candidate - 1, args.pairs, args.workers)
        rows.append(
            {
                "challenger_depth": candidate,
                "incumbent_depth": candidate - 1,
                "seconds_per_block": round(seconds, 1),
                "seconds_if_escalated": round(
                    seconds * (1 + args.extra_blocks), 1
                ),
                "seconds_per_search_challenger": round(_folded_seconds(candidate), 3),
                "seconds_per_search_incumbent": round(
                    _folded_seconds(candidate - 1), 3
                ),
            }
        )
    for candidate in range(args.start_depth - 1, args.min_depth - 1, -1):
        seconds = _project_seconds(candidate + 1, candidate, args.pairs, args.workers)
        rows.append(
            {
                "challenger_depth": candidate + 1,
                "incumbent_depth": candidate,
                "seconds_per_block": round(seconds, 1),
                "seconds_if_escalated": round(
                    seconds * (1 + args.extra_blocks), 1
                ),
                "direction": "down",
                "seconds_per_search_challenger": round(_folded_seconds(candidate + 1), 3),
                "seconds_per_search_incumbent": round(_folded_seconds(candidate), 3),
            }
        )
    return rows


def _decide(rate: float, args: argparse.Namespace, completed_blocks: int) -> str:
    """Return ``advance``, ``escalate`` or ``stop`` for a cumulative rate."""

    if rate > args.threshold:
        return "advance"
    if rate >= args.floor and completed_blocks < 1 + args.extra_blocks:
        return "escalate"
    return "stop"


def require_scored_pair(pair_result: ParallelPairResult) -> None:
    """Fail loudly instead of averaging a pair that did not produce two games."""

    for match in pair_result.matches:
        if not is_scored_status(match.status) or match.outcome_for_a is None:
            raise RuntimeError(
                f"pair {match.pair_id} seed {match.game_seed} game "
                f"{match.game_in_pair} did not score: status={match.status.value} "
                f"scene={match.failure_scene}"
            )


def _sum_stats(
    pairs: Mapping[Any, Sequence[dict[str, Any]]], key: str
) -> dict[str, int | float]:
    total: dict[str, int | float] = {}
    for games in pairs.values():
        for record in games:
            for name, value in record[key].items():
                if isinstance(value, (int, float)):
                    total[name] = total.get(name, 0) + value
    return total


def _outcome_by_action_change(
    pairs: Mapping[Any, Sequence[dict[str, Any]]], stats_key: str
) -> dict[str, float | int | None]:
    """Game score weighted by decisions that changed / kept the raw action."""

    changed_decisions = 0.0
    kept_decisions = 0.0
    changed_score_sum = 0.0
    kept_score_sum = 0.0
    for games in pairs.values():
        for record in games:
            score = _OUTCOME_SCORES[record["outcome_for_a"]]
            stats = record.get(stats_key, {})
            changed = float(stats.get("changed_action_count", 0))
            searchable = float(stats.get("searchable_decisions", 0))
            kept = max(0.0, searchable - changed)
            changed_decisions += changed
            kept_decisions += kept
            changed_score_sum += score * changed
            kept_score_sum += score * kept
    return {
        "changed_decisions": int(changed_decisions),
        "kept_decisions": int(kept_decisions),
        "score_when_search_changed_action": (
            changed_score_sum / changed_decisions if changed_decisions else None
        ),
        "score_when_search_kept_action": (
            kept_score_sum / kept_decisions if kept_decisions else None
        ),
    }


def _aggregate(
    pairs: Mapping[Any, Sequence[dict[str, Any]]],
    args: argparse.Namespace,
    *,
    resamples: int | None = None,
) -> dict[str, Any]:
    """Aggregate completed pairs; the bootstrap unit is one paired seed."""

    outcomes = [
        record["outcome_for_a"] for games in pairs.values() for record in games
    ]
    wins = sum(1 for outcome in outcomes if outcome == "a_win")
    losses = sum(1 for outcome in outcomes if outcome == "a_loss")
    draws = sum(1 for outcome in outcomes if outcome == "draw")
    games_count = len(outcomes)
    # One value per paired seed: the resampling unit of the bootstrap.
    pair_scores = [
        sum(_OUTCOME_SCORES[record["outcome_for_a"]] for record in games)
        / len(games)
        for games in pairs.values()
    ]
    bootstrap_resamples = (
        args.bootstrap_resamples if resamples is None else resamples
    )
    bootstrap = (
        paired_bootstrap_mean(
            pair_scores,
            resamples=bootstrap_resamples,
            seed=args.bootstrap_seed,
        ).to_dict()
        if bootstrap_resamples > 0 and pair_scores
        else None
    )
    challenger_stats = _sum_stats(pairs, "policy_a_stats")
    incumbent_stats = _sum_stats(pairs, "policy_b_stats")
    game_turns = [
        float(record.get("completed_player_turns", 0))
        for games in pairs.values()
        for record in games
    ]
    game_decisions = [
        float(record.get("decision_count", 0))
        for games in pairs.values()
        for record in games
    ]
    return {
        "pairs": len(pairs),
        "games": games_count,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "score_rate": (wins + 0.5 * draws) / games_count if games_count else 0.0,
        "pair_scores": pair_scores,
        "paired_bootstrap": bootstrap,
        "paired_bootstrap_unit": "one_paired_seed",
        "game_length": {
            "player_turns_mean": statistics.fmean(game_turns) if game_turns else 0.0,
            "player_turns_median": (
                statistics.median(game_turns) if game_turns else 0.0
            ),
            "decisions_mean": (
                statistics.fmean(game_decisions) if game_decisions else 0.0
            ),
        },
        "outcome_by_action_change": {
            "challenger": _outcome_by_action_change(pairs, "policy_a_stats"),
            "incumbent": _outcome_by_action_change(pairs, "policy_b_stats"),
        },
        "decision_quality_challenger": decision_quality_metrics(challenger_stats),
        "decision_quality_incumbent": decision_quality_metrics(incumbent_stats),
        "search_stats_challenger": challenger_stats,
        "search_stats_incumbent": incumbent_stats,
    }


@dataclass(slots=True)
class _Position:
    """Where the ladder stands, derived purely from the durable game log."""

    stages: list[dict[str, Any]]
    finished: bool
    rung: Rung | None = None
    next_block: int = 0
    stop_reason: str | None = None
    stopped_at_depth: int | None = None
    ceiling_depth: int | None = None


def _evaluate_rung(
    args: argparse.Namespace,
    store: GameStore,
    rung: Rung,
    *,
    resamples: int | None,
) -> tuple[dict[str, Any] | None, int]:
    """Aggregate one rung; returns (stage, next_block) with stage None if open."""

    max_blocks = 1 + args.extra_blocks
    block_pairs = [store.scored_pairs(rung.index, block) for block in range(max_blocks)]
    completed_blocks = [
        block for block, pairs in enumerate(block_pairs) if len(pairs) >= args.pairs
    ]
    if not completed_blocks:
        return None, 0
    merged: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for block in completed_blocks:
        for pair_id, games in block_pairs[block].items():
            merged[(block, pair_id)] = games
    cumulative = _aggregate(merged, args, resamples=resamples)
    decision = _decide(cumulative["score_rate"], args, len(completed_blocks))
    if decision == "escalate":
        return None, len(completed_blocks)
    stage = {
        "stage": rung.index,
        "mode": rung.mode,
        "incumbent_depth": rung.incumbent_depth,
        "challenger_depth": rung.challenger_depth,
        "blocks": [
            _aggregate(block_pairs[block], args, resamples=resamples)
            for block in completed_blocks
        ],
        "cumulative": cumulative,
        "decision": decision,
        "decision_reason": (
            f"{rung.mode} rung: score_rate {cumulative['score_rate']:.4f} vs "
            f"threshold {args.threshold:.4f} / floor {args.floor:.4f} after "
            f"{cumulative['games']} games"
        ),
    }
    return stage, len(completed_blocks)


def _rebuild(
    args: argparse.Namespace,
    store: GameStore,
    *,
    resamples: int | None = None,
) -> _Position:
    """Replay the up/down ladder plan over the durable game log."""

    stages: list[dict[str, Any]] = []
    index = 0
    mode = "up"
    depth = args.start_depth
    while True:
        if mode == "up":
            if depth >= args.max_depth:
                return _Position(
                    stages,
                    finished=True,
                    stop_reason="reached_max_depth",
                    ceiling_depth=depth,
                )
            rung = Rung(index, "up", depth, depth + 1)
        else:
            incumbent = depth - 1
            if incumbent < args.min_depth:
                return _Position(
                    stages,
                    finished=True,
                    stop_reason="reached_min_depth",
                    stopped_at_depth=depth,
                )
            rung = Rung(index, "down", incumbent, incumbent + 1)

        stage, next_block = _evaluate_rung(
            args, store, rung, resamples=resamples
        )
        if stage is None:
            return _Position(stages, finished=False, rung=rung, next_block=next_block)
        stages.append(stage)
        index += 1
        if rung.mode == "up":
            if stage["decision"] == "advance":
                depth = rung.challenger_depth
                continue
            mode = "down"
            depth = rung.incumbent_depth
            continue
        if stage["decision"] == "advance":
            return _Position(
                stages,
                finished=True,
                stop_reason="downward_probe_passed",
                stopped_at_depth=rung.incumbent_depth,
            )
        depth = rung.incumbent_depth


def _summary(
    store: GameStore,
    position: _Position,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        **store.header,
        "stages": position.stages,
        "finished": position.finished,
        "stop_reason": position.stop_reason,
        "stopped_at_depth": position.stopped_at_depth,
        "ceiling_depth": position.ceiling_depth,
        "next_rung": (
            None
            if position.rung is None
            else {
                "mode": position.rung.mode,
                "incumbent_depth": position.rung.incumbent_depth,
                "challenger_depth": position.rung.challenger_depth,
                "block": position.next_block,
            }
        ),
        "game_log_root": str(store.root),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "artifact_note": (
            "the per-game log under game_log_root is authoritative; this "
            "summary is rebuilt from it"
        ),
    }



def _run_pairs(
    args: argparse.Namespace,
    checkpoint: Path,
    rung: Rung,
    block: int,
    store: GameStore,
    on_pair: Callable[[], None],
) -> float:
    all_seeds = _block_seeds(args, block)
    started = time.perf_counter()

    def progress(completed: int, total: int) -> None:
        if completed % args.progress_every == 0 or completed == total:
            print(
                json.dumps(
                    {
                        "event": "ladder_progress",
                        "stage": rung.index,
                        "mode": rung.mode,
                        "incumbent_depth": rung.incumbent_depth,
                        "challenger_depth": rung.challenger_depth,
                        "block": block,
                        "completed_pairs": completed,
                        "total_pairs": total,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    def check_pair(pair_result: ParallelPairResult) -> None:
        require_scored_pair(pair_result)
        on_pair()

    config = FixedEvaluationConfig(
        policy_a=SerializablePolicySpec.search_checkpoint(
            f"search_depth_{rung.challenger_depth}",
            checkpoint,
            search_config=_search_config(args, rung.challenger_depth),
        ),
        policy_b=SerializablePolicySpec.search_checkpoint(
            f"search_depth_{rung.incumbent_depth}",
            checkpoint,
            search_config=_search_config(args, rung.incumbent_depth),
        ),
        game_seeds=all_seeds,
        workers=args.workers,
        device="cpu",
        policy_seed_base=args.policy_seed_base,
        max_completed_turns=args.max_completed_turns,
        adjudicate_deterministic_policy_cycles=True,
        metadata={
            "experiment": PROTOCOL,
            "stage": rung.index,
            "mode": rung.mode,
            "block": block,
            "incumbent_depth": rung.incumbent_depth,
            "challenger_depth": rung.challenger_depth,
        },
    )
    run_fixed_evaluation(
        config,
        store.block_dir(rung.index, block) / "evaluation.json",
        store_dir=store.block_dir(rung.index, block),
        failure_dir=args.output.parent / "failures" / args.output.stem,
        resume=args.resume,
        allow_legacy_resume=args.resume,
        require_scored_pairs=True,
        progress_every=args.progress_every,
        summary_every=args.summary_every,
        progress_callback=progress,
        pair_callback=check_pair,
    )
    store.invalidate(rung.index, block)
    return time.perf_counter() - started


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.pairs <= 0:
        raise ValueError("pairs must be positive")
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    if args.start_depth <= 0 or args.max_depth <= args.start_depth:
        raise ValueError("max_depth must be greater than start_depth")
    if not 1 <= args.min_depth <= args.start_depth:
        raise ValueError("require 1 <= min_depth <= start_depth")
    if args.extra_blocks < 0:
        raise ValueError("extra_blocks must not be negative")
    if args.workers > 1 and args.device != "cpu":
        raise ValueError("parallel evaluation requires --device cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if not 0.0 < args.floor <= args.threshold < 1.0:
        raise ValueError("require 0 < floor <= threshold < 1")
    if args.progress_every <= 0 or args.summary_every <= 0:
        raise ValueError("progress_every and summary_every must be positive")

    checkpoint = args.checkpoint.resolve()
    store = GameStore(
        args.output.with_suffix(args.output.suffix + ".games"),
        _run_header(args, checkpoint),
    )
    if store.root.exists() and any(store.root.iterdir()):
        if not args.resume:
            raise ValueError(
                f"{store.root} exists; pass --resume to continue it or remove it"
            )
        store.load()
        _check_resume_header(store.header, args, checkpoint)
        if (
            store.header.get("max_depth") != args.max_depth
            or store.header.get("min_depth") != args.min_depth
        ):
            print(
                json.dumps(
                    {
                        "event": "ladder_plan_extended",
                        "previous_min_depth": store.header.get("min_depth"),
                        "previous_max_depth": store.header.get("max_depth"),
                        "min_depth": args.min_depth,
                        "max_depth": args.max_depth,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            store.header["min_depth"] = args.min_depth
            store.header["max_depth"] = args.max_depth
            store.start()
    else:
        store.start()

    position = _rebuild(args, store)

    def count_completed_pairs() -> int:
        store.clear_cache()
        total = 0
        for stage_dir in sorted(store.root.glob("stage-*")):
            try:
                index = int(stage_dir.name.split("-")[1])
            except (IndexError, ValueError):  # pragma: no cover - defensive
                continue
            for block in range(1 + args.extra_blocks):
                total += len(store.scored_pairs(index, block))
        return total

    if not position.stages and position.next_block == 0:
        rows = _projection_rows(args)
        print(
            json.dumps(
                {
                    "event": "ladder_projection",
                    "workers": args.workers,
                    "pairs_per_block": args.pairs,
                    "note": (
                        "per-block seconds assume perfect worker scaling; an "
                        "escalated rung plays one extra block"
                    ),
                    "rungs": rows,
                    "game_log_root": str(store.root),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    elif position.stages or position.rung is not None:
        print(
            json.dumps(
                {
                    "event": "ladder_resumed",
                    "completed_pairs": count_completed_pairs(),
                    "finished": position.finished,
                    "next_rung": (
                        None
                        if position.rung is None
                        else {
                            "mode": position.rung.mode,
                            "incumbent_depth": position.rung.incumbent_depth,
                            "challenger_depth": position.rung.challenger_depth,
                            "block": position.next_block,
                        }
                    ),
                    "game_log_root": str(store.root),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    def write_summary(*, full: bool = False) -> None:
        store.clear_cache()
        atomic_write_json(
            args.output,
            _summary(
                store,
                _rebuild(args, store, resamples=None if full else 0),
                args,
            ),
        )

    pairs_since_summary = 0

    def on_pair() -> None:
        nonlocal pairs_since_summary
        pairs_since_summary += 1
        if pairs_since_summary >= args.summary_every:
            pairs_since_summary = 0
            write_summary()

    if position.finished:
        atomic_write_json(args.output, _summary(store, position, args))
        print(
            json.dumps(
                {
                    "event": "ladder_already_finished",
                    "stop_reason": position.stop_reason,
                    "stopped_at_depth": position.stopped_at_depth,
                    "ceiling_depth": position.ceiling_depth,
                    "output": str(args.output.resolve()),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0

    while not position.finished:
        rung = position.rung
        if rung is None:  # pragma: no cover - defensive
            raise RuntimeError("ladder position has no runnable rung")
        block = position.next_block
        complete = store.scored_pairs(rung.index, block)
        missing = [
            pair_id for pair_id in range(args.pairs) if pair_id not in complete
        ]
        if not missing:
            raise RuntimeError(
                f"ladder made no progress at stage {rung.index} block {block}"
            )
        elapsed = _run_pairs(args, checkpoint, rung, block, store, on_pair)
        print(
            json.dumps(
                {
                    "event": "ladder_block_finished",
                    "stage": rung.index,
                    "mode": rung.mode,
                    "incumbent_depth": rung.incumbent_depth,
                    "challenger_depth": rung.challenger_depth,
                    "block": block,
                    "ran_pairs": len(missing),
                    "elapsed_seconds": round(elapsed, 1),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        position = _rebuild(args, store)
        write_summary(full=True)
        if position.finished:
            break
        next_rung = position.rung
        stage = position.stages[-1] if position.stages else None
        print(
            json.dumps(
                {
                    "event": "ladder_stage_status",
                    "stage": None if stage is None else stage["stage"],
                    "mode": None if next_rung is None else next_rung.mode,
                    "next_incumbent_depth": (
                        None if next_rung is None else next_rung.incumbent_depth
                    ),
                    "next_challenger_depth": (
                        None if next_rung is None else next_rung.challenger_depth
                    ),
                    "decided_stages": [
                        {
                            "stage": item["stage"],
                            "mode": item["mode"],
                            "incumbent_depth": item["incumbent_depth"],
                            "challenger_depth": item["challenger_depth"],
                            "games": item["cumulative"]["games"],
                            "score_rate": round(item["cumulative"]["score_rate"], 4),
                            "decision": item["decision"],
                            "paired_ci": [
                                round(
                                    item["cumulative"]["paired_bootstrap"]["lower"], 4
                                ),
                                round(
                                    item["cumulative"]["paired_bootstrap"]["upper"], 4
                                ),
                            ],
                            "chosen_policy_rank": round(
                                item["cumulative"]["decision_quality_challenger"][
                                    "chosen_policy_rank"
                                ],
                                3,
                            ),
                            "search_changed_raw_action_rate": round(
                                item["cumulative"]["decision_quality_challenger"][
                                    "search_changed_raw_action_rate"
                                ],
                                4,
                            ),
                        }
                        for item in position.stages
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    atomic_write_json(args.output, _summary(store, position, args))
    print(
        json.dumps(
            {
                "event": "ladder_stopped",
                "stop_reason": position.stop_reason,
                "stopped_at_depth": position.stopped_at_depth,
                "ceiling_depth": position.ceiling_depth,
                "pairs": count_completed_pairs(),
                "output": str(args.output.resolve()),
                "game_log_root": str(store.root),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
