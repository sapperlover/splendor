"""Export reproducible diagnostics from a completed 5C training run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from .evaluation import PolicySpec, evaluate_policies
from .neural_policy import GreedyActorCriticPolicy
from .opponent_pool import load_frozen_actor_critic_checkpoint
from .train_5c import (
    _HISTORICAL_GAME_RNG_OFFSET,
    _HISTORICAL_POLICY_RNG_OFFSET,
    _evaluation_game_seeds,
    load_five_c_config,
)


_OPPONENTS = ("latest", "historical", "score", "development")
_ACTIONS = (
    "TakeDistinct",
    "TakePair",
    "ReserveMarket",
    "ReserveDeck",
    "BuyMarket",
    "BuyReserved",
)
_PHASE_COUNTS = (
    "payment_decisions",
    "return_tokens_decisions",
    "choose_noble_decisions",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: Sequence[str]) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)


def _scalar_fields(records: Sequence[dict[str, Any]]) -> list[str]:
    preferred = [
        "run_local_update_count",
        "lifetime_update_count",
        "run_local_step",
        "lifetime_step",
        "wall_seconds",
        "policy_loss",
        "value_loss",
        "total_loss",
        "entropy",
        "entropy_coef",
        "explained_variance",
        "approx_kl",
        "clip_fraction",
        "grad_norm",
        "learning_rate",
    ]
    keys = {
        key
        for record in records
        for key, value in record.items()
        if value is None or isinstance(value, (str, int, float, bool))
    }
    return [key for key in preferred if key in keys] + sorted(keys - set(preferred))


def _bin_end(step: int, size: int, total_steps: int) -> int:
    if size <= 0:
        raise ValueError("bin size must be positive")
    return min(total_steps, ((max(step, 1) - 1) // size + 1) * size)


def _group_updates(
    updates: Sequence[dict[str, Any]], size: int, total_steps: int
) -> list[tuple[int, list[dict[str, Any]]]]:
    bins: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for update in updates:
        bins[_bin_end(int(update["run_local_step"]), size, total_steps)].append(update)
    return sorted(bins.items())


def _weighted_mean(
    records: Sequence[dict[str, Any]], value_key: str, weight_key: str
) -> float | None:
    numerator = 0.0
    denominator = 0.0
    for record in records:
        value = record.get(value_key)
        weight = record.get(weight_key)
        if value is None or weight is None:
            continue
        numerator += float(value) * float(weight)
        denominator += float(weight)
    return numerator / denominator if denominator else None


def _sum(records: Sequence[dict[str, Any]], key: str) -> int:
    return sum(int(record.get(key, 0) or 0) for record in records)


def _opponent_bins(
    grouped: Sequence[tuple[int, list[dict[str, Any]]]], size: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for end, records in grouped:
        row: dict[str, Any] = {
            "bin_start_exclusive": max(0, end - size),
            "bin_end_inclusive": end,
            "updates": len(records),
            "games": _sum(records, "games_completed"),
        }
        total_games = sum(_sum(records, f"opponent_{name}_count") for name in _OPPONENTS)
        for name in _OPPONENTS:
            count_key = f"opponent_{name}_count"
            count = _sum(records, count_key)
            row[f"{name}_games"] = count
            row[f"{name}_fraction"] = count / total_games if total_games else None
            row[f"{name}_learner_score_rate"] = _weighted_mean(
                records, f"opponent_{name}_train_score", count_key
            )
        rows.append(row)
    return rows


def _action_bins(
    grouped: Sequence[tuple[int, list[dict[str, Any]]]], size: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for end, records in grouped:
        counts = {name: _sum(records, f"{name}_count") for name in _ACTIONS}
        total = sum(counts.values())
        row: dict[str, Any] = {
            "bin_start_exclusive": max(0, end - size),
            "bin_end_inclusive": end,
            "updates": len(records),
            "main_actions": total,
            "main_action_decisions": total,
        }
        for name in _ACTIONS:
            row[f"{name}_count"] = counts[name]
            row[f"{name}_fraction"] = counts[name] / total if total else None
        for phase in _PHASE_COUNTS:
            row[phase] = _sum(records, phase)
        rows.append(row)
    return rows


def _quality_bins(
    grouped: Sequence[tuple[int, list[dict[str, Any]]]], size: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    weighted = (
        "mean_completed_player_turns",
        "mean_learning_decisions_per_game",
        "mean_final_score",
        "mean_purchased_development_cards",
        "mean_nobles_acquired",
        "mean_buys_per_game",
        "mean_reserves_per_game",
        "mean_token_taking_per_game",
        "train_win_rate",
        "train_draw_rate",
        "train_loss_rate",
    )
    for end, records in grouped:
        games = _sum(records, "games_completed")
        row: dict[str, Any] = {
            "bin_start_exclusive": max(0, end - size),
            "bin_end_inclusive": end,
            "updates": len(records),
            "games": games,
            "forced_skips": _sum(records, "forced_skips"),
            "engineering_truncations": _sum(records, "engineering_truncations"),
            "boundary_drain_decisions_not_trained": _sum(
                records, "boundary_drain_decisions_not_trained"
            ),
        }
        for key in weighted:
            row[key] = _weighted_mean(records, key, "games_completed")
        if row["train_win_rate"] is not None and row["train_draw_rate"] is not None:
            row["learner_score_rate"] = (
                float(row["train_win_rate"]) + 0.5 * float(row["train_draw_rate"])
            )
        rows.append(row)
    return rows


def _formal_candidates(run_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted((run_root / "eval" / "formal").glob("*/cycle_summary.json")):
        summary = _read_json(summary_path)
        cycle = int(summary["cycle_run_local_step"])
        winner = summary.get("formal_winner_checkpoint")
        promoted = summary.get("promoted_checkpoint")
        for candidate in summary["formal_candidates_ranked"]:
            panel = candidate["panel_summary"]
            checkpoint = str(candidate["checkpoint"])
            row = {
                "formal_cycle_run_local_step": cycle,
                "formal_rank": candidate["formal_rank"],
                "candidate_run_local_step": panel["run_local_step"],
                "candidate_lifetime_step": panel["lifetime_step"],
                "candidate_checkpoint": checkpoint,
                "formal_eval_path": candidate["formal_eval_path"],
                "is_initial_baseline": cycle == 0,
                "is_formal_winner": checkpoint == winner,
                "promotion_eligible": candidate.get("promotion_eligible"),
                "was_promoted": cycle > 0 and checkpoint == promoted,
                "heuristic_score": panel["heuristic_score"],
                "score_score_rate": panel["score_policy_score_rate"],
                "score_wins": panel["score_wins"],
                "score_draws": panel["score_draws"],
                "score_losses": panel["score_losses"],
                "score_first_player_score": panel["score_first_player_score"],
                "score_second_player_score": panel["score_second_player_score"],
                "score_mean_turns": panel["score_mean_turns"],
                "development_score_rate": panel["development_policy_score_rate"],
                "development_wins": panel["development_wins"],
                "development_draws": panel["development_draws"],
                "development_losses": panel["development_losses"],
                "development_first_player_score": panel[
                    "development_first_player_score"
                ],
                "development_second_player_score": panel[
                    "development_second_player_score"
                ],
                "development_mean_turns": panel["development_mean_turns"],
                "anchor_512112_score_rate": panel["anchor_512112_score_rate"],
                "anchor_512112_wins": panel["anchor_512112_wins"],
                "anchor_512112_draws": panel["anchor_512112_draws"],
                "anchor_512112_losses": panel["anchor_512112_losses"],
                "anchor_512112_first_player_score": panel[
                    "anchor_512112_first_player_score"
                ],
                "anchor_512112_second_player_score": panel[
                    "anchor_512112_second_player_score"
                ],
                "anchor_512112_mean_turns": panel["anchor_512112_mean_turns"],
                "policy_cycle_draws": panel.get("policy_cycle_draws", 0),
                "evaluation_seconds": panel["evaluation_seconds"],
                "engineering_truncations": panel["engineering_truncations"],
                "illegal_actions": panel["illegal_actions"],
                "invariant_failures": panel["invariant_failures"],
                "exceptions": panel["exceptions"],
            }
            rows.append(row)
    return sorted(rows, key=lambda row: (row["formal_cycle_run_local_step"], row["formal_rank"]))


def _observed_historical(run_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    long_rows: list[dict[str, Any]] = []
    all_opponents: set[str] = set()
    for path in sorted((run_root / "eval" / "historical").glob("*.json")):
        payload = _read_json(path)
        panel = payload["panel_summary"]
        candidate = str(payload["checkpoint"])
        for opponent, result in payload["evaluations"].items():
            all_opponents.add(opponent)
            long_rows.append(
                {
                    "evaluation_cycle_run_local_step": panel[
                        "evaluation_cycle_run_local_step"
                    ],
                    "candidate_run_local_step": panel["run_local_step"],
                    "candidate_lifetime_step": panel["lifetime_step"],
                    "candidate_checkpoint": candidate,
                    "opponent_id": opponent,
                    "paired_seeds": result["total_pairs"],
                    "score_rate": result["win_rate_score"],
                    "wins": result["a_wins"],
                    "draws": result["draws"],
                    "losses": result["a_losses"],
                    "first_player_score": result["a_first"]["win_rate_score"],
                    "second_player_score": result["a_second"]["win_rate_score"],
                    "mean_completed_player_turns": result[
                        "mean_completed_player_turns"
                    ],
                    "policy_cycle_draws": result.get("policy_cycle_draws", 0),
                }
            )
    opponents = sorted(all_opponents)
    wide_by_cycle: dict[int, dict[str, Any]] = {}
    for row in long_rows:
        cycle = int(row["evaluation_cycle_run_local_step"])
        wide = wide_by_cycle.setdefault(
            cycle,
            {
                "evaluation_cycle_run_local_step": cycle,
                "candidate_run_local_step": row["candidate_run_local_step"],
                "candidate_lifetime_step": row["candidate_lifetime_step"],
                "candidate_checkpoint": row["candidate_checkpoint"],
            },
        )
        wide[str(row["opponent_id"])] = row["score_rate"]
    return long_rows, [wide_by_cycle[key] for key in sorted(wide_by_cycle)], opponents


def export_run(run_root: Path, output_dir: Path, bin_sizes: Sequence[int]) -> dict[str, Any]:
    summary = _read_json(run_root / "summary.json")
    total_steps = int(summary["actual_run_local_decisions"])
    records = _read_jsonl(run_root / "metrics.jsonl")
    updates = [record for record in records if record.get("record_type") == "update"]
    if len(updates) != int(summary["run_local_ppo_updates"]):
        raise ValueError("update count does not agree with summary.json")

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    update_path = output_dir / "ppo_updates.csv"
    _write_csv(update_path, updates, _scalar_fields(updates))
    outputs.append(str(update_path))

    formal = _formal_candidates(run_root)
    if len(formal) != 61:
        raise ValueError(f"expected 61 Formal candidates, found {len(formal)}")
    formal_path = output_dir / "formal_candidates.csv"
    _write_csv(formal_path, formal, list(formal[0]))
    outputs.append(str(formal_path))

    historical_long, historical_wide, historical_columns = _observed_historical(run_root)
    observed_long_path = output_dir / "historical_observed_matchups_long.csv"
    _write_csv(observed_long_path, historical_long, list(historical_long[0]))
    outputs.append(str(observed_long_path))
    observed_wide_path = output_dir / "historical_observed_matchups_wide.csv"
    _write_csv(
        observed_wide_path,
        historical_wide,
        [
            "evaluation_cycle_run_local_step",
            "candidate_run_local_step",
            "candidate_lifetime_step",
            "candidate_checkpoint",
            *historical_columns,
        ],
    )
    outputs.append(str(observed_wide_path))

    for size in bin_sizes:
        grouped = _group_updates(updates, size, total_steps)
        suffix = f"{size // 1000}k"
        opponent_rows = _opponent_bins(grouped, size)
        opponent_path = output_dir / f"opponent_training_{suffix}.csv"
        _write_csv(opponent_path, opponent_rows, list(opponent_rows[0]))
        outputs.append(str(opponent_path))

        action_rows = _action_bins(grouped, size)
        action_path = output_dir / f"action_distribution_{suffix}.csv"
        _write_csv(action_path, action_rows, list(action_rows[0]))
        outputs.append(str(action_path))

        quality_rows = _quality_bins(grouped, size)
        quality_path = output_dir / f"game_quality_{suffix}.csv"
        _write_csv(quality_path, quality_rows, list(quality_rows[0]))
        outputs.append(str(quality_path))

    manifest = {
        "source_run_root": str(run_root.resolve()),
        "source_metrics": str((run_root / "metrics.jsonl").resolve()),
        "run_local_steps": total_steps,
        "ppo_updates": len(updates),
        "formal_candidates": len(formal),
        "observed_historical_matchups": len(historical_long),
        "bin_sizes": list(bin_sizes),
        "outputs": outputs,
        "aggregation": {
            "opponent_score_rate": "sum(per-update score_rate * opponent games) / sum(opponent games)",
            "action_fraction": "sum(action count) / sum(all MAIN_ACTION counts)",
            "game_quality_means": "weighted by completed games in each update",
            "bin_membership": "update belongs to the interval ending at the smallest bin boundary >= update run-local step",
        },
    }
    _write_json(output_dir / "export_manifest.json", manifest)
    return manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _matrix_rows(
    entries: Sequence[dict[str, Any]], results: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    labels = [str(entry["opponent_id"]) for entry in entries]
    long_rows: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []
    for row_index, row_label in enumerate(labels):
        matrix: dict[str, Any] = {"model": row_label}
        for column_index, column_label in enumerate(labels):
            if row_index == column_index:
                score = 0.5
            elif row_index < column_index:
                score = results[f"{row_label}|||{column_label}"]["score_rate"]
            else:
                score = 1.0 - results[f"{column_label}|||{row_label}"]["score_rate"]
            matrix[column_label] = score
            long_rows.append(
                {
                    "row_model": row_label,
                    "column_model": column_label,
                    "row_score_rate": score,
                    "is_diagonal": row_index == column_index,
                }
            )
        matrix_rows.append(matrix)
    return long_rows, matrix_rows


def _historical_result_summary(
    result: dict[str, Any],
    *,
    row_model: str,
    column_model: str,
    reverse: bool,
    source: str,
) -> dict[str, Any]:
    if reverse:
        score_rate = 1.0 - float(result["win_rate_score"])
        wins = int(result["a_losses"])
        losses = int(result["a_wins"])
        first_player_score = 1.0 - float(result["a_second"]["win_rate_score"])
        second_player_score = 1.0 - float(result["a_first"]["win_rate_score"])
    else:
        score_rate = float(result["win_rate_score"])
        wins = int(result["a_wins"])
        losses = int(result["a_losses"])
        first_player_score = float(result["a_first"]["win_rate_score"])
        second_player_score = float(result["a_second"]["win_rate_score"])
    return {
        "row_model": row_model,
        "column_model": column_model,
        "score_rate": score_rate,
        "wins": wins,
        "draws": int(result["draws"]),
        "losses": losses,
        "first_player_score": first_player_score,
        "second_player_score": second_player_score,
        "mean_completed_player_turns": result["mean_completed_player_turns"],
        "policy_cycle_draws": int(result.get("policy_cycle_draws", 0)),
        "forced_skips": int(result["forced_skips"]),
        "engineering_truncations": int(result["engineering_truncations"]),
        "illegal_actions": int(result["illegal_actions"]),
        "invariant_failures": int(result["invariant_failures"]),
        "exceptions": int(result["exceptions"]),
        "elapsed_seconds": None,
        "source": source,
    }


def _reuse_logged_historical_matchups(
    run_root: Path,
    entries: Sequence[dict[str, Any]],
    results: dict[str, dict[str, Any]],
    *,
    pairs: int,
) -> int:
    labels = [str(entry["opponent_id"]) for entry in entries]
    indices = {label: index for index, label in enumerate(labels)}
    checkpoint_labels = {
        str(Path(entry["checkpoint_path"]).resolve()): str(entry["opponent_id"])
        for entry in entries
    }
    reused = 0
    for path in sorted((run_root / "eval" / "historical").glob("*.json")):
        payload = _read_json(path)
        candidate = checkpoint_labels.get(str(Path(payload["checkpoint"]).resolve()))
        if candidate is None:
            continue
        for opponent, result in payload["evaluations"].items():
            if opponent not in indices or opponent == candidate:
                continue
            if int(result["total_pairs"]) != pairs:
                continue
            candidate_index = indices[candidate]
            opponent_index = indices[opponent]
            if candidate_index < opponent_index:
                row_model, column_model = candidate, opponent
                reverse = False
            else:
                row_model, column_model = opponent, candidate
                reverse = True
            key = f"{row_model}|||{column_model}"
            if key in results:
                continue
            results[key] = _historical_result_summary(
                result,
                row_model=row_model,
                column_model=column_model,
                reverse=reverse,
                source=str(path),
            )
            reused += 1
    return reused


def evaluate_historical_matrix(
    run_root: Path,
    output_dir: Path,
    *,
    runtime_dir: Path,
    pairs: int,
    device: str,
) -> dict[str, Any]:
    if pairs <= 0:
        raise ValueError("pairs must be positive")
    config = load_five_c_config(run_root / "config.json")
    snapshot = _read_json(
        run_root
        / "historical_pool"
        / "local_005000000_lifetime_005512112.json"
    )
    entries = list(snapshot["historical"])
    if len(entries) != 12:
        raise ValueError(f"expected final active pool size 12, found {len(entries)}")
    labels = [str(entry["opponent_id"]) for entry in entries]
    if len(set(labels)) != len(labels):
        raise ValueError("historical opponent ids must be unique")
    for entry in entries:
        if not Path(entry["checkpoint_path"]).is_file():
            raise FileNotFoundError(entry["checkpoint_path"])

    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    progress_path = runtime_dir / "historical_matrix_progress.json"
    seeds = _evaluation_game_seeds(
        _HISTORICAL_GAME_RNG_OFFSET, config.historical_panel_seed, pairs
    )
    policy_seed_base = -(
        _HISTORICAL_POLICY_RNG_OFFSET + config.historical_panel_seed * 1_000_000
    )
    checkpoint_info = [
        {
            "opponent_id": entry["opponent_id"],
            "training_step": entry["training_step"],
            "historical_bucket": entry["historical_bucket"],
            "checkpoint_path": entry["checkpoint_path"],
            "sha256": _sha256(Path(entry["checkpoint_path"])),
        }
        for entry in entries
    ]
    identity = {
        "pairs": pairs,
        "game_seeds": list(seeds),
        "policy_seed_base": policy_seed_base,
        "policy": "greedy argmax",
        "adjudicate_deterministic_policy_cycles": True,
        "checkpoint_info": checkpoint_info,
    }
    if progress_path.exists():
        progress = _read_json(progress_path)
        if progress["identity"] != identity:
            raise ValueError("existing historical matrix progress has incompatible identity")
    else:
        progress = {"identity": identity, "results": {}}
        _write_json(progress_path, progress)

    for result in progress["results"].values():
        result.setdefault("source", "new_matrix_evaluation")
    reused = _reuse_logged_historical_matchups(
        run_root, entries, progress["results"], pairs=pairs
    )
    if reused:
        _write_json(progress_path, progress)
    print(
        json.dumps(
            {
                "event": "historical_matrix_existing_results_loaded",
                "completed": len(progress["results"]),
                "reused_from_training_diagnostics": reused,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    models: dict[str, Any] = {}

    def model_for(entry: dict[str, Any]) -> Any:
        label = str(entry["opponent_id"])
        if label not in models:
            models[label] = load_frozen_actor_critic_checkpoint(
                Path(entry["checkpoint_path"]), device=device
            )
        return models[label]

    total = len(entries) * (len(entries) - 1) // 2
    completed = len(progress["results"])
    started_all = time.perf_counter()
    for left_index, left in enumerate(entries):
        for right in entries[left_index + 1 :]:
            left_label = str(left["opponent_id"])
            right_label = str(right["opponent_id"])
            key = f"{left_label}|||{right_label}"
            if key in progress["results"]:
                continue
            left_model = model_for(left)
            right_model = model_for(right)
            left_spec = PolicySpec(
                left_label,
                lambda seed, frozen=left_model: GreedyActorCriticPolicy(
                    frozen, device=device
                ),
            )
            right_spec = PolicySpec(
                right_label,
                lambda seed, frozen=right_model: GreedyActorCriticPolicy(
                    frozen, device=device
                ),
            )
            started = time.perf_counter()
            result = evaluate_policies(
                left_spec,
                right_spec,
                seeds,
                policy_seed_base=policy_seed_base,
                max_completed_turns=config.max_completed_turns,
                failure_dir=runtime_dir / "historical_matrix_failures" / key,
                adjudicate_deterministic_policy_cycles=True,
            )
            elapsed = time.perf_counter() - started
            if not result.passed:
                raise RuntimeError(f"historical matchup failed: {key}")
            progress["results"][key] = {
                "row_model": left_label,
                "column_model": right_label,
                "score_rate": result.win_rate_score,
                "wins": result.a_wins,
                "draws": result.draws,
                "losses": result.a_losses,
                "first_player_score": result.a_first.win_rate_score,
                "second_player_score": result.a_second.win_rate_score,
                "mean_completed_player_turns": result.mean_completed_player_turns,
                "policy_cycle_draws": result.policy_cycle_draws,
                "forced_skips": result.forced_skips,
                "engineering_truncations": result.engineering_truncations,
                "illegal_actions": result.illegal_actions,
                "invariant_failures": result.invariant_failures,
                "exceptions": result.exceptions,
                "elapsed_seconds": elapsed,
                "source": "new_matrix_evaluation",
            }
            completed += 1
            _write_json(progress_path, progress)
            print(
                json.dumps(
                    {
                        "event": "historical_matrix_pair_completed",
                        "completed": completed,
                        "total": total,
                        "row": left_label,
                        "column": right_label,
                        "score_rate": result.win_rate_score,
                        "elapsed_seconds": elapsed,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    long_rows, matrix_rows = _matrix_rows(entries, progress["results"])
    matrix_path = output_dir / "historical_final_pool_matrix.csv"
    _write_csv(matrix_path, matrix_rows, ["model", *labels])
    long_path = output_dir / "historical_final_pool_matrix_long.csv"
    _write_csv(long_path, long_rows, list(long_rows[0]))
    pair_rows = [progress["results"][key] for key in sorted(progress["results"])]
    pairs_path = output_dir / "historical_final_pool_pair_results.csv"
    _write_csv(pairs_path, pair_rows, list(pair_rows[0]))
    newly_evaluated = sum(
        row["source"] == "new_matrix_evaluation" for row in pair_rows
    )
    reused_matchups = len(pair_rows) - newly_evaluated
    report = {
        **identity,
        "unordered_matchups": len(pair_rows),
        "games": len(pair_rows) * pairs * 2,
        "reused_training_diagnostic_matchups": reused_matchups,
        "reused_training_diagnostic_games": reused_matchups * pairs * 2,
        "new_matrix_evaluations": newly_evaluated,
        "new_matrix_evaluation_games": newly_evaluated * pairs * 2,
        "new_matrix_evaluation_device": device,
        "new_matrix_evaluation_seconds": sum(
            float(row["elapsed_seconds"] or 0.0)
            for row in pair_rows
            if row["source"] == "new_matrix_evaluation"
        ),
        "elapsed_seconds_this_invocation": time.perf_counter() - started_all,
        "engineering_truncations": sum(row["engineering_truncations"] for row in pair_rows),
        "illegal_actions": sum(row["illegal_actions"] for row in pair_rows),
        "invariant_failures": sum(row["invariant_failures"] for row in pair_rows),
        "exceptions": sum(row["exceptions"] for row in pair_rows),
        "policy_cycle_draws": sum(row["policy_cycle_draws"] for row in pair_rows),
        "matrix_csv": str(matrix_path),
        "long_csv": str(long_path),
        "pair_results_csv": str(pairs_path),
    }
    _write_json(output_dir / "historical_final_pool_matrix_manifest.json", report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="export logged update and evaluation data")
    export.add_argument("--run-root", type=Path, required=True)
    export.add_argument("--output-dir", type=Path, required=True)
    export.add_argument("--bin-size", type=int, action="append", dest="bin_sizes")
    matrix = commands.add_parser("matrix", help="evaluate the final active historical pool")
    matrix.add_argument("--run-root", type=Path, required=True)
    matrix.add_argument("--output-dir", type=Path, required=True)
    matrix.add_argument(
        "--runtime-dir",
        type=Path,
        default=Path("runtime/work_package_5c"),
    )
    matrix.add_argument("--pairs", type=int, default=50)
    matrix.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "export":
        report = export_run(
            args.run_root,
            args.output_dir,
            args.bin_sizes or (100_000, 250_000),
        )
    else:
        report = evaluate_historical_matrix(
            args.run_root,
            args.output_dir,
            runtime_dir=args.runtime_dir,
            pairs=args.pairs,
            device=args.device,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
