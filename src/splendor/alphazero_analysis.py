"""Read-only aggregation of committed AZ work and invocation wall time."""
from __future__ import annotations
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .alphazero_recording import distribution, read_json, unavailable, write_json
from .alphazero_train import metric_ratios


def jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            # Only complete append records count; an interrupted final line is uncommitted.
            continue
    return records


def analyze_run(output: Path, destination: Path) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    sessions = [read_json(p) for p in sorted((output / "recording" / "sessions").glob("*.json"))]
    resources = jsonl(output / "recording" / "resources.jsonl")
    phases = Counter()
    partial_phases = Counter()
    actual_wall = pause = incomplete_lower_bound = 0.0
    previous_end = None
    session_rows = []
    round_timings: dict[int, Counter] = {}
    round_intervals: dict[int, float] = {}
    for session in sorted(sessions, key=lambda r: r["start_unix"]):
        if previous_end is not None:
            pause += max(0, session["start_unix"] - previous_end)
        if "wall_seconds" in session:
            actual_wall += session["wall_seconds"]
            previous_end = session["end_unix"]
        else:
            samples = [r["unix_time"] for r in resources if r["session_id"] == session["session_id"]]
            incomplete_lower_bound += max(0.0, max(samples, default=session["start_unix"]) - session["start_unix"])
            previous_end = None  # A hard kill has no exact end time: do not invent pause or throughput.
        for phase in session["phases"]:
            if "seconds" not in phase:
                continue
            details = phase.get("details", {})
            nested = details.get("phase_seconds_this_session")
            phase_totals = phases if "wall_seconds" in session else partial_phases
            round_index = phase.get("round_index")
            round_totals = round_timings.setdefault(round_index, Counter()) if round_index is not None else Counter()
            if phase["phase"] == "learner_train" and nested:
                for name, seconds in nested.items():
                    phase_totals[f"learner_{name}"] += seconds
                    round_totals[f"learner_{name}"] += seconds
                phase_totals["learner_call_remainder"] += phase["seconds"] - sum(nested.values())
                round_totals["learner_call_remainder"] += phase["seconds"] - sum(nested.values())
            else:
                phase_totals[phase["phase"]] += phase["seconds"]
                round_totals[phase["phase"]] += phase["seconds"]
        for round_index in {p["round_index"] for p in session["phases"] if p.get("round_index") is not None}:
            ended = [p for p in session["phases"] if p.get("round_index") == round_index and "end_unix" in p]
            if ended:
                round_intervals[round_index] = round_intervals.get(round_index, 0) + max(p["end_unix"] for p in ended) - min(p["start_unix"] for p in ended)
        session_rows.append({k: session.get(k) for k in ("session_id", "status", "start_unix", "end_unix", "wall_seconds", "error_type", "error", "resource_error")})
    if actual_wall:
        phases["invocation_remainder"] = actual_wall - sum(phases.values())
    rounds = []
    curve_rows = []
    eval_rows = []
    for round_root in sorted((output / "rounds").glob("round-*")):
        round_index = int(round_root.name.split("-")[1])
        training = read_json(round_root / "training.json") if (round_root / "training.json").exists() else {}
        if (round_root / "checkpoints" / "final.json").exists():
            committed = read_json(round_root / "checkpoints" / "final.json")["round_train_step"]
        else:
            committed = max((int(p.stem.split("-")[1]) for p in (round_root / "checkpoints").glob("step-*.pt")), default=0)
        steps = {row["round_train_step"]: row for row in jsonl(round_root / "training_metrics.jsonl")
                 if row.get("event") == "train_step" and row["round_train_step"] <= committed}
        ordered = [steps[k] for k in sorted(steps)]
        losses = {}
        raw_fields = ("policy_ce_sum", "policy_weight_sum", "value_sq_sum", "value_weight_sum", "entropy_sum", "target_entropy_sum", "value_target_sum", "value_target_sq_sum")
        if ordered and all(all(key in row for key in raw_fields) for row in ordered):
            sums = {key: sum(row[key] for row in ordered) for key in raw_fields}
            losses = {**sums, **metric_ratios(sums)}
            constant_path = round_root / "diagnostics/training_constant.json"
            constant = read_json(constant_path) if constant_path.exists() else {}
            if constant.get("available") and sums["value_weight_sum"]:
                c = constant["value"]
                losses["value_train_constant_mse"] = (sums["value_target_sq_sum"] - 2 * c * sums["value_target_sum"] + c * c * sums["value_weight_sum"]) / sums["value_weight_sum"]
                losses["constant_baseline"] = constant
            else:
                losses["value_train_constant_mse"] = unavailable("no training-only fitted baseline yet")
        else:
            losses = unavailable("no committed steps or legacy metrics missing raw weighted sums")
        cache_windows = {}
        for name, window in (("first32", ordered[:32]), ("last32", ordered[-32:]), ("all", ordered)):
            if not window or not all("cache_delta" in row for row in window):
                cache_windows[name] = unavailable("no committed cache deltas")
                continue
            deltas = {key: sum(row["cache_delta"][key] for row in window) for key in window[0]["cache_delta"]}
            denom = deltas["hits"] + deltas["misses"]
            cache_windows[name] = {"steps": len(window), **deltas, "hit_rate": deltas["hits"] / denom if denom else None}
        attempted_rows = [row for path in sorted((round_root / "usage").glob("*.steps.jsonl")) for row in jsonl(path)
                          if row.get("event") == "train_step"]
        attempted_cost = sum(sum(r.get("seconds", {}).values()) for r in attempted_rows)
        committed_cost = sum(sum(r.get("seconds", {}).values()) for r in ordered)
        recovery = {"attempted_steps": len(attempted_rows), "committed_steps": committed,
                    "repeated_or_uncommitted_steps": len(attempted_rows) - committed,
                    "repeated_or_uncommitted_draw_collate_update_seconds": max(0.0, attempted_cost - committed_cost),
                    "scope": "fully logged attempts only; work interrupted inside a step remains in phase remainder"} if attempted_rows else unavailable("no per-session attempt records")
        resource_rows = [r for r in resources if r.get("round_index") == round_index]
        phase_errors = Counter(p["error_type"] for s in sessions for p in s["phases"]
                               if p.get("round_index") == round_index and p.get("error_type"))
        timing = round_timings.get(round_index, {})
        collection_path = round_root / "selfplay/summary.json"
        collection = read_json(collection_path) if collection_path.exists() else {}
        collected = len(list((round_root / "selfplay/games").glob("game-*.npz")))
        collect_seconds = timing.get("collect", 0)
        learner_seconds = sum(seconds for phase, seconds in timing.items() if phase.startswith("learner_") and phase != "learner_initialize")
        exact_wall = not any("wall_seconds" not in s and any(p.get("round_index") == round_index for p in s["phases"]) for s in sessions)
        rates = {"scope": "unique committed work / all recorded execution segments including recovery overhead; pause excluded",
                 "committed_games": collected,
                 "collection_games_per_minute": collected / collect_seconds * 60 if collect_seconds else unavailable("no recorded collection wall"),
                 "collection_policy_samples_per_second": collection["policy_samples"] / collect_seconds if collect_seconds and "policy_samples" in collection else unavailable("no cumulative collection sample count/wall"),
                 "collection_simulations_per_second": collection["simulations"] / collect_seconds if collect_seconds and "simulations" in collection else unavailable("no cumulative simulations count/wall"),
                 "learner_steps_per_second": committed / learner_seconds if learner_seconds else unavailable("no recorded learner wall"),
                 "learner_positions_per_second": sum(r["batch_size"] for r in ordered) / learner_seconds if learner_seconds else unavailable("no recorded learner wall")}
        if not exact_wall:
            rates = {**unavailable("hard-killed invocation has no exact end time; cumulative throughput is unavailable"),
                     "committed_games": collected, "committed_steps": committed}
        row = {"round_index": round_index, "committed_steps": committed, "logged_committed_steps": len(ordered), "training_batch_metrics": losses,
               "grad_norm": distribution(r["grad_norm"] for r in ordered),
               "clipped_steps": sum(r["grad_clipped"] for r in ordered) if ordered and all("grad_clipped" in r for r in ordered) else unavailable("no committed clip flags"),
               "clip_fraction": sum(r["grad_clipped"] for r in ordered) / len(ordered) if ordered and all("grad_clipped" in r for r in ordered) else unavailable("no committed clip flags"),
               "current_draws": sum(r["current_positions"] for r in ordered) if ordered and all("current_positions" in r for r in ordered) else unavailable("no committed current draw counters"),
               "history_draws": sum(r["history_positions"] for r in ordered) if ordered and all("history_positions" in r for r in ordered) else unavailable("no committed history draw counters"),
               "recorded_phase_errors_by_type": dict(phase_errors) if sessions else unavailable("no invocation error ledger"),
               "cache_windows": cache_windows, "resource_samples": len(resource_rows),
               "recovery_overhead": recovery,
               "phase_seconds_across_sessions": dict(round_timings.get(round_index, {})),
               "recorded_round_wall_seconds": round_intervals.get(round_index, unavailable("no ended round phase intervals")),
               "cumulative_rates": rates,
               "tree_rss_bytes": distribution(r["tree_rss_bytes"] for r in resource_rows), "tree_pss_bytes": distribution(r["tree_pss_bytes"] for r in resource_rows),
               "last_training_session": training, "complete": (round_root / "round_summary.json").exists()}
        rounds.append(row)
        for record in ordered:
            curve_rows.append({"round_index": round_index, **{k: record.get(k) for k in ("round_train_step", "total_train_step", "policy_loss", "target_entropy", "policy_entropy", "policy_kl", "value_loss", "total_loss", "grad_norm", "learning_rate", "current_positions", "history_positions", "reference_sha256")}})
        for panel in ("mcts_vs_parent", "mcts_vs_initial", "greedy_vs_initial"):
            path = round_root / "evaluation" / f"{panel}.json"
            if path.exists():
                summary = read_json(path)
                seconds = timing.get(f"eval_{panel}", 0)
                eval_rows.append({"round_index": round_index, "panel": panel, "purpose": "repeated DEV", "path": str(path.resolve()),
                                  "recorded_wall_seconds": seconds if seconds else unavailable("no invocation panel timing"),
                                  "pairs_per_minute_recorded": summary.get("completed_pairs", 0) / seconds * 60 if seconds and exact_wall else unavailable("no exact invocation panel timing"),
                                  **{k: summary.get(k) for k in ("win_rate_score", "paired_bootstrap", "completed_pairs", "total_pairs", "passed", "a_wins", "draws", "a_losses", "a_first", "a_second")}})
    evaluation_root = output / "anchors"
    for path in sorted(evaluation_root.glob("*.json")):
        summary = read_json(path)
        if "completed_pairs" in summary:
            wall_path = evaluation_root / "recording" / path.stem / "wall_summary.json"
            ledger = read_json(wall_path) if wall_path.exists() else unavailable("external panel used direct evaluation entry without invocation ledger")
            eval_rows.append({"panel": path.stem, "path": str(path.resolve()), "external_wall_ledger": ledger, **summary})
    diagnostic_rows = []
    for path in sorted(destination.glob("round-*/fixed_curve.json")):
        for point in read_json(path):
            for stage, metrics in point["metrics"].items():
                diagnostic_rows.append({"round_index": point["round_index"], "node": point["node"], "stage": stage,
                    "checkpoint_sha256": point["checkpoint_sha256"], "set_sha256": point["set_sha256"],
                    **{k: metrics.get(k) for k in ("available", "positions", "policy_loss", "target_entropy", "policy_entropy", "policy_kl", "value_loss", "value_zero_mse", "value_train_constant_mse")}})
    with (destination / "fixed_validation_curve.csv").open("w", encoding="utf-8", newline="") as handle:
        if diagnostic_rows:
            writer = csv.DictWriter(handle, fieldnames=list(diagnostic_rows[0]))
            writer.writeheader(); writer.writerows(diagnostic_rows)
    resource_groups: dict[str, list[dict[str, Any]]] = {}
    for record in resources:
        resource_groups.setdefault(f"round-{record.get('round_index')}:{record['phase']}", []).append(record)
    resource_summary = {}
    for name, rows in resource_groups.items():
        values = {"samples": len(rows), "scope": "few samples do not establish steady utilization; device memory includes other users",
                  "tree_cpu_percent_one_core": distribution(r["tree_cpu_percent_one_core"] for r in rows if isinstance(r.get("tree_cpu_percent_one_core"), (float, int))),
                  "tree_rss_bytes": distribution(r["tree_rss_bytes"] for r in rows), "tree_pss_bytes": distribution(r["tree_pss_bytes"] for r in rows)}
        columns = ("utilization_percent", "memory_used_mib", "temperature_celsius", "power_watts", "sm_clock_mhz", "memory_clock_mhz")
        gpu = {key: [] for key in columns}
        for row in rows:
            raw = row.get("gpu_csv")
            if not isinstance(raw, str):
                continue
            for line in raw.splitlines():
                fields = line.split(",")
                for key, value in zip(columns, fields[1:]):
                    try:
                        gpu[key].append(float(value))
                    except ValueError:
                        continue  # nvidia-smi reports N/A for unsupported sensors, never zero-fill.
        values["gpu_device_samples"] = {key: distribution(samples) for key, samples in gpu.items()}
        values["torch_allocator"] = {key: distribution(r["torch_allocator"][key] for r in rows if "torch_allocator" in r)
                                     for key in ("allocated_bytes", "reserved_bytes", "peak_allocated_bytes", "peak_reserved_bytes")}
        resource_summary[name] = values
    with (destination / "learning_curve.csv").open("w", encoding="utf-8", newline="") as handle:
        if curve_rows:
            writer = csv.DictWriter(handle, fieldnames=list(curve_rows[0]))
            writer.writeheader()
            writer.writerows(curve_rows)
    result = {"protocol": "az-run-analysis-v1", "run_path": str(output.resolve()), "sessions": session_rows, "rounds": rounds,
              "recorded_actual_wall_seconds": actual_wall if sessions else unavailable("legacy run has no invocation ledger"),
              "hard_kill_wall_lower_bound_seconds": incomplete_lower_bound,
              "hard_kill_completed_phase_seconds": dict(partial_phases),
              "pause_seconds_known": pause, "phase_seconds_mutually_exclusive": dict(phases),
              "phase_fraction_recorded_wall": {k: v / actual_wall for k, v in phases.items()} if actual_wall else {},
              "timing_scope": "all recorded invocations; pause excluded; hard-kill unobserved end not invented; nested worker/owner timers not summed",
              "evaluation": eval_rows, "selection_records": str((destination / "decisions.jsonl").resolve()),
              "resources_by_phase": resource_summary,
              "formal_experiment_state": "not_started" if not (output / "config.json").exists() else "complete" if rounds and all(r["complete"] for r in rounds) and len(rounds) == read_json(output / "config.json")["config"]["num_rounds"] else "incomplete"}
    write_json(destination / "summary.json", result)
    return result
