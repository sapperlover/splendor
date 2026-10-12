"""Extend the original round-nine optimizer trajectory, keeping its replay fixed."""
from __future__ import annotations

import argparse
import fcntl
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from splendor.alphazero_checkpoint import load_alphazero_checkpoint
from splendor.alphazero_recording import ReferenceWriter, environment_identity, read_json, sha256, source_identity, write_json
from splendor.alphazero_replay import PositionPool
from splendor.alphazero_train import AlphazeroLearner, LearnerConfig, run_training_round
from splendor.compact_game_cache import CompactGameCache
from splendor.hybrid_train import draw_mixed_batch
from splendor.offline_train import make_index
from splendor.pool_fitting import evaluate

ART = Path(__file__).resolve().parent
ROOT = ART.parents[1]
ORIGINAL = ROOT / "runs/entity_attention_p4_pool_iteration_20261010/rounds/round-008"
RUN = ROOT / "runs" / ART.name
RUNTIME = ROOT / "runtime" / ART.name
PLAN = ART / "plan.json"
POINTS = (500, 1000, 2000, 5000)
TARGET = 5000


def identity(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}


def freeze(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"frozen identity differs: {path}")
    else:
        write_json(path, value)


def prepare():
    snapshot = read_json(ORIGINAL / "replay_snapshot.json")
    checkpoint = ORIGINAL / "checkpoints/final.pt"
    if sha256(checkpoint) != "ff34660fece12590762da93d031523e4ae09db4fa6af5d2284c4af0dcd57237b":
        raise ValueError("original ninth-round model changed")
    payload = load_alphazero_checkpoint(checkpoint)
    if (payload.round_index, payload.round_train_step, payload.total_train_step, payload.round_steps_target) != (9, 500, 38898, 500):
        raise ValueError("original checkpoint counters differ")
    if payload.replay_identity["online"]["digest"] != snapshot["digest"] or snapshot["quotas"] != [0, 256]:
        raise ValueError("original replay identity or sampling quotas differ")
    if sha256(Path(snapshot["teacher"]["path"])) != snapshot["teacher"]["sha256"]:
        raise ValueError("round-eight teacher changed")
    records = snapshot["games"]
    heldout = [g for g in records if g["round_index"] == 8 and g["split"] == "validation"]
    train = [g for g in records if g["split"] == "train"]
    if len(records) != 6000 or len(heldout) != 100 or sum(g["decision_count"] for g in heldout) != 8174:
        raise ValueError("frozen population differs")
    plan = {"experiment": ART.name, "start": snapshot["teacher"], "reused_500_step_checkpoint": identity(checkpoint),
        "snapshot": identity(ORIGINAL / "replay_snapshot.json"), "learner": payload.learner_config,
        "replay_identity": payload.replay_identity, "target_steps": TARGET, "evaluated_steps": list(POINTS),
        "training": {"games": len(train), "positions": sum(g["decision_count"] for g in train),
            "generations": sorted({g["round_index"] + 1 for g in train}),
            "sampling": "original draw_mixed_batch(offline=0, online=256); uniform over training positions; AdamW and all RNG state continued from original step500"},
        "validation": {"games": len(heldout), "positions": sum(g["decision_count"] for g in heldout),
            "policy_positions": sum(g["policy_samples"] for g in heldout), "generation": 9,
            "records": heldout, "scope": "complete fixed ninth-generation whole-game DEV split; never sampled for training"},
        "runner": identity(Path(__file__)), "metric_implementation": identity(ROOT / "src/splendor/pool_fitting.py"),
        "original_fitting": {str(s): identity(ORIGINAL / "fitting" / f"step-{s:09d}.json") for s in (0, 500)},
        "scope": "Only update count changes; one optimizer trajectory with nested checkpoints, not independent training seeds. No new selfplay, no search battle."}
    freeze(PLAN, plan)
    return plan, payload, records


def paired_change(before, after):
    keys = sorted(k for k in before if k.endswith("/all"))
    if keys != sorted(k for k in after if k.endswith("/all")):
        raise ValueError("paired validation games differ")
    rng = np.random.default_rng(31050000)
    result = {"games": len(keys), "resamples": 10000, "seed": 31050000, "unit": "whole validation game"}
    for metric, numerator, denominator in (("policy_kl", "kl", "policy_weight"), ("value_mse", "sq_error", "positions")):
        weights = np.array([before[k][denominator] for k in keys])
        if any(before[k][denominator] != after[k][denominator] for k in keys):
            raise ValueError("paired metric population differs")
        diffs = np.array([after[k][numerator] - before[k][numerator] for k in keys])
        sampled = []
        for _ in range(40):
            indices = rng.integers(0, len(keys), size=(250, len(keys)))
            sampled.extend((diffs[indices].sum(1) / weights[indices].sum(1)).tolist())
        result[metric] = {"delta": float(diffs.sum() / weights.sum()), "ci95": np.quantile(sampled, [.025, .975]).tolist()}
    return result


def aggregate(plan):
    rows, baseline = [], None
    for step in POINTS:
        path = ART / "fitting" / f"step-{step:09d}.json"
        if not path.exists():
            rows.append({"step": step, "status": "pending"})
            continue
        raw = read_json(path)
        metrics = raw["evaluation"]["metrics"]["g009/validation/all"]
        if (metrics["positions"], metrics["policy_weight"]) != (8174, 6688):
            raise ValueError("evaluated population differs")
        row = {"step": step, "status": "complete", "checkpoint": raw["checkpoint"],
            "global_step": raw["global_step"], "policy_kl": metrics["policy_kl"], "value_mse": metrics["value_mse"],
            "result": identity(path), "seconds": raw["seconds"]}
        if step == 500:
            baseline = raw["evaluation"]["per_game"]
        else:
            row["minus_500"] = paired_change(baseline, raw["evaluation"]["per_game"])
        rows.append(row)
    result = {"status": "complete" if all(r["status"] == "complete" for r in rows) else "incomplete",
        "plan_sha256": sha256(PLAN), "points": rows, "scope": plan["scope"],
        "bootstrap_scope": "conditional on fixed DEV games and this training trajectory; exploratory, no multiplicity adjustment"}
    write_json(ART / "summary.json", result)
    lines = ["# 第九轮固定数据：训练步数延长", "", f"状态：`{result['status']}`。从第八轮结束模型38,398步出发，保留原第4–9轮6000局窗口、原AdamW及RNG；前500步复用原训练，新增4500步。",
        "", "评估完整第九轮固定留出：100局、8174个value局面、6688个有效policy局面。KL为KL(教师标签∥学生)，MSE对原终局value标签；越低越好。", "",
        "| 本轮总步数 | 全局步数 | 留出 KL | 留出 MSE | ΔKL 对500步 | ΔMSE 对500步 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        if row["status"] != "complete":
            lines.append(f"| {row['step']} | — | — | — | — | — |")
            continue
        delta = row.get("minus_500")
        dk = f"{delta['policy_kl']['delta']:+.6f}" if delta else "0"
        dm = f"{delta['value_mse']['delta']:+.6f}" if delta else "0"
        lines.append(f"| {row['step']} | {row['global_step']} | {row['policy_kl']:.6f} | {row['value_mse']:.6f} | {dk} | {dm} |")
    lines += ["", "相对500步的整局配对bootstrap 95%区间：", "", "| 总步数 | ΔKL 95% CI | ΔMSE 95% CI |", "| ---: | --- | --- |"]
    for row in rows:
        if "minus_500" in row:
            dk, dm = (row["minus_500"][k]["ci95"] for k in ("policy_kl", "value_mse"))
            lines.append(f"| {row['step']} | [{dk[0]:+.6f}, {dk[1]:+.6f}] | [{dm[0]:+.6f}, {dm[1]:+.6f}] |")
    lines += ["", "本实验只测固定标签的留出拟合。各点属于同一条连续训练轨迹；未复验训练seed，也未测试棋力，不能据此认定最优自迭代步数。", ""]
    (ART / "results.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "analyze"), nargs="?", default="run")
    parser.add_argument("--max-steps-this-session", type=int)
    args = parser.parse_args()
    RUNTIME.mkdir(parents=True, exist_ok=True)
    with (RUNTIME / "run.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        plan, payload, records = prepare()
        result = aggregate(plan)
        if args.action != "run" or result["status"] == "complete":
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        session = str(time.time_ns())
        source = source_identity(ROOT, RUNTIME / session)
        execution_path = ART / "execution.json"
        execution = read_json(execution_path) if execution_path.exists() else {"sessions": []}
        execution["sessions"].append({**source, "environment": environment_identity(), "session": session,
            "command": [sys.executable, *sys.argv], "start_unix": time.time(), "plan_sha256": sha256(PLAN)})
        write_json(execution_path, execution)
        config = LearnerConfig(**plan["learner"])
        saved = sorted((RUN / "checkpoints").glob("step-*.pt"))
        checkpoint = saved[-1] if saved else ROOT / plan["reused_500_step_checkpoint"]["path"]
        learner = AlphazeroLearner.resume(checkpoint, config, replay_identity=payload.replay_identity,
            round_index=9, round_steps_target=TARGET if saved else 500)
        learner.round_steps_target = TARGET
        index = make_index({"games": records}, 8 * 1024**3)
        index.cache = CompactGameCache(index.cache.max_bytes)
        pool = PositionPool(index.split_games("train"), index.cache, "online")
        validation_records = [{**g, "source_id": "g009"} for g in plan["validation"]["records"]]
        heldout = make_index({"games": validation_records}, index.cache.max_bytes)
        heldout.cache = index.cache

        def observe(current, path):
            step = current.round_train_step
            if step not in POINTS:
                return
            destination = ART / "fitting" / f"step-{step:09d}.json"
            if destination.exists():
                if read_json(destination)["checkpoint"]["sha256"] != sha256(path):
                    raise ValueError("fitting checkpoint differs on resume")
                return
            tick = time.perf_counter()
            evaluation = evaluate(current, heldout, validation_records)
            if step == 500:
                old = read_json(ORIGINAL / "fitting/step-000000500.json")["online_validation_full"]["metrics"]["g009/validation/all"]
                actual = evaluation["metrics"]["g009/validation/all"]
                if any(abs(actual[k] - old[k]) > 1e-6 for k in ("policy_kl", "value_mse")):
                    raise ValueError("recomputed original 500-step baseline differs")
            write_json(destination, {"step": step, "global_step": current.total_train_step, "checkpoint": identity(path),
                "plan_sha256": sha256(PLAN), "evaluation": evaluation, "seconds": time.perf_counter() - tick})
            print(json.dumps({"event": "fitting", "step": step, **evaluation["pool_weighted"]["validation"]}), flush=True)
            aggregate(plan)

        if not saved:
            observe(learner, checkpoint)
            learner.save(RUN / "checkpoints/step-000000500.pt", metadata={"experiment": ART.name, "original_500": plan["reused_500_step_checkpoint"]})
        elif learner.round_train_step in POINTS:
            # The reused baseline retains its original checkpoint identity.
            if learner.round_train_step != 500:
                observe(learner, checkpoint)
        writer = ReferenceWriter(RUN / "usage", session, config.batch_size, plan["replay_identity"]["online"])
        try:
            training = run_training_round(learner, steps_target=TARGET, current_pool=pool, history_pool=None,
                validation_samples=(), validation_games=0, checkpoint_dir=RUN / "checkpoints",
                metrics_path=RUN / "training_metrics.jsonl", reference_writer=writer,
                checkpoint_callback=observe, max_steps_this_session=args.max_steps_this_session,
                batch_drawer=lambda rng: draw_mixed_batch(pool, pool, rng, 0, config.batch_size, record_rounds=True),
                progress_callback=lambda row: print(json.dumps({"event": "train_progress", "step": row["round_train_step"],
                    "global_step": row["total_train_step"], "loss": row["total_loss"]}), flush=True) if row["round_train_step"] % 100 == 0 else None)
        finally:
            writer.close()
        write_json(ART / "training_sessions" / f"{session}.json", training)
        result = aggregate(plan)
        if result["status"] == "complete":
            write_json(RUN / "complete.json", {"plan_sha256": sha256(PLAN), "checkpoint": identity(RUN / "checkpoints/step-000005000.pt")})
        print(json.dumps({"status": result["status"], "step": learner.round_train_step}), flush=True)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
