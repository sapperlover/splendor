"""Round-nine one-factor experiments from the unchanged round-eight endpoint."""
from __future__ import annotations

import argparse
import fcntl
import json
import runpy
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch

from splendor.alphazero_checkpoint import load_alphazero_checkpoint
from splendor.alphazero_recording import ReferenceWriter, environment_identity, read_json, sha256, source_identity, write_json
from splendor.alphazero_replay import PositionPool
from splendor.alphazero_train import AlphazeroLearner, LearnerConfig, run_training_round
from splendor.compact_game_cache import CompactGameCache
from splendor.hybrid_train import draw_mixed_batch
from splendor.offline_train import _digest, make_index
from splendor.pool_fitting import evaluate

ART = Path(__file__).resolve().parent
ROOT = ART.parents[1]
ORIGINAL = ROOT / "runs/entity_attention_p4_pool_iteration_20261010/rounds"
EXTENSION = ROOT / "artifacts/entity_attention_r9_step_extension_20261010"
RUN = ROOT / "runs" / ART.name
RUNTIME = ROOT / "runtime" / ART.name
PLAN = ART / "plan.json"
STEPS = (0, 100, 250, 500, 1000, 2000)
BASE_STEPS = (*STEPS, 5000)
STATS_SOURCE = EXTENSION / "run.py"
paired_change = runpy.run_path(str(STATS_SOURCE))["paired_change"]


def identity(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}


def freeze(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"frozen identity differs: {path}")
    else:
        write_json(path, value)


def point_path(panel, step):
    return ART / "fitting" / panel / f"step-{step:09d}.json"


def prepare():
    stage = ORIGINAL / "round-008"
    snap = read_json(stage / "replay_snapshot.json")
    older_path = ORIGINAL / "round-002/replay_snapshot.json"
    older = read_json(older_path)
    records = sorted([*older["games"], *snap["games"]], key=lambda g: (g["round_index"], g["game_id"]))
    if len(records) != 9000 or len({(g["round_index"], g["game_id"]) for g in records}) != 9000:
        raise ValueError("nine-generation pool population differs")
    initial = stage / "checkpoints/initial.pt"
    payload = load_alphazero_checkpoint(initial)
    parent = load_alphazero_checkpoint(snap["teacher"]["path"])
    if (payload.round_index, payload.round_train_step, payload.total_train_step) != (9, 0, 38398):
        raise ValueError("round-eight endpoint counters differ")
    if payload.model_sha256 != parent.model_sha256 or sha256(Path(snap["teacher"]["path"])) != snap["teacher"]["sha256"]:
        raise ValueError("initial model differs from round-eight endpoint")
    base_config = LearnerConfig(**payload.learner_config)
    if (base_config.learning_rate, base_config.batch_size, snap["quotas"]) != (1e-4, 256, [0, 256]):
        raise ValueError("baseline hyperparameters differ")
    dataset = {"games": records, "digest": _digest(records), "sources": [identity(older_path), identity(stage / "replay_snapshot.json")]}
    freeze(ART / "dataset.json", dataset)
    refs_path = stage / "fitting/train_references.json"
    refs = [r for r in read_json(refs_path)["references"] if r["round_index"] == 8]
    if len(refs) != 2048 or any(r["split"] != "train" for r in refs):
        raise ValueError("latest-generation training diagnostic differs")
    freeze(ART / "latest_train_references.json", {"references": refs, "digest": _digest(refs), "source": identity(refs_path)})
    variants = [(f"lr_{lr:g}", "learning_rate", {"learning_rate": lr}) for lr in (5e-5, 2e-5, 1e-5)]
    variants += [(f"pool_{n}", "pool_games", {"pool_games": n}) for n in (9000, 3000, 1000)]
    variants += [(f"batch_{n}", "batch_size", {"batch_size": n}) for n in (64, 128, 512, 1024)]
    panels = []
    for name, axis, override in variants:
        count = override.get("pool_games", 6000)
        selected = records[-count:]
        batch = override.get("batch_size", 256)
        config = replace(base_config, learning_rate=override.get("learning_rate", 1e-4), batch_size=batch,
            history_batch_size=min(base_config.history_batch_size, batch))
        points = sorted({*STEPS, *([125] if batch == 1024 else [])})
        train = [g for g in selected if g["split"] == "train"]
        replay = {"digest": _digest(selected), "source": str((ART / "dataset.json").relative_to(ROOT)),
            "pool_games": count, "scope": "original whole-game splits; uniform Train-position sampling; no offline samples"}
        panels.append({"id": name, "axis": axis, "override": override, "pool_games": count,
            "learner": asdict(config), "steps": points, "target": 2000, "replay_identity": replay,
            "train_games": len(train), "train_positions": sum(g["decision_count"] for g in train),
            "generations": sorted({g["round_index"] + 1 for g in selected}),
            "inactive_history_field": "history_batch_size unused by explicit zero-history drawer; clipped to batch only for dataclass validation"})
    baselines = []
    for step in BASE_STEPS:
        if step <= 500:
            cp = stage / "checkpoints" / ("initial.pt" if step == 0 else f"step-{step:09d}.pt")
            raw = stage / "fitting" / f"step-{step:09d}.json"
        else:
            cp = ROOT / "runs/entity_attention_r9_step_extension_20261010/checkpoints" / f"step-{step:09d}.pt"
            raw = EXTENSION / "fitting" / f"step-{step:09d}.json"
        baselines.append({"step": step, "checkpoint": identity(cp), "original_evaluation": identity(raw)})
    heldout = [g for g in records if g["round_index"] == 8 and g["split"] == "validation"]
    if len(heldout) != 100 or sum(g["decision_count"] for g in heldout) != 8174:
        raise ValueError("fixed ninth-generation validation population differs")
    plan = {"experiment": ART.name, "start": snap["teacher"], "initial_checkpoint": identity(initial),
        "base_learner": asdict(base_config), "baseline_pool_games": 6000, "dataset": identity(ART / "dataset.json"),
        "latest_train_references": identity(ART / "latest_train_references.json"), "baselines": baselines, "panels": panels,
        "validation": {"generation": 9, "games": 100, "positions": 8174, "policy_positions": 6688},
        "runner": identity(Path(__file__)), "statistics_helper": identity(STATS_SOURCE),
        "equal_draw_budget": 128000, "equal_draw_steps": {"64": 2000, "128": 1000, "256": 500, "512": 250, "1024": 125},
        "scope": "One factor at a time versus original lr1e-4/pool6000/batch256 at matched update counts; all branches restore identical round-eight model, AdamW moments and RNG. Fixed ninth-generation DEV; no new labels or battles; no multiplicity correction."}
    freeze(PLAN, plan)
    return plan, payload, records


def read_points(panel, steps):
    return {s: read_json(point_path(panel, s)) for s in steps if point_path(panel, s).exists()}


def comparison(a, b):
    return paired_change(a["validation"]["per_game"], b["validation"]["per_game"])


def metric_row(raw):
    val = raw["validation"]["metrics"]["g009/validation/all"]
    train = raw["latest_train"]["metrics"]["g009/train/all"]
    return {"step": raw["step"], "checkpoint": raw["checkpoint"],
        "policy_kl": val["policy_kl"], "value_mse": val["value_mse"],
        "latest_train_kl": train["policy_kl"], "latest_train_mse": train["value_mse"],
        "validation_positions": val["positions"], "policy_positions": val["policy_weight"]}


def aggregate(plan):
    baseline = read_points("baseline", BASE_STEPS)
    panels = []
    for panel in plan["panels"]:
        points = read_points(panel["id"], panel["steps"])
        rows = []
        for step, raw in points.items():
            row = metric_row(raw)
            if step in baseline:
                cache = ART / "comparisons" / panel["id"] / f"step-{step:09d}.json"
                if not cache.exists():
                    write_json(cache, {"baseline_result": identity(point_path("baseline", step)),
                        "candidate_result": identity(point_path(panel["id"], step)),
                        "delta_vs_same_step": comparison(baseline[step], raw)})
                row["delta_vs_same_step"] = read_json(cache)["delta_vs_same_step"]
            rows.append(row)
        panels.append({"id": panel["id"], "axis": panel["axis"], "override": panel["override"],
            "status": "complete" if set(points) == set(panel["steps"]) else "incomplete", "points": rows})
    equal = []
    if 500 in baseline:
        for batch, step in plan["equal_draw_steps"].items():
            panel_id = "baseline" if batch == "256" else f"batch_{batch}"
            path = point_path(panel_id, step)
            if path.exists():
                raw = read_json(path)
                equal.append({"batch_size": int(batch), "updates": step, "draws": int(batch)*step,
                    **{k: v for k, v in metric_row(raw).items() if k != "step"},
                    "delta_vs_baseline_128000_draws": comparison(baseline[500], raw)})
    base_rows = []
    for step, raw in baseline.items():
        row = metric_row(raw)
        if 500 in baseline:
            row["delta_vs_500"] = comparison(baseline[500], raw)
        base_rows.append(row)
    result = {"status": "complete" if len(baseline) == len(BASE_STEPS) and all(p["status"] == "complete" for p in panels) else "incomplete",
        "plan_sha256": sha256(PLAN), "baseline": base_rows, "panels": panels, "equal_draw_batch_comparison": equal,
        "completed_panels": sum(p["status"] == "complete" for p in panels), "total_panels": len(panels), "scope": plan["scope"]}
    write_json(ART / "summary.json", result)
    lines = ["# 第九轮单变量实验", "", f"状态：`{result['status']}`；新训练分支完成{result['completed_panels']}/{result['total_panels']}。",
        "", "所有分支从第八轮结束38,398步出发，保留原AdamW与RNG。共同评估第九轮固定100局留出：8174个value局面、6688个policy局面。最新代Train诊断为固定2048局面。", "",
        "## 原配置步数曲线", "", "| 步数 | 留出 KL | 留出 MSE | 最新代Train KL | 最新代Train MSE |", "| ---: | ---: | ---: | ---: | ---: |"]
    for row in base_rows:
        lines.append(f"| {row['step']} | {row['policy_kl']:.6f} | {row['value_mse']:.6f} | {row['latest_train_kl']:.6f} | {row['latest_train_mse']:.6f} |")
    lines += ["", "## 固定500步单变量比较", "", "差值均为新分支减同为500步的原配置；负值为改善。区间按100个完整留出对局配对bootstrap，不把局面当独立样本。", "",
        "| 变量 | KL | MSE | ΔKL 95% CI | ΔMSE 95% CI |", "| --- | ---: | ---: | --- | --- |"]
    if 500 in baseline:
        b = metric_row(baseline[500])
        lines.append(f"| 基准lr1e-4 / 6k / B256 | {b['policy_kl']:.6f} | {b['value_mse']:.6f} | — | — |")
    for panel in panels:
        row = next((r for r in panel["points"] if r["step"] == 500), None)
        if row and "delta_vs_same_step" in row:
            d = row["delta_vs_same_step"]
            a, b = (d[k]["ci95"] for k in ("policy_kl", "value_mse"))
            lines.append(f"| {panel['id']} | {row['policy_kl']:.6f} | {row['value_mse']:.6f} | [{a[0]:+.6f}, {a[1]:+.6f}] | [{b[0]:+.6f}, {b[1]:+.6f}] |")
        else:
            lines.append(f"| {panel['id']} | — | — | — | — |")
    lines += ["", "## 各分支完整曲线", "", "每格为留出KL / MSE；125步仅用于B1024的相同抽样量比较，见下一表。", "",
        "| 配置 | 100步 | 250步 | 500步 | 1000步 | 2000步 |", "| --- | --- | --- | --- | --- | --- |"]
    for panel in panels:
        by_step = {r["step"]: r for r in panel["points"]}
        cells = [f"{by_step[s]['policy_kl']:.6f} / {by_step[s]['value_mse']:.6f}" if s in by_step else "—" for s in STEPS[1:]]
        lines.append(f"| {panel['id']} | " + " | ".join(cells) + " |")
    lines += ["", "## Batch按相同128000条训练抽样比较", "", "这张补充表按相同数据量对齐，步数随batch变化；不等同于固定步数的一变量比较。不随batch缩放学习率。", "",
        "| Batch | 步数 | 抽样数 | KL | MSE |", "| ---: | ---: | ---: | ---: | ---: |"]
    for row in equal:
        lines.append(f"| {row['batch_size']} | {row['updates']} | {row['draws']} | {row['policy_kl']:.6f} | {row['value_mse']:.6f} |")
    lines += ["", "所有结果为固定DEV、单起点及单采样种子的探索性结果，未作多重比较校正。池大小改变同时改变教师代际组成及最新代抽样比例，不能归结为容量本身；lr改变保留AdamW动量，测的是续训学习率。未测棋力、未证明配置最优。", ""]
    (ART / "results.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def observe(learner, checkpoint, panel_id, train_index, train_records, train_fixed, val_index, val_records):
    step = learner.round_train_step
    path = point_path(panel_id, step)
    if path.exists():
        if read_json(path)["checkpoint"] != identity(checkpoint):
            raise ValueError("evaluation checkpoint changed")
        return
    tick = time.perf_counter()
    result = {"step": step, "global_step": learner.total_train_step, "checkpoint": identity(checkpoint),
        "plan_sha256": sha256(PLAN), "validation": evaluate(learner, val_index, val_records),
        "latest_train": evaluate(learner, train_index, train_records, fixed=train_fixed)}
    row = metric_row(result)
    if (row["validation_positions"], row["policy_positions"]) != (8174, 6688):
        raise ValueError("fixed validation population differs")
    result["seconds"] = time.perf_counter() - tick
    write_json(path, result)
    print(json.dumps({"event": "fitting", "panel": panel_id, **row}), flush=True)


def train_panel(panel, plan, payload, records, cache, diagnostic, session, budget):
    name = panel["id"]
    config = LearnerConfig(**panel["learner"])
    root = RUN / name
    saved = sorted((root / "checkpoints").glob("step-*.pt"))
    resume_path = saved[-1] if saved else root / "checkpoints/initial.pt"
    if resume_path.exists():
        learner = AlphazeroLearner.resume(resume_path, config, replay_identity=panel["replay_identity"], round_index=9, round_steps_target=panel["target"])
    else:
        learner = AlphazeroLearner.resume(ROOT / plan["initial_checkpoint"]["path"], LearnerConfig(**payload.learner_config),
            replay_identity=payload.replay_identity, round_index=9, round_steps_target=payload.round_steps_target)
        learner.config = config
        for group in learner.optimizer.param_groups:
            group["lr"] = config.learning_rate
        learner.round_steps_target = panel["target"]
        learner.replay_identity = panel["replay_identity"]
        learner.save(root / "checkpoints/initial.pt", metadata={"panel": name, "source": plan["initial_checkpoint"]})
    index = make_index({"games": records[-panel["pool_games"]:]}, cache.max_bytes)
    index.cache = cache
    pool = PositionPool(index.split_games("train"), cache, "online")
    train_index, train_records, fixed, val_index, val_records = diagnostic
    def callback(current, path):
        if current.round_train_step in panel["steps"]:
            observe(current, path, name, train_index, train_records, fixed, val_index, val_records)
            aggregate(plan)
    if not point_path(name, 0).exists():
        initial = read_json(point_path("baseline", 0))
        freeze(point_path(name, 0), {**initial, "checkpoint": identity(root / "checkpoints/initial.pt"), "seconds": 0., "shared_zero_step_evaluation": identity(point_path("baseline", 0))})
    if saved and learner.round_train_step in panel["steps"]:
        callback(learner, saved[-1])
    start = learner.round_train_step
    writer = ReferenceWriter(root / "usage", session, config.batch_size, panel["replay_identity"])
    try:
        final_step = panel["target"] if budget is None else min(panel["target"], start + budget)
        # A requested node such as125 must be committed even between periodic checkpoints.
        stops = sorted({s for s in panel["steps"] if start < s < final_step and s % config.checkpoint_every_steps} | {final_step})
        for stop in stops:
            if learner.round_train_step >= stop:
                continue
            stats = run_training_round(learner, steps_target=panel["target"], current_pool=pool, history_pool=None,
                validation_samples=(), validation_games=0, checkpoint_dir=root / "checkpoints", metrics_path=root / "training_metrics.jsonl",
                reference_writer=writer, checkpoint_callback=callback, max_steps_this_session=stop-learner.round_train_step,
                batch_drawer=lambda rng: draw_mixed_batch(pool, pool, rng, 0, config.batch_size, record_rounds=True),
                progress_callback=lambda row: print(json.dumps({"event": "train", "panel": name, "step": row["round_train_step"]}), flush=True) if row["round_train_step"] % 500 == 0 else None)
            write_json(ART / "training_sessions" / name / f"{session}-step{stop}.json", stats)
    finally:
        writer.close()
    return learner.round_train_step - start


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
            print(json.dumps({"status": result["status"], "completed_panels": result["completed_panels"]}))
            return 0
        session = str(time.time_ns())
        source = source_identity(ROOT, RUNTIME / session)
        path = ART / "execution.json"
        execution = read_json(path) if path.exists() else {"sessions": []}
        execution["sessions"].append({**source, "session": session, "environment": environment_identity(),
            "command": [sys.executable, *sys.argv], "start_unix": time.time(), "plan_sha256": sha256(PLAN)})
        write_json(path, execution)
        cache = CompactGameCache(8 * 1024**3)
        latest = [{**g, "source_id": "g009"} for g in records if g["round_index"] == 8]
        train_records = [g for g in latest if g["split"] == "train"]
        val_records = [g for g in latest if g["split"] == "validation"]
        train_index, val_index = (make_index({"games": rows}, cache.max_bytes) for rows in (train_records, val_records))
        train_index.cache = val_index.cache = cache
        fixed = read_json(ART / "latest_train_references.json")
        diagnostic = train_index, train_records, fixed, val_index, val_records
        for point in plan["baselines"]:
            step = point["step"]
            if point_path("baseline", step).exists():
                continue
            cp = ROOT / point["checkpoint"]["path"]
            saved = load_alphazero_checkpoint(cp)
            learner = AlphazeroLearner.resume(cp, LearnerConfig(**saved.learner_config), replay_identity=saved.replay_identity,
                round_index=saved.round_index, round_steps_target=saved.round_steps_target)
            observe(learner, cp, "baseline", *diagnostic)
            raw = read_json(point_path("baseline", step))
            old = read_json(ROOT / point["original_evaluation"]["path"])
            val = old.get("online_validation_full", old.get("evaluation"))["metrics"]["g009/validation/all"]
            if any(abs(metric_row(raw)[k] - val[k]) > 1e-6 for k in ("policy_kl", "value_mse")):
                raise ValueError("recomputed baseline differs from original evaluation")
            del learner
        aggregate(plan)
        budget = args.max_steps_this_session
        for panel in plan["panels"]:
            if budget is not None and budget <= 0:
                break
            if all(point_path(panel["id"], s).exists() for s in panel["steps"]):
                continue
            print(json.dumps({"event": "start_panel", "panel": panel["id"], "override": panel["override"]}), flush=True)
            used = train_panel(panel, plan, payload, records, cache, diagnostic, session, budget)
            if budget is not None:
                budget -= used
        result = aggregate(plan)
        if result["status"] == "complete":
            write_json(RUN / "complete.json", {"plan_sha256": sha256(PLAN), "completed_panels": result["completed_panels"]})
        print(json.dumps({"status": result["status"], "completed_panels": result["completed_panels"]}), flush=True)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
