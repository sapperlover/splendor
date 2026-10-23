"""User-requested stop: select reached validation minimum and run two greedy arenas."""
from __future__ import annotations

import argparse
import fcntl
import importlib.util
import os
from pathlib import Path
import signal
import sys
import time

from splendor.alphazero_recording import read_json, sha256, write_json

ART = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("original_offline_runner", ART / "run.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
GREEDY = tuple(p for p in base.PANELS if p["kind"] == "greedy")
OVERRIDE = ART / "manual_stop_greedy_plan.json"
EXECUTION = ART / "greedy_after_stop_execution.json"


def prepare():
    plan = read_json(ART / "plan.json")
    if sha256(ART / "run.py") != plan["runner"]["sha256"]:
        raise ValueError("Original runner differs from frozen plan")
    source = base.frozen_source()
    progress = read_json(base.OUTPUT / "progress.json")
    if progress["status"] != "incomplete":
        raise ValueError("Expected user-interrupted, unfinished training")
    live = read_json(base.RUNTIME / "live.json")
    if live["status"] == "running":
        raise ValueError("Another experiment process is running")
    dataset = read_json(base.OUTPUT / "dataset.json")
    rows = []
    for step in plan["validation_steps"]:
        if step > progress["step"]:
            break
        row = read_json(base.OUTPUT / "full_validation" / f"step-{step:09d}.json")
        checkpoint = base.OUTPUT / "checkpoints" / f"step-{step:09d}.pt"
        if (row["step"] != step or row["checkpoint_sha256"] != sha256(checkpoint)
                or row["dataset_digest"] != dataset["digest"]
                or row["metrics"]["positions"] != plan["splits"]["validation"]["positions"]):
            raise ValueError("Reached validation checkpoint/data identity differs")
        row["selection_objective"] = row["metrics"]["policy_kl"] + row["metrics"]["value_loss"]
        rows.append(row)
    best = min(rows, key=lambda row: (row["selection_objective"], row["step"]))
    selection = {
        "criterion": "full_validation.policy_kl + full_validation.value_loss; ties earlier step",
        "candidate_rule": "all reached post-update percentage checkpoints before user-requested stop",
        "validation_percent_step": 5, "candidates": rows,
        "selected_step": best["step"], "checkpoint": best["checkpoint"],
        "checkpoint_sha256": best["checkpoint_sha256"],
        "selection_objective": best["selection_objective"],
        "scope": "single training seed; validation-selected full checkpoint, no policy/value composition",
    }
    base.freeze(base.OUTPUT / "selection.json", selection)
    base.freeze(ART / "selected_candidate.json", selection)
    override = {
        "authorization": "User: 停止训练，直接开始两组greedy",
        "training_stop_reason": "user_requested", "saved_step": progress["step"],
        "original_steps_target": progress["steps_target"],
        "training_budget_completed": False,
        "early_stopping_triggered": read_json(base.OUTPUT / "early_stopping.json")["triggered"],
        "original_execution": base.identity(ART / "execution.json"),
        "original_plan": base.identity(ART / "plan.json"),
        "runner": base.identity(Path(__file__)),
        "source": base.identity(ART / "source.json"),
        "selection": base.identity(ART / "selected_candidate.json"),
        "panels": list(GREEDY),
        "search_panels": "cancelled_by_user; this runner never launches search or training",
    }
    base.freeze(OVERRIDE, override)
    for panel in GREEDY:
        base.freeze(ART / "evaluation_configs" / f"{panel['id']}.json",
                    base.evaluation_config(plan, selection, panel))
    return source, selection, rows


def report(selection, rows, status):
    panels = [base.summarize_panel(panel) for panel in GREEDY]
    panels += [{**panel, "status": "cancelled_by_user", "completed_pairs": 0}
               for panel in base.PANELS if panel["kind"] == "search"]
    if all(p["status"] == "complete" for p in panels[:2]):
        status = "user_stopped_training_greedy_complete"
    summary = {"status": status, "training": read_json(base.OUTPUT / "progress.json"),
               "training_stop_reason": "user_requested", "selection": selection,
               "full_validation": rows, "panels": panels,
               "scope_override": base.identity(OVERRIDE)}
    write_json(ART / "summary.json", summary)
    lines = ["# Attention128：第21–31轮固定池离线训练", "",
             f"状态：`{status}`。用户主动停止训练，保留84000步恢复点；原148632步预算未完成，未触发自动早停。", "",
             f"对战前按已完成完整留出KL＋MSE固定整个checkpoint：step={selection['selected_step']}，SHA `{selection['checkpoint_sha256']}`。", "",
             "| 步数 | 完整留出KL | 胜负MSE | KL＋MSE |",
             "| ---: | ---: | ---: | ---: |"]
    for row in rows:
        m = row["metrics"]
        lines.append(f"| {row['step']} | {m['policy_kl']:.6f} | {m['value_loss']:.6f} | {row['selection_objective']:.6f} |")
    lines += ["", "| 对战 | 完成／计划pair | 候选得分 | 95%配对区间 |", "| --- | ---: | ---: | --- |"]
    for panel in panels[:2]:
        score = f"{100*panel['score']:.3f}%" if panel["status"] == "complete" else "—"
        ci = (f"[{100*panel['ci95']['lower']:.3f}%,{100*panel['ci95']['upper']:.3f}%]"
              if panel["status"] == "complete" else "—")
        lines.append(f"| {panel['id']} | {panel['completed_pairs']}/{panel['pairs']} | {score} | {ci} |")
    lines += ["", "本次仅两组greedy，各1000pair／2000局；12贵族、CPU8worker，同一开局交换先后手，和棋半分。",
              "两组共用44000000–44000999开局，整pair bootstrap10000次；双方模型／规则身份、完整交换先后手及零搜索已核对后才计为complete。",
              "原两组S1024N1搜索按用户最新指令取消，未运行。原配置／源码／计划保留，未将中断训练标记为预算完成。", ""]
    (ART / "results.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "status"), nargs="?", default="run")
    args = parser.parse_args()
    if args.action == "status":
        selection = read_json(ART / "selected_candidate.json")
        print(report(selection, selection["candidates"], "user_stopped_training_greedy_pending")["status"])
        return
    with (base.RUNTIME / "chain.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source, selection, rows = prepare()
        report(selection, rows, "user_stopped_training_greedy_pending")
        if args.action == "prepare":
            print({"selected_step": selection["selected_step"], "panels": [p["id"] for p in GREEDY]}, flush=True)
            return
        signal.signal(signal.SIGINT, base.interrupt)
        signal.signal(signal.SIGTERM, base.interrupt)
        execution = read_json(EXECUTION) if EXECUTION.exists() else {"sessions": []}
        record = {"command": [sys.executable, *sys.argv], "pid": os.getpid(), "start_unix": time.time(),
                  "status": "running", "scope_override": base.identity(OVERRIDE)}
        execution["sessions"].append(record)
        write_json(EXECUTION, execution)
        try:
            for panel in GREEDY:
                if base.summarize_panel(panel)["status"] == "complete":
                    continue
                config = ART / "evaluation_configs" / f"{panel['id']}.json"
                output = ART / "panels" / f"{panel['id']}.json"
                command = [sys.executable, "-u", "-m", "splendor.evaluation", "--config", str(config),
                           "--output", str(output), "--progress-every", "10", "--summary-every", "10"]
                if output.exists() or output.with_suffix(".json.games").exists():
                    command.append("--resume")
                base.run_child(command, panel["id"], source)
                if base.summarize_panel(panel)["status"] != "complete":
                    raise ValueError("Greedy panel stopped before completion")
                report(selection, rows, "user_stopped_training_greedy_running")
            record["status"] = "complete"
        except KeyboardInterrupt:
            record["status"] = "interrupted"
        except Exception as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            record["end_unix"] = time.time()
            write_json(EXECUTION, execution)
            report(selection, rows, f"user_stopped_training_greedy_{record['status']}")
            write_json(base.RUNTIME / "live.json", {"status": record["status"], "pid": os.getpid(),
                       "scope": "greedy_only_after_user_stop", "error": record.get("error")})


if __name__ == "__main__":
    main()
