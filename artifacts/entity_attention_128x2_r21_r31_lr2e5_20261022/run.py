"""Prepare or continue the fixed selected attention checkpoint; no strength arenas."""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from splendor.alphazero_recording import environment_identity, read_json, sha256, source_identity, write_json
from splendor.offline_finetune import FinetuneOfflineConfig, prepare_inputs

ART = Path(__file__).resolve().parent
ROOT = ART.parents[1]
OUTPUT = ROOT / "runs" / ART.name
RUNTIME = ROOT / "runtime" / ART.name
PARENT = ROOT / "runs/entity_attention_128x2_offline_r21_r31_20261022"
CONFIG = ROOT / "configs/entity_attention_v1/offline_128x2_r21_r31_lr2e5_20261022.json"


def identity(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}


def freeze(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Frozen continuation input changed: {path}")
    else:
        write_json(path, value)


def prepare():
    selected = read_json(PARENT / "selection.json")
    checkpoint = Path(selected["checkpoint"])
    if sha256(checkpoint) != selected["checkpoint_sha256"]:
        raise ValueError("Selected parent checkpoint changed")
    rows = [read_json(p) for p in sorted((PARENT / "full_validation").glob("step-*.json"))]
    best = min(rows, key=lambda p: (p["metrics"]["policy_kl"]+p["metrics"]["value_loss"], p["step"]))
    if best["step"] != selected["selected_step"] or best["checkpoint_sha256"] != selected["checkpoint_sha256"]:
        raise ValueError("Parent selection is not the minimum completed full-holdout objective")
    raw = read_json(PARENT / "config.json")
    model = raw.pop("model_config")
    raw.pop("initialization_seed")
    raw.pop("validation_percent_step")
    raw.update(initial_checkpoint=str(checkpoint), initial_checkpoint_sha256=selected["checkpoint_sha256"],
        validation_interval_steps=2000)
    raw["learner"]["learning_rate"] = 2e-5
    freeze(CONFIG, raw)
    config = FinetuneOfflineConfig.from_dict(raw)
    _, _, _, fixed, training = prepare_inputs(config, OUTPUT)
    if read_json(OUTPUT / "dataset.json") != read_json(PARENT / "dataset.json"):
        raise ValueError("Continuation dataset or whole-game splits differ")
    if fixed != read_json(PARENT / "fixed_validation.json"):
        raise ValueError("Continuation fixed validation references differ")
    plan = {**training, "status": "prepared_not_started", "experiment": ART.name,
        "config": identity(CONFIG), "runner": identity(Path(__file__)), "parent_selection": identity(PARENT / "selection.json"),
        "parent_checkpoint": identity(checkpoint), "parent_step": selected["selected_step"],
        "parent_objective": selected["selection_objective"], "parent_full_validation": best,
        "parent_completed_full_nodes": len(rows), "model_config": model,
        "dataset_digest": read_json(OUTPUT / "dataset.json")["digest"],
        "scope": "Prepared command only. Same44k pool, labels, splits, fixed holdout, batches, cache and optimizer settings; inherit selected model/AdamW/all RNG. Only lr2e-5 and full-validation interval2000 change. Preserve3 strict-increase stopping, step0 excluded. Additional12 value exposures; no arena code or automatic strength evaluations."}
    freeze(ART / "plan.json", plan)
    return plan


def frozen_source():
    destination, record_path = RUNTIME / "source", ART / "source.json"
    if not record_path.exists():
        record = source_identity(ROOT, destination)
        for path in (CONFIG, Path(__file__), ROOT / "src/splendor/native_search.cpp",
                     *ROOT.glob("src/splendor/_native_search*.so"), *(ROOT / "data").glob("splendor_*")):
            target = destination / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        record["snapshot_files"] = {str(p.relative_to(destination)): sha256(p)
            for p in sorted(destination.rglob("*")) if p.is_file()}
        freeze(record_path, record)
    for path, expected in read_json(record_path)["snapshot_files"].items():
        if sha256(destination / path) != expected:
            raise ValueError(f"Frozen source changed: {path}")
    return destination


def report(plan):
    progress = read_json(OUTPUT / "progress.json") if (OUTPUT / "progress.json").exists() else {"status": "prepared_not_started", "step": 0}
    points = [read_json(p) for p in sorted((OUTPUT / "full_validation").glob("step-*.json"))]
    selected = read_json(OUTPUT / "selection.json") if (OUTPUT / "selection.json").exists() else None
    result = {"status": progress["status"], "training": progress, "selection": selected,
        "full_validation": points, "parent_step": plan["parent_step"], "arenas": "not configured"}
    write_json(ART / "summary.json", result)
    lines = ["# 第21–31轮attention128：最低节点lr2e-5续训", "", f"状态：`{progress['status']}`，本段已提交{progress['step']}/{plan['steps_target']}步；起点为原{plan['parent_step']}步。", "",
        "继承完整AdamW／采样及Torch RNG，lr2e-5；每2000步评估固定4096子集与完整留出KL/MSE，连续三次联合指标严格上升早停。无对战流程。", "",
        "| 本段步数 | 累计步数 | 完整KL | 胜负MSE | KL+MSE |", "| ---: | ---: | ---: | ---: | ---: |"]
    for point in points:
        m = point["metrics"]
        lines.append(f"| {point['step']} | {plan['parent_step']+point['step']} | {m['policy_kl']:.6f} | {m['value_loss']:.6f} | {m['policy_kl']+m['value_loss']:.6f} |")
    if selected:
        lines += ["", f"选中本段step={selected['selected_step']}，累计{plan['parent_step']+selected['selected_step']}，KL+MSE={selected['selection_objective']:.9f}。"]
    (ART / "results.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "status"), nargs="?", default="run")
    args = parser.parse_args()
    RUNTIME.mkdir(parents=True, exist_ok=True)
    if args.action == "status":
        print(report(read_json(ART / "plan.json"))["status"], flush=True)
        return 0
    with (RUNTIME / "chain.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == "prepare":
            plan = prepare()
            report(plan)
            print({"status": "prepared_not_started", "parent_step": plan["parent_step"], "steps_target": plan["steps_target"]}, flush=True)
            return 0
        plan = read_json(ART / "plan.json")
        if identity(CONFIG) != plan["config"] or identity(Path(__file__)) != plan["runner"]:
            raise ValueError("Prepared continuation configuration or entry point changed")
        if report(plan)["status"] == "complete":
            return 0
        source = frozen_source()
        command = [sys.executable, "-u", "-m", "splendor.offline_finetune", "--config",
            str(source / CONFIG.relative_to(ROOT)), "--output", str(OUTPUT)]
        execution_path = ART / "execution.json"
        execution = read_json(execution_path) if execution_path.exists() else {"sessions": []}
        record = {"status": "running", "start_unix": time.time(), "command": command,
            "source": identity(ART / "source.json"), "environment": environment_identity()}
        execution["sessions"].append(record)
        write_json(execution_path, execution)
        process = None

        def interrupt(signum, frame):
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)

        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, interrupt)
        try:
            with (RUNTIME / "train.log").open("a", encoding="utf-8") as log:
                process = subprocess.Popen(command, cwd=source, env={**os.environ, "PYTHONPATH": str(source / "src")},
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
                write_json(RUNTIME / "live.json", {"status": "running", "phase": "train", "pid": os.getpid(),
                    "child_pid": process.pid, "command": command})
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
                code = process.wait()
            record["status"] = "complete" if code == 0 else "interrupted" if code in (130, -signal.SIGINT) else "failed"
            return code if code >= 0 else 128-code
        finally:
            record["end_unix"] = time.time()
            write_json(execution_path, execution)
            write_json(RUNTIME / "live.json", {"status": record["status"], "pid": os.getpid()})
            report(plan)


if __name__ == "__main__":
    raise SystemExit(main())
