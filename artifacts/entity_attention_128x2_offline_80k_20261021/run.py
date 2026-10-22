"""Prepare/train attention128 on exactly the stopped set-MLP run's data and schedule."""
from __future__ import annotations

import argparse
import copy
import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from splendor.actor_critic import build_actor_critic
from splendor.alphazero_recording import environment_identity, read_json, sha256, source_identity, write_json
from splendor.offline_mixed_budget import MixedBudgetOfflineConfig, prepare_inputs

ART = Path(__file__).resolve().parent
ROOT = ART.parents[1]
OUTPUT = ROOT / "runs" / ART.name
RUNTIME = ROOT / "runtime" / ART.name
BASE = ROOT / "runs/entity_set_mlp_256x4_offline_80k_20261021"
BASE_ART = ROOT / "artifacts/entity_set_mlp_256x4_offline_80k_20261021"
CONFIG = ROOT / "configs/entity_attention_v1/offline_128x2_no_aux_80k_lr1e4_20261021.json"
PRESET = ROOT / "configs/entity_attention_v1/model_relations_128x2.json"


def identity(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}


def freeze(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Frozen experiment input changed: {path}")
    else:
        write_json(path, value)


def prepare():
    baseline = read_json(BASE / "config.json")
    manifest = OUTPUT / "inputs/pool_games.jsonl"
    source_manifest = Path(baseline["manifest"])
    if sha256(source_manifest) != baseline["manifest_sha256"]:
        raise ValueError("Set-MLP source manifest changed")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    if not manifest.exists():
        shutil.copyfile(source_manifest, manifest)
    if sha256(manifest) != baseline["manifest_sha256"]:
        raise ValueError("Attention source manifest differs from set-MLP")
    raw = copy.deepcopy(baseline)
    raw.update(manifest=str(manifest), model_config=read_json(PRESET))
    freeze(CONFIG, raw)
    config = MixedBudgetOfflineConfig.from_dict(raw)
    _, _, _, fixed, prepared = prepare_inputs(config, OUTPUT)
    old_fixed = read_json(BASE / "fixed_validation.json")
    if fixed != old_fixed:
        raise ValueError("Attention and set-MLP fixed validation references differ")
    dataset = read_json(OUTPUT / "dataset.json")
    if dataset != read_json(BASE / "dataset.json"):
        raise ValueError("Attention and set-MLP data/splits differ")
    plan = copy.deepcopy(read_json(BASE_ART / "plan.json"))
    for key in ("opponent", "search_template"):
        plan.pop(key)
    plan.update(experiment=ART.name, config=identity(CONFIG), runner=identity(Path(__file__)),
        preset=identity(PRESET), model_implementation=identity(ROOT / "src/splendor/entity_attention.py"),
        model_config=config.model_config,
        parameters=build_actor_critic(config.model_config).trainable_parameter_count(),
        initialization="random attention128x2 with existing relation features and global/players/market value readout; fresh AdamW",
        panels=[], comparison={"baseline_config": identity(BASE / "config.json"),
            "baseline_plan": identity(BASE_ART / "plan.json"), "dataset_digest": dataset["digest"],
            "fixed_validation_digest": fixed["digest"], "differences": ["model_config", "independent output/manifest path"]},
        scope="Prepared training only; no formal training until train. Same source games, splits, labels, sampler, seeds, optimizer, budget and validation as set-MLP; no arenas.")
    if prepared["steps_target"] != plan["steps_target"] or prepared["validation_steps"] != plan["validation_steps"]:
        raise ValueError("Attention and set-MLP training/validation schedule differ")
    freeze(ART / "plan.json", plan)
    return plan


def frozen_source():
    destination, record_path = RUNTIME / "source", ART / "source.json"
    if not record_path.exists():
        record = source_identity(ROOT, destination)
        paths = [CONFIG, PRESET, ART / "plan.json", Path(__file__),
                 ROOT / "src/splendor/native_search.cpp", *ROOT.glob("src/splendor/_native_search*.so")]
        for name, expected in read_json(ART / "plan.json")["catalog_files"].items():
            path = ROOT / "data" / name
            if sha256(path) != expected:
                raise ValueError(f"Frozen catalog changed: {path}")
            paths.append(path)
        for path in paths:
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


def report():
    progress = read_json(OUTPUT / "progress.json") if (OUTPUT / "progress.json").exists() else {
        "status": "prepared_not_started", "step": 0}
    result = {"training": progress,
        "full_validation": [read_json(p) for p in sorted((OUTPUT / "full_validation").glob("step-*.json"))],
        "selection": read_json(OUTPUT / "selection.json") if (OUTPUT / "selection.json").exists() else None,
        "arenas": "not configured"}
    write_json(ART / "summary.json", result)
    return result


def run_child(command, phase, source):
    child = subprocess.Popen(command, cwd=source, env={**os.environ, "PYTHONPATH": str(source / "src")},
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
    write_json(RUNTIME / "live.json", {"status": "running", "phase": phase,
        "pid": os.getpid(), "child_pid": child.pid, "command": command})

    def interrupt(signum, frame):
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGINT)

    previous = {s: signal.signal(s, interrupt) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        with (RUNTIME / f"{phase}.log").open("a", encoding="utf-8") as log:
            for line in child.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
        code = child.wait()
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    if code in (130, -signal.SIGINT):
        raise KeyboardInterrupt
    if code:
        raise subprocess.CalledProcessError(code, command)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "train", "select", "status"), nargs="?", default="train")
    args = parser.parse_args()
    RUNTIME.mkdir(parents=True, exist_ok=True)
    if args.action == "status":
        print(report(), flush=True)
        return 0
    with (RUNTIME / "chain.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = prepare()
        report()
        if args.action == "prepare":
            print({"status": "prepared_not_started", "parameters": plan["parameters"],
                "steps": plan["steps_target"], "validation_steps": plan["validation_steps"]}, flush=True)
            return 0
        source = frozen_source()
        command = [sys.executable, "-u", "-m", "splendor.offline_mixed_budget", "--config",
            str(source / CONFIG.relative_to(ROOT)), "--output", str(OUTPUT)]
        path = ART / "execution.json"
        execution = read_json(path) if path.exists() else {"sessions": []}
        record = {"command": [sys.executable, *sys.argv], "start_unix": time.time(),
            "source": identity(ART / "source.json"), "environment": environment_identity(), "status": "running"}
        execution["sessions"].append(record)
        write_json(path, execution)
        try:
            if args.action == "train":
                progress = read_json(OUTPUT / "progress.json") if (OUTPUT / "progress.json").exists() else {}
                if progress.get("status") != "complete":
                    run_child(command, "train", source)
            run_child(command + ["--select-only"], "select", source)
            record["status"] = "complete"
            return 0
        except KeyboardInterrupt:
            record["status"] = "interrupted"
            return 130
        except Exception as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            record["end_unix"] = time.time()
            write_json(path, execution)
            write_json(RUNTIME / "live.json", {"status": record["status"], "pid": os.getpid()})
            report()


if __name__ == "__main__":
    raise SystemExit(main())
