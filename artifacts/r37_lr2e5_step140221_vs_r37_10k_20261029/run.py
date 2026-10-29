"""Fixed R37-initialized104k/lr2e-5 step140221 vs rawR37: greedy then bothS128N1."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from splendor.alphazero_checkpoint import load_alphazero_checkpoint
from splendor.alphazero_recording import environment_identity, read_json, sha256, write_json
from splendor.catalog import load_default_catalog

ROOT = Path(__file__).resolve().parents[2]
ART = Path(__file__).resolve().parent
RUNTIME = ROOT / "runtime" / ART.name
TRAIN_NAME = "entity_attention_r37_104k_lr2e5_20261027"
TRAIN = ROOT / "runs" / TRAIN_NAME
SHARED = ROOT / "artifacts/offline5_vs_r37_10k_20261027/run.py"
STEP = 140221
CANDIDATE_SHA = "42fbd07c413190389ac5f2cab73ef41ff4a10f5402a13e099a37d49aa3cb6caa"


def configured_entry():
    spec = importlib.util.spec_from_file_location("fixed_r37_lr2e5_evaluation", SHARED)
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    entry.ROOT, entry.ART, entry.RUNTIME = ROOT, ART, RUNTIME
    original_report = entry.write_report

    def prepare():
        reference_path = ROOT / "artifacts" / TRAIN_NAME / "plan.json"
        reference, training_config = read_json(reference_path), read_json(TRAIN / "config.json")
        anchor = reference["opponents"]["R37"]
        if (training_config["learner"]["learning_rate"] != 2e-5
                or training_config["initial_checkpoint_sha256"] != anchor["sha256"]
                or reference["games"] != 104000):
            raise ValueError("Expected the original R37-initialized104k/lr2e-5 training branch")
        path = TRAIN / "checkpoints" / f"step-{STEP:09d}.pt"
        validation_path = TRAIN / "full_validation" / f"step-{STEP:09d}.json"
        validation = read_json(validation_path)
        if sha256(path) != CANDIDATE_SHA or validation["checkpoint_sha256"] != CANDIDATE_SHA:
            raise ValueError("The user-fixed140221-step checkpoint identity changed")
        payload = load_alphazero_checkpoint(path)
        if (payload.total_train_step != STEP or payload.round_train_step != STEP
                or payload.learner_config["learning_rate"] != 2e-5
                or payload.model_config["architecture"] != "entity_attention_v1"
                or payload.parent_checkpoint["sha256"] != anchor["sha256"]):
            raise ValueError("Fixed checkpoint lineage, step, LR or architecture differs")
        if sha256(Path(anchor["path"])) != anchor["sha256"]:
            raise ValueError("Raw R37 opponent changed")
        model = {"key": "r37_lr2e5_step140221", "experiment": TRAIN_NAME,
            "checkpoint": str(path), "sha256": CANDIDATE_SHA, "selected_step": STEP,
            "model_config": payload.model_config, "selection_kind": "user_fixed_checkpoint",
            "validation": entry.identity(validation_path),
            "policy_kl": validation["metrics"]["policy_kl"], "value_mse": validation["metrics"]["value_loss"],
            "selection_objective": validation["metrics"]["policy_kl"] + validation["metrics"]["value_loss"]}
        opponent = {"key": "raw_R37", "checkpoint": anchor["path"], "sha256": anchor["sha256"],
            "model_config": load_alphazero_checkpoint(anchor["path"]).model_config}
        catalog = load_default_catalog()
        if catalog.data_fingerprint != reference["catalog_fingerprint"]:
            raise ValueError("Current catalog differs from the original12-noble rules")
        templates = {kind: read_json(ROOT / "configs/entity_attention_v1" / filename) for kind, filename in
            (("greedy", "evaluation_greedy_r43.json"), ("search", "evaluation_search_a_r43.json"))}
        entry.freeze(ART / "evaluation_templates.json", templates)
        for filename, expected in reference["catalog_files"].items():
            source, target = ROOT / "data" / filename, ART / "catalog" / filename
            if sha256(source) != expected:
                raise ValueError("Original catalog changed")
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copyfile(source, target)
            if sha256(target) != expected:
                raise ValueError("Frozen catalog changed")
        panels = []
        for kind in ("greedy", "search"):
            panel_id = f"{len(panels)+1:02d}_{kind}_step140221_vs_R37"
            config = entry.panel_config(model, opponent, kind, templates, catalog.data_fingerprint)
            config["metadata"]["selection"] = "User-fixed140221-step checkpoint; never follow latest progress or reselect"
            config_path = ART / "configs" / f"{panel_id}.json"
            entry.freeze(config_path, config)
            panels.append({"id": panel_id, "kind": kind, "model": model["key"], "pairs": 10000,
                "config": entry.identity(config_path), "output": str((ART / "panels" / f"{panel_id}.json").relative_to(ROOT))})
        plan = {"experiment": ART.name, "runner": entry.identity(Path(__file__)),
            "shared_runner": entry.identity(SHARED), "training_plan": entry.identity(reference_path),
            "models": [model], "opponent": opponent, "panels": panels,
            "catalog_fingerprint": catalog.data_fingerprint, "catalog_files": reference["catalog_files"],
            "pairs_per_panel": 10000, "total_pairs": 20000, "total_games": 40000,
            "seeds": {"game_seed_start": entry.SEED_START, "game_seed_end": entry.SEED_START+9999,
                "policy_seed_base": entry.POLICY_SEED_BASE, "search_seed": entry.SEARCH_SEED,
                "bootstrap_seed": entry.BOOTSTRAP_SEED},
            "progress_every": entry.PROGRESS_EVERY, "summary_every": entry.SUMMARY_EVERY,
            "selection": "User-fixed step140221 (about140k), not the latest checkpoint; never auto-reselect",
            "scope": "Original R37-initialized104k/lr2e-5 checkpoint140221 vs rawR37. Greedy10000pair, then bothS128N1(total128each)10000pair, same seed bank as prior offline5 comparison. Strictly serial and resumable full pairs. No training or dependency on training completion/selection.json."}
        entry.freeze(ART / "plan.json", plan)
        return plan

    def write_report(plan, rows):
        original_report(plan, rows)
        path = ART / "results.md"
        text = path.read_text().replace("五个离线best对raw R37", "R37初值lr2e-5的固定140221步对raw R37")
        text = text.replace("先全部greedy，再全部搜索", "先greedy，再双方S128N1")
        text = text.replace("模型在本次对战前已按原完整留出主KL＋MSE选定", "模型由用户明确固定为140221步，恢复时不跟随最新节点或重选")
        path.write_text(text, encoding="utf-8")

    def run_panel(panel, plan, source):
        command = entry.panel_command(panel, source)
        record_path = RUNTIME / f"{panel['id']}_execution.json"
        execution = read_json(record_path) if record_path.exists() else {"sessions": []}
        record = {"command": command, "start_unix": time.time(), "status": "running",
            "source": entry.identity(ART / "source.json"), "config": panel["config"]}
        execution["sessions"].append(record)
        write_json(record_path, execution)
        try:
            with (RUNTIME / f"{panel['id']}.log").open("a", encoding="utf-8") as log:
                entry.child = subprocess.Popen(command, cwd=source,
                    env={**os.environ, "PYTHONPATH": str(source / "src")}, stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
                write_json(RUNTIME / "live.json", {"status": "running", "panel": panel["id"],
                    "pid": os.getpid(), "child_pid": entry.child.pid, "command": command})
                for line in entry.child.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
                code = entry.child.wait()
                entry.child = None
            if entry.stopping or code in (130, -signal.SIGINT):
                raise KeyboardInterrupt
            if code:
                raise subprocess.CalledProcessError(code, command)
            record["status"] = "complete"
        except BaseException as error:
            record.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
            raise
        finally:
            record["end_unix"] = time.time()
            write_json(record_path, execution)

    entry.prepare, entry.write_report, entry.run_panel = prepare, write_report, run_panel
    return entry


def frozen_source(entry):
    destination, record_path = RUNTIME / "source", ART / "source.json"
    if not record_path.exists():
        scopes = ["src", "scripts", str(ART.relative_to(ROOT)), str(SHARED.relative_to(ROOT)),
                  "configs/entity_attention_v1/evaluation_greedy_r43.json", "configs/entity_attention_v1/evaluation_search_a_r43.json"]
        patch = subprocess.check_output(["git", "diff", "HEAD", "--", *scopes], cwd=ROOT)
        files = set(ROOT.glob("src/splendor/**/*.py")) | set(ROOT.glob("scripts/*.py"))
        files.update([Path(__file__), ART / "validate.py", SHARED,
            ROOT / "scripts/run_r37_lr2e5_step140221_vs_r37_10k_20261029.sh",
            ROOT / "src/splendor/native_search.cpp", *ROOT.glob("src/splendor/_native_search*.so"),
            *(ART / "configs").glob("*.json")])
        for source in sorted(files):
            target = destination / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        for source in (ART / "catalog").iterdir():
            target = destination / "data" / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        (destination / "dirty.patch").write_bytes(patch)
        entry.freeze(record_path, {"git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "patch_scope": scopes, "dirty_diff_sha256": hashlib.sha256(patch).hexdigest(),
            "snapshot_files": {str(p.relative_to(destination)): sha256(p) for p in sorted(destination.rglob("*"))
                               if p.is_file() and "__pycache__" not in p.parts}})
    for relative, expected in read_json(record_path)["snapshot_files"].items():
        if sha256(destination / relative) != expected:
            raise ValueError(f"Frozen source changed: {relative}")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", default="run", choices=("prepare", "run", "status"))
    args = parser.parse_args()
    entry = configured_entry()
    if args.action == "status":
        plan = read_json(ART / "plan.json")
        rows = [entry.panel_status(p, plan) for p in plan["panels"]]
        print({"panels": rows, "completed_pairs": sum(r["completed_pairs"] for r in rows), "planned_pairs": 20000}, flush=True)
        return 0
    RUNTIME.mkdir(parents=True, exist_ok=True)
    with (RUNTIME / "chain.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = entry.prepare()
        source = frozen_source(entry)
        if args.action == "prepare":
            entry.write_report(plan, [entry.panel_status(p, plan) for p in plan["panels"]])
            print({"status": "prepared_not_started", "fixed_step": STEP, "panels": 2, "pairs": 20000}, flush=True)
            return 0
        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, entry.interrupt)
        path = ART / "execution.json"
        execution = read_json(path) if path.exists() else {"sessions": []}
        record = {"command": [sys.executable, *sys.argv], "start_unix": time.time(), "status": "running",
            "source": entry.identity(ART / "source.json"), "environment": environment_identity()}
        execution["sessions"].append(record)
        write_json(path, execution)
        try:
            entry.run_sequence(plan, source)
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
            write_json(RUNTIME / "live.json", {"status": record["status"], "pid": os.getpid(), "error": record.get("error")})


if __name__ == "__main__":
    raise SystemExit(main())
