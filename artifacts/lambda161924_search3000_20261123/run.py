"""Fixed scalar-lambda161924 vs same-poolMC70842 or rawR37, bothS128N1/3000pairs."""
from __future__ import annotations

import argparse
import copy
import fcntl
import importlib.util
from pathlib import Path
import shutil
import signal
import sys
import time

from splendor.alphazero_checkpoint import load_alphazero_checkpoint
from splendor.alphazero_recording import environment_identity, read_json, sha256, source_identity, write_json
from splendor.catalog import load_default_catalog
from splendor.evaluation import FixedEvaluationConfig

ROOT = Path(__file__).resolve().parents[2]
ART = Path(__file__).resolve().parent
RUNTIME = ROOT / "runtime" / ART.name
TRAIN_ART = ROOT / "artifacts/value_outcome_60k_20261120"
SHARED = ROOT / "artifacts/offline5_vs_r37_10k_20261027/run.py"
FIXED = {
    "lambda": ("scalar_lambda", 161924, "561f055af56719ad7fbfa9779e93955ce139d5651b6a2b9b9dbc7e10b7182b3b"),
    "mc": ("scalar_mc", 70842, "37cbd0bf0082e530bf303fe56e027b99af63b9f9a7afea15ccf4d236a9c69cec"),
}
PAIRS, SEED_START, POLICY_SEED, SEARCH_SEED, BOOTSTRAP_SEED = 3000, 55230000, 55240000, 55250000, 55260000


def configured_entry():
    spec = importlib.util.spec_from_file_location("lambda3000_evaluation", SHARED)
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    entry.ROOT, entry.ART, entry.RUNTIME = ROOT, ART, RUNTIME
    entry.PROGRESS_EVERY, entry.SUMMARY_EVERY = 100, 500
    return entry


def training_model(key, pool):
    arm, step, expected = FIXED[key]
    run = ROOT / "runs" / f"outcome_60k_{arm}_20261120"
    path, validation_path = run / "checkpoints" / f"step-{step:09d}.pt", run / "full_validation" / f"step-{step:09d}.json"
    config, validation = read_json(run / "config.json"), read_json(validation_path)
    if config["outcome_objective"] != arm or config["manifest_sha256"] != pool["manifest_sha256"]:
        raise ValueError("Wrong scalar outcome objective or same-pool identity")
    if key == "lambda" and (config["outcome_lambda_source"] != "archive_search" or config["outcome_lambda"] != .8):
        raise ValueError("Expected the current archived-search lambda0.8 branch")
    if sha256(path) != expected or validation["checkpoint_sha256"] != expected or validation["step"] != step:
        raise ValueError(f"User-fixed checkpoint or validation changed: {key}")
    payload = load_alphazero_checkpoint(path)
    if payload.total_train_step != step or payload.model_config["architecture"] != "entity_attention_v1":
        raise ValueError("Expected a fixed-step scalar attention checkpoint")
    return {"key": key, "checkpoint": str(path), "sha256": expected, "selected_step": step,
        "model_config": payload.model_config, "manifest_sha256": config["manifest_sha256"],
        "dataset_digest": validation["dataset_digest"], "outcome_objective": arm,
        "validation": {"path": str(validation_path.relative_to(ROOT)), "sha256": sha256(validation_path)},
        "policy_kl": validation["metrics"]["policy_kl"], "value_mse": validation["metrics"]["value_loss"]}


def prepare(entry):
    pool = read_json(TRAIN_ART / "pool_identity.json")
    if pool["games"] != 60000:
        raise ValueError("Expected the same recent60k pool")
    models = {key: training_model(key, pool) for key in FIXED}
    if models["lambda"]["dataset_digest"] != models["mc"]["dataset_digest"]:
        raise ValueError("Lambda and MC checkpoint validation datasets differ")
    opponent = read_json(TRAIN_ART / "opponent.json")
    if sha256(Path(opponent["path"])) != opponent["sha256"]:
        raise ValueError("Raw R37 changed")
    r37 = load_alphazero_checkpoint(opponent["path"])
    models["r37"] = {"key": "r37", "checkpoint": opponent["path"], "sha256": opponent["sha256"],
                     "model_config": r37.model_config}
    fingerprint = load_default_catalog().data_fingerprint
    template = read_json(TRAIN_ART / "evaluation_templates.json")["search_S128N1"]
    entry.freeze(ART / "template.json", template)
    panels = {}
    for key in ("mc", "r37"):
        raw = copy.deepcopy(template)
        raw.pop("pairs", None)
        raw.pop("seed_start", None)
        raw.update(game_seeds=list(range(SEED_START, SEED_START+PAIRS)), policy_seed_base=POLICY_SEED,
                   bootstrap_resamples=10000, bootstrap_seed=BOOTSTRAP_SEED)
        for side, model in (("policy_a", models["lambda"]), ("policy_b", models[key])):
            raw[side].update(checkpoint=model["checkpoint"], name=f"{model['key']}_{model.get('selected_step','original')}")
            raw[side]["search_config"].update(simulations=128, determinizations=1, seed=SEARCH_SEED,
                root_noise_fraction=0., forced_playouts=0., policy_target_pruning=False)
        raw["metadata"] = {"experiment": ART.name, "opponent": key,
            "candidate_sha256": models["lambda"]["sha256"], "opponent_sha256": models[key]["sha256"],
            "candidate_step": 161924, "opponent_step": models[key].get("selected_step"),
            "catalog_fingerprint": fingerprint, "budget": "both S128N1:128 per world*1world=total128each",
            "selection": "User-fixed lambda161924 and same-poolMC70842; never follow later checkpoints"}
        config = ART / "configs" / f"lambda161924_vs_{key}_S128N1_3000pair.json"
        entry.freeze(config, FixedEvaluationConfig.from_dict(raw).to_dict())
        panels[key] = {"id": f"lambda161924_vs_{key}_S128N1_3000pair", "kind": "search", "model": "lambda161924",
            "pairs": PAIRS, "config": entry.identity(config),
            "output": str((RUNTIME / "panels" / f"lambda161924_vs_{key}.json").relative_to(ROOT))}
    plan = {"experiment": ART.name, "runner": entry.identity(Path(__file__)), "shared_runner": entry.identity(SHARED),
        "models": models, "pool_identity": entry.identity(TRAIN_ART / "pool_identity.json"),
        "catalog_fingerprint": fingerprint, "panels": panels, "pairs_each": PAIRS, "games_each": 2*PAIRS,
        "seeds": {"game_seed_start": SEED_START, "game_seed_end": SEED_START+PAIRS-1,
            "policy_seed_base": POLICY_SEED, "search_seed": SEARCH_SEED, "bootstrap_seed": BOOTSTRAP_SEED},
        "scope": "Two independent fixed-checkpoint both-S128N1 searches. Lambda161924 vs same60k-poolMC70842 or rawR37. No training, greedy, automatic model selection or queue. Full-pair resume."}
    entry.freeze(ART / "plan.json", plan)
    return plan


def frozen_source(entry):
    destination, record_path = RUNTIME / "source", ART / "source.json"
    if not record_path.exists():
        record = source_identity(ROOT, destination)
        for path in [Path(__file__), ART / "validate.py", SHARED, ROOT / "src/splendor/native_search.cpp",
                *ROOT.glob("src/splendor/_native_search*.so"), *(ART / "configs").glob("*.json"),
                *ROOT.glob("scripts/run_lambda161924_*_s128n1_3000pair_20261123.sh"),
                *(ROOT / "data").glob("splendor_*")]:
            target = destination / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        record["snapshot_files"] = {str(p.relative_to(destination)): sha256(p) for p in sorted(destination.rglob("*"))
                                   if p.is_file() and "__pycache__" not in p.parts}
        entry.freeze(record_path, record)
    for relative, expected in read_json(record_path)["snapshot_files"].items():
        if sha256(destination / relative) != expected:
            raise ValueError(f"Frozen execution changed: {relative}")
    return destination


def report(key, row):
    write_json(ART / f"summary_{key}.json", row)
    score = f"{100*row['score']:.3f}%" if row["status"] == "complete" else "—"
    ci = f"[{100*row['ci95']['lower']:.3f}%,{100*row['ci95']['upper']:.3f}%]" if row["status"] == "complete" else "—"
    (ART / f"results_{key}.md").write_text(f"# λ161924 vs {'同池MC70842' if key=='mc' else '原R37'}\n\n"
        f"双方S128N1（每世界128×1世界，总128），计划3000pair／6000局。\n\n"
        f"状态：{row['status']}；完成{row['completed_pairs']}/3000pair；得分率{score}；95%整pair区间{ci}。\n\n"
        "未完成结果不作正式棋力结论。\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("opponent", choices=("mc", "r37"))
    parser.add_argument("action", nargs="?", default="run", choices=("prepare", "run", "status"))
    args = parser.parse_args()
    entry = configured_entry()
    if args.action == "status":
        plan = read_json(ART / "plan.json")
        print(entry.panel_status(plan["panels"][args.opponent], plan), flush=True)
        return 0
    RUNTIME.mkdir(parents=True, exist_ok=True)
    with (RUNTIME / "prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        plan = prepare(entry)
        source = frozen_source(entry)
    panel = plan["panels"][args.opponent]
    with (RUNTIME / f"{args.opponent}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        row = entry.panel_status(panel, plan)
        if args.action == "prepare" or row["status"] == "complete":
            report(args.opponent, row)
            print({"status": row["status"], "opponent": args.opponent, "pairs": PAIRS}, flush=True)
            return 0
        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, entry.interrupt)
        record_path = ART / f"execution_{args.opponent}.json"
        execution = read_json(record_path) if record_path.exists() else {"sessions": []}
        record = {"command": [sys.executable, *sys.argv], "start_unix": time.time(), "status": "running",
            "environment": environment_identity(), "source": entry.identity(ART / "source.json")}
        execution["sessions"].append(record)
        write_json(record_path, execution)
        try:
            entry.run_panel(panel, plan, source)
            row = entry.panel_status(panel, plan)
            if row["status"] != "complete":
                raise ValueError("Evaluation did not complete all3000pairs")
            report(args.opponent, row)
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
            write_json(record_path, execution)
            write_json(RUNTIME / "live.json", {"status": record["status"], "opponent": args.opponent})


if __name__ == "__main__":
    raise SystemExit(main())
