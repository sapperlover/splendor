"""Lambda161924 with prior best FPU or historical search vs historical L12,2000pairs."""
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
SHARED = ROOT / "artifacts/offline5_vs_r37_10k_20261027/run.py"
PAST = ROOT / "artifacts/search_fpu_cpuct_l12_20261104"
BEST_ID = "04_2b_fpu0.3_c1"
LAMBDA = ("runs/outcome_60k_scalar_lambda_20261120/checkpoints/step-000161924.pt",
          "561f055af56719ad7fbfa9779e93955ce139d5651b6a2b9b9dbc7e10b7182b3b")
PAIRS, GAME_SEED, POLICY_SEED, SEARCH_SEED, BOOTSTRAP_SEED = 2000, 55300000, 55310000, 55320000, 55330000


def configured_entry():
    spec = importlib.util.spec_from_file_location("lambda_l12_fpu_evaluation", SHARED)
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    entry.ROOT, entry.ART, entry.RUNTIME = ROOT, ART, RUNTIME
    entry.PROGRESS_EVERY, entry.SUMMARY_EVERY = 100, 500
    return entry


def prepare(entry):
    history = read_json(PAST / "summary.json")
    if history["status"] != "complete" or any(p["status"] != "complete" for p in history["panels"]):
        raise ValueError("FPU reference screening is not complete")
    winner = max(history["panels"], key=lambda p: p["score"])
    if winner["id"] != BEST_ID:
        raise ValueError("Historical best FPU screening identity changed")
    config_path = PAST / "configs" / f"{BEST_ID}.json"
    template = read_json(config_path)
    if template["policy_a"]["search_config"]["fpu_reduction"] != .3 or template["policy_a"]["search_config"]["c_puct"] != 1.0:
        raise ValueError("Expected the prior highest-score FPU0.3/c_puct1.0 recipe")
    past_plan = read_json(PAST / "plan.json")
    l12 = past_plan["checkpoint"]
    models = {}
    for key, path, expected in (("lambda", ROOT / LAMBDA[0], LAMBDA[1]),
                                ("l12", ROOT / l12["path"], l12["sha256"])):
        if sha256(path) != expected:
            raise ValueError(f"Fixed model changed: {key}")
        payload = load_alphazero_checkpoint(path)
        if key == "lambda" and payload.total_train_step != 161924:
            raise ValueError("Expected the user-fixed lambda161924 checkpoint")
        models[key] = {"checkpoint": str(path), "sha256": expected,
            "total_train_step": payload.total_train_step, "model_config": payload.model_config}
    catalog = load_default_catalog()
    if catalog.data_fingerprint != past_plan["catalog_fingerprint"]:
        raise ValueError("Current catalog differs from the original L12/FPU experiment")
    entry.freeze(ART / "reference_template.json", template)
    historical = copy.deepcopy(template["policy_b"]["search_config"])
    historical.update(c_puct=1.5, fpu_reduction=None, simulations=128, determinizations=1,
        seed=SEARCH_SEED, root_noise_fraction=0., forced_playouts=0., policy_target_pruning=False)
    panels = {}
    for key in ("fpu", "nofpu"):
        raw = copy.deepcopy(template)
        raw.pop("pairs", None)
        raw.pop("seed_start", None)
        raw.update(game_seeds=list(range(GAME_SEED, GAME_SEED+PAIRS)), policy_seed_base=POLICY_SEED,
            bootstrap_resamples=10000, bootstrap_seed=BOOTSTRAP_SEED)
        candidate_search = dict(historical)
        if key == "fpu":
            candidate_search.update(fpu_reduction=.3, c_puct=1.)
        raw["policy_a"].update(checkpoint=models["lambda"]["checkpoint"], name=f"lambda161924_{key}", search_config=candidate_search)
        raw["policy_b"].update(checkpoint=models["l12"]["checkpoint"], name="L12_historical_search", search_config=historical)
        raw["metadata"] = {"experiment": ART.name, "variant": key, "candidate_step":161924,
            "candidate_sha256":models["lambda"]["sha256"], "opponent_sha256":models["l12"]["sha256"],
            "catalog_fingerprint":catalog.data_fingerprint, "budget":"both S128N1=128perworld*1world=total128each",
            "candidate_search":candidate_search, "opponent_search":"L12 historical: FPU None,c_puct1.5",
            "selection":"User-fixed lambda161924; FPU recipe is prior screening point-estimate winner"}
        config = ART / "configs" / f"lambda161924_{key}_vs_L12_S128N1_2000pair.json"
        entry.freeze(config, FixedEvaluationConfig.from_dict(raw).to_dict())
        panels[key] = {"id":f"lambda161924_{key}_vs_L12_S128N1_2000pair", "kind":"search", "model":"lambda161924",
            "pairs":PAIRS, "config":entry.identity(config),
            "output":str((RUNTIME / "panels" / f"lambda161924_{key}_vs_L12.json").relative_to(ROOT))}
    plan = {"experiment":ART.name, "runner":entry.identity(Path(__file__)), "shared_runner":entry.identity(SHARED),
        "models":models, "panels":panels, "pairs_each":PAIRS, "games_each":PAIRS*2,
        "catalog_fingerprint":catalog.data_fingerprint,
        "prior_best_fpu":{"panel":BEST_ID, "score":winner["score"], "fpu_reduction":.3, "c_puct":1.,
            "summary":entry.identity(PAST / "summary.json"), "config":entry.identity(config_path), "original_result":winner["result"]},
        "no_fpu_search":{"fpu_reduction":None,"c_puct":1.5},
        "opponent_search":{"fpu_reduction":None,"c_puct":1.5},
        "seeds":{"game_seed_start":GAME_SEED,"game_seed_end":GAME_SEED+PAIRS-1,
            "policy_seed_base":POLICY_SEED,"search_seed":SEARCH_SEED,"bootstrap_seed":BOOTSTRAP_SEED},
        "scope":"Two independent fixed-lambda161924 vs L12 search panels,2000pair each,bothS128N1. Candidate either prior best FPU0.3/c_puct1.0 recipe or original noFPU/c_puct1.5; L12 always original search. Whole-recipe comparison,not FPU-only attribution. No training/greedy/queue/model reselection."}
    entry.freeze(ART / "plan.json", plan)
    return plan


def frozen_source(entry):
    destination, record_path = RUNTIME / "source", ART / "source.json"
    if not record_path.exists():
        record = source_identity(ROOT, destination)
        for path in [Path(__file__), ART / "validate.py", SHARED, ROOT / "src/splendor/native_search.cpp",
            *ROOT.glob("src/splendor/_native_search*.so"), *(ART / "configs").glob("*.json"),
            *ROOT.glob("scripts/run_lambda161924_*_vs_l12_s128n1_2000pair_20261123.sh"), *(ROOT / "data").glob("splendor_*")]:
            target = destination / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path,target)
        record["snapshot_files"] = {str(p.relative_to(destination)):sha256(p) for p in sorted(destination.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts}
        entry.freeze(record_path,record)
    for path, expected in read_json(record_path)["snapshot_files"].items():
        if sha256(destination / path) != expected:
            raise ValueError(f"Frozen execution changed: {path}")
    return destination


def report(key,row):
    write_json(ART / f"summary_{key}.json",row)
    score = f"{row['score']*100:.3f}%" if row["status"]=="complete" else "—"
    ci = f"[{row['ci95']['lower']*100:.3f}%,{row['ci95']['upper']*100:.3f}%]" if row["status"]=="complete" else "—"
    (ART / f"results_{key}.md").write_text(f"# λ161924 {key} vs L12\n\n双方S128N1（各总128），2000pair／4000局。\n\n"
        f"状态{row['status']}；完成{row['completed_pairs']}/2000pair；得分率{score}；95%整pair区间{ci}。\n\n"
        "FPU组为r=0.3/c_puct=1.0；无FPU组为None/c_puct=1.5；L12始终None/c_puct=1.5。\n",encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant",choices=("fpu","nofpu"))
    parser.add_argument("action",nargs="?",default="run",choices=("prepare","run","status"))
    args=parser.parse_args()
    entry=configured_entry()
    if args.action=="status":
        plan=read_json(ART / "plan.json")
        print(entry.panel_status(plan["panels"][args.variant],plan),flush=True)
        return 0
    RUNTIME.mkdir(parents=True,exist_ok=True)
    with (RUNTIME / "prepare.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        plan=prepare(entry)
        source=frozen_source(entry)
    panel=plan["panels"][args.variant]
    with (RUNTIME / f"{args.variant}.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        row=entry.panel_status(panel,plan)
        if args.action=="prepare" or row["status"]=="complete":
            report(args.variant,row)
            print({"status":row["status"],"variant":args.variant,"pairs":PAIRS},flush=True)
            return 0
        for signum in (signal.SIGINT,signal.SIGTERM):signal.signal(signum,entry.interrupt)
        path=ART / f"execution_{args.variant}.json"
        execution=read_json(path) if path.exists() else {"sessions":[]}
        record={"command":[sys.executable,*sys.argv],"start_unix":time.time(),"status":"running",
                "source":entry.identity(ART / "source.json"),"environment":environment_identity()}
        execution["sessions"].append(record)
        write_json(path,execution)
        try:
            entry.run_panel(panel,plan,source)
            row=entry.panel_status(panel,plan)
            if row["status"]!="complete":raise ValueError("Evaluation did not complete2000pairs")
            report(args.variant,row)
            record["status"]="complete"
            return 0
        except KeyboardInterrupt:
            record["status"]="interrupted"
            return 130
        except Exception as error:
            record.update(status="failed",error=f"{type(error).__name__}: {error}")
            raise
        finally:
            record["end_unix"]=time.time()
            write_json(path,execution)
            write_json(RUNTIME / "live.json",{"status":record["status"],"variant":args.variant})


if __name__=="__main__":raise SystemExit(main())
