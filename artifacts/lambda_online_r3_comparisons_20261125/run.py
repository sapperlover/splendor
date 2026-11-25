"""Fixed round3-used checkpoint vs lambda161924/round1-used/round3-step3000; all FPU."""
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
BRANCH = ROOT / "runs/lambda161924_fpu_p60k_iteration_20261124"
SHARED = ROOT / "artifacts/offline5_vs_r37_10k_20261027/run.py"
TEMPLATE = BRANCH / "rounds/round-002/search_config.json"
OPPONENTS = ("lambda", "r1", "r3_step3000")
FIXED = {
    "r3_used": ("runs/lambda161924_fpu_p60k_iteration_20261124/rounds/round-002/checkpoints/final.pt",
        "8b6798463c23feb1745faaff0364b9de7a92e27ba825d507063355309e6c1667", 10000, 191924),
    "lambda": ("runs/outcome_60k_scalar_lambda_20261120/checkpoints/step-000161924.pt",
        "561f055af56719ad7fbfa9779e93955ce139d5651b6a2b9b9dbc7e10b7182b3b", 161924, 161924),
    "r1": ("runs/lambda161924_fpu_p60k_iteration_20261124/rounds/round-000/checkpoints/final.pt",
        "79e601e2084a0b58020db82e66b4ed3271efd1c4f881757b7280a05b43de1e42", 10000, 171924),
    "r3_step3000": ("runs/lambda161924_fpu_p60k_iteration_20261124/rounds/round-002/checkpoints/step-000003000.pt",
        "68396a6caa8b6df1d1189e0b025503d777083098f95e1009afca146f9845e077", 3000, 184924),
}
PAIRS, GAME_SEED, POLICY_SEED, SEARCH_SEED, BOOTSTRAP_SEED = 2000, 55500000, 55510000, 55520000, 55530000


def configured_entry():
    spec = importlib.util.spec_from_file_location("lambda_r3_evaluation", SHARED)
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    entry.ROOT, entry.ART, entry.RUNTIME = ROOT, ART, RUNTIME
    entry.PROGRESS_EVERY, entry.SUMMARY_EVERY = 100, 500
    return entry


def prepare(entry):
    state = read_json(BRANCH / "state.json")
    rounds = {row["iteration"]: row for row in state["completed_rounds"]}
    models, actual_use, payloads = {}, {}, {}
    for key, (relative, expected, local_step, total_step) in FIXED.items():
        path = ROOT / relative
        if sha256(path) != expected:
            raise ValueError(f"Fixed checkpoint changed: {key}")
        payload = load_alphazero_checkpoint(path)
        if payload.round_train_step != local_step or payload.total_train_step != total_step:
            raise ValueError(f"Checkpoint local/cumulative step differs: {key}")
        payloads[key] = payload
        models[key] = {"checkpoint":str(path), "sha256":expected, "local_train_step":local_step,
            "total_train_step":total_step, "round_index":payload.round_index,
            "model_config":payload.model_config, "replay_digest":payload.replay_identity["digest"]}
    for key, iteration, next_stage in (("r1",1,"round-001"),("r3_used",3,"round-003")):
        expected = {"path":models[key]["checkpoint"],"sha256":models[key]["sha256"]}
        manifest_path = BRANCH / "rounds" / next_stage / "selfplay/manifest.json"
        manifest = read_json(manifest_path)
        if rounds[iteration]["checkpoint"] != expected or manifest["checkpoint_identity"] != expected:
            raise ValueError(f"The requested actually-used checkpoint is not the committed next-round teacher: {key}")
        if rounds[iteration]["evaluations"]["greedy"]["checkpoint_identity"]["policy_a"] != expected:
            raise ValueError(f"Completed-round evaluation used a different checkpoint: {key}")
        actual_use[key] = {"iteration":iteration,"committed_checkpoint":expected,
            "state_path":str((BRANCH / "state.json").relative_to(ROOT)),
            "next_collection_manifest":str(manifest_path.relative_to(ROOT)),"next_collection_teacher":expected}
    if (payloads["r3_used"].round_index != payloads["r3_step3000"].round_index
            or payloads["r3_used"].replay_identity != payloads["r3_step3000"].replay_identity
            or payloads["r3_used"].parent_checkpoint != payloads["r3_step3000"].parent_checkpoint):
        raise ValueError("Round3 final and3000-step checkpoints are not from the same training round")
    template = read_json(TEMPLATE)
    fingerprint = load_default_catalog().data_fingerprint
    manifest_fingerprint = read_json(BRANCH / "rounds/round-003/selfplay/manifest.json")["catalog_fingerprint"]
    if fingerprint != manifest_fingerprint:
        raise ValueError("Current catalog differs from the actual online branch")
    entry.freeze(ART / "reference_template.json",template)
    search = copy.deepcopy(template["policy_a"]["search_config"])
    search.update(simulations=128,determinizations=1,c_puct=1.,fpu_reduction=.3,seed=SEARCH_SEED,
                  root_noise_fraction=0.,forced_playouts=0.,policy_target_pruning=False)
    panels = {}
    for key in OPPONENTS:
        raw = copy.deepcopy(template)
        raw.pop("pairs",None); raw.pop("seed_start",None)
        raw.update(game_seeds=list(range(GAME_SEED,GAME_SEED+PAIRS)),policy_seed_base=POLICY_SEED,
                   bootstrap_resamples=10000,bootstrap_seed=BOOTSTRAP_SEED)
        for side, model_key in (("policy_a","r3_used"),("policy_b",key)):
            raw[side].update(checkpoint=models[model_key]["checkpoint"],name=model_key,search_config=copy.deepcopy(search))
        raw["metadata"] = {"experiment":ART.name,"opponent":key,
            "candidate_sha256":models["r3_used"]["sha256"],"opponent_sha256":models[key]["sha256"],
            "catalog_fingerprint":fingerprint,"budget":"bothS128N1=128perworld*1world=total128each",
            "search_recipe":"both sides FPU0.3/c_puct1.0,no noise,T0,FP/PTPoff",
            "selection":"Fixed completed-round3 actually-used final; never follow later rounds or reselect"}
        config = ART / "configs" / f"r3_used_vs_{key}_FPU_S128N1_2000pair.json"
        entry.freeze(config,FixedEvaluationConfig.from_dict(raw).to_dict())
        panels[key] = {"id":f"r3_used_vs_{key}_FPU_S128N1_2000pair","kind":"search","model":"r3_used",
            "pairs":PAIRS,"config":entry.identity(config),
            "output":str((RUNTIME / "panels" / f"r3_used_vs_{key}.json").relative_to(ROOT))}
    plan = {"experiment":ART.name,"runner":entry.identity(Path(__file__)),"shared_runner":entry.identity(SHARED),
        "models":models,"actual_use_evidence":actual_use,"panels":panels,"pairs_each":PAIRS,"games_each":PAIRS*2,
        "catalog_fingerprint":fingerprint,"template":entry.identity(TEMPLATE),
        "search_recipe":{"fpu_reduction":.3,"c_puct":1.,"simulations":128,"determinizations":1,"applies_to":"both sides of all three panels"},
        "seeds":{"game_seed_start":GAME_SEED,"game_seed_end":GAME_SEED+PAIRS-1,
            "policy_seed_base":POLICY_SEED,"search_seed":SEARCH_SEED,"bootstrap_seed":BOOTSTRAP_SEED},
        "scope":"Three independent search evaluations. Fixed round3-used final(local10000,total191924) vs lambda161924,round1-used final(local10000,total171924),or same-round3 local3000(total184924). All sidesFPU0.3/c_puct1.0,S128N1total128,2000pair each. No training,automatic model selection,latest-round following or queue."}
    entry.freeze(ART / "plan.json",plan)
    return plan


def frozen_source(entry):
    destination, record_path = RUNTIME / "source", ART / "source.json"
    if not record_path.exists():
        record = source_identity(ROOT,destination)
        for path in [Path(__file__),ART / "validate.py",SHARED,ROOT / "src/splendor/native_search.cpp",
            *ROOT.glob("src/splendor/_native_search*.so"),*(ART / "configs").glob("*.json"),
            *ROOT.glob("scripts/run_lambda_r3_used_vs_*_fpu_s128n1_2000pair_20261125.sh"),*(ROOT / "data").glob("splendor_*")]:
            target = destination / path.relative_to(ROOT)
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
        record["snapshot_files"] = {str(p.relative_to(destination)):sha256(p) for p in sorted(destination.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts}
        entry.freeze(record_path,record)
    for path, expected in read_json(record_path)["snapshot_files"].items():
        if sha256(destination / path) != expected:raise ValueError(f"Frozen execution changed: {path}")
    return destination


def report(key,row):
    write_json(ART / f"summary_{key}.json",row)
    score = f"{100*row['score']:.3f}%" if row["status"]=="complete" else "—"
    ci = f"[{100*row['ci95']['lower']:.3f}%,{100*row['ci95']['upper']:.3f}%]" if row["status"]=="complete" else "—"
    (ART / f"results_{key}.md").write_text(f"# 第三轮实际使用final vs {key}\n\n双方FPU0.3／c_puct1.0，S128N1（各总128），2000pair／4000局。\n\n"
        f"状态{row['status']}；完成{row['completed_pairs']}/2000pair；得分率{score}；95%整pair区间{ci}。\n",encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("opponent",choices=OPPONENTS)
    parser.add_argument("action",nargs="?",default="run",choices=("prepare","run","status"))
    args=parser.parse_args();entry=configured_entry()
    if args.action=="status":
        plan=read_json(ART / "plan.json")
        print(entry.panel_status(plan["panels"][args.opponent],plan),flush=True);return 0
    RUNTIME.mkdir(parents=True,exist_ok=True)
    with (RUNTIME / "prepare.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);plan=prepare(entry);source=frozen_source(entry)
    panel=plan["panels"][args.opponent]
    with (RUNTIME / f"{args.opponent}.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);row=entry.panel_status(panel,plan)
        if args.action=="prepare" or row["status"]=="complete":
            report(args.opponent,row);print({"status":row["status"],"opponent":args.opponent,"pairs":PAIRS},flush=True);return 0
        for signum in (signal.SIGINT,signal.SIGTERM):signal.signal(signum,entry.interrupt)
        path=ART / f"execution_{args.opponent}.json"
        execution=read_json(path) if path.exists() else {"sessions":[]}
        record={"command":[sys.executable,*sys.argv],"start_unix":time.time(),"status":"running",
                "source":entry.identity(ART / "source.json"),"environment":environment_identity()}
        execution["sessions"].append(record);write_json(path,execution)
        try:
            entry.run_panel(panel,plan,source);row=entry.panel_status(panel,plan)
            if row["status"]!="complete":raise ValueError("Evaluation did not complete2000pairs")
            report(args.opponent,row);record["status"]="complete";return 0
        except KeyboardInterrupt:
            record["status"]="interrupted";return 130
        except Exception as error:
            record.update(status="failed",error=f"{type(error).__name__}: {error}");raise
        finally:
            record["end_unix"]=time.time();write_json(path,execution)
            write_json(RUNTIME / "live.json",{"status":record["status"],"opponent":args.opponent})


if __name__=="__main__":raise SystemExit(main())
