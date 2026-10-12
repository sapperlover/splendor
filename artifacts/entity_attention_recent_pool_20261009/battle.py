"""Final offline student versus theta5000: one matched 200-pair S1024/N1 panel."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from splendor.alphazero_recording import read_json, sha256, source_identity, write_json
from splendor.evaluation import FixedEvaluationConfig

ROOT = Path(__file__).resolve().parents[2]
ART = Path(__file__).resolve().parent
RUNTIME = ROOT / "runtime/entity_attention_recent_pool_20261009/battle"
TEMPLATE = ROOT / "artifacts/entity_attention_hybrid_offline9500_20261009/battles/configs/offline9500_vs_theta5000.json"
CONFIG = ART / "battle_config.json"
OUTPUT = ART / "battles/final103194_vs_theta5000.json"
CANDIDATE = ROOT / "runs/entity_attention_recent_pool_20261009/checkpoints/final.pt"
PREVIOUS = ROOT / "artifacts/entity_attention_hybrid_offline9500_20261009/battles"


def prepare():
    if read_json(ART / "progress.json")["status"] != "complete":
        raise ValueError("offline training is not complete")
    config = read_json(TEMPLATE)
    config["policy_a"]["checkpoint"] = str(CANDIDATE)
    config["policy_a"]["name"] = "recent_pool_final103194"
    config["metadata"] = {"experiment":"entity_attention_recent_pool_20261009",
        "panel":"final103194_vs_theta5000", "checkpoint_selection":"predeclared final training step, not minimum heldout MSE",
        "candidate_step":103194, "candidate_sha256":sha256(CANDIDATE),
        "template_config":str(TEMPLATE.relative_to(ROOT)), "template_sha256":sha256(TEMPLATE),
        "seeds":"shared DEV panel: game 14900000+, policy 14910000+, search 14930000+"}
    FixedEvaluationConfig.from_dict(config)
    if CONFIG.exists() and read_json(CONFIG) != config:
        raise ValueError("frozen battle config changed")
    write_json(CONFIG, config)
    OUTPUT.parent.mkdir(exist_ok=True)
    return config


def scores(data):
    if not data["passed"] or data["completed_pairs"] != 200:
        raise ValueError("battle panel incomplete or invalid")
    result = {}
    for pair in data["pairs"]:
        if len(pair["games"]) != 2:
            raise ValueError("paired seed does not have both seats")
        result[int(pair["game_seed"])] = sum({"a_win":1.,"draw":.5,"a_loss":0.}[g["outcome_for_a"]] for g in pair["games"]) / 2
    if len(result) != 200:
        raise ValueError("duplicate or missing paired seeds")
    return result


def analyze():
    data = read_json(OUTPUT)
    current = scores(data)
    result = {"protocol":"recent-pool-final-vs-theta5000-v1", "status":"complete", "candidate_step":103194,
        "pairs":200,"games":400,"score":data["win_rate_score"],"ci95":data["paired_bootstrap"],
        "a_wins":data["a_wins"],"draws":data["draws"],"a_losses":data["a_losses"],
        "a_first":data["a_first"],"a_second":data["a_second"],
        "engineering_truncations":data["engineering_truncations"],"exceptions":data["exceptions"],
        "policy_cycle_draws":data["policy_cycle_draws"],"output":str(OUTPUT.relative_to(ROOT)),
        "output_sha256":sha256(OUTPUT), "matched_contrasts":{},
        "scope":"one final checkpoint, one opponent and search budget, reused DEV seeds; not a checkpoint selection sweep or independent FINAL"}
    for i,name in enumerate(("offline9500_vs_theta5000","hybrid_r3_vs_theta5000","hybrid_r6_vs_theta5000","hybrid_r9_vs_theta5000")):
        path=PREVIOUS/f"{name}.json"
        previous=scores(read_json(path))
        if current.keys() != previous.keys():
            raise ValueError("old and new panels have different paired seeds")
        delta=np.asarray([current[k]-previous[k] for k in sorted(current)],dtype=np.float64)
        rng=np.random.default_rng(30200000+i)
        boot=delta[rng.integers(0,200,size=(10000,200))].mean(-1)
        result["matched_contrasts"][f"new_minus_{name}"]={"pairs":200,"estimate":float(delta.mean()),
            "lower":float(np.percentile(boot,2.5)),"upper":float(np.percentile(boot,97.5)),
            "seed":30200000+i,"resamples":10000,"source":str(path.relative_to(ROOT)),"source_sha256":sha256(path)}
    write_json(ART/"battle_summary.json",result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


def run():
    prepare()
    session=RUNTIME/str(time.time_ns())
    identity=source_identity(ROOT,session)
    for name in ("battle.py","battle_config.json"):
        import shutil
        shutil.copyfile(ART/name,session/name)
    command=[sys.executable,"-u","-m","splendor.evaluation","--config",str(CONFIG),
        "--output",str(OUTPUT),"--progress-every","10","--summary-every","10"]
    if OUTPUT.exists() or OUTPUT.with_suffix(OUTPUT.suffix+".games").exists():
        command.append("--resume")
    write_json(session/"identity.json",{**identity,"command":command,"config_sha256":sha256(CONFIG),
        "candidate_sha256":sha256(CANDIDATE),"runner_sha256":sha256(Path(__file__)),"start_unix":time.time()})
    subprocess.run(command,cwd=ROOT,check=True)
    analyze()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("prepare","run","analyze"),nargs="?",default="run")
    args=parser.parse_args()
    if args.action=="prepare":
        prepare()
    elif args.action=="run":
        run()
    else:
        analyze()
