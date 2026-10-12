"""One-command preparation, resumed training and publication of learning curves."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

from pool import HERE, prepare
from splendor.alphazero_recording import read_json, sha256, write_json


def config_for(summary):
    return {"protocol":"recent-attention-fixed-pool-offline-v1",
        "manifest":str((HERE/"pool_games.jsonl").resolve()),"manifest_sha256":summary["manifest_sha256"],
        "output":"runs/entity_attention_recent_pool_20261009",
        "artifacts":"artifacts/entity_attention_recent_pool_20261009",
        "model_config":{"architecture":"entity_attention_v1"},
        "initialization_seed":13100000,"sampling_seed":30110000,"validation_seed":30120000,
        "positions_per_source_split":2048,"cache_mib":8192,
        "draw_exposures":12,"lr_drop_exposure":8,"late_lr_multiplier":.25,
        "full_holdout_exposures":[4,8,12],
        "baseline_checkpoints":{
            "hybrid9500":"runs/entity_attention_hybrid_static_graph_seed1/rounds/round-008/checkpoints/final.pt",
            "latest17700":"runs/entity_attention_avg4_behavior_20261009/seed1_avg4/rounds/round-011/checkpoints/final.pt"},
        "learner":{"device":"cuda","batch_size":256,"history_batch_size":128,
            "learning_rate":1e-4,"weight_decay":1e-4,"policy_loss_weight":1.,"value_loss_weight":1.,
            "max_grad_norm":1.,"steps_per_new_position":4.,"checkpoint_every_steps":1000,
            "validation_batch_size":256,"max_steps_per_round":None}}


def analyze():
    path = HERE/"curve.jsonl"
    if not path.exists():
        return
    curve = [json.loads(line) for line in path.read_text().splitlines()]
    data = [{"step":p["step"],"draw_exposure":p["draw_exposure"],"learning_rate":p["learning_rate"],
        "train":p["evaluation"]["pool_weighted"]["train"],
        "validation":p["evaluation"]["pool_weighted"]["validation"]} for p in curve]
    summary = {"status":read_json(HERE/"progress.json")["status"],"curve":data,
        "best_validation_kl":min(data,key=lambda p:p["validation"]["policy_kl"]),
        "best_validation_mse":min(data,key=lambda p:p["validation"]["value_mse"]),
        "scope":"fixed stored behavior-search labels; DEV fitting diagnostics, strength unmeasured",
        "full_holdouts":[{"path":str(p),"sha256":sha256(p),"step":read_json(p)["step"],
            "metrics":read_json(p)["evaluation"]["pool_weighted"]["validation"]}
            for p in sorted(HERE.glob("full_holdout_step_*.json"))]}
    write_json(HERE/"training_summary.json",summary)
    # matplotlib is installed in the inspected base environment. The training
    # environment stays unchanged; plotting runs only after a training session.
    subprocess.run(["conda","run","--no-capture-output","-n","base","env",
        "MPLCONFIGDIR=/tmp/splendor_recent_pool_matplotlib","python",str(HERE/"plot.py")],check=True)
    print(json.dumps({"status":summary["status"],"last":data[-1],"best_validation_kl":summary["best_validation_kl"]},ensure_ascii=False,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("prepare","run","status","analyze"),default="run",nargs="?")
    parser.add_argument("--max-steps-this-session",type=int)
    args=parser.parse_args()
    if args.action in ("status","analyze"):
        if args.action=="analyze":
            analyze()
        if (HERE/"progress.json").exists():
            p=read_json(HERE/"progress.json")
            print(json.dumps({k:p[k] for k in ("status","step","steps_target","train_positions")},ensure_ascii=False))
        live=Path("runs/entity_attention_recent_pool_20261009/live.json")
        if live.exists():
            print(json.dumps({"live":read_json(live)},ensure_ascii=False))
        return
    summary=prepare()
    config=config_for(summary)
    config_path=HERE/"config.json"
    if config_path.exists() and read_json(config_path)!=config:
        raise ValueError("frozen experiment config changed")
    write_json(config_path,config)
    if args.action=="prepare":
        print(json.dumps({"games":summary["totals"]["games"],"train_positions":summary["totals"]["train"]["positions"],
            "steps":math.ceil(12*summary["totals"]["train"]["positions"]/256)},ensure_ascii=False))
        return
    command=[sys.executable,"-u",str(HERE/"train.py"),"--config",str(config_path)]
    if args.max_steps_this_session is not None:
        command.extend(("--max-steps-this-session",str(args.max_steps_this_session)))
    subprocess.run(command,check=True)
    analyze()


if __name__=="__main__":
    main()
