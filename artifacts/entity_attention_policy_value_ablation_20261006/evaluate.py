"""Run one declared panel, optionally composing policy with the fixed initial value."""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from splendor.alphazero_recording import environment_identity, sha256, source_identity, write_json
from splendor.evaluation import _evaluation_config_from_file, run_fixed_evaluation
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.offline_train import make_index
from splendor.selfplay_data import collate_selfplay_samples
from splendor.search_inference import FrozenSearchInference, InferenceConfig
from core import FixedValueModel
from train import check_inputs, samples_from_refs

ROOT=Path(__file__).resolve().parents[2]
ART=Path(__file__).resolve().parent


def validate_composition(model,policy,value,batch):
    with torch.inference_mode():
        logits,values=model(batch.decisions)
        expected_logits,_=policy(batch.decisions)
        _,expected_values=value(batch.decisions)
    torch.testing.assert_close(logits,expected_logits,rtol=0,atol=0)
    torch.testing.assert_close(values,expected_values,rtol=0,atol=0)
    return {"policy_max_abs_difference":0.,"value_max_abs_difference":0.}


def install_fixed_value(panel,spec,config):
    import splendor.evaluation_batch as batch_module
    policy_path=Path(config.policy_a.checkpoint).resolve()
    value_path=(ROOT/spec["initial"]["path"]).resolve()
    policy=load_frozen_actor_critic_checkpoint(policy_path,device="cuda")
    value=load_frozen_actor_critic_checkpoint(value_path,device="cuda")
    composed=FixedValueModel(policy,value).to("cuda")
    index=make_index({"games":spec["games"]},256*1024**2)
    refs=np.load(ROOT/spec["batch_references"]["path"])[0,:16]
    batch=collate_selfplay_samples(samples_from_refs(index,refs))
    checks=validate_composition(composed,policy,value,batch)
    original_loader=batch_module.load_frozen_actor_critic_checkpoint
    def loader(path,*,device):
        if Path(path).resolve()==policy_path:
            assert device=="cuda"
            return composed
        return original_loader(path,device=device)
    batch_module.load_frozen_actor_critic_checkpoint=loader
    return checks | {"policy_checkpoint_sha256":sha256(policy_path),
                     "value_checkpoint_sha256":sha256(value_path),"adapter_sha256":sha256(ART/"core.py")}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--case",required=True)
    parser.add_argument("--smoke-fixed-value",action="store_true")
    args=parser.parse_args()
    spec=json.loads((ART/"specification.json").read_text())
    panel=next(p for p in spec["panels"] if p["id"]==args.case)
    config_path=ART/panel["config"]
    config,raw=_evaluation_config_from_file(config_path)
    torch.set_num_threads(1)
    check_inputs(spec)
    if args.smoke_fixed_value:
        assert panel["fixed_value"]
        from splendor.evaluation import FixedEvaluationConfig
        raw["pairs"]=1
        for key in ("policy_a","policy_b"):
            raw[key]["search_config"]["simulations"]=8
        config=FixedEvaluationConfig.from_dict(raw)
        name="fixed_value_smoke"
    else:
        name=panel["id"]
    runtime=ROOT/"runtime/entity_attention_policy_value_ablation_20261006"/name
    output=ART/"evaluations"/f"{name}.json"
    session=str(time.time_ns())
    source=source_identity(ROOT,runtime/"source"/session)
    composition=install_fixed_value(panel,spec,config) if panel["fixed_value"] else None
    recording={"case":name,"command":[sys.executable,*sys.argv],"source":source,
               "environment":environment_identity(),"config_sha256":sha256(config_path),
               "runner_sha256":sha256(Path(__file__)),"adapter_sha256":sha256(ART/"core.py"),
               "native_source_sha256":sha256(ROOT/"src/splendor/native_search.cpp"),
               "specification_sha256":sha256(ART/"specification.json"),"composition":composition,
               "scope":"Repeated DEV; same PPO latest; smoke is not a strength panel." if args.smoke_fixed_value
                       else "Repeated DEV fixed-data diagnostic, not independent FINAL or self-iteration."}
    path=ART/"recording"/f"{name}-{session}.json"
    write_json(path,recording)
    print(json.dumps({"event":"evaluation_start","case":name,"games":len(config.game_seeds)*2}),flush=True)
    start=time.perf_counter()
    def progress(done,total):
        print(json.dumps({"event":"evaluation_progress","case":name,"completed_pairs":done,"pairs":total}),flush=True)
    result=run_fixed_evaluation(config,output,store_dir=runtime/"games",failure_dir=runtime/"failures",
              resume=output.exists(),require_scored_pairs=True,progress_every=10,summary_every=10,
              progress_callback=progress)
    check_inputs(spec)
    assert result.summary["passed"]
    recording.update(passed=True,inputs_unchanged=True,elapsed_seconds=time.perf_counter()-start,
                     output_sha256=sha256(output))
    write_json(path,recording)
    print(json.dumps({"event":"evaluation_completed","case":name,
                      "score":result.summary["win_rate_score"],"valid_games":result.summary["valid_games"],
                      "elapsed_seconds":recording["elapsed_seconds"]}),flush=True)


if __name__=="__main__":
    main()
