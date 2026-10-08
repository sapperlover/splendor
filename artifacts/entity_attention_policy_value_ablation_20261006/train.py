"""Run finite common-data ablations; no collection or formal hybrid resume."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from splendor.alphazero_checkpoint import save_alphazero_checkpoint
from splendor.alphazero_recording import append_json, environment_identity, sha256, source_identity, write_json
from splendor.offline_train import make_index
from splendor.selfplay_data import collate_selfplay_samples
from core import (ARMS, FixedValueModel, actual_learner_config, diagnostic_step,
                  gradient_probe, restored_learner, shared_parameters, terms_for)

ROOT=Path(__file__).resolve().parents[2]
ART=Path(__file__).resolve().parent


def check_inputs(spec):
    for item in [*spec["inputs"],spec["batch_references"],spec["gradient_probe"]]:
        assert sha256(ROOT/item["path"])==item["sha256"],item["path"]


def samples_from_refs(index, refs):
    games={g.game_id:g for g in index.split_games("train")}
    return [index.cache.sample(games[int(gid)],int(pos)) for gid,pos in refs]


def smoke(spec):
    torch.set_num_threads(1)
    index=make_index({"games":spec["games"]},256*1024**2)
    refs=np.load(ROOT/spec["batch_references"]["path"])[0,:4]
    samples=samples_from_refs(index,refs)
    batch=collate_selfplay_samples(samples)
    first,_=restored_learner(ROOT/spec["initial"]["path"],"cpu")
    second,_=restored_learner(ROOT/spec["initial"]["path"],"cpu")
    references=[index.position_pool(split="train",round_index=spec["data_round"]).sample(first.sampling_rng)[1]
                for _ in range(4)]
    first.train_step(batch,references=references)
    diagnostic_step(second,batch,"joint")
    difference=max(float((v-second.model.state_dict()[k]).abs().max()) for k,v in first.model.state_dict().items())
    assert difference<=1e-7
    third,_=restored_learner(ROOT/spec["initial"]["path"],"cpu")
    original={k:v.clone() for k,v in third.model.critic.state_dict().items()}
    third.model.critic.requires_grad_(False)
    diagnostic_step(third,batch,"policy_only")
    assert all(p.grad is None for p in third.model.critic.parameters())
    assert all(torch.equal(original[k],v) for k,v in third.model.critic.state_dict().items())
    fourth,_=restored_learner(ROOT/spec["initial"]["path"],"cpu")
    terms,_,_=terms_for(fourth,batch,detach_value=True)
    gv=torch.autograd.grad(terms.value_loss,shared_parameters(fourth.model),allow_unused=True)
    assert all(g is None for g in gv)
    composed=FixedValueModel(third.model,fourth.model)
    with torch.inference_mode():
        logits,values=composed(batch.decisions)
        p,_=third.model(batch.decisions)
        _,v=fourth.model(batch.decisions)
    torch.testing.assert_close(logits,p,rtol=0,atol=0)
    torch.testing.assert_close(values,v,rtol=0,atol=0)
    result={"passed":True,"joint_vs_existing_train_step_max_abs_difference":difference,
            "policy_only_critic_parameters_unchanged":True,"policy_only_critic_gradient_none":True,
            "detached_value_shared_gradient_none":True,"composite_policy_and_value_exact":True}
    write_json(ART/"smoke.json",result)
    print(json.dumps(result),flush=True)


def extras(learner,samples,baseline=None):
    outputs=[]
    correct=mass=denom=0.
    learner.model.eval()
    with torch.inference_mode():
        for start in range(0,len(samples),256):
            batch=collate_selfplay_samples(samples[start:start+256])
            logits,values=learner.model(batch.decisions)
            pred=logits.argmax(dim=1).cpu().numpy()
            target=batch.policy_targets.argmax(axis=1)
            w=batch.policy_weights
            correct+=float(np.sum((pred==target)*w))
            mass+=float(np.sum(batch.policy_targets[np.arange(len(pred)),pred]*w))
            denom+=float(w.sum())
            outputs.append(values.cpu().numpy())
    values=np.concatenate(outputs)
    metrics={"policy_top1_agreement":correct/denom,"policy_target_mass_on_greedy":mass/denom,
             "predicted_value_mean":float(values.mean()),"predicted_value_std":float(values.std()),
             "predicted_value_min":float(values.min()),"predicted_value_max":float(values.max())}
    if baseline is not None:
        metrics["value_function_drift_rms"]=float(np.sqrt(np.mean((values-baseline)**2)))
    return metrics,values


def train(spec):
    torch.set_num_threads(1)
    assert torch.cuda.is_available()
    output=ROOT/spec["output"]
    check_inputs(spec)
    session=str(time.time_ns())
    snapshot=ROOT/"runtime/entity_attention_policy_value_ablation_20261006/source"/session
    source=source_identity(ROOT,snapshot)
    record={"command":[sys.executable,*sys.argv],"source":source,"environment":environment_identity(),
            "artifact_sources":{p.name:sha256(p) for p in ART.glob("*.py")},
            "specification_sha256":sha256(ART/"specification.json"),"inputs_unchanged":None}
    write_json(ART/"training_recording.json",record)
    index=make_index({"games":spec["games"]},1024**3)
    samples={split:[index.cache.sample(g,i) for g in index.split_games(split) for i in range(g.decision_count)]
             for split in ("train","validation")}
    baseline_learner,_=restored_learner(ROOT/spec["initial"]["path"],"cuda")
    baseline={split:extras(baseline_learner,rows)[1] for split,rows in samples.items()}
    del baseline_learner
    gc.collect()
    torch.cuda.empty_cache()
    refs=np.load(ROOT/spec["batch_references"]["path"])
    probe_refs=json.loads((ROOT/spec["gradient_probe"]["path"]).read_text())["references"]
    probe_batches=[collate_selfplay_samples(samples_from_refs(index,probe_refs[i:i+256]))
                   for i in range(0,len(probe_refs),256)]
    pool=index.position_pool(split="train",round_index=spec["data_round"])
    results={}
    for arm in ARMS:
        stage=output/arm
        stage.mkdir(exist_ok=False)
        learner,payload=restored_learner(ROOT/spec["initial"]["path"],"cuda")
        if arm=="policy_only":
            learner.model.critic.requires_grad_(False)
        critic_before={k:v.detach().clone() for k,v in learner.model.critic.state_dict().items()}
        started=time.perf_counter()
        ledger=hashlib.sha256()
        curve=[]
        for step in range(501):
            if step in spec["observations"]:
                observation={"arm":arm,"step":step,"total_train_step":learner.total_train_step,
                             "elapsed_seconds":time.perf_counter()-started}
                for split,rows in samples.items():
                    observation[split]=learner.evaluate_samples(rows)|extras(learner,rows,baseline[split])[0]
                observation["gradient_probe"]= [gradient_probe(learner,b) for b in probe_batches]
                curve.append(observation)
                append_json(stage/"curve.jsonl",observation)
                print(json.dumps({"event":"fit_observation","arm":arm,"step":step,
                                  "train_kl":observation["train"]["policy_kl"],
                                  "validation_kl":observation["validation"]["policy_kl"],
                                  "validation_value_mse":observation["validation"]["value_loss"]}),flush=True)
                if step:
                    checkpoint=stage/f"step-{step:06d}.pt"
                    save_alphazero_checkpoint(checkpoint,model=learner.model,optimizer=learner.optimizer,
                        learner_config=actual_learner_config(learner,arm),round_index=payload.round_index,
                        round_train_step=step,total_train_step=learner.total_train_step,round_steps_target=500,
                        parent_checkpoint=spec["initial"],replay_identity={"protocol":"diagnostic-fixed-round-v1",
                            "digest":sha256(ART/"specification.json"),"data_round":spec["data_round"]},
                        sampling_rng_state=learner.sampling_rng.snapshot(),metadata={"diagnostic_only":True,
                            "arm":arm,"value_trunk_detached":arm=="value_detached",
                            "original_learner_config":payload.learner_config,
                            "batch_references_sha256":spec["batch_references"]["sha256"]})
            if step==500:
                break
            drawn=[pool.sample(learner.sampling_rng) for _ in range(256)]
            actual=np.asarray([[r.game_id,r.decision_index] for _,r in drawn],dtype="<i8")
            assert np.array_equal(actual,refs[step])
            ledger.update(actual.tobytes())
            batch=collate_selfplay_samples([s for s,_ in drawn])
            append_json(stage/"training_metrics.jsonl",diagnostic_step(learner,batch,arm))
        critic_difference=max(float((v-critic_before[k]).abs().max()) for k,v in learner.model.critic.state_dict().items())
        if arm=="policy_only":
            assert critic_difference==0
        assert asdict(learner.sampling_rng.snapshot())==spec["sampling_rng_end"]
        results[arm]={"curve":curve,"elapsed_seconds":time.perf_counter()-started,
                      "actual_batch_ledger_sha256":ledger.hexdigest(),"critic_parameter_max_abs_difference":critic_difference}
        del learner,payload,critic_before
        gc.collect()
        torch.cuda.empty_cache()
    assert len({r["actual_batch_ledger_sha256"] for r in results.values()})==1
    check_inputs(spec)
    write_json(ART/"fitting.json",{"arms":results,"inputs_unchanged":True,"cache":index.cache.stats(),
                                  "all_arms_share_identical_batches":True})
    record.update(inputs_unchanged=True,passed=True,output_sha256=sha256(ART/"fitting.json"))
    write_json(ART/"training_recording.json",record)
    print(json.dumps({"event":"training_completed","arms":len(ARMS),"updates_per_arm":500}),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args()
    spec=json.loads((ART/"specification.json").read_text())
    check_inputs(spec)
    smoke(spec) if args.smoke else train(spec)
