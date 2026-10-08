"""Verify the finite experiment's completed artifacts and preserved formal inputs."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

from splendor.alphazero_recording import sha256,write_json
from splendor.alphazero_checkpoint import load_alphazero_checkpoint
from train import check_inputs

ROOT=Path(__file__).resolve().parents[2]
ART=Path(__file__).resolve().parent
spec=json.loads((ART/"specification.json").read_text())
check_inputs(spec)
fit=json.loads((ART/"fitting.json").read_text())
analysis=json.loads((ART/"analysis.json").read_text())
assert analysis["complete"] and analysis["completed_panels"]==17
assert fit["all_arms_share_identical_batches"] and fit["inputs_unchanged"]
refs=np.load(ROOT/spec["batch_references"]["path"])
assert refs.shape==(500,256,2)
ledger=hashlib.sha256(refs.tobytes()).hexdigest()
initial=load_alphazero_checkpoint(ROOT/spec["initial"]["path"])
teacher=load_alphazero_checkpoint(spec["teacher_checkpoint"]["path"])
assert initial.model_state_dict.keys()==teacher.model_state_dict.keys()
assert all(torch.equal(value,teacher.model_state_dict[name]) for name,value in initial.model_state_dict.items())
candidate_records=[]
for arm in spec["arms"]:
    result=fit["arms"][arm]
    assert result["actual_batch_ledger_sha256"]==ledger
    assert [c["step"] for c in result["curve"]]==[0,100,300,500]
    for step in (100,300,500):
        path=ROOT/spec["output"]/arm/f"step-{step:06d}.pt"
        payload=load_alphazero_checkpoint(path)
        assert payload.metadata["diagnostic_only"] and payload.metadata["arm"]==arm
        assert payload.total_train_step==14400+step
        assert payload.round_train_step==step and payload.round_steps_target==500
        if arm=="policy_only":
            assert payload.learner_config["value_loss_weight"]==0
            for name,value in payload.model_state_dict.items():
                if name.startswith("critic."):
                    assert torch.equal(initial.model_state_dict[name],value)
        if arm=="value_detached":
            assert payload.metadata["value_trunk_detached"]
        candidate_records.append({"path":str(path.relative_to(ROOT)),"sha256":sha256(path)})
    metrics=[json.loads(line) for line in (ROOT/spec["output"]/arm/"training_metrics.jsonl").read_text().splitlines()]
    assert [m["step"] for m in metrics]==list(range(1,501))
assert len({fit["arms"][arm]["actual_batch_ledger_sha256"] for arm in spec["arms"]})==1
longer=json.loads((ART/"policy_only_longer.json").read_text())
assert longer["passed"] and longer["inputs_unchanged"] and longer["critic_parameters_unchanged"]
assert longer["script_sha256"]==sha256(ART/"policy_longer.py")
assert longer["core_sha256"]==sha256(ART/"core.py")
assert longer["sampling_rng_start"]==spec["sampling_rng_end"]
assert longer["additional_updates"]==1500 and [c["step"] for c in longer["curve"]]==[500,1000,2000]
extra_refs_path=ROOT/longer["actual_batch_references"]["path"]
assert sha256(extra_refs_path)==longer["actual_batch_references"]["sha256"]
extra_refs=np.load(extra_refs_path)
assert extra_refs.shape==(1500,256,2)
assert hashlib.sha256(extra_refs.tobytes()).hexdigest()==longer["actual_batch_ledger_sha256"]
extra_stage=ROOT/spec["output"]/"policy_only_longer"
extra_metrics=[json.loads(s) for s in (extra_stage/"training_metrics.jsonl").read_text().splitlines()]
assert [m["step"] for m in extra_metrics]==list(range(501,2001))
for cp in longer["checkpoints"]:
    path=ROOT/cp["path"]
    assert sha256(path)==cp["sha256"]
    payload=load_alphazero_checkpoint(path)
    assert payload.metadata["diagnostic_only"] and payload.metadata["exploratory_continuation"]
    assert payload.total_train_step==14400+payload.round_train_step
    for name,value in payload.model_state_dict.items():
        if name.startswith("critic."):
            assert torch.equal(initial.model_state_dict[name],value)
valid_games=0
for panel in spec["panels"]:
    path=ART/"evaluations"/f"{panel['id']}.json"
    data=json.loads(path.read_text())
    assert data["passed"] and data["valid_games"]==panel["pairs"]*2
    assert data["completed_pairs"]==panel["pairs"]
    assert all(data[key]==0 for key in ("exceptions","illegal_actions","invariant_failures","engineering_truncations"))
    valid_games+=data["valid_games"]
    recordings=[json.loads(p.read_text()) for p in (ART/"recording").glob(f"{panel['id']}-*.json")]
    passed=[r for r in recordings if r.get("passed")]
    assert len(passed)==1
    record=passed[0]
    assert record["inputs_unchanged"] and record["output_sha256"]==sha256(path)
    assert record["runner_sha256"]==sha256(ART/"evaluate.py")
    assert record["adapter_sha256"]==sha256(ART/"core.py")
    if panel["fixed_value"]:
        assert record["composition"]["value_checkpoint_sha256"]==spec["initial"]["sha256"]
assert valid_games==2700
assert json.loads((ART/"smoke.json").read_text())["passed"]
profile=json.loads((ART/"value_profile.json").read_text())
assert profile["inputs_unchanged"] and profile["validation_games"]==25
assert profile["script_sha256"]==sha256(ART/"value_profile.py")
assert profile["native_search_source_sha256"]==sha256(ROOT/"src/splendor/native_search.cpp")
for name,p in profile["profiles"].items():
    assert p["positions"]==2051 and p["search_clipped_mse"]<=p["raw_mse"]+1e-6
    arm="joint" if name=="initial" else name.removesuffix("_500")
    step=0 if name=="initial" else 500
    curve=next(c for c in fit["arms"][arm]["curve"] if c["step"]==step)
    assert abs(p["raw_mse"]-curve["validation"]["value_loss"])<1e-6
compiled=json.loads((ART/"compiled_composition_verification.json").read_text())
assert compiled["passed"] and compiled["core_sha256"]==sha256(ART/"core.py")
for key in ("policy_max_abs_difference","value_max_abs_difference","value_only_max_abs_difference"):
    assert compiled[key]<1e-4
for rel,hashed in analysis["inputs_sha256"].items():
    assert sha256(ROOT/rel)==hashed
record=json.loads((ART/"training_recording.json").read_text())
assert record["passed"] and record["inputs_unchanged"]
for rel,hashed in record["source"]["source_files"].items():
    assert sha256(ROOT/rel)==hashed
for name,hashed in record["artifact_sources"].items():
    assert sha256(ART/name)==hashed
for rel,hashed in record["source"]["native_extensions"].items():
    assert sha256(Path(rel))==hashed
for path in ART.glob("*.py"):
    compile(path.read_text(),str(path),"exec")
outputs=[p for p in ART.rglob("*.json") if p.name not in ("verification.json","partial_analysis.json")]
write_json(ART/"verification.json",{"protocol":"policy-value-ablation-verification-v1","passed":True,
    "main_training_updates":1500,"main_training_arms":3,"same_batch_references":True,
    "exploratory_full_pool_additional_updates":1500,"exploratory_subset_updates":1000,
    "strength_panels":17,"valid_strength_games":valid_games,
    "smoke_strength_games_excluded":2,"formal_inputs_unchanged":True,
    "frozen_input_count":len(spec["inputs"]),"compiled_composition":compiled,
    "initial_model_matches_collection_teacher":True,
    "report_sha256":sha256(ART/"README.md"),
    "candidate_checkpoints":candidate_records,
    "evidence_sha256":{str(p.relative_to(ROOT)):sha256(p) for p in outputs},
    "python_sources_sha256":{p.name:sha256(p) for p in ART.glob("*.py")},
    "scope":"Fixed-data diagnostics only; no new collection, formal resume, long-run sampling or architecture change."})
print(json.dumps({"passed":True,"panels":17,"valid_games":valid_games,"same_batches":True,"formal_inputs_unchanged":True}))
