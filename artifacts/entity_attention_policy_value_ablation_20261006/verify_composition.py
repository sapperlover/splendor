"""Verify the compiled complete fixed-value function, without running extra games."""
import json
from pathlib import Path
import numpy as np
import torch

from splendor.alphazero_recording import sha256, write_json
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.offline_train import make_index
from splendor.actor_critic import TorchStateBatch
from splendor.search_inference import FrozenSearchInference, InferenceConfig, _STATE_FIELDS
from splendor.selfplay_data import collate_selfplay_samples
from core import FixedValueModel
from train import check_inputs,samples_from_refs

ROOT=Path(__file__).resolve().parents[2]
ART=Path(__file__).resolve().parent
spec=json.loads((ART/"specification.json").read_text())
check_inputs(spec)
torch.set_num_threads(1)
policy=load_frozen_actor_critic_checkpoint(ROOT/spec["output"]/"policy_only/step-000500.pt",device="cuda")
value=load_frozen_actor_critic_checkpoint(ROOT/spec["initial"]["path"],device="cuda")
model=FixedValueModel(policy,value).to("cuda")
index=make_index({"games":spec["games"]},256*1024**2)
batch=collate_selfplay_samples(samples_from_refs(index,np.load(ROOT/spec["batch_references"]["path"])[0,:16]))
prepared=policy._prepare(batch.decisions)
state=TorchStateBatch(feature_schema_version=prepared.feature_schema_version,
                      **{field:getattr(prepared,field) for field in _STATE_FIELDS})
evaluator=FrozenSearchInference(model,device="cuda",config=InferenceConfig(mode="compiled",actor_projection=False,
                           candidate_buckets=(16,32,48,64),native_transport="reused",compile_failure_policy="fail_fast"))
with torch.inference_mode():
    expected_logits,_=policy(batch.decisions)
    _,expected_values=value(batch.decisions)
    logits,values=evaluator.compiled.policy_value(prepared)
    only_values=evaluator.compiled.value_only(state)
requested_width=expected_logits.shape[1]
compiled_width=logits.shape[1]
assert compiled_width==evaluator.compiled.width_for(requested_width)
logits=logits[:,:requested_width]
torch.testing.assert_close(logits,expected_logits,rtol=1e-5,atol=1e-5)
torch.testing.assert_close(values,expected_values,rtol=1e-5,atol=1e-5)
torch.testing.assert_close(only_values,expected_values,rtol=1e-5,atol=1e-5)
write_json(ART/"compiled_composition_verification.json",{"passed":True,
    "policy_max_abs_difference":float((logits-expected_logits).abs().max()),
    "value_max_abs_difference":float((values-expected_values).abs().max()),
    "value_only_max_abs_difference":float((only_values-expected_values).abs().max()),
    "requested_candidate_width":requested_width,"compiled_candidate_width":compiled_width,
    "comparison_scope":"Same requested candidate columns; compiled API retains padded candidate bucket columns.",
    "core_sha256":sha256(ART/"core.py"),"script_sha256":sha256(Path(__file__)),
    "inputs_unchanged":True})
