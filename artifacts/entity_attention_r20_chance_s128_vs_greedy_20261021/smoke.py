"""One engineering root; not part of the 1000-pair strength panel."""
from dataclasses import asdict
import time
import torch

from run import ART, CHECKPOINT, MODEL_SHA, SEED
from splendor.alphazero_recording import sha256, write_json
from splendor.catalog import load_default_catalog
from splendor.engine import GameEngine
from splendor.mcts_policy import MCTSConfig
from splendor.native_policy import NativeBatchRunner, NativeChancePUCTSearchPolicy
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.search_inference import FrozenSearchInference, InferenceConfig

assert torch.cuda.is_available()
assert sha256(CHECKPOINT)==MODEL_SHA
torch.set_num_threads(1)
catalog=load_default_catalog()
engine=GameEngine.new(SEED+90000,first_player=0,mode='checked')
request=engine.decision()
model=load_frozen_actor_critic_checkpoint(CHECKPOINT,device='cuda')
evaluator=FrozenSearchInference(model,device='cuda',config=InferenceConfig(mode='eager',actor_projection=False))
policy=NativeChancePUCTSearchPolicy(model,evaluator=evaluator,catalog=catalog,
    config=MCTSConfig(simulations=128,determinizations=1,seed=SEED+30000,
        root_noise_fraction=0.,forced_playouts=0.,policy_target_pruning=False))
assert policy.chance_widening is None
job=policy.start_job(request,add_root_noise=False,temperature=0.)
runner=NativeBatchRunner(evaluator,catalog,capacity=32,buckets=(16,32,48,64),slots=1)
start=time.perf_counter()
while not job.done:
    assert runner.run([job])>0
result=job.result
diagnostics=job.core.chance_diagnostics()
assert diagnostics['stochastic'] and diagnostics['root_hidden_samples']==128
assert sum(result.root_visit_counts)==128 and result.transition_cap_hits==0
engine.apply(request.legal_actions[result.chosen_candidate_index])
write_json(ART/'smoke.json',{'scope':'One CUDA eager engineering root only; excluded from formal match.',
    'checkpoint_sha256':MODEL_SHA,'gpu':torch.cuda.get_device_name(0),
    'elapsed_seconds':time.perf_counter()-start,'result':asdict(result),
    'diagnostics':diagnostics,'checked_action_applied':True,'cpw_disabled':True})
print({'engineering_smoke':'passed','simulations':128,'hidden_samples':128,'cpw':False},flush=True)
