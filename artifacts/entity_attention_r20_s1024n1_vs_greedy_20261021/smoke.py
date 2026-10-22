"""One small-model CPU root checks the formal N1 configuration, no arena score."""
from dataclasses import asdict
import time
import torch
from run import ART,CHECKPOINT,prepare,MODEL_SHA
from splendor.alphazero_recording import sha256,write_json
from splendor.engine import GameEngine
from splendor.mcts_policy import MCTSConfig
from splendor.native_policy import NativePUCTSearchPolicy,NativeGreedyPolicy
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint

torch.set_num_threads(1)
config=prepare();assert sha256(CHECKPOINT)==MODEL_SHA
request=GameEngine.new(42593000,first_player=0,mode='checked').decision()
model=load_frozen_actor_critic_checkpoint(CHECKPOINT,device='cpu')
search=NativePUCTSearchPolicy(model,config=MCTSConfig(**config['policy_a']['search_config']))
start=time.perf_counter();result=search.search(request)
greedy=NativeGreedyPolicy(model).search(request)
assert result.simulations==1024 and result.simulation_budgets==[1024]
assert result.transition_cap_hits==0 and sum(result.root_visit_counts)==1024
assert greedy.simulations==0 and not greedy.searchable
write_json(ART/'smoke.json',dict(scope='One engineering CPU root, excluded from formal panel.',passed=True,
    elapsed_seconds=time.perf_counter()-start,search=asdict(result),greedy=asdict(greedy)))
print('CPU S1024N1 / zero-simulation greedy configuration smoke passed.',flush=True)
