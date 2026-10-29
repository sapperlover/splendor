"""Information-safe S1024N4 responses on both sides of the two tactical branches."""
from collections import Counter
from dataclasses import asdict,replace
from pathlib import Path
import sys,time
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import analyze as a
from splendor.actions import BuyMarket,ReserveDeck,ReserveMarket
from splendor.alphazero_recording import read_json,write_json,sha256
from splendor.catalog import load_default_catalog
from splendor.contracts import DecisionRequest
from splendor.model import DecisionPhase,Tier
from splendor.native_policy import NativeBatchRunner,NativePUCTSearchPolicy
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.replay import _decode_state,to_primitive
from splendor.result import TerminalResult
from splendor.search_inference import FrozenSearchInference

ART=Path(__file__).resolve().parent
RAW=a.RAW/'defense'
torch.set_num_threads(1)
catalog=load_default_catalog()
model=load_frozen_actor_critic_checkpoint(a.MODEL,device='cpu')
evaluator=FrozenSearchInference(model,device='cpu')
policy=NativePUCTSearchPolicy(model,config=a.CFG,catalog=catalog,evaluator=evaluator)
runner=NativeBatchRunner(evaluator,catalog,capacity=128,buckets=(16,32,48,64))
pre=_decode_state(read_json(a.RAW/'roots.json')[2]['state'])
branches=[]
for label,first in [('buy035',BuyMarket('035')),('actual_blind3',ReserveDeck(Tier.THREE))]:
 engine=a.fast(pre,catalog)
 engine.apply(first)
 while isinstance(engine.decision(),DecisionRequest) and engine.current_player==1:
  request=engine.decision()
  assert request.observation.phase is DecisionPhase.PAYMENT
  engine.apply(min(request.legal_actions,key=lambda x:x.tokens.gold))
 assert engine.current_player==0
 for manual in (False,True):
  copy=engine.fork()
  if manual:copy.apply(ReserveMarket('070'))
  branches.append(dict(label=label+('_manual070' if manual else '_both_search'),engine=copy,trace=[],done=False))
# The first response itself is also checked at 4x simulations, retaining N4.
human_roots=[b['engine'].decision() for b in branches if b['label'].endswith('both_search')]
strong=NativePUCTSearchPolicy(model,config=replace(a.CFG,simulations=16384),catalog=catalog,evaluator=evaluator)
jobs=[strong.start_job(request,add_root_noise=False,temperature=0.) for request in human_roots]
started=time.perf_counter()
while any(not j.done for j in jobs):
 assert runner.run([j for j in jobs if not j.done])>0
high=[]
for label,req,job in zip(['after_buy035','after_actual_blind3'],human_roots,jobs,strict=True):
 result=job.result
 # root_record enforces total4096, so serialize this higher-budget root directly.
 visits=np.asarray(result.root_visit_counts);sums=np.asarray(result.per_world_root_value_sums).sum(axis=0)
 order=sorted(range(len(visits)),key=lambda i:(-visits[i],-result.root_network_prior[i],i))
 assert visits.sum()==16384
 high.append(dict(label=label,budget='S4096N4 (16384 total)',chosen=to_primitive(req.legal_actions[result.chosen_candidate_index]),top=[dict(action=to_primitive(req.legal_actions[i]),visits=int(visits[i]),q=float(sums[i]/visits[i]) if visits[i] else None,prior=result.root_network_prior[i]) for i in order[:5]]))
write_json(RAW/'high_roots.json',high)
print('high roots',[x['chosen'] for x in high],flush=True)
cycle=0
while any(not b['done'] for b in branches):
 waiting=[]
 for b in branches:
  if b['done']:continue
  engine=b['engine'];request=engine.decision()
  if isinstance(request,TerminalResult) or min(engine._debug_state().completed_turns)>=31:
   b['done']=True;continue
  waiting.append((b,request))
 if not waiting:break
 results=a.search_all(policy,runner,[req for _,req in waiting])
 for (b,req),result in zip(waiting,results,strict=True):
  engine=b['engine'];action=req.legal_actions[result.chosen_candidate_index]
  logits,value=evaluator.policy_value((req,))[0]
  b['trace'].append(dict(actor=engine.current_player,turn=req.observation.own_completed_turns+1,phase=req.observation.phase.value,action=to_primitive(action),network_value=value,root=a.root_record(req,result)))
  engine.apply(action)
 cycle+=1
 for b in branches:
  engine=b['engine']
  write_json(RAW/f"search_{b['label']}.json",dict(label=b['label'],done=b['done'],trace=b['trace'],state=to_primitive(engine._debug_state()),terminal=to_primitive(engine._debug_state().result) if engine._debug_state().result else None))
 print('cycle',cycle,'seconds',time.perf_counter()-started,flush=True)
summary=[]
for b in branches:
 e=b['engine']
 mains=[{k:r[k] for k in ('actor','turn','action','network_value')} for r in b['trace'] if r['phase']=='main_action']
 summary.append(dict(label=b['label'],main_actions=mains,terminal=to_primitive(e._debug_state().result) if e._debug_state().result else None,source=str((RAW/f"search_{b['label']}.json").relative_to(a.ROOT))))
report=dict(scope='A few deterministic tactical continuations, not a playing-strength match. Both policies see only their own DecisionRequests; no root noise, seed0, fixed current strongest checkpoint.',search=asdict(a.CFG),device='cpu',torch_threads=1,higher_budget_human_roots=high,continuations=summary,seconds=time.perf_counter()-started,script_sha256=sha256(Path(__file__)))
write_json(ART/'search_responses.json',report)
print('COMPLETE',report['seconds'],flush=True)
