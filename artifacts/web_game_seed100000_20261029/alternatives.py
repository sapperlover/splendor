"""Bounded adaptive replies and 4x-budget recheck at the final critical root."""
from dataclasses import asdict,replace
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'artifacts/web_blind_reservation_20261029'))
import analyze as a
from splendor.actions import ReserveMarket
from splendor.alphazero_recording import read_json,sha256,write_json
from splendor.catalog import load_default_catalog
from splendor.model import DecisionPhase
from splendor.native_policy import NativeBatchRunner,NativePUCTSearchPolicy
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.replay import _decode_state,to_primitive
from splendor.result import TerminalResult
from splendor.search_inference import FrozenSearchInference

ART=Path(__file__).resolve().parent
RAW=ROOT/'runtime'/ART.name
a.identity_check()
torch.set_num_threads(1)
catalog=load_default_catalog()
model=load_frozen_actor_critic_checkpoint(a.MODEL,device='cpu')
evaluator=FrozenSearchInference(model,device='cpu')
policy=NativePUCTSearchPolicy(model,config=a.CFG,catalog=catalog,evaluator=evaluator)
runner=NativeBatchRunner(evaluator,catalog,capacity=128,buckets=(16,32,48,64),slots=1)
state=_decode_state(next(r['state'] for r in read_json(RAW/'states.json') if r['event']==69))
request=a.fast(state,catalog).decision()
started=time.perf_counter()
strong=NativePUCTSearchPolicy(model,config=replace(a.CFG,simulations=16384),catalog=catalog,evaluator=evaluator)
job=strong.start_job(request,add_root_noise=False,temperature=0.)
while not job.done:
    assert runner.run([job])>0
r=job.result
counts=np.asarray(r.root_visit_counts);sums=np.asarray(r.per_world_root_value_sums).sum(axis=0)
assert counts.sum()==16384
order=sorted(range(len(counts)),key=lambda i:(-counts[i],-r.root_network_prior[i],i))
high=dict(budget='S4096N4, total16384',chosen=to_primitive(request.legal_actions[r.chosen_candidate_index]),
          top=[dict(action=to_primitive(request.legal_actions[i]),visits=int(counts[i]),
                    q=float(sums[i]/counts[i]) if counts[i] else None,prior=r.root_network_prior[i]) for i in order[:8]])
print('higher',high,flush=True)
branches=[]
for label,card in [('actual046','046'),('block077','077')]:
    engine=a.fast(state,catalog);action=ReserveMarket(card)
    assert action in engine.decision().legal_actions
    engine.apply(action)
    branches.append(dict(label=label,first=to_primitive(action),engine=engine,trace=[],done=False))
while any(not b['done'] for b in branches):
    waiting=[]
    for b in branches:
        if b['done']:continue
        e=b['engine'];req=e.decision()
        if isinstance(req,TerminalResult) or min(e._debug_state().completed_turns)>=33:
            b['done']=True
        else:waiting.append((b,req))
    if not waiting:break
    results=a.search_all(policy,runner,[req for _,req in waiting])
    for (b,req),r in zip(waiting,results,strict=True):
        e=b['engine'];action=req.legal_actions[r.chosen_candidate_index]
        b['trace'].append(dict(actor=e.current_player,turn=req.observation.own_completed_turns+1,
                               phase=req.observation.phase.value,action=to_primitive(action),root=a.root_record(req,r)))
        e.apply(action)
    print('steps',[len(b['trace']) for b in branches],'seconds',round(time.perf_counter()-started,1),flush=True)
summary=[]
for b in branches:
    e=b['engine']
    row=dict(label=b['label'],first=b['first'],trace=b['trace'],state=to_primitive(e._debug_state()),
             terminal=to_primitive(e._debug_state().result))
    write_json(RAW/f"alternative_{b['label']}.json",row)
    summary.append(dict(label=b['label'],first=b['first'],
                        main_actions=[{k:r[k] for k in ('actor','turn','action')}
                                      for r in b['trace'] if r['phase']=='main_action'],
                        terminal=row['terminal'],raw=str((RAW/f"alternative_{b['label']}.json").relative_to(ROOT))))
write_json(ART/'alternatives.json',dict(scope='Two finite deterministic continuations, both players use S1024N4 '
           'with their own private requests. Actual initial deck; not win rates or optimal-play proof.',
           search=asdict(a.CFG),device='cpu',torch_threads=1,higher_root=high,branches=summary,
           seconds=time.perf_counter()-started,script_sha256=sha256(Path(__file__))))
print('COMPLETE',summary,flush=True)
