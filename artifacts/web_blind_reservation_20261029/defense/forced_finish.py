"""Check the genuine forced finish after the human fills the third reserve slot."""
import sys,time
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import analyze as a
from splendor.actions import BuyMarket,ReserveMarket
from splendor.alphazero_recording import read_json,write_json,sha256
from splendor.catalog import load_default_catalog
from splendor.contracts import DecisionRequest
from splendor.model import DecisionPhase
from splendor.native_policy import NativeBatchRunner,NativePUCTSearchPolicy
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.replay import _decode_state,to_primitive
from splendor.result import TerminalResult
from splendor.search_inference import FrozenSearchInference

ART=Path(__file__).resolve().parent
RAW=a.RAW/'defense'
catalog=load_default_catalog()
pre=_decode_state(read_json(a.RAW/'roots.json')[2]['state'])
def make(state):return a.fast(state,catalog)
def apply(engine,action):
 req=engine.decision();assert isinstance(req,DecisionRequest) and action in req.legal_actions
 engine.apply(action)
def buy_min(engine,card):
 apply(engine,BuyMarket(card))
 req=engine.decision();assert req.observation.phase is DecisionPhase.PAYMENT
 apply(engine,min(req.legal_actions,key=lambda x:x.tokens.gold))
engine=make(pre)
buy_min(engine,'035')
apply(engine,ReserveMarket('083'))
start=engine._debug_state()
assert engine.current_player==1 and start.completed_turns==(24,23)
assert len(start.players[0].reserved_cards)==3
# Prove buy038 followed by buy070 wins against EVERY intervening human action.
buy_min(engine,'038')
def settled(state,actor,trail):
 e=make(state);req=e.decision()
 if isinstance(req,TerminalResult) or e.current_player!=actor:
  yield e._debug_state(),trail
 else:
  assert req.observation.phase is not DecisionPhase.MAIN_ACTION
  for action in req.legal_actions:
   child=e.fork();child.apply(action)
   yield from settled(child._debug_state(),actor,trail+(action,))
responses=[]
for action in engine.decision().legal_actions:
 child=engine.fork();child.apply(action)
 for state,tail in settled(child._debug_state(),0,(action,)):
  reply=make(state)
  assert not isinstance(reply.decision(),TerminalResult),'human can end first'
  assert reply.current_player==1
  buy_min(reply,'070')
  result=reply.decision()
  assert isinstance(result,TerminalResult) and result.winner_ids==(1,)
  responses.append(dict(human_actions=[to_primitive(x) for x in tail],result=to_primitive(result)))
write_json(RAW/'forced_finish_all_human_responses.json',responses)
# Does the actual model find this line, when the human does leave it available?
torch.set_num_threads(1)
model=load_frozen_actor_critic_checkpoint(a.MODEL,device='cpu')
evaluator=FrozenSearchInference(model,device='cpu')
policy=NativePUCTSearchPolicy(model,config=a.CFG,catalog=catalog,evaluator=evaluator)
runner=NativeBatchRunner(evaluator,catalog,capacity=128,buckets=(16,32,48,64))
engine=make(start);trace=[]
for number in range(2):
 req=engine.decision()
 result=a.search_all(policy,runner,[req])[0]
 row=a.root_record(req,result);trace.append(row)
 action=req.legal_actions[result.chosen_candidate_index]
 apply(engine,action)
 while isinstance(engine.decision(),DecisionRequest) and engine.current_player==1:
  req=engine.decision();r=a.search_all(policy,runner,[req])[0]
  apply(engine,req.legal_actions[r.chosen_candidate_index])
 if isinstance(engine.decision(),TerminalResult):break
 if number==0:
  # Same legal human turn as the original game, now unable to reserve070.
  for event in a.GameEngine.restore(a.REPLAY).replay_record.events:
   if event.actor==0 and event.sequence in (68,69):apply(engine,event.action)
summary=dict(scope='After AI23 buys035 and human24 reserves083, human has three reserve slots occupied. Forced line038→070 checked against all legal human25 main actions and subdecisions, for the actual remaining deck.',human_responses_checked=len(responses),human_main_actions_checked=len({str(r['human_actions'][0]) for r in responses}),all_AI_win=True,model_search='S1024N4 (4096 total), CPU FP32, seed0, no noise',model_choices=[r['chosen'] for r in trace],model_terminal=to_primitive(engine._debug_state().result),model_roots=[{'chosen':r['chosen'],'top':r['top']} for r in trace],raw=str((RAW/'forced_finish_all_human_responses.json').relative_to(a.ROOT)),script_sha256=sha256(Path(__file__)))
write_json(ART/'forced_finish.json',summary)
print(summary,flush=True)
