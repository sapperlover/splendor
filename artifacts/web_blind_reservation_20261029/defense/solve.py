"""Exact finite defense verification on the recorded realized deck, not a policy claim."""
from __future__ import annotations
from collections import Counter
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
import json,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import analyze as a
from splendor.actions import BuyMarket,BuyReserved,ReserveDeck,ReserveMarket,TakeDistinct,ReturnTokens,Pay
from splendor.alphazero_recording import read_json,write_json,sha256
from splendor.catalog import load_default_catalog
from splendor.contracts import DecisionRequest
from splendor.model import Color,DecisionPhase,Tier,TokenVector
from splendor.replay import _decode_state,to_primitive,state_digest
from splendor.result import TerminalResult
from splendor.rules import eligible_noble_ids

ART=Path(__file__).resolve().parent
RAW=a.RAW/'defense'
RAW.mkdir(parents=True,exist_ok=True)
catalog=load_default_catalog()
pre=_decode_state(read_json(a.RAW/'roots.json')[2]['state'])
counts=Counter()
started=time.perf_counter()


def canonical(s):
 players=tuple(replace(p,purchased_card_ids=tuple(sorted(p.purchased_card_ids)),reserved_cards=tuple(sorted(p.reserved_cards,key=lambda r:(r.card_id,r.source.value))),noble_ids=tuple(sorted(p.noble_ids))) for p in s.players)
 market=replace(s.market,tier_one=tuple(sorted(s.market.tier_one)),tier_two=tuple(sorted(s.market.tier_two)),tier_three=tuple(sorted(s.market.tier_three)))
 return replace(s,players=players,market=market,nobles_in_play=tuple(sorted(s.nobles_in_play)),nobles_out_of_game=tuple(sorted(s.nobles_out_of_game)))


def engine(s):return a.fast(s,catalog)


def advance(s,action):
 e=engine(s)
 request=e.decision()
 assert isinstance(request,DecisionRequest) and action in request.legal_actions,(s.completed_turns,action)
 e.apply(action)
 counts['applied_transitions']+=1
 return e._debug_state()


def turn_finishes(s,trail):
 """All payments, returns and noble choices, including gold overpayment."""
 e=engine(s);request=e.decision();s=e._debug_state()
 if isinstance(request,TerminalResult) or s.current_player!=1:
  yield canonical(s),trail
  return
 assert request.observation.phase is not DecisionPhase.MAIN_ACTION
 for action in request.legal_actions:
  yield from turn_finishes(advance(s,action),trail+(action,))


def human_defense(s):
 e=engine(s);request=e.decision();s=e._debug_state()
 if isinstance(request,TerminalResult):return s,()
 assert s.current_player==0
 turn=s.completed_turns[0]+1
 if turn==24:actions=[ReserveMarket('070')]
 elif turn==25:actions=[TakeDistinct((Color.WHITE,Color.BLUE,Color.GREEN)),ReturnTokens(TokenVector(white=1))]
 elif turn==26:
  actions=[BuyReserved('086')]
 else:raise AssertionError(turn)
 trace=[]
 for action in actions:
  s=advance(s,action);trace.append(action)
 if turn==26:
  e=engine(s);request=e.decision()
  assert request.observation.phase is DecisionPhase.PAYMENT
  action=min(request.legal_actions,key=lambda x:x.tokens.gold)
  s=advance(s,action);trace.append(action)
 e=engine(s);request=e.decision();s=e._debug_state()
 assert isinstance(request,TerminalResult) or s.current_player==1
 if turn==26:assert e.observation_for(0).self_public.score==15
 return canonical(s),tuple(trace)


def terminal_rank(result):
 # Maximization is from AI's perspective. A draw would refute guaranteed human win.
 payoff=result.payoff(1)
 return (int(payoff),result.scores[1]-result.scores[0],-result.purchased_card_counts[1])


@lru_cache(None)
def solve(s):
 counts['unique_positions']+=1
 e=engine(s);request=e.decision();s=e._debug_state()
 if isinstance(request,TerminalResult):
  counts['terminal_positions']+=1
  counts[f'AI_terminal_payoff_{request.payoff(1)}']+=1
  return terminal_rank(request),(),to_primitive(request)
 if s.current_player==0:
  nxt,actions=human_defense(s)
  best,line,result=solve(nxt)
  return best,tuple((0,x) for x in actions)+line,result
 assert s.current_player==1 and s.phase is DecisionPhase.MAIN_ACTION
 assert s.completed_turns[1]+1<=26
 actions=list(request.legal_actions)
 # Once human has 15, a non-purchase cannot change bonuses. If no noble remains
 # eligible and current AI payoff is already losing, such actions cannot save it.
 if s.final_round_triggered_by is not None and e.observation_for(0).self_public.score==15 and e.observation_for(1).self_public.score<15 and not eligible_noble_ids(s.players[1],s.nobles_in_play,catalog):
  kept=[x for x in actions if isinstance(x,(BuyMarket,BuyReserved))]
  counts['proven_nonwinning_final_nonpurchase_prunes']+=len(actions)-len(kept)
  actions=kept
 best=None;best_line=();best_result=None
 successors={}
 for action in actions:
  for nxt,subactions in turn_finishes(advance(s,action),(action,)):
   successors.setdefault(nxt,subactions)
 counts['unique_turn_outcomes']+=len(successors)
 for nxt,subactions in successors.items():
  value,line,result=solve(nxt)
  if best is None or value>best:
   best=value;best_line=tuple((1,x) for x in subactions)+line;best_result=result
 # If no purchase is legal in the last turn, every remaining action was proved
 # non-winning above. Settle one actual legal move to retain a concrete terminal.
 if best is None:
  assert s.final_round_triggered_by is not None
  action=request.legal_actions[0]
  for nxt,subactions in turn_finishes(advance(s,action),(action,)):
   value,line,result=solve(nxt)
   if best is None or value>best:
    best=value;best_line=tuple((1,x) for x in subactions)+line;best_result=result
 assert best is not None
 return best,best_line,best_result

reports=[]
for label,first in [('buy035',BuyMarket('035')),('actual_blind3',ReserveDeck(Tier.THREE))]:
 initial=advance(pre,first)
 outcomes=list(turn_finishes(initial,(first,)))
 best=None;line=None;result=None
 for nxt,actions in outcomes:
  value,tail,terminal=solve(nxt)
  if best is None or value>best:
   best=value;line=tuple((1,x) for x in actions)+tail;result=terminal
 # Validate the selected best-response witness on the uncanonicalized original.
 e=engine(pre);witness=[]
 for actor,action in line:
  req=e.decision()
  assert isinstance(req,DecisionRequest) and e.current_player==actor and action in req.legal_actions
  witness.append(dict(actor=actor,turn=req.observation.own_completed_turns+1,phase=req.observation.phase.value,action=to_primitive(action)))
  e.apply(action)
 assert isinstance(e.decision(),TerminalResult)
 assert to_primitive(e.decision())==result
 reports.append(dict(initial_AI_action=label,all_initial_payment_outcomes=len(outcomes),AI_best_payoff=best[0],human_forces_win=best[0]<0,best_AI_terminal=result,best_AI_witness=witness))
 write_json(RAW/f'{label}.json',reports[-1])
 print(json.dumps(dict(stage=label,counts=counts,seconds=time.perf_counter()-started,AI_best_payoff=best[0],terminal=result)),flush=True)
report=dict(scope='Exhaustive AI best responses against one specified legal human defense; fixed actual remaining deck and private cards. Analysis grants AI full knowledge when choosing best responses, but human defense uses only public card070 and its own blind086. Does not solve every hidden world or every possible initial AI action.',root_event=64,root_state_sha256=state_digest(pre),human_strategy=['H24 ReserveMarket070','H25 TakeDistinct white blue green, return white1','H26 BuyReserved086, minimum-gold payment'],AI_search='All legal main actions, payments, returns and noble choices through final reply turn26; no neural value cutoff',pruning='Only final-turn nonpurchases that cannot improve score or acquire any eligible noble, when human already has15 and AI has less15, are omitted as proven losing.',branches=reports,counts=counts,cache=solve.cache_info()._asdict(),seconds=time.perf_counter()-started,source=dict(script_sha256=sha256(Path(__file__)),replay_sha256=sha256(a.REPLAY),catalog_fingerprint=catalog.data_fingerprint))
write_json(ART/'exact_defense.json',report)
print('COMPLETE',json.dumps({'seconds':report['seconds'],'counts':counts}),flush=True)
