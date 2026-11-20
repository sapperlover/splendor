"""Audit the known finishing threat separately in each sampled root world."""
from collections import Counter
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'artifacts/web_blind_reservation_20261029'))
import analyze as a
from splendor.actions import ReserveMarket,BuyReserved
from splendor.alphazero_recording import read_json,sha256,write_json
from splendor.catalog import load_default_catalog
from splendor.model import DecisionPhase
from splendor.replay import _decode_state,to_primitive
from splendor.result import TerminalResult
from splendor.search_policy import _request_seed,determinize_request

ART=Path(__file__).resolve().parent
RAW=ROOT/'runtime'/ART.name
catalog=load_default_catalog()
a.identity_check()


def settled(engine,actor,trail):
    req=engine.decision()
    if isinstance(req,TerminalResult) or engine.current_player!=actor:
        yield engine,trail
        return
    assert req.observation.phase is not DecisionPhase.MAIN_ACTION
    for action in req.legal_actions:
        child=engine.fork();child.apply(action)
        yield from settled(child,actor,trail+[to_primitive(action)])


def all_main(engine):
    actor=engine.current_player
    for action in engine.decision().legal_actions:
        child=engine.fork();child.apply(action)
        yield from settled(child,actor,[to_primitive(action)])


def finish(engine):
    req=engine.decision();action=BuyReserved('077')
    assert action in req.legal_actions
    engine.apply(action)
    while not isinstance(engine.decision(),TerminalResult) and engine.current_player==1:
        req=engine.decision()
        assert req.observation.phase is DecisionPhase.PAYMENT
        engine.apply(min(req.legal_actions,key=lambda x:x.tokens.gold))


pre=_decode_state(next(r['state'] for r in read_json(RAW/'states.json') if r['event']==69))
request=a.fast(pre,catalog).decision()
started=time.perf_counter();worlds=[]
for index in range(4):
    engine=determinize_request(request,seed=_request_seed(request,0,index),catalog=catalog,engine_mode='fast')
    assert engine.current_player==0 and engine.decision()==request
    replacement=engine._debug_state().decks.tier_two[-1]
    engine.apply(ReserveMarket('046'))
    assert engine.current_player==1
    engine.apply(ReserveMarket('077'))
    results=Counter();wins=[];best=None;complete26=0
    for after26,trail26 in all_main(engine):
        complete26+=1
        if isinstance(after26.decision(),TerminalResult):
            finals=[(after26,[])]
        else:
            finish(after26)
            finals=[(after26,[])] if isinstance(after26.decision(),TerminalResult) else all_main(after26)
        for final,trail27 in finals:
            result=final.decision();assert isinstance(result,TerminalResult)
            results[','.join(map(str,result.winner_ids))]+=1
            row=dict(AI26=trail26,AI27=trail27,result=to_primitive(result))
            rank=(int(0 in result.winner_ids),result.scores[0],-result.purchased_card_counts[0])
            if best is None or rank>best[0]:best=(rank,row)
            if result.winner_ids==(0,):wins.append(row)
    worlds.append(dict(index=index,seed=_request_seed(request,0,index),
                       replacement_after_reserve046=a.describe(replacement,catalog),
                       complete_AI26_outcomes=complete26,terminal_outcomes=sum(results.values()),winner_hist=results,
                       AI_has_winning_reply_to_human077=bool(wins),best_AI_reply=best[1]))
    print(worlds[-1],flush=True)
write_json(ART/'worlds_audit.json',dict(scope='Root-relative actors: AI=0,human=1. In each of the four '
           'sampled AI25 root worlds, force AI25 reserve046 and human26 reserve077, then enumerate every '
           'AI26/27 response to human27 buy077. This checks that specific finishing line, not arbitrary human defense.',
           worlds=worlds,seconds=time.perf_counter()-started,script_sha256=sha256(Path(__file__))))
