"""Exact certificate for the visible reserve077 -> buy077 finishing line."""
from collections import Counter
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'artifacts/web_blind_reservation_20261029'))
import analyze as a
from splendor.actions import BuyReserved
from splendor.alphazero_recording import read_json,sha256,write_json
from splendor.catalog import load_default_catalog
from splendor.contracts import DecisionRequest
from splendor.engine import GameEngine
from splendor.model import DecisionPhase
from splendor.replay import _decode_state,to_primitive
from splendor.result import TerminalResult

ART=Path(__file__).resolve().parent
RAW=ROOT/'runtime'/ART.name
REPLAY=ROOT/'runtime/web_replays/6c2654de6ed04a1db9b77da7131cf2c3.json'
catalog=load_default_catalog()
a.identity_check()
states={r['event']:_decode_state(r['state']) for r in read_json(RAW/'states.json')}


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
    req=engine.decision();assert req.observation.phase is DecisionPhase.MAIN_ACTION
    actor=engine.current_player
    for action in req.legal_actions:
        child=engine.fork();child.apply(action)
        yield from settled(child,actor,[to_primitive(action)])


started=time.perf_counter()
actual=GameEngine.restore(REPLAY).replay_record
human_finish=[event.action for event in actual.events if event.sequence in (72,73) and not event.automatic]
assert isinstance(human_finish[0],BuyReserved)
counts=Counter();results=Counter();best=None;defenses=[]
first=a.fast(states[71],catalog)
for after26,trail26 in all_main(first):
    counts['AI26_complete_action_outcomes']+=1
    if isinstance(after26.decision(),TerminalResult):
        finals=[(after26,[])]
    else:
        assert after26.current_player==0
        for action in human_finish:
            assert action in after26.decision().legal_actions
            after26.apply(action)
        assert after26.observation_for(0).self_public.score==15
        if isinstance(after26.decision(),TerminalResult):
            finals=[(after26,[])]
        else:
            assert after26.current_player==1
            finals=all_main(after26)
    for final,trail27 in finals:
        result=final.decision()
        assert isinstance(result,TerminalResult)
        counts['terminal_action_outcomes']+=1
        results[','.join(map(str,result.winner_ids))]+=1
        row=dict(AI26=trail26,AI27=trail27,result=to_primitive(result))
        if result.winner_ids!=(0,):
            defenses.append(row)
        rank=(result.scores[1],-result.purchased_card_counts[1])
        if best is None or rank>best[0]:
            best=(rank,row)
report=dict(scope='After actual AI25 reserves046 and human26 reserves077. '
            'Enumerate every legal AI26 and AI27 main action, all payment/return/noble branches. '
            'Human27 buys077 using the actual legal payment. Actual remaining deck fixed.',
            replay_sha256=sha256(REPLAY),script_sha256=sha256(Path(__file__)),
            all_human_win=not defenses,counts=counts,winner_hist=results,
            best_AI_result=best[1],defenses=defenses,seconds=time.perf_counter()-started)
write_json(ART/'forced_finish.json',report)
print(report,flush=True)
