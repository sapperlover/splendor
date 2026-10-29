"""Legal line holding the recorded human replies fixed; not a forced-win proof."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import analyze as a
from splendor.actions import BuyMarket
from splendor.alphazero_recording import read_json,write_json,sha256
from splendor.catalog import load_default_catalog
from splendor.contracts import DecisionRequest
from splendor.engine import GameEngine
from splendor.model import DecisionPhase
from splendor.replay import _decode_state,to_primitive
from splendor.result import TerminalResult

catalog=load_default_catalog()
state=_decode_state(read_json(a.RAW/'roots.json')[2]['state'])
engine=a.fast(state,catalog)
record=GameEngine.restore(a.REPLAY).replay_record
trace=[]
def apply(action,source):
 request=engine.decision()
 assert isinstance(request,DecisionRequest) and action in request.legal_actions
 trace.append(dict(actor=engine.current_player,turn=request.observation.own_completed_turns+1,action=to_primitive(action),source=source))
 engine.apply(action)
def buy(card):
 apply(BuyMarket(card),'counterfactual chosen legal line, not a network prediction')
 while isinstance(engine.decision(),DecisionRequest) and engine.current_player==1:
  request=engine.decision()
  assert request.observation.phase is DecisionPhase.PAYMENT
  apply(min(request.legal_actions,key=lambda x:x.tokens.gold),'minimum-gold payment')
buy('035')
for event in record.events:
 if event.actor==0 and event.sequence==65:apply(event.action,'recorded human reply')
buy('038')
for event in record.events:
 if event.actor==0 and event.sequence in (68,69):apply(event.action,'recorded human reply')
buy('070')
assert isinstance(engine.decision(),TerminalResult)
report=dict(scope='Omit third blind and buy 035,038,070, holding recorded human replies fixed. Availability of this legal line does not prove a forced win against an adapting opponent or that the network would select it.',trace=trace,result=to_primitive(engine.decision()),script_sha256=sha256(Path(__file__)))
write_json(a.ART/'tempo_control.json',report)
print(report)
