"""Replay diagnostic branches and check stored source identities."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'artifacts/web_blind_reservation_20261029'))
import analyze as a
from splendor.alphazero_recording import read_json,sha256,write_json
from splendor.catalog import load_default_catalog
from splendor.replay import _decode_action,_decode_state,to_primitive

ART=Path(__file__).resolve().parent
RAW=ROOT/'runtime'/ART.name
a.identity_check()
catalog=load_default_catalog()
curve=read_json(ART/'curve.json')
assert curve['script_sha256']==sha256(ART/'analyze.py')
assert sha256(ROOT/curve['replay'])==curve['replay_sha256']
assert len(curve['rows'])==54
states={r['event']:_decode_state(r['state']) for r in read_json(RAW/'states.json')}
alternatives=read_json(ART/'alternatives.json')
assert alternatives['script_sha256']==sha256(ART/'alternatives.py')
checks=[]
for branch in alternatives['branches']:
    raw=read_json(ROOT/branch['raw'])
    engine=a.fast(states[69],catalog)
    first=_decode_action(branch['first'])
    assert first in engine.decision().legal_actions
    engine.apply(first)
    for row in raw['trace']:
        req=engine.decision()
        assert engine.current_player==row['actor']
        assert req.observation.own_completed_turns+1==row['turn']
        action=_decode_action(row['action'])
        assert action in req.legal_actions
        engine.apply(action)
    assert to_primitive(engine._debug_state())==raw['state']
    assert to_primitive(engine._debug_state().result)==branch['terminal']
    checks.append(dict(label=branch['label'],actions=len(raw['trace'])+1,terminal_exact=True))
root_rows=[]
for path in ART.glob('roots_*.json'):
    root_rows.extend(read_json(path)['rows'])
assert len(root_rows)==9 and all(r['reproduces_actual'] for r in root_rows)
exact=read_json(ART/'forced_finish.json')
assert exact['script_sha256']==sha256(ART/'endgame.py')
assert exact['all_human_win'] and exact['winner_hist']=={'0':1088}
write_json(ART/'verification.json',dict(actual_main_decisions=54,actual_search_roots_reproduced=9,
           branches=checks,exact_terminal_outcomes_checked=1088,driver_hashes_match=True,
           replay_sha256=curve['replay_sha256'],checkpoint_sha256=curve['checkpoint_sha256']))
print('Verified 54 actual main decisions, 9 reproduced roots, 2 exact branch replays, 1088 enumerated final outcomes.')
