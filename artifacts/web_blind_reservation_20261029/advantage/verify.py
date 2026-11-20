"""Verify finite diagnostic traces against the frozen rule engine."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import analyze as a
from splendor.alphazero_recording import read_json,sha256,write_json
from splendor.catalog import load_default_catalog
from splendor.replay import _decode_action,_decode_state,to_primitive

ART=Path(__file__).resolve().parent
RAW=a.RAW/'advantage'
a.identity_check()
catalog=load_default_catalog()
states={r['event']:_decode_state(r['state']) for r in read_json(RAW/'states.json')}
report=read_json(ART/'continuations.json')
assert sha256(ART/'continuations.py')==report['script_sha256']
assert sha256(ART/'analyze.py')==read_json(ART/'curve.json')['script_sha256']
assert sha256(ART/'knowledge.py')==read_json(ART/'knowledge.json')['script_sha256']
checks=[]
for row in report['branches']:
    raw=read_json(a.ROOT/row['raw'])
    engine=a.fast(states[row['event']],catalog)
    first=_decode_action(row['first'])
    assert first in engine.decision().legal_actions
    engine.apply(first)
    for step in raw['trace']:
        req=engine.decision()
        assert engine.current_player==step['actor']
        assert req.observation.own_completed_turns+1==step['turn']
        assert req.observation.phase.value==step['phase']
        action=_decode_action(step['action'])
        assert action in req.legal_actions
        engine.apply(action)
    assert to_primitive(engine._debug_state())==raw['state']
    assert to_primitive(engine._debug_state().result)==row['terminal']
    assert row['terminal'] is not None
    # The last batch was flushed before the runner observed all terminals.
    # Finalize that progress marker only after exact legal replay succeeds.
    if not raw['done']:
        raw['done']=True
        raw['completion_verified_by']=sha256(Path(__file__))
        write_json(a.ROOT/row['raw'],raw)
    checks.append(dict(label=row['label'],actions=len(raw['trace'])+1,terminal_exact=True))
root_report=read_json(ART/'roots_49_55_57_59_60_61.json')
assert all(r['reproduces_actual'] for r in root_report['rows'])
write_json(ART/'verification.json',dict(actual_main_decisions=len(states),replayed_branches=checks,
           replayed_action_count=sum(r['actions'] for r in checks),
           actual_root_choices_reproduced=len(root_report['rows']),
           driver_hashes_match=True,checkpoint_sha256=sha256(a.MODEL),replay_sha256=sha256(a.REPLAY)))
print('Verified',len(states),'actual main decisions;',sum(r['actions'] for r in checks),'counterfactual actions;',len(checks),'exact terminal states.')
