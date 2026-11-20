"""Small adaptive counterfactuals; not a strength match or minimax proof."""
from dataclasses import asdict
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze as a
from splendor.actions import BuyMarket, BuyReserved, ReserveMarket
from splendor.alphazero_recording import read_json, sha256, write_json
from splendor.catalog import load_default_catalog
from splendor.contracts import DecisionRequest
from splendor.model import DecisionPhase
from splendor.native_policy import NativeBatchRunner, NativePUCTSearchPolicy
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.replay import _decode_state, to_primitive
from splendor.result import TerminalResult
from splendor.search_inference import FrozenSearchInference

ART = Path(__file__).resolve().parent
RAW = a.RAW/'advantage'
a.identity_check()
torch.set_num_threads(1)
catalog = load_default_catalog()
model = load_frozen_actor_critic_checkpoint(a.MODEL, device='cpu')
evaluator = FrozenSearchInference(model, device='cpu')
policy = NativePUCTSearchPolicy(model, config=a.CFG, catalog=catalog, evaluator=evaluator)
runner = NativeBatchRunner(evaluator, catalog, capacity=128, buckets=(16,32,48,64), slots=1)
states = {r['event']:_decode_state(r['state']) for r in read_json(RAW/'states.json')}
branches = []
for event, label, action in [
    (49, 'A18_actual_buy045', BuyReserved('045')),
    (49, 'A18_reserve060', ReserveMarket('060')),
    (49, 'A18_buy060', BuyMarket('060')),
    (61, 'A22_actual_reserve080', ReserveMarket('080')),
    (61, 'A22_buy035', BuyMarket('035')),
    (61, 'A22_buy047', BuyMarket('047')),
]:
    engine = a.fast(states[event], catalog)
    assert action in engine.decision().legal_actions
    engine.apply(action)
    branches.append(dict(event=event, label=label, first=to_primitive(action), engine=engine, trace=[], done=False))
started = time.perf_counter()
cycle = 0
while any(not b['done'] for b in branches):
    waiting = []
    for b in branches:
        if b['done']:
            continue
        engine = b['engine']
        req = engine.decision()
        if isinstance(req, TerminalResult) or min(engine._debug_state().completed_turns)>=30:
            b['done'] = True
        else:
            waiting.append((b, req))
    if not waiting:
        break
    results = a.search_all(policy, runner, [r for _,r in waiting])
    for (b, req), result in zip(waiting, results, strict=True):
        engine = b['engine']
        action = req.legal_actions[result.chosen_candidate_index]
        _, value = evaluator.policy_value((req,))[0]
        b['trace'].append(dict(actor=engine.current_player, turn=req.observation.own_completed_turns+1,
                               phase=req.observation.phase.value, action=to_primitive(action),
                               value=float(value), root=a.root_record(req,result)))
        engine.apply(action)
    cycle += 1
    for b in branches:
        engine = b['engine']
        write_json(RAW/f"continuation_{b['label']}.json", dict(event=b['event'], label=b['label'],
                   first=b['first'], trace=b['trace'], done=b['done'], state=to_primitive(engine._debug_state())))
    print('cycle',cycle,'seconds',round(time.perf_counter()-started,1),flush=True)
summary = []
for b in branches:
    engine = b['engine']
    summary.append(dict(event=b['event'],label=b['label'],first=b['first'],
                        main_actions=[{k:r[k] for k in ('actor','turn','action','value')}
                                      for r in b['trace'] if r['phase']=='main_action'],
                        terminal=to_primitive(engine._debug_state().result),
                        raw=str((RAW/f"continuation_{b['label']}.json").relative_to(a.ROOT))))
write_json(ART/'continuations.json',dict(scope='One deterministic adaptive continuation per first action, '
           'actual initial hidden cards/deck, both players use their own private DecisionRequest with S1024N4. '
           'Not a win-rate estimate or proof against optimal defense.',search=asdict(a.CFG),device='cpu',
           torch_threads=1,script_sha256=sha256(Path(__file__)),seconds=time.perf_counter()-started,branches=summary))
print('COMPLETE',time.perf_counter()-started,flush=True)
