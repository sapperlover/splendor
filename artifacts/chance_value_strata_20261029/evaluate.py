"""Enumerate every conditional next-card outcome; evaluate only valid turn owners."""
from dataclasses import replace
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'artifacts/web_blind_reservation_20261029'))
import analyze as a
from splendor.alphazero_recording import read_json,sha256,write_json
from splendor.catalog import load_default_catalog
from splendor.engine import GameEngine
from splendor.model import DecisionPhase,Tier
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.replay import _decode_action,_decode_state
from splendor.search_inference import FrozenSearchInference

ART=Path(__file__).resolve().parent
RAW=ROOT/'runtime'/ART.name


def main():
    a.identity_check();torch.set_num_threads(1)
    plan=read_json(ART/'plan.json')
    assert sha256(ROOT/plan['cases'])==plan['cases_sha256']
    assert sha256(ROOT/plan['model12'])==plan['model12_sha256']
    cases=read_json(ROOT/plan['cases']);catalog=load_default_catalog()
    evaluator9=FrozenSearchInference(load_frozen_actor_critic_checkpoint(a.MODEL,device='cpu'),device='cpu')
    evaluator12=FrozenSearchInference(load_frozen_actor_critic_checkpoint(ROOT/plan['model12'],device='cpu'),device='cpu')
    rows=[];batch=[];indices=[];calls=0
    started=time.perf_counter()

    def flush():
        nonlocal calls
        if not batch:return
        v9=evaluator9.value_only(batch);v12=evaluator12.value_only(batch)
        for (r,i,sign),x,y in zip(indices,v9,v12,strict=True):
            rows[r]['value9'][i]=sign*x;rows[r]['value12'][i]=sign*y
        calls+=len(batch);batch.clear();indices.clear()

    for number,case in enumerate(cases):
        state=_decode_state(case['state']);tier=Tier(case['tier'])
        field={1:'tier_one',2:'tier_two',3:'tier_three'}[case['tier']]
        deck=list(getattr(state.decks,field));actions=[_decode_action(x) for x in case['actions']]
        original=GameEngine(state,catalog,None,RAW/'failures',mode='fast').decision()
        row={k:case[k] for k in ('id','game_id','index','stage','mode','tier','actor','max_score')}
        row.update(candidates=deck.copy(),actual_draw=deck[-1],value9=[None]*len(deck),value12=[None]*len(deck),
                   action_kind=case['actions'][0]['__type__'],anchor_actor=case['target_actor'])
        rows.append(row)
        baseline=None
        for i,card in enumerate(deck):
            changed=deck.copy();changed[i],changed[-1]=changed[-1],changed[i]
            branch=replace(state,decks=replace(state.decks,**{field:tuple(changed)}))
            engine=GameEngine(branch,catalog,None,RAW/'failures',mode='fast')
            assert engine.decision()==original
            for action in actions:
                assert action in engine.decision().legal_actions
                engine.apply(action)
            req=engine.decision()
            assert req.observation.phase is DecisionPhase.MAIN_ACTION
            assert engine.current_player==case['target_actor']
            assert (case['mode']!='blind') or engine.current_player==case['actor']
            anchor=engine._debug_state()
            invariant=(anchor.completed_turns,anchor.supply,
                       tuple((p.tokens,p.purchased_card_ids,p.noble_ids) for p in anchor.players))
            if baseline is None:baseline=invariant
            else:assert invariant==baseline
            sign=1 if engine.current_player==case['actor'] else -1
            batch.append(req);indices.append((number,i,sign))
            if len(batch)==128:flush()
        if (number+1)%100==0:
            print('cases',number+1,'evaluated',calls,'seconds',round(time.perf_counter()-started,1),flush=True)
    flush()
    assert all(all(v is not None for v in row['value9']+row['value12']) for row in rows)
    write_json(RAW/'outcomes.json',rows)
    write_json(ART/'execution.json',dict(cases=len(rows),outcomes=calls,network_evaluations=2*calls,
               model9_sha256=plan['model9_sha256'],model12_sha256=plan['model12_sha256'],
               device='cpu',torch_threads=1,inference='FP32 state-only value, batches<=128',
               invariant_checks='Every fixed action legal; same initial observation; same public scores, tokens, bonuses, turns after settlement; valid own-turn evaluation.',
               outcomes_sha256=sha256(RAW/'outcomes.json'),script_sha256=sha256(Path(__file__)),
               seconds=time.perf_counter()-started))
    print('COMPLETE',calls,time.perf_counter()-started,flush=True)


if __name__=='__main__':main()
