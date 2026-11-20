"""Actual replay value curve and bounded decision diagnostics for seed100000."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'artifacts/web_blind_reservation_20261029'))
import analyze as a
from splendor.actions import ReserveDeck
from splendor.alphazero_recording import read_json,sha256,write_json
from splendor.catalog import load_default_catalog
from splendor.engine import GameEngine
from splendor.model import DecisionPhase
from splendor.native_policy import NativeBatchRunner,NativePUCTSearchPolicy
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.replay import _decode_state,to_primitive
from splendor.search_inference import FrozenSearchInference

ART=Path(__file__).resolve().parent
RAW=ROOT/'runtime'/ART.name
REPLAY=ROOT/'runtime/web_replays/6c2654de6ed04a1db9b77da7131cf2c3.json'


def positions(catalog):
    saved=GameEngine.restore(REPLAY)
    record=saved.replay_record
    engine=GameEngine.new(record.header.game_seed,record.header.first_player,catalog=catalog)
    rows=[];requests=[];states=[]
    for event in record.events:
        if event.automatic:
            continue
        req=engine.decision();state=engine._debug_state()
        assert engine.current_player==event.actor and event.action in req.legal_actions
        if req.observation.phase is DecisionPhase.MAIN_ACTION:
            row=dict(event=event.sequence,actor=event.actor,turn=state.completed_turns[event.actor]+1,
                     action=to_primitive(event.action),scores=[engine.observation_for(i).self_public.score for i in (0,1)],
                     players=[to_primitive(engine.observation_for(i).self_public) for i in (0,1)],
                     private_reserved=[[r.card_id for r in p.reserved_cards] for p in state.players])
            if isinstance(event.action,ReserveDeck):
                row['draw']=state.decks.for_tier(event.action.tier)[-1]
            rows.append(row);requests.append(req);states.append(to_primitive(state))
        engine.apply(event.action)
    assert engine.current_state_digest==saved.current_state_digest
    return rows,requests,states


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('curve','roots'))
    parser.add_argument('--events',type=int,nargs='+',default=[])
    args=parser.parse_args()
    a.identity_check()
    torch.set_num_threads(1)
    catalog=load_default_catalog()
    model=load_frozen_actor_critic_checkpoint(a.MODEL,device='cpu')
    evaluator=FrozenSearchInference(model,device='cpu')
    rows,requests,states=positions(catalog)
    RAW.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    identity=dict(replay=str(REPLAY.relative_to(ROOT)),replay_sha256=sha256(REPLAY),
                  checkpoint=str(a.MODEL.relative_to(ROOT)),checkpoint_sha256=sha256(a.MODEL),
                  script_sha256=sha256(Path(__file__)),source_plan=str((a.ART/'plan.json').relative_to(ROOT)),
                  search=asdict(a.CFG),device='cpu',torch_threads=1)
    if args.mode=='curve':
        for row,req,(logits,value) in zip(rows,requests,evaluator.policy_value(requests),strict=True):
            row.update(value=float(value),estimated_score_rate=(float(np.clip(value,-1,1))+1)/2,
                       greedy=to_primitive(req.legal_actions[int(np.argmax(logits))]))
            print(row['event'],'HA'[row['actor']],row['turn'],row['scores'],
                  round(row['estimated_score_rate']*100,2),row['action'],flush=True)
        write_json(ART/'curve.json',dict(**identity,scope='Uncalibrated value, evaluated at own actual MAIN_ACTION decisions only.',rows=rows))
        write_json(RAW/'states.json',[dict(event=row['event'],state=state) for row,state in zip(rows,states,strict=True)])
    else:
        selected=[(r,q) for r,q in zip(rows,requests,strict=True) if r['event'] in args.events]
        assert len(selected)==len(set(args.events))
        policy=NativePUCTSearchPolicy(model,config=a.CFG,catalog=catalog,evaluator=evaluator)
        runner=NativeBatchRunner(evaluator,catalog,capacity=128,buckets=(16,32,48,64),slots=1)
        results=a.search_all(policy,runner,[req for _,req in selected])
        compact=[];raw=[]
        for (row,req),result in zip(selected,results,strict=True):
            record=a.root_record(req,result)
            raw.append(dict(event=row['event'],root=record))
            compact.append(dict(event=row['event'],actor=row['actor'],turn=row['turn'],actual=row['action'],
                                chosen=record['chosen'],top=record['top'],reproduces_actual=record['chosen']==row['action']))
            print(compact[-1],flush=True)
        suffix='_'.join(map(str,args.events))
        write_json(RAW/f'roots_{suffix}.json',raw)
        write_json(ART/f'roots_{suffix}.json',dict(**identity,rows=compact,seconds=time.perf_counter()-started))


if __name__=='__main__':
    main()
