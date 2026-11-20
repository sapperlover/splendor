"""Show four value strata for the two previously analyzed human-game roots."""
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'artifacts/web_blind_reservation_20261029'))
import analyze as a
from splendor.actions import ReserveMarket,ReserveDeck
from splendor.alphazero_recording import read_json,sha256,write_json
from splendor.catalog import load_default_catalog
from splendor.model import DecisionPhase,Tier
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.replay import _decode_action,_decode_state
from splendor.search_inference import FrozenSearchInference

ART=Path(__file__).resolve().parent
a.identity_check();torch.set_num_threads(1)
catalog=load_default_catalog()
evaluator=FrozenSearchInference(load_frozen_actor_critic_checkpoint(a.MODEL,device='cpu'),device='cpu')
first=read_json(a.RAW/'roots.json')[0]
last=next(r['state'] for r in read_json(ROOT/'runtime/web_game_seed100000_20261029/states.json') if r['event']==69)
cases=[('first_blind3',_decode_state(first['state']),3,[ReserveDeck(Tier.THREE)]+[_decode_action(x) for x in first['fixed_human_reply']]),
       ('last_game_reserve046',_decode_state(last),2,[ReserveMarket('046')])]
result=[]
for label,state,tier,actions in cases:
    field={2:'tier_two',3:'tier_three'}[tier];deck=list(getattr(state.decks,field))
    requests=[];signs=[]
    for index,card in enumerate(deck):
        changed=deck.copy();changed[index],changed[-1]=changed[-1],changed[index]
        e=a.fast(replace(state,decks=replace(state.decks,**{field:tuple(changed)})),catalog)
        for action in actions:
            assert action in e.decision().legal_actions;e.apply(action)
        req=e.decision();assert req.observation.phase is DecisionPhase.MAIN_ACTION
        requests.append(req);signs.append(1 if e.current_player==state.current_player else -1)
    values=np.asarray(evaluator.value_only(requests))*signs
    groups=np.array_split(np.argsort(values,kind='stable'),4)
    rates=(np.clip(values,-1,1)+1)*50
    row=dict(label=label,scope='Actor-relative value at the same valid anchor as the population study.',
             candidates=[dict(card=card,value=float(value),score_rate_pp=float(rate)) for card,value,rate in zip(deck,values,rates,strict=True)],
             groups=[dict(cards=[deck[i] for i in group],probability=len(group)/len(deck),mean_score_rate_pp=float(rates[group].mean())) for group in groups])
    result.append(row)
    print(label,row['groups'],flush=True)
write_json(ART/'examples.json',dict(cases=result,script_sha256=sha256(Path(__file__)),checkpoint_sha256=a.MODEL_SHA))
