"""Separate luck, private knowledge and public development in the actual game."""
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import analyze as a
from splendor.actions import ReserveDeck
from splendor.alphazero_recording import read_json,sha256,write_json
from splendor.catalog import load_default_catalog
from splendor.engine import GameEngine
from splendor.model import ReservationSource,Tier
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.replay import _decode_state,to_primitive
from splendor.search_inference import FrozenSearchInference
from splendor.search_policy import _request_seed,determinize_request

ART=Path(__file__).resolve().parent
RAW=a.RAW/'advantage'
a.identity_check()
torch.set_num_threads(1)
catalog=load_default_catalog()
model=load_frozen_actor_critic_checkpoint(a.MODEL,device='cpu')
evaluator=FrozenSearchInference(model,device='cpu')
states={r['event']:_decode_state(r['state']) for r in read_json(RAW/'states.json')}
worlds=[]
for event in (55,59,61,64):
    engine=a.fast(states[event],catalog)
    request=engine.decision()
    assert engine.current_player==1
    sampled=[]
    for j in range(4):
        sampled_engine=determinize_request(request,seed=_request_seed(request,0,j),catalog=catalog,engine_mode='fast')
        # The Python determinizer maps the observing player to seat0.
        assert sampled_engine.current_player==0
        assert sampled_engine.decision()==request
        hidden=[r.card_id for r in sampled_engine._debug_state().players[1].reserved_cards
                if r.source is ReservationSource.DECK]
        sampled.append(hidden)
    worlds.append(dict(event=event,AI_turn=request.observation.own_completed_turns+1,
                       hypothetical_human_blind=sampled,hits_086=sum('086' in h for h in sampled)))
pre=states[54]
engine=a.fast(pre,catalog)
request=engine.decision()
assert engine.current_player==0 and pre.decks.tier_three[-1]=='086'
events=GameEngine.restore(a.REPLAY).replay_record.events
replies=[e.action for e in events if e.sequence in (55,56) and not e.automatic]
rows=[]
for candidate in pre.decks.tier_three:
    deck=list(pre.decks.tier_three)
    index=deck.index(candidate)
    deck[index],deck[-1]=deck[-1],deck[index]
    state=replace(pre,decks=replace(pre.decks,tier_three=tuple(deck)))
    changed=a.fast(state,catalog)
    assert changed.decision()==request
    changed.apply(ReserveDeck(Tier.THREE))
    # The AI cannot distinguish the actual private draw at its turn20 root.
    assert changed.decision()==a.fast(states[55],catalog).decision()
    for action in replies:
        assert action in changed.decision().legal_actions
        changed.apply(action)
    assert changed.current_player==0
    req=changed.decision()
    logits,value=evaluator.policy_value((req,))[0]
    rows.append(dict(card=a.describe(candidate,catalog),network_value=float(value),
                     estimated_score_rate=(float(np.clip(value,-1,1))+1)/2,
                     greedy=to_primitive(req.legal_actions[int(np.argmax(logits))])))
write_json(ART/'knowledge.json',dict(scope='Actual thirteen cards remaining in deck before human20 blind draw; '
           'AI private071 fixed. Swap top with each actual remaining card; fix actual AI20 buy069 and payment. '
           'Evaluate human at next own turn21. All thirteen AI20 input requests are identical.',
           hypothetical_world_scope='Python/C++ deterministic world construction was audited previously; '
           'worlds here are AI-root actor-relative (opponent=seat1).',
           actual_draw='086',conditional_draws=rows,AI_worlds=worlds,
           checkpoint_sha256=sha256(a.MODEL),replay_sha256=sha256(a.REPLAY),script_sha256=sha256(Path(__file__))))
print('worlds',worlds)
print('draws',[(r['card']['id'],round(r['estimated_score_rate']*100,2)) for r in rows])
