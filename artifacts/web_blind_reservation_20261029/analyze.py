"""Local root/continuation diagnostics for the saved human game; no strength claims."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

from splendor.actions import BuyMarket, BuyReserved, ReserveDeck, ReserveMarket
from splendor.alphazero_recording import read_json, sha256, write_json
from splendor.catalog import load_default_catalog
from splendor.contracts import DecisionRequest
from splendor.engine import GameEngine, replay_game
from splendor.invariants import assert_game_state
from splendor.mcts_policy import MCTSConfig
from splendor.model import DecisionPhase, ReservationSource, Tier
from splendor.native_policy import NativeBatchRunner, NativePUCTSearchPolicy
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.replay import _decode_action, _decode_state, to_primitive
from splendor.result import TerminalResult
from splendor.search_inference import FrozenSearchInference
from splendor.search_policy import _request_seed, determinize_request

ROOT = Path(__file__).resolve().parents[2]
ART = Path(__file__).resolve().parent
RAW = ROOT / 'runtime' / ART.name
REPLAY = ROOT / 'runtime/web_replays/ec42f9b0c8a043ddb0218bb72a2531ac.json'
MODEL = ROOT / 'runs/entity_attention_policy140221_r31_r37_p60k_iteration_20261025/rounds/round-008/checkpoints/final.pt'
MODEL_SHA = '852cb0420e94d77c146bc2330600f48d52f638e4cca96a5c1578e9dfe54940ec'
WHITE = {'076', '077', '078'}
CFG = MCTSConfig(simulations=4096, determinizations=4, seed=0, root_noise_fraction=0.)


def fast(state, catalog):
    assert_game_state(state, catalog)
    return GameEngine(state, catalog, None, RAW / 'failures', mode='fast')


def describe(card, catalog):
    c = catalog.card(card)
    return dict(id=card, bonus=c.bonus.value, points=c.points, cost=to_primitive(c.cost))


def snapshot(engine):
    return to_primitive(engine._debug_state())


def knowledge_swap(state, candidate, catalog):
    """Condition this draw by swapping the actual top with an unseen card.

    The candidate may be in the human's hidden reservation (third blind only).
    Then that hidden identity is swapped as well; AI's pre-draw request is fixed.
    """
    deck = list(state.decks.tier_three)
    original = deck[-1]
    if candidate in deck:
        index = deck.index(candidate)
        deck[index], deck[-1] = deck[-1], deck[index]
        players = state.players
    else:
        hidden = list(state.players[0].reserved_cards)
        index = next(i for i,r in enumerate(hidden) if r.source is ReservationSource.DECK and r.card_id == candidate)
        hidden[index] = replace(hidden[index], card_id=original)
        players = (replace(state.players[0], reserved_cards=tuple(hidden)), state.players[1])
        deck[-1] = candidate
    swapped = replace(state, decks=replace(state.decks,tier_three=tuple(deck)),players=players)
    assert_game_state(swapped,catalog)
    assert fast(swapped,catalog).decision() == fast(state,catalog).decision()
    return swapped


def prepare():
    RAW.mkdir(parents=True,exist_ok=True)
    assert sha256(MODEL) == MODEL_SHA
    catalog=load_default_catalog()
    original=GameEngine.restore(REPLAY)
    assert replay_game(original.replay_record).current_state_digest == original.current_state_digest
    record=original.replay_record
    engine=GameEngine.new(record.header.game_seed,record.header.first_player,catalog=catalog)
    roots=[]
    actual=[]
    for event in record.events:
        if event.automatic:
            continue
        request=engine.decision()
        if event.actor == 1 and isinstance(event.action,ReserveDeck):
            state=engine._debug_state()
            number=len(roots)+1
            next_ai=next(e.sequence for e in record.events[event.sequence+1:] if e.actor==1 and not e.automatic)
            replies=[to_primitive(e.action) for e in record.events[event.sequence+1:next_ai] if not e.automatic]
            assert all(e.actor==0 for e in record.events[event.sequence+1:next_ai])
            own_known={r.card_id for r in state.players[1].reserved_cards}
            public=set(state.market.all_cards()) | {x for p in state.players for x in p.purchased_card_ids} | {r.card_id for p in state.players for r in p.reserved_cards if r.source is ReservationSource.MARKET}
            candidates=[c.card_id for c in catalog.cards if c.tier is Tier.THREE and c.card_id not in public|own_known]
            tops=[determinize_request(request,seed=_request_seed(request,0,j),catalog=catalog,engine_mode='fast')._debug_state().decks.tier_three[-1] for j in range(4)]
            root=dict(number=number,event=event.sequence,turn=state.completed_turns[1]+1,state=to_primitive(state),actual_draw=state.decks.tier_three[-1],candidates=candidates,fixed_human_reply=replies,sampled_world_tops=tops,white_hits=[x for x in candidates if x in WHITE])
            roots.append(root)
        action=event.action
        row=dict(event=event.sequence,actor=event.actor,turn=engine._debug_state().completed_turns[event.actor]+1,phase=event.phase.value,action=to_primitive(action))
        if isinstance(action,ReserveDeck):row['draw']=engine._debug_state().decks.for_tier(action.tier)[-1]
        engine.apply(action)
        row['after_scores']=[engine.observation_for(i).self_public.score for i in (0,1)]
        actual.append(row)
    write_json(RAW/'roots.json',roots)
    write_json(RAW/'actual.json',actual)
    source=RAW/'source'
    for path in sorted((ROOT/'src/splendor').iterdir()):
        if path.is_file() and path.suffix in ('.py','.cpp','.so'):
            target=source/'src/splendor'/path.name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,target)
    shutil.copytree(ROOT/'data',source/'data',dirs_exist_ok=True)
    target=source/'artifacts'/ART.name/'analyze.py'
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(Path(__file__),target)
    files={str(p.relative_to(source)):sha256(p) for p in source.rglob('*') if p.is_file()}
    plan=dict(checkpoint=str(MODEL.relative_to(ROOT)),checkpoint_sha256=MODEL_SHA,replay=str(REPLAY.relative_to(ROOT)),replay_sha256=sha256(REPLAY),catalog_fingerprint=catalog.data_fingerprint,git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source_snapshot=str(source.relative_to(ROOT)),source_sha256=files,search=asdict(CFG),device='cpu',torch_threads=4,batch_capacity=128,candidate_buckets=[16,32,48,64],root_noise=False,temperature=0,continuation=dict(human='same checkpoint greedy after one fixed actual human reply',ai='S1024N4',first_blind_ai_turns=12,other_blind_ai_turns=6,remaining_deck='swap selected draw with actual top; no reshuffle',warning='One deterministic continuation per card, not a win rate or an adaptive human response'),roots=[{k:v for k,v in r.items() if k not in ('state','fixed_human_reply')} for r in roots])
    write_json(ART/'plan.json',plan)
    print(json.dumps({'prepared':True,'candidate_counts':[len(r['candidates']) for r in roots],'world_tops':[r['sampled_world_tops'] for r in roots]}),flush=True)


def identity_check():
    plan=read_json(ART/'plan.json')
    assert sha256(MODEL)==plan['checkpoint_sha256'] and sha256(REPLAY)==plan['replay_sha256']
    for path,digest in plan['source_sha256'].items():
        assert sha256(ROOT/plan['source_snapshot']/path)==digest,path
    assert load_default_catalog().data_fingerprint==plan['catalog_fingerprint']


def root_record(request,result):
    count=np.array(result.root_visit_counts,dtype=np.int64)
    sums=np.asarray(result.per_world_root_value_sums,dtype=float)
    visits=np.asarray(result.per_world_root_visit_counts,dtype=np.int64)
    if result.searchable:
        assert int(count.sum())==4096 and np.array_equal(visits.sum(axis=0),count)
        q=np.divide(sums.sum(axis=0),count,out=np.zeros(len(count)),where=count>0)
    else:q=np.zeros(len(count))
    order=sorted(range(len(count)),key=lambda i:(-count[i],-result.root_network_prior[i],i))
    return dict(chosen=to_primitive(request.legal_actions[result.chosen_candidate_index]),chosen_index=result.chosen_candidate_index,prior=list(result.root_network_prior),visits=count.tolist(),per_world_visits=visits.tolist(),per_world_value_sums=sums.tolist(),top=[dict(action=to_primitive(request.legal_actions[i]),visits=int(count[i]),q=float(q[i]),prior=result.root_network_prior[i]) for i in order[:5]],max_tree_depth=result.max_tree_depth)


def search_all(policy,runner,requests):
    jobs=[policy.start_job(request,add_root_noise=False,temperature=0.) for request in requests]
    while any(not job.done for job in jobs):
        assert runner.run([job for job in jobs if not job.done])>0
    return [job.result for job in jobs]


def run(device='cpu'):
    identity_check()
    torch.set_num_threads(4 if device=='cpu' else 1)
    if device=='cuda':
        torch.backends.cuda.matmul.allow_tf32=False
    catalog=load_default_catalog()
    model=load_frozen_actor_critic_checkpoint(MODEL,device=device)
    evaluator=FrozenSearchInference(model,device=device)
    policy=NativePUCTSearchPolicy(model,config=CFG,catalog=catalog,evaluator=evaluator)
    runner=NativeBatchRunner(evaluator,catalog,capacity=128,buckets=(16,32,48,64),slots=1)
    roots=read_json(RAW/'roots.json')
    started=time.perf_counter()
    driver=RAW/'drivers'/f'{sha256(Path(__file__))}.py'
    driver.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(Path(__file__),driver)
    write_json(RAW/f'execution_{time.time_ns()}.json',dict(device=device,precision='float32',tf32=False,driver=str(driver.relative_to(ROOT)),driver_sha256=sha256(driver),command=sys.argv,torch_threads=torch.get_num_threads()))
    requests=[fast(_decode_state(r['state']),catalog).decision() for r in roots]
    originals=[]
    if (RAW/'original_roots.json').exists():
        originals=read_json(RAW/'original_roots.json')
    else:
        for root,request,result in zip(roots,requests,search_all(policy,runner,requests),strict=True):
            row=root_record(request,result)
            row.update(number=root['number'],event=root['event'],sampled_world_tops=root['sampled_world_tops'],reproduces_actual=isinstance(request.legal_actions[result.chosen_candidate_index],ReserveDeck) and request.legal_actions[result.chosen_candidate_index].tier is Tier.THREE)
            originals.append(row)
        write_json(RAW/'original_roots.json',originals)
    print(json.dumps({'stage':'original_roots','seconds':time.perf_counter()-started,'all_reproduced':all(x['reproduces_actual'] for x in originals)}),flush=True)
    branches=[]
    for root in roots:
        state=_decode_state(root['state'])
        for candidate in root['candidates']:
            changed=knowledge_swap(state,candidate,catalog)
            engine=fast(changed,catalog)
            engine.apply(ReserveDeck(Tier.THREE))
            assert engine._debug_state().players[1].reserved_cards[-1].card_id==candidate
            for raw in root['fixed_human_reply']:
                request=engine.decision();action=_decode_action(raw)
                assert engine.current_player==0 and action in request.legal_actions
                engine.apply(action)
            assert engine.current_player==1
            request=engine.decision()
            logits,value=evaluator.policy_value((request,))[0]
            branch=dict(id=f"blind{root['number']}_{candidate}",number=root['number'],draw=candidate,card=describe(candidate,catalog),group='white_633_7_73' if candidate in WHITE else 'other',is_actual=candidate==root['actual_draw'],start_turn=engine._debug_state().completed_turns[1],limit=12 if root['number']==1 else 6,engine=engine,initial_value=value,initial_greedy=to_primitive(request.legal_actions[int(np.argmax(logits))]),initial_state=snapshot(engine),trace=[],main_actions=[],finished=False,device=device)
            saved_path=RAW/'branches'/f"{branch['id']}.json"
            if saved_path.exists():
                saved=read_json(saved_path)
                assert saved['draw']==candidate and saved['number']==root['number']
                if not saved['finished']:
                    assert saved.get('device','cpu')==device
                branch.update(saved)
                branch['device']=saved.get('device','cpu')
                branch['engine']=fast(_decode_state(saved['final_state']),catalog)
            branches.append(branch)
    cycle=0
    while any(not b['finished'] for b in branches):
        waiting=[]
        for branch in branches:
            if branch['finished']:continue
            engine=branch['engine']
            while True:
                request=engine.decision()
                if isinstance(request,TerminalResult) or engine._debug_state().completed_turns[1]-branch['start_turn']>=branch['limit']:
                    branch['finished']=True
                    break
                if engine.current_player==1:
                    waiting.append((branch,request))
                    break
                logits,value=evaluator.policy_value((request,))[0]
                action=request.legal_actions[int(np.argmax(logits))]
                branch['trace'].append(dict(actor=0,turn=request.observation.own_completed_turns+1,phase=request.observation.phase.value,action=to_primitive(action),network_value=value))
                engine.apply(action)
        if not waiting:break
        results=search_all(policy,runner,[req for _,req in waiting])
        for (branch,request),result in zip(waiting,results,strict=True):
            engine=branch['engine']
            logits,value=evaluator.policy_value((request,))[0]
            row=dict(actor=1,turn=request.observation.own_completed_turns+1,phase=request.observation.phase.value,network_value=value,search=root_record(request,result))
            action=request.legal_actions[result.chosen_candidate_index]
            row['action']=to_primitive(action)
            if isinstance(action,ReserveDeck):row['draw']=engine._debug_state().decks.for_tier(action.tier)[-1]
            engine.apply(action)
            row['after']=dict(scores=[engine.observation_for(i).self_public.score for i in (0,1)],ai_bonuses=to_primitive(engine.observation_for(1).self_public.bonuses),ai_tokens=to_primitive(engine._debug_state().players[1].tokens))
            branch['trace'].append(row)
            if request.observation.phase is DecisionPhase.MAIN_ACTION:branch['main_actions'].append(dict(turn=row['turn'],action=row['action'],draw=row.get('draw')))
        cycle+=1
        for branch in branches:
            save_branch(branch,catalog)
        status=dict(stage='continuations',cycle=cycle,active=sum(not b['finished'] for b in branches),done=sum(b['finished'] for b in branches),total=len(branches),seconds=time.perf_counter()-started,runner_rows=runner.rows,runner_batches=runner.batches)
        write_json(RAW/'live.json',status)
        print(json.dumps(status),flush=True)
    for branch in branches:save_branch(branch,catalog)
    write_json(RAW/'complete.json',dict(seconds=time.perf_counter()-started,branches=len(branches),runner_rows=runner.rows,runner_batches=runner.batches))
    identity_check()
    summarize()


def save_branch(branch,catalog):
    engine=branch['engine']
    state=engine._debug_state()
    ai=engine.observation_for(1)
    terminal=state.result
    record={k:v for k,v in branch.items() if k not in ('engine','written')}
    bought=branch['draw'] in state.players[1].purchased_card_ids
    record.update(final_state=snapshot(engine),final_scores=[engine.observation_for(i).self_public.score for i in (0,1)],final_ai_bonuses=to_primitive(ai.self_public.bonuses),ai_turns=state.completed_turns[1]-branch['start_turn'],blind_card_bought=bought,terminal=to_primitive(terminal) if terminal else None,draw_still_reserved=any(r.card_id==branch['draw'] for r in state.players[1].reserved_cards))
    assert_game_state(state,catalog)
    write_json(RAW/'branches'/f"{branch['id']}.json",record)


def summarize():
    rows=[read_json(p) for p in sorted((RAW/'branches').glob('*.json'))]
    compact=[]
    for r in rows:
        first=next(t for t in r['trace'] if t['actor']==1)
        compact.append({**{k:r[k] for k in ('id','number','draw','card','group','is_actual','initial_value','initial_greedy','main_actions','final_scores','final_ai_bonuses','ai_turns','blind_card_bought','draw_still_reserved','terminal')},'device':r.get('device','cpu'),'first_search':first['search']['top'],'source':str((RAW/'branches'/f"{r['id']}.json").relative_to(ROOT))})
    original_roots=[{k:v for k,v in r.items() if k in ('number','event','sampled_world_tops','reproduces_actual','chosen','top')} for r in read_json(RAW/'original_roots.json')]
    write_json(ART/'summary.json',dict(plan='plan.json',execution=read_json(RAW/'complete.json') if (RAW/'complete.json').exists() else None,original_roots=original_roots,branches=compact))
    print(json.dumps({'stage':'summary','branches':len(compact)}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','run','summarize'])
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=parser.parse_args()
    if args.command=='run':run(args.device)
    else:{'prepare':prepare,'summarize':summarize}[args.command]()
