"""One recovered pre-purchase root; no game continuation or future-card input."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from splendor.actions import BuyMarket, ReserveMarket, TakeDistinct
from splendor.bga_live import DEFAULT_CHECKPOINT, ReservationHistory, request_from_snapshot, runtime_catalog
from splendor.compact_trace import decode_request
from splendor.mcts_policy import MCTSConfig
from splendor.model import Color
from splendor.native_policy import (ChanceWideningConfig, NativeBatchRunner,
    NativeChancePUCTSearchPolicy, NativePUCTSearchPolicy)
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.recommend import decision_identity, describe_action
from splendor.replay import to_primitive
from splendor.search_inference import FrozenSearchInference, InferenceConfig
from splendor.search_policy import determinize_request

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
RAW=ROOT/'runtime/bga_934713322_061_20261021/live_dom.json'
CHECKPOINT_SHA='fe7b71449b6ada28978c824d3e9e22b851fadf8bebf77261f961d85e27e394bd'
BUDGETS=(1024,4096,16384)
SEEDS=(0,1,2)
METHODS=(('native',1),('native',4),('chance',1),('cpw',1))


def save(name,value):
    path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def load_model():
    assert sha(DEFAULT_CHECKPOINT)==CHECKPOINT_SHA
    torch.set_num_threads(1)
    model=load_frozen_actor_critic_checkpoint(DEFAULT_CHECKPOINT,device='cpu')
    return model,FrozenSearchInference(model,device='cpu',config=InferenceConfig(mode='eager',actor_projection=False))


def prepare():
    raw=json.loads(RAW.read_text())
    snapshots=[r['snapshot'] for r in raw if r.get('snapshot')]
    assert len(snapshots)==1
    before=snapshots[0]
    assert before['move_text']=='19' and before['gamedatas']['gamestate']['active_player']==99846887
    save('root19_snapshot.json',before)
    events=[json.loads(line) for line in (ROOT/'runtime/bga_advisor/evaluations.jsonl').read_text().splitlines()
            if json.loads(line).get('table')==934713322 and int(json.loads(line)['move'])<=22]
    after=json.loads(json.dumps(before));gd=after['gamedatas']
    assert any(r['move']=='20' and r['event'].get('kind')=='reserveCard' and r['event'].get('card_id')=='45' for r in events)
    assert any(r['move']=='20' and r['event'].get('kind')=='revealCard' and r['event'].get('card_id')=='53' for r in events)
    row=gd['market']['row_2'];removed=row['cards'].pop('45')
    row['cards']['53']={**removed,'id':'53','type':'53','location_arg':0}
    row['count']-=1
    other=gd['players']['99846887'];other['nb_cards_hidden']['2']+=1
    other['coins']['G']+=1;gd['market']['pool']['G']-=1
    gd['gamestate']['active_player']=100878205
    after['move_text']='20'
    # Original DOM logs stop at move 19. Add only witnessed public reservations
    # up to move 20, never later identities from the terminal history file.
    history=ReservationHistory.start(after)
    for r in events:
        e=r['event']
        if int(r['move'])<=20 and e.get('kind')=='reserveCard' and e.get('public_reservation'):
            history.market_reserves[str(e['card_id'])]=str(e['actor_id'])
    request,catalog,_=request_from_snapshot(after,history)
    assert request.observation.phase.value=='main_action'
    assert BuyMarket('061') in request.legal_actions
    for seed in SEEDS:assert determinize_request(request,seed=seed,catalog=catalog,engine_mode='checked').decision()==request
    model,evaluator=load_model()
    logits,value=evaluator.policy_value((request,))[0]
    prior=np.exp(np.asarray(logits)-max(logits));prior/=prior.sum()
    quoted={BuyMarket('061'):.066,TakeDistinct((Color.BLUE,Color.GREEN,Color.RED)):.333,ReserveMarket('061'):.042}
    checks=[{'action':describe_action(a,catalog),'observed_rounded_probability':p,
             'recovered_probability':float(prior[request.legal_actions.index(a)])} for a,p in quoted.items()]
    assert all(abs(r['observed_rounded_probability']-r['recovered_probability'])<.0005 for r in checks),checks
    assert abs(value-.44121766090393066)<1e-5,value
    save('root_snapshot.json',after);save('root_history.json',history.to_dict())
    save('root_request.json',to_primitive(request));save('source_events.json',events)
    save('greedy.json',{'prior':prior.tolist(),'value':value,'expected_score':(value+1)/2,
         'ranking':sorted(range(len(prior)),key=lambda i:(-prior[i],i)),
         'actions':[describe_action(a,catalog) for a in request.legal_actions]})
    policy=NativePUCTSearchPolicy(model,evaluator=evaluator,catalog=catalog,
        config=MCTSConfig(simulations=1024,determinizations=1,seed=0,root_noise_fraction=0.))
    reference=policy.search(request,temperature=0.,add_root_noise=False)
    save('reference_S1024_seed0.json',asdict(reference))
    recovered_counts=[reference.root_visit_counts[request.legal_actions.index(a)] for a in quoted]
    save('input_validation.json',{'table':934713322,'move':20,'decision_sha256':decision_identity(request),
        'root_provenance':'captured replay move19 + witnessed opponent reserve45/reveal53/gold transfer, no later information',
        'raw_capture':str(RAW.relative_to(ROOT)),'raw_capture_sha256':sha(RAW),
        'checks':checks,'root_value':value,'live_root_value':.44121766090393066,
        'quoted_counts':[607,255,52],'recovered_reference_counts':recovered_counts,
        'exact_count_reproduction':recovered_counts==[607,255,52],
        'three_checked_hidden_world_reconstructions':True,'no_future_deck_order_in_input':True,
        'checkpoint':str(DEFAULT_CHECKPOINT),'checkpoint_sha256':CHECKPOINT_SHA,
        'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()})
    print(json.dumps({'prepared':True,'checks':checks,'value':value,'reference_counts':recovered_counts},ensure_ascii=False),flush=True)


def run():
    snapshot=json.loads((OUT/'root_snapshot.json').read_text())
    catalog,_,_=runtime_catalog(snapshot['gamedatas'])
    request=decode_request(json.loads((OUT/'root_request.json').read_text()))
    validation=json.loads((OUT/'input_validation.json').read_text())
    assert decision_identity(request)==validation['decision_sha256']
    model,evaluator=load_model()
    runner=NativeBatchRunner(evaluator,catalog,capacity=12,buckets=(16,32,48,64),slots=1)
    jobs=[]
    for budget in BUDGETS:
        for method,worlds in METHODS:
            for seed in SEEDS:
                name=f'{method}_S{budget}_N{worlds}'
                path=OUT/'search'/f'{name}_seed{seed}.json'
                if path.exists():continue
                config=MCTSConfig(simulations=budget,determinizations=worlds,seed=seed,root_noise_fraction=0.)
                cls=NativePUCTSearchPolicy if method=='native' else NativeChancePUCTSearchPolicy
                kw={'chance_widening':ChanceWideningConfig(k=.25,alpha=.5)} if method=='cpw' else {}
                policy=cls(model,evaluator=evaluator,catalog=catalog,config=config,**kw)
                job=policy.start_job(request,add_root_noise=False,temperature=0.)
                jobs.append((name,seed,job))
    save('plan.json',{'scope':'one pre-purchase root only, no continuation or strength evaluation',
        'budgets':BUDGETS,'seeds':SEEDS,'methods':METHODS,'cpw':{'k':.25,'alpha':.5},
        'total_searches':len(BUDGETS)*len(METHODS)*len(SEEDS),'total_simulations':sum(BUDGETS)*len(METHODS)*len(SEEDS),
        'execution':{'device':'cpu','threads':1,'native_batch_capacity':12,'pipeline_slots':1},
        'checkpoint_sha256':CHECKPOINT_SHA,'decision_sha256':validation['decision_sha256'],
        'source_sha256':{p:sha(ROOT/p) for p in ('src/splendor/native_search.cpp','src/splendor/native_policy.py',
            'src/splendor/features.py','src/splendor/search_inference.py')},'script_sha256':sha(Path(__file__))})
    pending=list(jobs);start=time.perf_counter();last=0.
    while pending:
        assert runner.run([j for _,_,j in pending])>0
        for name,seed,job in list(pending):
            if not job.done:continue
            result=asdict(job.result)
            visits=np.asarray(result['root_visit_counts']);world=np.asarray(result['per_world_root_visit_counts'])
            sums=np.asarray(result['per_world_root_value_sums']).sum(axis=0)
            assert sum(visits)==sum(result['simulation_budgets'])==result['simulations']
            np.testing.assert_array_equal(visits,world.sum(axis=0))
            assert np.all(np.abs(sums)<=visits+1e-6)
            result.update(panel=name,seed=seed,q=[float(v/n) if n else None for v,n in zip(sums,visits)],
                          diagnostics=job.core.chance_diagnostics())
            save(f'search/{name}_seed{seed}.json',result)
            pending.remove((name,seed,job))
        now=time.perf_counter()-start
        if now-last>10 or not pending:
            save('live.json',{'pending':len(pending),'elapsed_seconds':now})
            print(json.dumps({'pending':len(pending),'elapsed_seconds':round(now,1)}),flush=True);last=now
    summarize(request,catalog)


def summarize(request,catalog):
    greedy=json.loads((OUT/'greedy.json').read_text());actions=greedy['actions']
    target=request.legal_actions.index(BuyMarket('061'))
    summaries=[]
    for budget in BUDGETS:
        for method,worlds in METHODS:
            name=f'{method}_S{budget}_N{worlds}'
            rows=[json.loads((OUT/'search'/f'{name}_seed{seed}.json').read_text()) for seed in SEEDS]
            visits=np.asarray([r['root_visit_counts'] for r in rows]);means=visits.mean(axis=0)/budget
            ranking=sorted(range(len(means)),key=lambda i:(-means[i],-greedy['prior'][i],i))
            ranks=[sorted(range(len(actions)),key=lambda i:(-r['root_visit_counts'][i],-greedy['prior'][i],i)).index(target)+1 for r in rows]
            counts=Counter(actions[r['chosen_candidate_index']] for r in rows)
            summaries.append({'panel':name,'buy061_top1_seeds':sum(r['chosen_candidate_index']==target for r in rows),
                'buy061_rank_by_seed':ranks,'buy061_mean_visit_share':float(means[target]),
                'choice_counts':dict(counts),'consensus_choice':actions[ranking[0]],
                'top3':[{'action':actions[i],'visit_share':float(means[i]),
                         'q_by_seed':[r['q'][i] for r in rows]} for i in ranking[:3]],
                'buy061_q_by_seed':[r['q'][target] for r in rows]})
    save('summary.json',{'scope':'one root, three seeds per panel; visits are not win probabilities',
        'greedy_buy061_probability':greedy['prior'][target],
        'greedy_buy061_rank':greedy['ranking'].index(target)+1,'rows':summaries})
    lines=['# 买061之前的单局面搜索对比','',
           '仅分析桌934713322第20步主行动；不含支付比较，不进行续局或棋力赛。','',
           f"Greedy 首选：{actions[greedy['ranking'][0]]}；买061概率 {greedy['prior'][target]:.2%}，排名 {greedy['ranking'].index(target)+1}。",'',
           '| 搜索 | 买061首选种子数 | 买061名次（seed0/1/2） | 买061平均访问占比 | 按平均访问量首选 |',
           '|---|---:|---|---:|---|']
    for r in summaries:
        lines.append(f"| {r['panel']} | {r['buy061_top1_seeds']}/3 | {r['buy061_rank_by_seed']} | {r['buy061_mean_visit_share']:.2%} | {r['consensus_choice']} |")
    lines+=['','N4为相同总预算分成4个独立世界；chance每次模拟采样并共享历史树；CPW k=0.25、alpha=0.5。',
            '无根噪声、T=0，首选按访问量决定。访问占比和Q均不是校准胜率；不同方法一致不等于真实最优。','']
    (OUT/'results.md').write_text('\n'.join(lines))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('prepare','run'));a=p.parse_args()
    prepare() if a.stage=='prepare' else run()
