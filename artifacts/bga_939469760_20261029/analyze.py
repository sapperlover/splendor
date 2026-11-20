"""Greedy and native S1024N4 analysis on every captured human decision."""
from collections import Counter
from dataclasses import asdict
import hashlib
import html
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

from splendor.alphazero_recording import write_json
from splendor.catalog import load_default_catalog
from splendor.compact_trace import decode_request
from splendor.mcts_policy import MCTSConfig
from splendor.native_policy import NativeBatchRunner, NativePUCTSearchPolicy
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.search_inference import FrozenSearchInference
from splendor.replay import to_primitive

ROOT=Path(__file__).resolve().parents[2]
ART=Path(__file__).resolve().parent
RAW=ROOT/'runtime'/ART.name
MODEL=ROOT/'runs/entity_attention_policy140221_r31_r37_p60k_iteration_20261025/rounds/round-008/checkpoints/final.pt'
MODEL_SHA='852cb0420e94d77c146bc2330600f48d52f638e4cca96a5c1578e9dfe54940ec'
CFG=MCTSConfig(simulations=4096,determinizations=4,seed=0,root_noise_fraction=0.)
PLAYERS=['ZeroDimensional','blue_walker']
COLORS={'white':'白','blue':'蓝','green':'绿','red':'红','black':'黑','gold':'金'}


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def records():return [json.loads(s) for s in (RAW/'model_decisions.jsonl').read_text().splitlines()]


def label(a,cat):
    kind=a['__type__']
    if kind=='TakeDistinct':return '拿'+''.join(COLORS[c] for c in a['colors'])
    if kind=='TakePair':return '拿2'+COLORS[a['color']]
    if kind=='ReserveDeck':return f"盲留{a['tier']}级"
    if kind in ('Pay','ReturnTokens'):
        return ('支付' if kind=='Pay' else '退还')+(''.join(f'{COLORS[c]}{v}' for c,v in a['tokens'].items() if c!='__type__' and v) or '0')
    if kind in ('ReserveMarket','BuyReserved','BuyMarket'):
        card=cat.card(a['card_id'])
        verb={'ReserveMarket':'保留','BuyReserved':'买保留','BuyMarket':'买市场'}[kind]
        cost=''.join(f'{COLORS[c]}{v}' for c,v in to_primitive(card.cost).items() if c!='__type__' and v)
        return f'{verb}{card.card_id}（{COLORS[card.bonus.value]}+，{card.points}分，{cost}）'
    raise ValueError(kind)


def run():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    assert sha(MODEL)==MODEL_SHA
    cat=load_default_catalog();rows=records()
    assert len(rows)==86 and all(to_primitive(decode_request(r['request']).legal_actions[r['candidate_index']])==r['action'] for r in rows)
    source=RAW/'source'
    for p in (ROOT/'src/splendor').iterdir():
        if p.is_file() and p.suffix in ('.py','.cpp','.so'):
            dest=source/'src/splendor'/p.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    shutil.copytree(ROOT/'data',source/'data',dirs_exist_ok=True)
    model=load_frozen_actor_critic_checkpoint(MODEL,device='cpu')
    evaluator=FrozenSearchInference(model,device='cpu')
    requests=[decode_request(r['request']) for r in rows]
    tick=time.perf_counter()
    greedy=[]
    for row,(logits,value) in zip(rows,evaluator.policy_value(requests),strict=True):
        logits=np.asarray(logits,dtype=float);prior=np.exp(logits-logits.max());prior/=prior.sum()
        order=sorted(range(len(prior)),key=lambda i:(-prior[i],i));actual=row['candidate_index']
        greedy.append(dict(decision_index=row['decision_index'],choice=order[0],prior=prior.tolist(),
            actual_rank=order.index(actual)+1,value_actor_raw=float(value)))
    write_json(RAW/'greedy.json',greedy)
    plan=dict(table=939469760,checkpoint=str(MODEL.relative_to(ROOT)),checkpoint_sha256=MODEL_SHA,
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        input_sha256={str(p.relative_to(ROOT)):sha(p) for p in (RAW/'model_decisions.jsonl',Path(__file__),ART/'reconstruct.py',RAW/'capture_zero.json')},
        source_snapshot=str(source.relative_to(ROOT)),source_sha256={str(p.relative_to(source)):sha(p) for p in source.rglob('*') if p.is_file()},
        catalog_fingerprint=cat.data_fingerprint,model_config=asdict(model.config),search=asdict(CFG),
        search_label='S1024N4; total 4096',seed=0,root_noise=False,temperature=0,device='cpu',
        torch_threads=1,precision='float32',batch_roots=16,batch_capacity=64,candidate_buckets=[16,32,48,64],pipeline_slots=1,
        python=sys.version,torch=torch.__version__,command=sys.argv,greedy_seconds=time.perf_counter()-tick,
        scope='One seed per captured decision. Model opinion, not a proof of human errors or calibrated winning chances. No reconstructed hidden deck enters search.')
    write_json(ART/'plan.json',plan)
    policy=NativePUCTSearchPolicy(model,config=CFG,catalog=cat,evaluator=evaluator)
    runner=NativeBatchRunner(evaluator,cat,capacity=64,buckets=(16,32,48,64),slots=1)
    path=RAW/'search.json'
    saved=json.loads(path.read_text()) if path.exists() else []
    done={r['decision_index'] for r in saved}
    pending=[r for r in rows if len(r['request']['legal_actions'])>1 and r['decision_index'] not in done]
    started=time.perf_counter()
    for start in range(0,len(pending),16):
        group=pending[start:start+16]
        jobs=[policy.start_job(requests[r['decision_index']],add_root_noise=False,temperature=0.) for r in group]
        while any(not j.done for j in jobs):
            assert runner.run([j for j in jobs if not j.done])>0
        for row,job in zip(group,jobs,strict=True):
            result=job.result;v=np.asarray(result.root_visit_counts,dtype=np.int64)
            wv=np.asarray(result.per_world_root_visit_counts,dtype=np.int64)
            ws=np.asarray(result.per_world_root_value_sums,dtype=float)
            assert sum(v)==4096 and tuple(result.simulation_budgets)==(1024,)*4
            np.testing.assert_array_equal(v,wv.sum(axis=0))
            q=[float(s/n) if n else None for s,n in zip(ws.sum(axis=0),v)]
            saved.append(dict(decision_index=row['decision_index'],choice=result.chosen_candidate_index,
                visits=v.tolist(),per_world_visits=wv.tolist(),per_world_value_sums=ws.tolist(),q=q,
                prior=list(result.root_network_prior),max_tree_depth=result.max_tree_depth,
                simulation_budgets=list(result.simulation_budgets)))
        write_json(path,saved)
        live=dict(done=len(saved),expected=sum(len(r['request']['legal_actions'])>1 for r in rows),
            seconds=time.perf_counter()-started,stage='search')
        write_json(RAW/'live.json',live);print(json.dumps(live),flush=True)
    for name,digest in plan['input_sha256'].items():assert sha(ROOT/name)==digest,name
    assert sha(MODEL)==MODEL_SHA
    write_json(RAW/'live.json',dict(stage='complete',seconds=time.perf_counter()-started,roots=len(saved),inference=evaluator.stats.to_dict()))
    report()


def report():
    cat=load_default_catalog();rows=records();g=json.loads((RAW/'greedy.json').read_text())
    ss={r['decision_index']:r for r in json.loads((RAW/'search.json').read_text())}
    assert set(ss)=={r['decision_index'] for r in rows if len(r['request']['legal_actions'])>1}
    details=[]
    for r in rows:
        i=r['decision_index'];actual=r['candidate_index'];search=ss.get(i)
        actions=r['request']['legal_actions'];prior=g[i]['prior'];choice=search['choice'] if search else 0
        order=sorted(range(len(actions)),key=lambda j:(-(search['visits'][j] if search else 1),-prior[j],j))
        q=search['q'] if search else [None]
        details.append(dict(decision_index=i,move=r['bga_move'],actor=r['actor'],player=PLAYERS[r['actor']],turn=r['turn'],phase=r['phase'],
            candidates=len(actions),actual=label(r['action'],cat),greedy=label(actions[g[i]['choice']],cat),search=label(actions[choice],cat),
            greedy_agree=g[i]['choice']==actual,search_agree=choice==actual,greedy_rank=g[i]['actual_rank'],search_rank=order.index(actual)+1,
            actual_prior=prior[actual],actual_visit_share=search['visits'][actual]/4096 if search else 1.,
            value_actor=g[i]['value_actor_raw'],root_search_value=float(np.dot(search['visits'],[x or 0 for x in q])/4096) if search else None,
            actual_q=q[actual],chosen_q=q[choice],q_gap=q[choice]-q[actual] if q[choice] is not None and q[actual] is not None else None,
            actual_visits=search['visits'][actual] if search else 0,
            top3=[dict(action=label(actions[j],cat),visits=search['visits'][j],q=q[j]) for j in order[:3]] if search else []))
    write_json(RAW/'details.json',details)
    stats={}
    for phase in ('main_action','payment','return_tokens'):
        stats[phase]={}
        for actor in (0,1):
            group=[d for d in details if d['phase']==phase and d['actor']==actor]
            stats[phase][PLAYERS[actor]]=dict(n=len(group),choice_n=sum(d['candidates']>1 for d in group),
                greedy_top1=sum(d['greedy_agree'] for d in group),greedy_top3=sum(d['greedy_rank']<=3 for d in group),
                search_top1=sum(d['search_agree'] for d in group),search_top3=sum(d['search_rank']<=3 for d in group),
                prior_mean=float(np.mean([d['actual_prior'] for d in group])),search_share_mean=float(np.mean([d['actual_visit_share'] for d in group])))
    main=[d for d in details if d['phase']=='main_action']
    notable=sorted([d for d in main if d['q_gap'] is not None and not d['search_agree'] and d['actual_visits']>=32],key=lambda d:-d['q_gap'])[:10]
    summary=dict(complete=True,decisions=len(rows),searched=len(ss),simulations=4096*len(ss),stats=stats,
        main_search_changes_from_greedy=sum(d['search']!=d['greedy'] for d in main),
        notable_disagreements=[{k:d[k] for k in ('move','player','turn','actual','greedy','search','greedy_rank','search_rank','actual_visit_share','q_gap','actual_visits')} for d in notable],
        wall_seconds=json.loads((RAW/'live.json').read_text()).get('seconds'))
    write_json(ART/'summary.json',summary)
    lines=['# BGA #939469760 · greedy / S1024N4','',
        'ZeroDimensional 先手13分（11卡），blue_walker 后手16分（13卡）；各29回合。',
        '模型：policy在线第9轮；CPU FP32，种子0，native固定4世界，每世界1024次，总4096，无根噪声、T=0。',
        '搜索只接收行动者可见请求。原始回放、私有盲留身份和重建未来牌序不进入搜索。',
        '以下为模型意见：吻合度不是棋手实力，Q差不是已证明损失，单种子可能受抽世界影响。V和Q均未校准为胜率。','',
        '| 玩家 | 主行动 | greedy首选/前三吻合 | S1024N4首选/前三吻合 |',
        '|---|---:|---:|---:|']
    for player,s in stats['main_action'].items():lines.append(f"|{player}|{s['n']}|{s['greedy_top1']} / {s['greedy_top3']}|{s['search_top1']} / {s['search_top3']}|")
    lines+=['','## 全部主行动','', '| BGA步 | 玩家/回合 | 实际 | greedy | 搜索 | 实际访问份额 | Q差 |', '|---:|---|---|---|---|---:|---:|']
    for d in main:
        gap='—' if d['q_gap'] is None else f"{d['q_gap']:+.3f}"
        lines.append(f"|{d['move']}|{d['player']} {d['turn']}|{d['actual']}|{d['greedy']}|{d['search']}|{d['actual_visit_share']:.1%}|{gap}|")
    (ART/'results.md').write_text('\n'.join(lines)+'\n')
    blocks=[]
    for d in details:
        top=''.join(f"<li>{html.escape(t['action'])}：{t['visits']}次；Q={t['q'] if t['q'] is not None else '未访问'}</li>" for t in d['top3'])
        blocks.append(f"<details><summary>BGA {d['move']} · {d['player']} 第{d['turn']}回合 · {d['phase']} · 实际：{html.escape(d['actual'])}</summary><p>Greedy：{html.escape(d['greedy'])}；实际先验{d['actual_prior']:.2%}，排名{d['greedy_rank']}。<br>搜索：{html.escape(d['search'])}；实际访问{d['actual_visit_share']:.2%}，排名{d['search_rank']}。行动者V={d['value_actor']:+.3f}。</p><ul>{top}</ul></details>")
    (ART/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>BGA 939469760分析</title><style>body{max-width:1200px;margin:32px auto;font:16px/1.7 sans-serif;padding:0 20px;background:#f7f8fa}details{background:white;padding:12px;margin:10px 0;border:1px solid #ddd}summary{cursor:pointer}h1{font-size:26px}</style><h1>BGA #939469760：greedy 与 S1024N4</h1><p>ZeroDimensional 13:16 blue_walker。policy在线第9轮，单种子0、CPU FP32、无根噪声、4096次/根。模型意见不等于已证实的棋手失误，V/Q未校准。全部86个决策：</p>'+''.join(blocks))
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='report':report()
    else:run()
