"""S1024N16 follow-up, preserving the completed greedy/S1024N4 panel."""
from dataclasses import asdict
import html
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import splendor

from splendor.alphazero_recording import write_json
from splendor.catalog import load_default_catalog
from splendor.compact_trace import decode_request
from splendor.mcts_policy import MCTSConfig
from splendor.native_policy import NativeBatchRunner, NativePUCTSearchPolicy
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.search_inference import FrozenSearchInference

import analyze as base

ART=base.ART/'s1024n16'
RAW=base.RAW/'s1024n16'
TOTAL=16384
CFG=MCTSConfig(simulations=TOTAL,determinizations=16,seed=0,root_noise_fraction=0.)


def identity():
    plan=json.loads((base.ART/'plan.json').read_text())
    assert base.sha(base.MODEL)==base.MODEL_SHA
    for name,digest in plan['input_sha256'].items():assert base.sha(base.ROOT/name)==digest,name
    source=base.ROOT/plan['source_snapshot']
    for name,digest in plan['source_sha256'].items():assert base.sha(source/name)==digest,name
    assert Path(splendor.__file__).resolve().is_relative_to(source.resolve()),'Use the frozen source PYTHONPATH'
    assert load_default_catalog().data_fingerprint==plan['catalog_fingerprint']
    return plan


def run():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    original=identity()
    inputs={str(p.relative_to(base.ROOT)):base.sha(p) for p in
        (Path(__file__),base.ART/'plan.json',base.RAW/'model_decisions.jsonl',base.RAW/'greedy.json',base.RAW/'search.json')}
    plan=dict(table=939469760,checkpoint=str(base.MODEL.relative_to(base.ROOT)),checkpoint_sha256=base.MODEL_SHA,
        source_snapshot=original['source_snapshot'],source_identity_reference=str((base.ART/'plan.json').relative_to(base.ROOT)),
        input_sha256=inputs,search=asdict(CFG),label='S1024N16: 1024 per world x 16; total 16384',
        device='cpu',precision='float32',torch_threads=1,batch_roots=4,batch_capacity=64,pipeline_slots=1,
        candidate_buckets=[16,32,48,64],root_noise=False,temperature=0,seed=0,command=sys.argv,
        python=sys.version,torch=torch.__version__,git_head=original['git_head'],
        comparison='Same checkpoint, safe requests, source, seed and per-world budget as S1024N4; 4x total budget. Not an equal-compute comparison.')
    if (ART/'plan.json').exists():
        assert json.loads((ART/'plan.json').read_text())==plan,'Existing panel identity changed'
    else:write_json(ART/'plan.json',plan)
    rows=base.records();greedy=json.loads((base.RAW/'greedy.json').read_text())
    assert len(rows)==86
    cat=load_default_catalog()
    model=load_frozen_actor_critic_checkpoint(base.MODEL,device='cpu')
    evaluator=FrozenSearchInference(model,device='cpu')
    policy=NativePUCTSearchPolicy(model,config=CFG,catalog=cat,evaluator=evaluator)
    runner=NativeBatchRunner(evaluator,cat,capacity=64,buckets=(16,32,48,64),slots=1)
    path=RAW/'search.json';saved=json.loads(path.read_text()) if path.exists() else []
    done={r['decision_index'] for r in saved}
    pending=[r for r in rows if len(r['request']['legal_actions'])>1 and r['decision_index'] not in done]
    # First check four previously discussed disagreements, then all other roots.
    focus={8:0,21:1,26:2,40:3}
    pending.sort(key=lambda r:(focus.get(r['bga_move'],4) if r['phase']=='main_action' else 4,r['decision_index']))
    started=time.perf_counter();prior_seconds=sum(r['seconds'] for r in saved)
    for start in range(0,len(pending),4):
        group=pending[start:start+4];tick=time.perf_counter()
        jobs=[policy.start_job(decode_request(r['request']),add_root_noise=False,temperature=0.) for r in group]
        while any(not j.done for j in jobs):assert runner.run([j for j in jobs if not j.done])>0
        seconds=(time.perf_counter()-tick)/len(group)
        for row,job in zip(group,jobs,strict=True):
            result=job.result;v=np.asarray(result.root_visit_counts,dtype=np.int64)
            wv=np.asarray(result.per_world_root_visit_counts,dtype=np.int64)
            ws=np.asarray(result.per_world_root_value_sums,dtype=float)
            assert v.sum()==TOTAL and tuple(result.simulation_budgets)==(1024,)*16
            assert wv.shape==(16,len(v)) and np.all(wv.sum(axis=1)==1024)
            np.testing.assert_array_equal(v,wv.sum(axis=0))
            np.testing.assert_allclose(v/TOTAL,(wv/1024).mean(axis=0),rtol=0,atol=1e-14)
            np.testing.assert_allclose(result.root_network_prior,greedy[row['decision_index']]['prior'],rtol=0,atol=2e-6)
            assert result.chosen_candidate_index==max(range(len(v)),key=lambda i:(v[i],result.root_network_prior[i],-i))
            saved.append(dict(decision_index=row['decision_index'],move=row['bga_move'],phase=row['phase'],actor=row['actor'],
                choice=result.chosen_candidate_index,visits=v.tolist(),per_world_visits=wv.tolist(),per_world_value_sums=ws.tolist(),
                q=[float(s/n) if n else None for s,n in zip(ws.sum(axis=0),v)],prior=list(result.root_network_prior),
                simulation_budgets=list(result.simulation_budgets),max_tree_depth=result.max_tree_depth,
                transition_cap_hits=result.transition_cap_hits,seconds=seconds))
        write_json(path,saved)
        live=dict(stage='search',done=len(saved),expected=71,session_seconds=time.perf_counter()-started,
            total_search_seconds=prior_seconds+time.perf_counter()-started)
        write_json(RAW/'live.json',live);print(json.dumps(live),flush=True)
    identity()
    for name,digest in inputs.items():assert base.sha(base.ROOT/name)==digest,name
    write_json(RAW/'live.json',dict(stage='complete',roots=len(saved),total_search_seconds=sum(r['seconds'] for r in saved),
        session_seconds=time.perf_counter()-started,inference=evaluator.stats.to_dict()))
    report()


def report():
    identity();plan=json.loads((ART/'plan.json').read_text())
    for name,digest in plan['input_sha256'].items():assert base.sha(base.ROOT/name)==digest,name
    rows=base.records();g=json.loads((base.RAW/'greedy.json').read_text())
    old={r['decision_index']:r for r in json.loads((base.RAW/'search.json').read_text())}
    new={r['decision_index']:r for r in json.loads((RAW/'search.json').read_text())}
    expected={r['decision_index'] for r in rows if len(r['request']['legal_actions'])>1}
    assert set(old)==set(new)==expected and len(expected)==71
    cat=load_default_catalog();details=[];exact_prefix=0;prefix_l1=0
    for row in rows:
        i=row['decision_index'];actions=row['request']['legal_actions'];actual=row['candidate_index']
        versions={}
        for name,record,total in (('S1024N4',old.get(i),4096),('S1024N16',new.get(i),TOTAL)):
            if record:
                v=record['visits'];prior=record['prior'];q=record['q'];choice=record['choice']
                order=sorted(range(len(v)),key=lambda j:(-v[j],-prior[j],j))
                versions[name]=dict(choice=choice,chosen=base.label(actions[choice],cat),agree=choice==actual,
                    rank=order.index(actual)+1,actual_visits=v[actual],actual_share=v[actual]/total,
                    actual_q=q[actual],chosen_q=q[choice],q_gap=q[choice]-q[actual] if q[choice] is not None and q[actual] is not None else None,
                    root_mean_return=float(sum(n*(qj or 0) for n,qj in zip(v,q))/total),
                    top3=[dict(action=base.label(actions[j],cat),share=v[j]/total,q=q[j]) for j in order[:3]])
            else:versions[name]=dict(choice=0,chosen=base.label(actions[0],cat),agree=True,rank=1,actual_visits=0,actual_share=1.,actual_q=None,chosen_q=None,q_gap=None,root_mean_return=None,top3=[])
        if i in new:
            a=np.asarray(old[i]['per_world_visits']);b=np.asarray(new[i]['per_world_visits'])[:4]
            exact_prefix+=int(np.array_equal(a,b));prefix_l1+=int(np.abs(a-b).sum())
        details.append(dict(decision_index=i,move=row['bga_move'],turn=row['turn'],phase=row['phase'],actor=row['actor'],
            player=base.PLAYERS[row['actor']],actual=base.label(row['action'],cat),
            greedy=base.label(actions[g[i]['choice']],cat),greedy_choice=g[i]['choice'],greedy_rank=g[i]['actual_rank'],
            actual_prior=g[i]['prior'][actual],versions=versions))
    write_json(RAW/'details.json',details)
    summary=dict(complete=True,decisions=86,searched=71,total_simulations=71*TOTAL,
        seconds=json.loads((RAW/'live.json').read_text())['total_search_seconds'],stats={},
        prefix_exact_roots=exact_prefix,prefix_visit_L1=prefix_l1)
    main=[d for d in details if d['phase']=='main_action']
    for player in base.PLAYERS:
        group=[d for d in main if d['player']==player];stats={}
        stats['greedy']=dict(top1=sum(d['greedy_rank']==1 for d in group),top3=sum(d['greedy_rank']<=3 for d in group),n=len(group))
        for name in ('S1024N4','S1024N16'):
            stats[name]=dict(top1=sum(d['versions'][name]['agree'] for d in group),
                top3=sum(d['versions'][name]['rank']<=3 for d in group),n=len(group),
                mean_actual_share=float(np.mean([d['versions'][name]['actual_share'] for d in group])))
        summary['stats'][player]=stats
    changed=[d for d in main if d['versions']['S1024N4']['choice']!=d['versions']['S1024N16']['choice']]
    summary['changed_main_moves']=[d['move'] for d in changed]
    summary['n4_disagree_n16_agree']=[d['move'] for d in main if not d['versions']['S1024N4']['agree'] and d['versions']['S1024N16']['agree']]
    summary['n4_agree_n16_disagree']=[d['move'] for d in main if d['versions']['S1024N4']['agree'] and not d['versions']['S1024N16']['agree']]
    summary['zero_visit_actual_main']={name:[d['move'] for d in main if d['versions'][name]['actual_visits']==0] for name in ('S1024N4','S1024N16')}
    summary['focus']=[d for d in main if d['move'] in (8,21,26,40)]
    write_json(ART/'summary.json',summary)
    lines=['# BGA #939469760：S1024N16复核','',
        '固定policy在线第9轮、同一86个安全请求、相同冻结源码、种子0、无噪声、T=0。',
        '**S1024N16 = 每世界1024次 × 16世界 = 总16384次/根。** 相比S1024N4增加世界数且总算力增为4倍，不是等算力比较。',
        f"71个有选择根全部完成，共{71*TOTAL}次模拟，CPU FP32耗时{summary['seconds']:.2f}秒。15个强制支付点直接沿用唯一合法动作。",'',
        '| 玩家 | greedy首选/前三 | S1024N4首选/前三 | S1024N16首选/前三 |', '|---|---:|---:|---:|']
    for player,s in summary['stats'].items():lines.append('|'+player+'|'+'|'.join(f"{s[n]['top1']} / {s[n]['top3']}" for n in ('greedy','S1024N4','S1024N16'))+'|')
    lines+=['','各玩家分母均29。搜索首选按平均访问份额决定，未访问动作的Q留空。吻合度和Q差均为模型意见，不是已证明的人类走法损失。','',
        f"相对N4，主行动首选改变{len(changed)}/58；从不吻合转吻合BGA步{summary['n4_disagree_n16_agree']}，反向变化{summary['n4_agree_n16_disagree']}。",'',
        '| BGA步 | 玩家/回合 | 实际 | N4首选 | N16首选 | 实际N4份额 | 实际N16份额 | 实际N4/N16排名 |',
        '|---:|---|---|---|---|---:|---:|---:|']
    for d in main:
        a=d['versions']['S1024N4'];b=d['versions']['S1024N16']
        lines.append(f"|{d['move']}|{d['player']} {d['turn']}|{d['actual']}|{a['chosen']}|{b['chosen']}|{a['actual_share']:.2%}|{b['actual_share']:.2%}|{a['rank']}/{b['rank']}|")
    (ART/'results.md').write_text('\n'.join(lines)+'\n')
    blocks=[]
    for d in details:
        body=f"<p>Greedy：{html.escape(d['greedy'])}；实际先验{d['actual_prior']:.2%}，排名{d['greedy_rank']}。</p>"
        for name,v in d['versions'].items():
            body+=f"<h3>{name}</h3><p>首选：{html.escape(v['chosen'])}；实际访问{v['actual_visits']}次，份额{v['actual_share']:.2%}，排名{v['rank']}；实际Q={v['actual_q']}，首选Q={v['chosen_q']}。</p><ul>"
            body+=''.join(f"<li>{html.escape(t['action'])}：{t['share']:.2%}，Q={t['q']}</li>" for t in v['top3'])+'</ul>'
        blocks.append(f"<details><summary>BGA {d['move']} · {d['player']} 第{d['turn']}回合 · {d['phase']} · 实际：{html.escape(d['actual'])}</summary>{body}</details>")
    (ART/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>BGA N16复核</title><style>body{max-width:1150px;margin:30px auto;padding:0 20px;font:16px/1.7 sans-serif;background:#f7f8fa}details{background:white;border:1px solid #ddd;margin:10px 0;padding:12px}summary{cursor:pointer}h3{font-size:17px}</style><h1>BGA #939469760：S1024N4与S1024N16</h1><p>同第9轮模型、种子0、每世界1024次；总预算4096→16384。单盘单种子模型意见，不是棋手失误证明。</p>'+''.join(blocks))
    write_json(ART/'verification.json',dict(status='passed',source_and_input_sha_verified=True,
        only_safe_requests=True,roots=71,simulations=71*TOTAL,per_world_budget=[1024]*16,
        normalized_world_weights=True,prior_matches_greedy=True,n4_original_preserved=True,
        search_sha256=base.sha(RAW/'search.json'),raw_path=str(RAW.relative_to(base.ROOT))))
    print(json.dumps({k:v for k,v in summary.items() if k!='focus'},ensure_ascii=False),flush=True)


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='report':report()
    else:run()
