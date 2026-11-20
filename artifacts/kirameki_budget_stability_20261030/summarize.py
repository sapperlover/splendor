"""Aggregate the frozen four-game budget matrix; raw root arrays stay in runtime."""
from collections import Counter
from datetime import datetime
from itertools import combinations
from pathlib import Path
from zoneinfo import ZoneInfo
import csv
import hashlib
import json

ROOT=Path(__file__).resolve().parents[2]
ART=Path(__file__).resolve().parent
RAW=ROOT/'runtime/kirameki_budget_matrix_20261030'
BUDGETS=(4096,8192,16384,32768)
PAIRS=list(combinations(BUDGETS,2))

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def short(action):
    kind=action['__type__']
    if kind in ('BuyMarket','BuyReserved'):return ('买市场' if kind=='BuyMarket' else '买保留')+action['card_id']
    if kind=='ReserveMarket':return '保留'+action['card_id']
    if kind=='ReserveDeck':return f'{action["tier"]}级盲留'
    if kind in ('Pay','ReturnTokens'):
        return ('付' if kind=='Pay' else '还')+''.join(c+str(action['tokens'][key]) for c,key in zip('白蓝绿红黑金',('white','blue','green','red','black','gold')) if action['tokens'][key])
    if kind=='TakeDistinct':return '拿'+''.join({'white':'白','blue':'蓝','green':'绿','red':'红','black':'黑'}[c] for c in action['colors'])
    if kind=='TakePair':return '拿'+{'white':'白','blue':'蓝','green':'绿','red':'红','black':'黑'}[action['color']]+'2'
    if kind=='ClaimNoble':return '贵族'+action['noble_id']
    raise ValueError(kind)
def count(items,low,high):return sum(row['choices'][low]!=row['choices'][high] for row in items)
def aggregate(items):
    n=len(items)
    return dict(total=n,any_change=sum(len(set(r['choices'].values()))>1 for r in items),
        pairwise={f'{a}_vs_{b}':dict(changed=count(items,a,b),total=n,rate=count(items,a,b)/n if n else None) for a,b in PAIRS},
        returned_to_4096=sum(r['choices'][4096]==r['choices'][32768] and len(set(r['choices'].values()))>1 for r in items))
def grouping(items,low,high):
    result={}
    for name,predicate in (
        ('top1_lt_60',lambda p:p['top1_share']<.60),
        ('top1_60_to_85',lambda p:.60<=p['top1_share']<.85),
        ('top1_ge_85',lambda p:p['top1_share']>=.85),
        ('proposed_trigger',lambda p:p['top1_share']<.60 or p['top2_ratio']>=.70),
        ('not_triggered',lambda p:p['top1_share']>=.60 and p['top2_ratio']<.70)):
        rows=[r for r in items if predicate(r['points'][low])]
        result[name]=dict(total=len(rows),changed=count(rows,low,high),
            rate=count(rows,low,high)/len(rows) if rows else None)
    return result

manifest=json.loads((RAW/'manifest.json').read_text())
rows=[];raw_files=[];reused=single=reruns=0
for spec in manifest['cases']:
    path=RAW/'results'/(spec['case']+'.json')
    r=json.loads(path.read_text());assert r['spec']==spec
    assert len(r['checkpoints'])==4
    parent=json.loads((RAW/spec['input_directory']/'recommendation.json').read_text())
    original=parent if spec['kind']=='main' else parent['followup']
    points={p['budget']:p for p in r['checkpoints']};assert set(points)==set(BUDGETS)
    unique=r.get('unique_legal_action',False)
    if unique:
        single+=1;assert spec['candidate_count']==1
    else:
        assert r['original_8192_visits_reproduced'] and r['original_8192_q_sums_match_within_1e_5']
        assert r['final_native_result_verified']
        assert r['checkpoint_sha256']==manifest['checkpoint_sha256']
        ordered=sorted(points[8192]['candidates'],key=lambda c:c['candidate_index'])
        assert [c['visits'] for c in ordered]==original['per_world_root_visit_counts'][0]
        assert max(abs(c['value_sum']-original['per_world_root_value_sums'][0][i]) for i,c in enumerate(ordered))<1e-5
        reused+=bool(r.get('reused_verified_result'))
        reruns+=len(r.get('exact_budget_reruns',[]))
    snapshot=json.loads((RAW/spec['input_directory']/'snapshot.json').read_text())
    row=dict(case=spec['case'],game=spec['game'],round=spec['round'],move_number=len(snapshot['moves'])+1,kind=spec['kind'],phase=spec['phase'],
        candidate_count=spec['candidate_count'],choices={},points={})
    for budget,point in points.items():
        candidates=point['candidates'];assert candidates[0]['candidate_index']==point['chosen_candidate_index']
        assert point['completed_simulations']==(0 if unique else budget)
        assert sum(c['visits'] for c in candidates)==point['completed_simulations']
        assert sorted(c['candidate_index'] for c in candidates)==list(range(spec['candidate_count']))
        c=candidates[0]
        row['choices'][budget]=c['candidate_index']
        row['points'][budget]=dict(choice=short(c['action']),description=c['description'],top1_share=c['share'],q=c['q'],
            runner_up=short(candidates[1]['action']) if len(candidates)>1 else None,
            top2_share=candidates[1]['share'] if len(candidates)>1 else None,
            top2_ratio=candidates[1]['visits']/c['visits'] if len(candidates)>1 and c['visits'] else None,
            terminal_hits=point.get('stats',{}).get('terminal_hits'),
            acquisition=point.get('acquisition','unique_legal_action'))
    rows.append(row);raw_files.append(dict(path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,sha256=sha(path)))
assert len(rows)==168
main=[r for r in rows if r['kind']=='main'];conditional=[r for r in rows if r['kind']=='followup']
assert len(main)==115 and len(conditional)==53
active_conditional=[r for r in conditional if r['candidate_count']>1]
summary=dict(schema='kirameki-budget-stability-summary-v1',status='complete',
    completed_at=datetime.now(ZoneInfo('Asia/Singapore')).isoformat(),games=[3,4,5,6],budgets=BUDGETS,
    execution_wall_seconds=json.loads((RAW/'progress.json').read_text())['wall_seconds'],workers=manifest['workers'],
    checkpoint_sha256=manifest['checkpoint_sha256'],main=aggregate(main),conditional=aggregate(conditional),
    conditional_with_choice=aggregate(active_conditional),
    by_game={str(game):dict(main=aggregate([r for r in main if r['game']==game]),
        conditional=aggregate([r for r in conditional if r['game']==game])) for game in manifest['requested_games']},
    main_closeness={f'{a}_vs_{b}':grouping(main,a,b) for a,b in ((4096,8192),(8192,16384),(16384,32768),(4096,32768),(8192,32768))},
    validation=dict(all_searching_roots_reproduced_8192=len(main)+len(active_conditional),
        reused_verified_roots=reused,unique_legal_actions=single,terminal_crossed_checkpoints_rerun_at_exact_budget=reruns),
    raw_location=str(RAW.relative_to(ROOT)),raw_manifest_sha256=sha(RAW/'manifest.json'),
    raw_results_count=len(raw_files),raw_results_bytes=sum(x['bytes'] for x in raw_files),
    limitations=['four specific human games, only saved advisor decision points',
        'one seed; visit-choice stability, not independently verified move quality or playing strength',
        'conditional decisions hold the original main action fixed; not a counterfactual full game',
        'near-saturated value estimates may accompany changing visit winners; they are not independently proven wins/losses'])
(ART/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
(RAW/'derived_rows.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
(RAW/'result_inventory.json').write_text(json.dumps(raw_files,ensure_ascii=False,indent=2)+'\n')
with (RAW/'choices.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.writer(f);writer.writerow(['case','game','round','move_number','kind','phase',*[f'{b}_{field}' for b in BUDGETS for field in ('choice','share','q')]])
    for row in rows:writer.writerow([row[k] for k in ('case','game','round','move_number','kind','phase')]+[row['points'][b][field] for b in BUDGETS for field in ('choice','top1_share','q')])
lines=['# 四盘完整推荐点的随机节点预算稳定性','',
    '2026-10-30，已完成。用户指定第3、4、5、6盘，115个本人主行动点全部纳入，另列53个固定原主行动下的支付／归还决策（31个有选择，22个唯一合法）。只涵盖本机保存的推荐点，不含未保存的终局或对方私有决策；各预算在同一历史观察上判招，不沿改选路线重走整盘。','',
    '固定policy在线第9轮权重，SHA256 `'+manifest['checkpoint_sha256']+'`。S4096N1／S8192N1／S16384N1／S32768N1，总预算分别4096／8192／16384／32768；随机节点共享树，无CPW、无根噪声，seed=0。CPU8进程、每进程1线程、单根推理batch1；冻结源码、数据目录及原生扩展见runtime清单。','',
    '每根优先在一棵32768次树上记录精确中间检查点；若原生搜索连续结算终局而越过某档，则按相同seed单独精确补跑该预算。所有有选择根的原8192次全部访问数精确复现，收益分子绝对误差小于1e-5。','',
    '## 主行动改首选数量','',
    '| 对比 | 改首选／115 | 比例 |','| --- | ---: | ---: |']
for a,b in PAIRS:
    p=summary['main']['pairwise'][f'{a}_vs_{b}'];lines.append(f'| {a} → {b} | {p["changed"]}/115 | {p["rate"]:.2%} |')
lines+=['',f'四档中至少改变一次的主行动点：**{summary["main"]["any_change"]}/115**。首末一致、但中间曾改选的点：{summary["main"]["returned_to_4096"]}个。统计的是检查点首选变化，不追踪两个检查点之间短暂的访问领先切换。','',
    '| 对局 | 主行动点 | 4096→8192 | 8192→16384 | 16384→32768 | 4096→32768 | 四档至少一变 |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
for game in manifest['requested_games']:
    s=summary['by_game'][str(game)]['main'];p=s['pairwise']
    lines.append(f'| 第{game}盘 | {s["total"]} | {p["4096_vs_8192"]["changed"]} | {p["8192_vs_16384"]["changed"]} | {p["16384_vs_32768"]["changed"]} | {p["4096_vs_32768"]["changed"]} | {s["any_change"]} |')
lines+=['','## 接近程度与后续改选','',
    '以下分组使用较低预算当时的首选访问份额；这些不是胜率或置信度。提议触发条件为首选份额<60%，或次选访问数／首选访问数≥70%。','',
    '| 对比 | 首选<60% | 60%–85% | 首选≥85% | 提议触发组 | 未触发组 |','| --- | ---: | ---: | ---: | ---: | ---: |']
for pair,data in summary['main_closeness'].items():
    cells=[]
    for name in ('top1_lt_60','top1_60_to_85','top1_ge_85','proposed_trigger','not_triggered'):
        p=data[name];cells.append(f'{p["changed"]}/{p["total"]}')
    lines.append('| '+pair.replace('_vs_',' → ')+' | '+' | '.join(cells)+' |')
lines+=['','本批原8192次首选份额≥85%的41个主行动点，到32768次均未改选；首选份额<60%的43点有13点改选，60%–85%的31点有6点改选。因此，分歧程度能区分本批的预算敏感性，但不能证明高份额动作正确或改选后更强。']
lines+=['','## 固定主行动下的条件决策','',
    '支付／归还与主行动分开统计，且固定原来的主行动／购买目标；不代表某预算改变主行动后重新选择的完整网站原子招法。','',
    '| 对比 | 有选择的31个 | 含唯一动作的53个 |','| --- | ---: | ---: |']
for a,b in PAIRS:
    key=f'{a}_vs_{b}';p=summary['conditional_with_choice']['pairwise'][key];q=summary['conditional']['pairwise'][key]
    lines.append(f'| {a} → {b} | {p["changed"]}/31 | {q["changed"]}/53 |')
lines+=['','## 所有发生改选的主行动点','',
    '| 对局／轮次／手数 | 4096 | 8192 | 16384 | 32768 |','| --- | --- | --- | --- | --- |']
for row in main:
    if len(set(row['choices'].values()))<=1:continue
    cells=[f'{row["points"][b]["choice"]} {row["points"][b]["top1_share"]:.1%}' for b in BUDGETS]
    lines.append(f'| 第{row["game"]}盘／R{row["round"]}／move{row["move_number"]:03d} | '+' | '.join(cells)+' |')
lines+=['','## 解释边界与原件','',
    '改选不等于改进。本次没有独立最优行动标签，没有评估改招后胜负，也未运行正式棋力对战；因此不能把改选率当成错误修复率。接近可能源于算力不足、行动价值接近或共享估值误差，单seed四盘不能将这些原因分离。','',
    '原件在本机 `runtime/kirameki_budget_matrix_20261030/`：`manifest.json`冻结输入和代码身份，`inputs/`安全快照及历史推荐，`source/`实际执行源码／数据／扩展，`results/`168个决策的完整候选结果，`choices.csv`与`derived_rows.json`为全部四档明细，`result_inventory.json`保存原件SHA及字节数。尚未归档至T7。Git仅保留本目录脚本、聚合summary及报告。','',
    '汇总复现：`python artifacts/kirameki_budget_stability_20261030/summarize.py`。原运行命令：','',
    '```bash','conda run --no-capture-output -n splendor env PYTHONPATH=runtime/kirameki_budget_matrix_20261030/source/src python -u runtime/kirameki_budget_matrix_20261030/run.py --workers 8','```','',
    f'验证：{len(main)+len(active_conditional)}个有选择根全部8192访问匹配，{single}个唯一动作保持零模拟；复用{reused}个已验证根，终局越过检查点的精确补跑共{reruns}次。主批CPU8进程墙钟约{summary["execution_wall_seconds"]/60:.1f}分钟（含汇报进度，不含准备与此前复用的三点计算）。']
(ART/'README.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({k:summary[k] for k in ('status','main','conditional_with_choice','validation')},ensure_ascii=False,indent=2))
