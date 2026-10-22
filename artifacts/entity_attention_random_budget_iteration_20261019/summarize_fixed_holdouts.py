"""Summarize fixed-cohort forward metrics and paired whole-game uncertainty."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

STEPS = (0, 100, 250, 500, 1000)
KEYS = ('policy_kl', 'policy_ce', 'teacher_top1_agreement', 'value_mse',
        'value_correlation', 'value_prediction_std', 'value_bias', 'kl_plus_mse')


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def resampling_weights(games, seed, count):
    """Resample whole games within each source iteration; preserve source mix."""
    rng = np.random.default_rng(seed)
    result = np.zeros((count, len(games)), dtype=np.float64)
    sources = np.array([g['source_iteration'] for g in games])
    for source in np.unique(sources):
        ids = np.flatnonzero(sources == source)
        result[:, ids] = rng.multinomial(len(ids), np.full(len(ids), 1/len(ids)), size=count)
    return result


def comparison(first, last, weights, columns):
    a, b = first['per_game'], last['per_game']
    if not np.array_equal(a[:, columns.index('positions')], b[:, columns.index('positions')]):
        raise ValueError('paired comparison changed value population')
    if not np.array_equal(a[:, columns.index('policy_weight')], b[:, columns.index('policy_weight')]):
        raise ValueError('paired comparison changed policy population')
    answer = {}
    for name, numerator, denominator in (('policy_kl','kl','policy_weight'),
        ('value_mse','sq_error','positions'), ('teacher_top1_agreement','top1','policy_weight')):
        change = b[:, columns.index(numerator)]-a[:, columns.index(numerator)]
        count = a[:, columns.index(denominator)]
        draws = (weights@change)/(weights@count)
        low, high = np.quantile(draws, (.025, .975))
        answer[name] = {'delta':float(change.sum()/count.sum()),
            'lower':float(low), 'upper':float(high)}
    return answer


def plot(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary = json.loads((output/'summary.json').read_text())
    fig, axes = plt.subplots(2,2,figsize=(11,6),constrained_layout=True)
    for row, group in enumerate(summary['groups']):
        points = group['points']
        initial = points[0]['total_train_step']
        for ax, metric, label in zip(axes[row], ('policy_kl','value_mse'), ('Policy KL','Win/loss MSE')):
            xs=[(p['total_train_step']-initial)/1000 for p in points]
            ys=[p[metric] for p in points]
            ax.plot(xs,ys,color='#1675b8',marker='o',markersize=2,linewidth=1.4)
            ax.set(xlabel='Updates since block start (thousands)', ylabel=label,
                title=f"Models {group['model_iterations'][0]}-{group['model_iterations'][-1]}, data {group['source_iterations'][0]}-{group['source_iterations'][-1]}")
            ax.grid(alpha=.2)
    fig.savefig(output/'fixed_curves.png',dpi=180)
    fig.savefig(output/'fixed_curves.svg')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=4000)
    parser.add_argument('--seed', type=int, default=38620000)
    parser.add_argument('--plot-only', action='store_true')
    args=parser.parse_args()
    if args.plot_only:
        plot(args.output)
        return
    plan=json.loads((args.input/'plan.json').read_text())
    complete=json.loads((args.input/'complete.json').read_text())
    if complete['smoke'] or complete['checkpoints'] != 60:
        raise ValueError('formal summary requires all60 planned checkpoint evaluations')
    args.output.mkdir(parents=True, exist_ok=True)
    groups=[]
    csv_points=[]
    csv_rounds=[]
    for group_index, group in enumerate(plan['groups']):
        root=args.input/group['name']
        points=[]
        loaded={}
        for cp in group['checkpoints']:
            point=json.loads((root/f'round-{cp["iteration"]:03d}-step-{cp["step"]:04d}.json').read_text())
            if point['checkpoint'] != cp or point['panel_digest'] != group['digest']:
                raise ValueError('point identity differs from frozen plan')
            key=point['weights_sha256']
            if key not in loaded:
                raw=json.loads((root/f'weights-{key}.json').read_text())
                with np.load(root/f'weights-{key}.npz') as archive:
                    per_game=archive['per_game']
                loaded[key]={'raw':raw,'per_game':per_game}
            item=loaded[key]
            points.append({**cp, **{k:point['metrics'][k] for k in KEYS},
                'value_zero_mse':point['metrics']['value_zero_mse'],
                'value_prediction_mean':point['metrics']['value_prediction_mean'],
                'weights_sha256':key, 'per_game':item['per_game'],
                'value_by_budget':item['raw']['value_by_budget']})
        columns=next(iter(loaded.values()))['raw']['columns']
        weights=resampling_weights(group['games'],args.seed+group_index,args.bootstrap)
        whole=comparison(points[0],points[-1],weights,columns)
        last_mid=next(p for p in points if p['iteration']==group['model_iterations'][-1] and p['step']==500)
        penultimate_end=next(p for p in points if p['iteration']==group['model_iterations'][-2] and p['step']==1000)
        tail_comparisons={
            'block_start_to_penultimate_end':comparison(points[0],penultimate_end,weights,columns),
            'block_start_to_last_500':comparison(points[0],last_mid,weights,columns),
            'last_500_to_last_1000':comparison(last_mid,points[-1],weights,columns)}
        bias_change=points[-1]['value_bias']**2-last_mid['value_bias']**2
        mse_change=points[-1]['value_mse']-last_mid['value_mse']
        round_changes=[]
        for iteration in group['model_iterations']:
            subset=[p for p in points if p['iteration']==iteration]
            change=comparison(subset[0],subset[-1],weights,columns)
            row={'iteration':iteration, 'comparison':change,
                'best_steps':{key:min(subset,key=lambda p:p[key])['step']
                    for key in ('policy_kl','value_mse','kl_plus_mse')}}
            round_changes.append(row)
            csv_rounds.append({'group':group['name'],'iteration':iteration,
                **{f'{key}_{stat}':v for key,d in change.items() for stat,v in d.items()},
                **{f'{key}_best_step':v for key,v in row['best_steps'].items()}})
        clean=[{k:v for k,v in p.items() if k!='per_game'} for p in points]
        for p in clean:
            csv_points.append({'group':group['name'], **{k:v for k,v in p.items() if k!='value_by_budget'}})
        groups.append({'name':group['name'],'source_iterations':group['source_iterations'],
            'model_iterations':group['model_iterations'],'games':len(group['games']),
            'policy_positions':group['policy_positions'],'value_positions':group['value_positions'],
            'block_comparison':whole,'round_changes':round_changes, 'points':clean,
            'tail_comparisons':tail_comparisons,
            'last_500_mse_decomposition':{'mse_change':mse_change,
                'squared_mean_error_change':bias_change,
                'error_variance_change':mse_change-bias_change},
            'mean_within_round_change':{key:float(np.mean([r['comparison'][key]['delta'] for r in round_changes]))
                for key in ('policy_kl','value_mse')},
            'best_step_histogram':{key:{str(step):sum(r['best_steps'][key]==step for r in round_changes) for step in STEPS}
                for key in ('policy_kl','value_mse','kl_plus_mse')},
            'reference_verifications':[item['raw']['reference_verification'] for item in loaded.values()
                if 'reference_verification' in item['raw']]})
    summary={'scope':'fixed historical validation labels, not current-policy continuation truth or playing strength',
        'input':str(args.input), 'plan_digest':plan['digest'], 'checkpoints':60,
        'bootstrap':{'resamples':args.bootstrap,'seed':args.seed,'numpy_version':np.__version__,
            'method':'paired whole-game bootstrap, stratified by source iteration; percentile95%; conditional on these source teachers and one training trajectory'},
        'groups':groups,'analysis_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write(args.output/'summary.json',summary)
    for name,rows in (('checkpoints.csv',csv_points),('round_changes.csv',csv_rounds)):
        with (args.output/name).open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

    lines=['# 六轮固定整局留出回补评估','',
        '2026-10-20。只读已有NPZ和checkpoint，未训练、未重新采集或搜索，未新增棋力赛。两组分别使用固定的原始整局validation全量局面。', '',
        '第6–11轮没有本分支第0轮可用，因此第一组取第1–6轮留出；第6轮数据在第6轮训练0步之前已采集完成，未使用未来数据。第二组严格取第6–11轮留出评估第12–17轮。两组绝对KL/MSE不能直接比较。', '',
        '每组起点为首轮训练前0步，终点为末轮1000步，共6000次更新；不是首轮1000步到末轮1000步。轮号指37423步起点的新分支，不是原R24–R43。', '',
        '全部原始留出局按唯一身份去重，并与已保存的17轮Train快照检查无交集；冻结数据SHA、checkpoint SHA、原始标签和权重。使用FP32、关闭TF32，批大小512；state-only value路径和仅正权重policy路径在每组首个模型上与现有联合forward损失实现逐项核对。相同参数的0步／前轮1000步结果复用，但两个数据面板之间不复用。', '',
        '![固定留出轨迹](fixed_curves.png)', '',
        '## 六轮累计变化','',
        '| 模型轮次 | 固定数据 | 留出对局 | Policy局面 | Value局面 | KL起点→终点 | MSE起点→终点 | Value相关性起点→终点 |',
        '| --- | --- | ---: | ---: | ---: | --- | --- | --- |']
    for g in groups:
        a,b=g['points'][0],g['points'][-1]
        lines.append(f"| {g['model_iterations'][0]}–{g['model_iterations'][-1]} | {g['source_iterations'][0]}–{g['source_iterations'][-1]} | {g['games']} | {g['policy_positions']} | {g['value_positions']} | {a['policy_kl']:.6f}→{b['policy_kl']:.6f} | {a['value_mse']:.6f}→{b['value_mse']:.6f} | {a['value_correlation']:.6f}→{b['value_correlation']:.6f} |")
    lines+=['','| 模型轮次 | ΔKL [95%区间] | ΔMSE [95%区间] | Δpolicy首选一致率 [95%区间] |',
            '| --- | --- | --- | --- |']
    for g in groups:
        cells=[]
        for key in ('policy_kl','value_mse','teacher_top1_agreement'):
            v=g['block_comparison'][key];scale=100 if key=='teacher_top1_agreement' else 1
            cells.append(f"{v['delta']*scale:+.6f} [{v['lower']*scale:+.6f}, {v['upper']*scale:+.6f}]"+(' pp' if scale==100 else ''))
        lines.append(f"| {g['model_iterations'][0]}–{g['model_iterations'][-1]} | "+' | '.join(cells)+' |')
    lines+=['','区间按整局配对重抽，并在6个数据来源轮次内分层；不是把约19万局面视为独立样本。区间只反映这批对局下的抽样不确定性，不包括不同训练seed或教师的变异。','']
    late=groups[-1]
    end=late['points'][-1]
    mid=next(p for p in late['points'] if p['iteration']==17 and p['step']==500)
    tail=late['tail_comparisons']
    lines+=['## 主要发现','',
        '1. **第6–11轮：value确有历史留出泛化改善，policy对这批旧标签总体持平。** MSE累计下降0.012263，整局配对区间不跨0；相关性0.4812→0.4911，不能仅解释为缩小预测幅度。KL先在第6–7轮下降，之后逐渐回升，六轮净变化−0.000331，区间跨0。', '',
        '2. **第12–17轮：最终value MSE未确认改善，policy与旧teacher的分歧增加。** MSE净变化+0.001989，区间跨0；KL增加0.007422，区间全为正，首选一致率61.890%→61.260%。这能确认对同一旧标签的变化，不能单凭它判定当前棋力变弱，或区分遗忘与教师策略更新。', '',
        '3. **后六轮不是一直没有value收益。** 第12–16轮每轮终点MSE均下降，累计0.768743→0.763936；第17轮500步进一步到0.762603，1000步回升到0.770732。后六轮的最终结果被第17轮末尾波动明显影响。最后500步的MSE增加0.008129，其中平均误差平方增加约0.005861，占约72%；这是MSE=误差方差＋平均误差平方的描述性分解，不是对训练故障的因果定位。', '',
        '4. **对先前滚动留出的判断作限定。** “后期收益不稳定”仍成立，但不能扩展成“第12轮起value不再学习”。固定大面板显示第12–16轮持续的小幅改善及第17轮明显反复；当前更直接的诊断线索是更新／校准稳定性，还不足以归因于独立对局数不够。EMA、降lr或checkpoint选择是否改善需另做受控验证，本次未实施。', '',
        '5. **固定历史面板与滚动面板回答不同问题。** 原滚动面板policy KL均值小降，固定旧teacher面板却持平／升高，说明变化不是单一“全局拟合更好”。应同时观察新近标签的吸收与旧分布保持；也不能把两组面板不同造成的绝对差异，当成训练阶段间的强弱差异。', '',
        '### 第17轮末尾的配对变化','',
        '| 比较 | ΔKL [95%区间] | ΔMSE [95%区间] |','| --- | --- | --- |']
    for label,key in (('第12轮0步→第16轮1000步','block_start_to_penultimate_end'),
        ('第12轮0步→第17轮500步','block_start_to_last_500'),
        ('第17轮500步→1000步','last_500_to_last_1000')):
        cells=[]
        for metric in ('policy_kl','value_mse'):
            v=tail[key][metric]
            cells.append(f"{v['delta']:+.6f} [{v['lower']:+.6f}, {v['upper']:+.6f}]")
        lines.append(f'| {label} | '+' | '.join(cells)+' |')
    lines+=['','该末尾分析是看过轨迹后的探索性比较，未校正多重比较，不能用于宣称500步普遍最优。', '',
        '| 第17轮节点 | Value相关性 | 预测标准差 | 平均误差 | MSE |',
        '| --- | ---: | ---: | ---: | ---: |']
    for p in (mid,end):
        lines.append(f"| {p['step']} | {p['value_correlation']:.6f} | {p['value_prediction_std']:.6f} | {p['value_bias']:+.6f} | {p['value_mse']:.6f} |")
    lines+=['','### 按原采集预算分层的value MSE','',
        '| 模型轮次 | 留出步骤类型 | 局面数 | 起点→终点MSE |',
        '| --- | --- | ---: | --- |']
    for g in groups:
        for key,title in (('high_policy','高预算有效policy'),('low','低预算'),('forced_or_other','单候选等其他')):
            a,b=(p['value_by_budget'][key] for p in (g['points'][0],g['points'][-1]))
            lines.append(f"| {g['model_iterations'][0]}–{g['model_iterations'][-1]} | {title} | {a['positions']} | {a['mse']:.6f}→{b['mse']:.6f} |")
    lines+=['','各层同向变化，未见仅低预算步骤独自恶化的证据；这是按原始步骤来源分层的描述，不能分离搜索预算的因果效应。','']
    for g in groups:
        lines.append('')
        lines += [f"## 第{g['model_iterations'][0]}–{g['model_iterations'][-1]}轮各节点",'',
            '| 轮／指标 | 0 | 100 | 250 | 500 | 1000 | Δ1000−0 |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for iteration in g['model_iterations']:
            points=[p for p in g['points'] if p['iteration']==iteration]
            for key,title in (('policy_kl','KL'),('value_mse','MSE')):
                lines.append(f'| {iteration} {title} | '+' | '.join(f'{p[key]:.6f}' for p in points)+f" | {points[-1][key]-points[0][key]:+.6f} |")
        lines+=['','| 轮 | ΔKL [95%区间] | ΔMSE [95%区间] |','| ---: | --- | --- |']
        for row in g['round_changes']:
            values=[]
            for key in ('policy_kl','value_mse'):
                v=row['comparison'][key];values.append(f"{v['delta']:+.6f} [{v['lower']:+.6f}, {v['upper']:+.6f}]")
            lines.append(f"| {row['iteration']} | "+' | '.join(values)+' |')
    lines+=['','## 解释边界与复核','',
        '固定面板回答对同一批历史搜索policy／终局胜负标签的拟合与泛化是否改变；它不提供当前模型独立续局的真值，也不是棋力胜负。历史teacher的标签可能随策略变强而过时。需要与滚动留出、greedy和搜索对战共同解释。', '',
        '逐节点CE、熵、top1一致率、value方差／偏差／相关性见`checkpoints.csv`与`summary.json`；原始逐局充分统计量、冻结清单和源码身份在原始输出目录，长期副本见`raw/`。']
    (args.output/'results.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'groups':[{k:g[k] for k in ('name','block_comparison','mean_within_round_change','best_step_histogram')} for g in groups]},ensure_ascii=False))


if __name__=='__main__':
    main()
