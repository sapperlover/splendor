"""Summarize saved diagnostics; never load networks or generate new games."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import mean

STEPS = (0, 100, 250, 500, 1000)
METRICS = {'policy_kl': 'kl', 'policy_loss': 'ce', 'value_loss': 'mse',
           'policy_entropy': 'policy_entropy', 'target_entropy': 'target_entropy',
           'value_zero_mse': 'zero_mse'}


def plot(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary = json.loads((output/'summary.json').read_text())
    with (output/'round_summary.csv').open() as handle:
        rounds = list(csv.DictReader(handle))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    for ax, key, title in zip(axes, ('kl','mse'), ('Validation policy KL', 'Validation win/loss MSE')):
        for r in rounds:
            ax.plot(STEPS, [float(r[f'val_{key}_{s}'])-float(r[f'val_{key}_0']) for s in STEPS],
                    color='#9ba5ad', alpha=.4, linewidth=.8)
        for group,color,label in (('all','#151e27','Mean: all 17 rounds'),
                                  ('all_online_6_17','#1675b8','Mean: rounds 6-17')):
            values = summary['groups'][group]['means_by_step']
            ax.plot(STEPS, [values[str(s)][key]-values['0'][key] for s in STEPS],
                    color=color, linewidth=2.2, marker='o', markersize=3, label=label)
        ax.axhline(0,color='#8b949c',linewidth=.8,linestyle='--')
        ax.set(title=title, xlabel='Updates within round', ylabel='Change from step 0 (lower is better)')
        ax.set_xticks(STEPS)
        ax.grid(alpha=.15)
    axes[0].legend(fontsize=8)
    fig.savefig(output/'within_round_changes.png',dpi=180)
    fig.savefig(output/'within_round_changes.svg')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path,
        default=Path('runs/entity_attention_37423_random_budget_p6k_v24k_20261019'))
    parser.add_argument('--output', type=Path,
        default=Path(__file__).parent / 'learning_curves_20261020')
    parser.add_argument('--plot-only', action='store_true', help='Plot existing CSV/JSON using Matplotlib')
    args = parser.parse_args()
    if args.plot_only:
        plot(args.output)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    inputs = {}

    def read(path):
        data = path.read_bytes()
        inputs[str(path)] = hashlib.sha256(data).hexdigest()
        return json.loads(data)

    state = read(args.run / 'state.json')
    assert len(state['completed_rounds']) >= 17
    initial_step = None
    validation, rounds = [], []
    # Freeze this report to rounds1-17 even if the live run later advances.
    for iteration in range(1, 18):
        stage = args.run / 'rounds' / f'round-{iteration-1:03d}'
        refs = read(stage / 'validation_references.json')
        snapshot = read(stage / 'replay_snapshot.json')
        metrics_path = stage / 'training_metrics.jsonl'
        data = metrics_path.read_bytes()
        inputs[str(metrics_path)] = hashlib.sha256(data).hexdigest()
        train = [row for line in data.splitlines() if (row := json.loads(line))['event'] == 'train_step']
        assert [row['round_train_step'] for row in train] == list(range(1, 1001))
        if iteration == 1:
            initial_step = train[0]['total_train_step']-1
        assert train[0]['total_train_step'] == initial_step+(iteration-1)*1000+1
        points = []
        for step in STEPS:
            raw = read(stage / 'validation' / f'step-{step:09d}.json')
            assert raw['step'] == step
            assert raw['validation_references_digest'] == refs['digest']
            assert raw['snapshot_digest'] == snapshot['digest']
            m = raw['metrics']
            assert abs(m['policy_loss'] - m['policy_kl'] - m['target_entropy']) < 1e-10
            point = {'iteration': iteration, 'step': step,
                'total_train_step': initial_step + (iteration-1)*1000 + step,
                'checkpoint_sha256': raw['checkpoint_sha256'],
                **{short: m[key] for key, short in METRICS.items()},
                'policy_positions': m['policy_positions'], 'value_positions': m['value_positions'],
                'relative_mse_reduction_vs_zero': 1-m['value_loss']/m['value_zero_mse']}
            point['kl_plus_mse'] = point['kl'] + point['mse']
            points.append(point)
            validation.append(point)
        for key in ('target_entropy', 'policy_positions', 'value_positions'):
            assert len({point[key] for point in points}) == 1
        row = {'iteration': iteration, 'historical_games': snapshot['historical_games'],
            'new_games': snapshot['new_games'], 'policy_new_fraction': snapshot['policy_new_fraction'],
            'policy_train_positions': sum(snapshot['policy_positions'].values()),
            'value_train_positions': snapshot['value_positions'],
            'unique_validation_games': len({(ref['run_id'], ref['round_index'], ref['game_id'])
                for ref in refs['references']}),
            'unique_validation_positions': len({tuple(sorted(ref.items())) for ref in refs['references']}),
            'validation_policy_positions': points[0]['policy_positions'],
            'validation_value_positions': points[0]['value_positions'],
            'train_grad_norm_mean': mean(m['grad_norm'] for m in train),
            'train_grad_clip_fraction': mean(m['grad_clipped'] for m in train),
            'policy_position_exposures': sum(m['policy_positions'] for m in train)/sum(snapshot['policy_positions'].values()),
            'value_position_exposures': sum(m['value_positions'] for m in train)/snapshot['value_positions']}
        for key in ('kl', 'ce', 'mse', 'policy_entropy', 'target_entropy', 'kl_plus_mse'):
            row.update({f'val_{key}_{p["step"]}': p[key] for p in points})
            row[f'val_{key}_delta'] = points[-1][key] - points[0][key]
            row[f'val_{key}_best_step'] = min(points, key=lambda p: p[key])['step']
        for key in ('policy_kl', 'policy_loss', 'value_loss', 'total_loss', 'target_entropy', 'policy_entropy'):
            for name, selected in (('first250', train[:250]), ('last250', train[-250:])):
                row[f'train_{key}_{name}'] = mean(m[key] for m in selected)
            row[f'train_{key}_delta'] = row[f'train_{key}_last250'] - row[f'train_{key}_first250']
        for name in ('greedy', 'search'):
            path = stage / f'{name}.json'
            if path.exists():
                scored = read(path)
                assert scored['passed']
                row[f'{name}_score'] = scored['win_rate_score']
                row[f'{name}_pairs'] = scored['completed_pairs']
            else:
                row[f'{name}_score'] = row[f'{name}_pairs'] = None
        rounds.append(row)

    def summarize(selected):
        return {'iterations': [r['iteration'] for r in selected],
            'means_by_step': {str(step): {key: mean(r[f'val_{key}_{step}'] for r in selected)
                for key in ('kl', 'ce', 'mse', 'policy_entropy', 'target_entropy')}
                for step in STEPS},
            'mean_within_round_delta': {key: mean(r[f'val_{key}_delta'] for r in selected)
                for key in ('kl', 'ce', 'mse', 'kl_plus_mse')},
            'improved_at_final': {key: sum(r[f'val_{key}_delta'] < 0 for r in selected)
                for key in ('kl', 'mse', 'kl_plus_mse')},
            'best_step_histogram': {key: dict(sorted(Counter(r[f'val_{key}_best_step'] for r in selected).items()))
                for key in ('kl', 'mse', 'kl_plus_mse')},
            'mean_train_window_delta': {key: mean(r[f'train_{key}_delta'] for r in selected)
                for key in ('policy_kl', 'policy_loss', 'value_loss', 'total_loss')},
            'late_delta_100_to_1000': {key: mean(r[f'val_{key}_1000']-r[f'val_{key}_100'] for r in selected)
                for key in ('kl', 'mse')},
            'late_delta_250_to_1000': {key: mean(r[f'val_{key}_1000']-r[f'val_{key}_250'] for r in selected)
                for key in ('kl', 'mse')},
            'late_250_to_1000_worse_rounds': {key: sum(r[f'val_{key}_1000']>r[f'val_{key}_250'] for r in selected)
                for key in ('kl', 'mse')},
            'train_improves_val_worsens_rounds': {key: [r['iteration'] for r in selected
                if r[f'train_{train_key}_delta']<0 and r[f'val_{key}_delta']>0]
                for key,train_key in (('kl','policy_kl'),('mse','value_loss'))}}

    summary = {'scope': 'existing diagnostics only; no new model evaluation',
        'analysis_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'run': str(args.run), 'completed_iterations': len(rounds),
        'steps': STEPS, 'initial_total_step': initial_step,
        'metric_definitions': {'kl': 'KL(search policy || model policy), positive archive policy weights',
            'ce': 'policy cross entropy = target entropy + KL', 'mse': 'win/loss value MSE',
            'train_windows': 'mean minibatch metrics, steps1-250 vs751-1000; not a fixed train subset',
            'validation': '2048 value-drawn positions fixed within each round; policy uses only positive weights; new subset every round'},
        'groups': {'all': summarize(rounds), 'historical_replacement_1_5': summarize(rounds[:5]),
            'all_online_6_17': summarize(rounds[5:]), 'online_6_11': summarize(rounds[5:11]),
            'online_12_17': summarize(rounds[11:])},
        'input_sha256': inputs}
    for name, rows in (('validation_by_step.csv', validation), ('round_summary.csv', rounds)):
        with (args.output/name).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    (args.output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')

    lines = ['# 混合预算持续迭代：17轮训练轨迹', '',
        '2026-10-20，仅汇总已有日志；未启动训练、重算网络指标或新增棋力赛。轮号为本分支第1–17轮，不是原R24–R43。', '',
        '## 口径', '',
        '- 每轮0/100/250/500/1000步使用同一份2048局面；脚本逐文件核对references与pool digest。跨轮重新抽样、池与教师也变化，不能把跨轮绝对值视为固定测试集上的趋势。',
        '- policy只使用2048局面中正权重的高预算／历史有效步骤；value使用全部2048。前5轮还在替换旧数据，第6轮起全为新分支数据。',
        '- Train为前250/后250步随机minibatch均值，不是固定Train子集。KL单位为nat，CE是policy loss，MSE是value loss；KL+MSE是诊断量，不是棋力。',
        '- 日志没有value预测方差、相关系数或top1准确率；不能从这些聚合量反推这些指标，或把MSE直接分解为不可约噪声。', '',
        '![每轮相对0步变化](within_round_changes.png)', '',
        '细灰线为各轮，黑线为17轮等权均值，蓝线为第6–17轮均值；这些轮次的池相互重叠，不将它们当独立样本生成显著性结论。', '',
        '## 平均轮内轨迹', '', '| 范围／指标 | 0 | 100 | 250 | 500 | 1000 | 终点减起点 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for group,label in (('all','1–17轮'),('all_online_6_17','6–17轮')):
        s=summary['groups'][group]
        for key in ('kl','ce','mse'):
            lines.append('| '+label+' '+key.upper()+' | '+' | '.join(f"{s['means_by_step'][str(step)][key]:.6f}" for step in STEPS)+f" | {s['mean_within_round_delta'][key]:+.6f} |")
    all_stats = summary['groups']['all']
    earlier, later = (summary['groups'][key] for key in ('online_6_11','online_12_17'))
    lines += ['', '## 观察到的规律与边界', '',
        f"1. **policy仍在学习，但每轮吸收幅度小。** 终点KL优于起点的轮数为{all_stats['improved_at_final']['kl']}/17，平均下降{abs(all_stats['mean_within_round_delta']['kl']):.6f}，约为起点KL的0.52%；Train前250到后250步平均下降{abs(all_stats['mean_train_window_delta']['policy_kl']):.6f}。第6–17轮中9/12轮终点KL改善。不能把小幅学习说成完全停学。",
        '2. **早期KL较好，不等于训练只该跑100步。** 17轮平均KL在100步已下降0.002487，100到1000步平均反升0.000194；12/17轮观察到的最低KL在100或250步。但KL＋MSE最低点有9/17轮在1000步，它的17轮平均值也是1000步最低。单一KL不足以决定联合训练的停止点，更不能代替棋力评测。',
        f"3. **value不是每轮都过拟合，后期泛化收益却在减弱。** 全17轮终点MSE改善{all_stats['improved_at_final']['mse']}/17轮，平均变化{all_stats['mean_within_round_delta']['mse']:+.6f}。第6–11轮6/6轮终点改善，平均{earlier['mean_within_round_delta']['mse']:+.6f}；第12–17轮仅3/6轮改善，平均{later['mean_within_round_delta']['mse']:+.6f}，同期Train窗口MSE平均变化{later['mean_train_window_delta']['value_loss']:+.6f}。第13/14/17轮为Train降而留出升，其中第14轮仅+0.000119，不能都当成明确退化。",
        '4. **跨轮绝对值混合了模型变化与面板变化。** 例如第11轮终点MSE为0.808990，第16轮为0.732222，相差0.076768；每轮训练造成的MSE变化通常在0.01以内。这些幅度不能用来估计纯抽样噪声，因为池、标签教师和模型都变了。更适合比较同轮0→1000的配对变化。',
        '5. **policy诊断样本量随着预算随机化下降。** 第1–5轮有效policy局面为1485/1305/1114/869/656，第6–17轮只有406–446；value仍为2048个局面，来自约1337–1396个独立对局。跨轮留出KL的起伏不能直接解释成模型来回变强变弱。',
        f"6. **loss与梯度补充信息。** CE等于目标熵＋KL；同轮目标熵不变，CE与KL的增减完全相同，并非另一份独立证据。Train总loss窗口平均下降{abs(all_stats['mean_train_window_delta']['total_loss']):.6f}。每轮梯度范数均值从{rounds[0]['train_grad_norm_mean']:.3f}升到{rounds[-1]['train_grad_norm_mean']:.3f}，全部17000步都触发clip1；现有日志没有分头梯度，不能据此认定value主导或裁剪就是主因。",
        '7. **现阶段判断与下一步建议。** 支持“policy小幅持续吸收，value后期泛化收益减弱”；不支持“学习完全停止”“所有轮次都value过拟合”或“100步一定优于1000步”。下一步优先固定一套跨轮不变、直接抽有效policy步的诊断集，并把value按高/低预算及数据代次分层，补预测方差／相关系数；固定Train子集也有助于区分拟合不足与泛化不足。本次只提出建议，没有修改训练或新增网络评测。']
    for key,title in (('kl','各轮留出KL'),('mse','各轮留出MSE')):
        lines += ['', '## '+title, '', '| 轮 | 0 | 100 | 250 | 500 | 1000 | Δ1000−0 | 最低点 |',
            '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for r in rounds:
            lines.append(f"| {r['iteration']} | "+' | '.join(f"{r[f'val_{key}_{step}']:.6f}" for step in STEPS)+f" | {r[f'val_{key}_delta']:+.6f} | {r[f'val_{key}_best_step']} |")
    lines += ['', '## Train与留出方向', '',
        '| 轮 | Train ΔKL | 留出 ΔKL | Train ΔMSE | 留出 ΔMSE | 有效policy局面 | 留出来源局数 | greedy得分 |',
        '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in rounds:
        score='—' if r['greedy_score'] is None else f"{r['greedy_score']*100:.3f}%"
        lines.append(f"| {r['iteration']} | {r['train_policy_kl_delta']:+.6f} | {r['val_kl_delta']:+.6f} | {r['train_value_loss_delta']:+.6f} | {r['val_mse_delta']:+.6f} | {r['validation_policy_positions']} | {r['unique_validation_games']} | {score} |")
    lines += ['', '## 可复核产物', '',
        '- `validation_by_step.csv`：85个checkpoint的KL、CE、MSE、熵、有效数和模型SHA。',
        '- `round_summary.csv`：每轮Train窗口、留出变化、观察到的最低点、梯度、池大小与已完成对战。',
        '- `summary.json`：分组均值、方向计数、最低点分布、输入文件SHA。观察到的最低点经过5次挑选，天然偏乐观；不能据此直接确定最优更新步数或最强checkpoint。', '',
        '复现统计：`conda run -n splendor env PYTHONPATH=src python artifacts/entity_attention_random_budget_iteration_20261019/analyze_learning_curves.py`。绘图：在已有Matplotlib的base环境执行同一脚本并加`--plot-only`；没有修改splendor环境依赖。']
    (args.output/'results.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'output':str(args.output), 'rounds':len(rounds), 'checkpoints':len(validation),
        'all_rounds':summary['groups']['all']},ensure_ascii=False))


if __name__ == '__main__':
    main()
