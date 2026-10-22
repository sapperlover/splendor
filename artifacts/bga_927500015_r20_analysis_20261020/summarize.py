"""Summarize completed BGA root searches; no inference, searches, or training."""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np

ART = Path(__file__).resolve().parent
ROOT = ART.parents[1]
DATA = ROOT / 'artifacts/bga_927500015_20261020'
OUT = ART / 'statistics'
NAMES = {'greedy': '原Policy', 'native_S1024_N1': '原搜索 S1024 N1', 'native_S1024_N4': '原搜索 S1024 N4',
         'chance_S1024_N1': 'Chance S1024', 'cpw_S1024_N1': 'CPW S1024',
         'native_S16384_N1': '原搜索 S16384 N1', 'native_S16384_N4': '原搜索 S16384 N4',
         'chance_S16384_N1': 'Chance S16384', 'cpw_S16384_N1': 'CPW S16384'}
PLAYERS = ['VnEpCxmR（先手、胜方）', 'FourDimensional（后手）']


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def js(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    a, b = a / a.sum(), b / b.sum(); m = (a + b) / 2
    return .5 * sum(float(np.sum(p[p > 0] * np.log(p[p > 0] / m[p > 0]))) for p in (a, b))


def rank_order(values, prior):
    return sorted(range(len(values)), key=lambda i: (-values[i], -prior[i], i))


def estimate(prior, actual, searches):
    """Root equal-seed estimates; value differences use paired observed Q only."""
    distributions = [np.asarray(s['visits'], dtype=float) / sum(s['visits']) for s in searches]
    probs = np.asarray(distributions).mean(axis=0).tolist()
    order = rank_order(probs, prior); best = order[0]
    pairs = [s for s in searches if s['q'][actual] is not None and s['q'][best] is not None]
    changes = [50 * (s['q'][actual] - s['q'][best]) for s in pairs]
    independent_changes = [50 * (s['q'][actual] - s['q'][s['choice']]) for s in searches if s['q'][actual] is not None]
    qactual = [s['q'][actual] for s in searches if s['q'][actual] is not None]
    versus_distribution = [50 * (s['q'][actual] - sum(s['value_sums']) / sum(s['visits'])) for s in searches if s['q'][actual] is not None]
    return dict(choice=best, actual_probability=probs[actual], actual_rank=order.index(actual) + 1,
                recommendation_probability=probs[best], top3_mass=sum(probs[i] for i in order[:3]),
                entropy_nats=-sum(p * math.log(p) for p in probs if p > 0),
                mean_q_actual=mean(qactual), mean_q_recommended_paired=mean(s['q'][best] for s in pairs),
                mean_q_actual_paired=mean(s['q'][actual] for s in pairs),
                delta_score_pp=mean(changes), delta_score_seed_min_pp=min(changes) if changes else None,
                delta_score_seed_max_pp=max(changes) if changes else None,
                delta_vs_each_seed_choice_pp=mean(independent_changes), delta_vs_search_distribution_pp=mean(versus_distribution),
                value_pair_seeds=len(pairs), actual_visited_seeds=len(qactual), seeds=len(searches),
                actual_min_visits=min(s['visits'][actual] for s in searches),
                recommendation_min_visits=min(s['visits'][best] for s in searches),
                actual_mean_visits=mean(s['visits'][actual] for s in searches),
                actual_probability_seed_sd=float(np.std([p[actual] for p in distributions])),
                actual_q_seed_sd=float(np.std(qactual)) if len(qactual) == len(searches) else None,
                seed_choices=[s['choice'] for s in searches], choice_support=sum(s['choice'] == best for s in searches),
                seed_unstable=len({s['choice'] for s in searches}) > 1,
                seed_pair_disagreement=mean(a['choice'] != b['choice'] for a, b in itertools.combinations(searches, 2)),
                seed_pair_js_nats=mean(js(a, b) for a, b in itertools.combinations(distributions, 2)),
                visit_q_argmax_conflicts=sum(s['choice'] != max((i for i, q in enumerate(s['q']) if q is not None), key=lambda i: (s['q'][i], -i)) for s in searches),
                max_depth=mean(s['max_tree_depth'] for s in searches), cap_hits=sum(s['transition_cap_hits'] for s in searches),
                stable_disagreement=best != actual and len(pairs) == len(searches) == 3
                    and all(s['choice'] == best for s in searches) and max(changes) < 0
                    and min(s['visits'][actual] for s in searches) >= 16)


def aggregate(rows):
    valid = [r for r in rows if r['delta_score_pp'] is not None]
    reliable = [r for r in valid if r['value_pair_seeds'] == 3 and r['actual_min_visits'] >= 16 and r['recommendation_min_visits'] >= 16]
    delta = [r['delta_score_pp'] for r in valid]
    return dict(decisions=len(rows), human_top1=sum(r['agree'] for r in rows), human_top3=sum(r['actual_rank'] <= 3 for r in rows),
                mean_human_probability_pct=100 * mean(r['actual_probability'] for r in rows),
                median_human_probability_pct=100 * statistics.median(r['actual_probability'] for r in rows),
                mean_human_rank=mean(r['actual_rank'] for r in rows),
                mean_top3_mass_pct=100 * mean(r['top3_mass'] for r in rows), mean_entropy_nats=mean(r['entropy_nats'] for r in rows),
                greedy_agree=sum(r['choice'] == r['greedy_choice'] for r in rows),
                mean_delta_score_pp=mean(delta), median_delta_score_pp=statistics.median(delta) if delta else None,
                mean_positive_delta_pp=mean(max(v, 0) for v in delta), mean_negative_delta_pp=mean(min(v, 0) for v in delta),
                better_than_consensus_gt1pp=sum(v > 1 for v in delta), worse_than_consensus_gt1pp=sum(v < -1 for v in delta),
                within1pp=sum(abs(v) <= 1 for v in delta), value_covered_roots=len(valid),
                all3_seed_pairs=sum(r['value_pair_seeds'] == 3 for r in rows),
                total_paired_seeds=sum(r['value_pair_seeds'] for r in rows),
                mean_delta_vs_each_seed_choice_pp=mean(r['delta_vs_each_seed_choice_pp'] for r in rows),
                mean_delta_vs_search_distribution_pp=mean(r['delta_vs_search_distribution_pp'] for r in rows),
                mean_q_actual=mean(r['mean_q_actual'] for r in rows),
                reliable_roots=len(reliable), reliable_mean_delta_pp=mean(r['delta_score_pp'] for r in reliable),
                low_actual_visits_roots=sum(r['actual_min_visits'] < 16 for r in rows),
                no_q_bga_moves=[r['bga_move'] for r in rows if r['delta_score_pp'] is None],
                unstable_roots=sum(r['seed_unstable'] for r in rows), stable_disagreements=sum(r['stable_disagreement'] for r in rows),
                mean_seed_pair_disagreement=mean(r['seed_pair_disagreement'] for r in rows),
                mean_seed_js_nats=mean(r['seed_pair_js_nats'] for r in rows),
                mean_human_probability_seed_sd_pp=100 * mean(r['actual_probability_seed_sd'] for r in rows),
                mean_human_q_seed_sd=mean(r['actual_q_seed_sd'] for r in rows),
                visit_q_argmax_conflicts=sum(r['visit_q_argmax_conflicts'] for r in rows),
                mean_max_depth=mean(r['max_depth'] for r in rows), transition_cap_hits=sum(r['cap_hits'] for r in rows))


def main():
    plan = read(ART / 'plan.json'); previous = read(ART / 'summary.json'); live = read(ART / 'live.json')
    assert previous['complete'] and live['stage'] == 'complete'
    for name, expected in plan['inputs'].items():
        assert digest(ROOT / name) == expected, name
    records = [json.loads(line) for line in (DATA / 'model_decisions.jsonl').read_text().splitlines()]
    record_map = {r['decision_index']: r for r in records}
    searchable = {i for i, r in record_map.items() if len(r['request']['legal_actions']) > 1}
    greedy = {r['decision_index']: r for r in read(ART / 'greedy.json')['rows']}
    labels = {}
    with (ART / 'action_details.csv').open(encoding='utf-8-sig') as handle:
        for r in csv.DictReader(handle):
            labels[int(r['decision_index']), int(r['candidate_index'])] = r['action']
    inputs = [ART / 'plan.json', ART / 'summary.json', ART / 'live.json', ART / 'greedy.json',
              DATA / 'model_decisions.jsonl', ART / 'action_details.csv', ART / 'decisions.csv']
    names = ['greedy'] + [p['name'] for p in plan['panels']]
    all_rows, panels, raw, timings = [], {}, {}, {}
    for panel in plan['panels']:
        name = panel['name']; data = {}
        for seed in plan['seeds']:
            path = ART / 'search' / name / f'{seed}.json'; inputs.append(path)
            doc = read(path); assert doc['panel'] == panel and doc['seed'] == seed
            assert {r['decision_index'] for r in doc['rows']} == searchable and len(doc['rows']) == len(searchable)
            data[seed] = {r['decision_index']: r for r in doc['rows']}
            for i, r in data[seed].items():
                visits = np.asarray(r['visits']); counts = np.asarray(r['per_world_visits']); sums = np.asarray(r['per_world_value_sums'])
                assert sum(visits) == panel['simulations'] and len(counts) == panel['determinizations']
                np.testing.assert_array_equal(visits, counts.sum(axis=0))
                np.testing.assert_array_equal(counts.sum(axis=1), r['simulation_budgets'])
                np.testing.assert_allclose(sums.sum(axis=0), r['value_sums'], atol=1e-10)
                for j, n in enumerate(visits):
                    if n:
                        assert abs(r['q'][j] - r['value_sums'][j] / n) < 1e-10
                    else:
                        assert r['q'][j] is None
            timings.setdefault(name, []).extend(c['seconds'] for c in doc['chunks'])
        raw[name] = data
    for name in names:
        panel_rows = []
        for i in sorted(searchable):
            rec = record_map[i]; actual = rec['candidate_index']; prior = greedy[i]['prior']
            base = dict(panel=name, decision_index=i, bga_move=rec['bga_move'], actor=rec['actor'],
                        phase=rec['request']['observation']['phase'], actual=labels[i, actual], greedy_choice=greedy[i]['choice'])
            if name == 'greedy':
                order = rank_order(prior, prior); best = order[0]
                value = dict(choice=best, actual_probability=prior[actual], actual_rank=order.index(actual) + 1,
                             top3_mass=sum(prior[j] for j in order[:3]), entropy_nats=-sum(p * math.log(p) for p in prior if p > 0),
                             delta_score_pp=None, value_pair_seeds=0, delta_vs_each_seed_choice_pp=None, delta_vs_search_distribution_pp=None,
                             actual_min_visits=0, recommendation_min_visits=0, mean_q_actual=None, seed_unstable=False,
                             stable_disagreement=False, seed_pair_disagreement=None, seed_pair_js_nats=None,
                             actual_probability_seed_sd=0., actual_q_seed_sd=None, visit_q_argmax_conflicts=0, max_depth=None, cap_hits=0)
            else:
                value = estimate(prior, actual, [raw[name][seed][i] for seed in plan['seeds']])
            row = dict(**base, **value, recommendation=labels[i, value['choice']], agree=value['choice'] == actual)
            panel_rows.append(row); all_rows.append(row)
        phase_stats = {}
        for phase in ['main_action', 'payment', 'all_choices']:
            selected = [r for r in panel_rows if phase == 'all_choices' or r['phase'] == phase]
            phase_stats[phase] = {who: aggregate([r for r in selected if actor is None or r['actor'] == actor]) for who, actor in [('all', None), ('p0', 0), ('p1', 1)]}
        panels[name] = phase_stats
        if name != 'greedy':
            for who, key in [('all', 'None'), ('p0', '0'), ('p1', '1')]:
                assert phase_stats['main_action'][who]['human_top1'] == previous['panels'][name][key]['human_agree']
    with (ART / 'decisions.csv').open(encoding='utf-8-sig') as handle:
        original = {(int(r['decision_index']), r['panel']): r for r in csv.DictReader(handle)}
    checked = 0; maximum_delta_error = 0.
    for r in all_rows:
        if r['panel'] == 'greedy': continue
        old = original[r['decision_index'], r['panel']]
        assert old['recommendation'] == r['recommendation']
        assert bool(old['paired_seed_q_gap']) == (r['delta_score_pp'] is not None)
        if r['delta_score_pp'] is not None:
            error = abs(-50 * float(old['paired_seed_q_gap']) - r['delta_score_pp'])
            assert error < 1e-9
            maximum_delta_error = max(maximum_delta_error, error)
        checked += 1
    shared = set.intersection(*[{r['decision_index'] for r in all_rows if r['panel'] == name and r['phase'] == 'main_action'
                               and r['value_pair_seeds'] == 3 and r['actual_min_visits'] >= 16 and r['recommendation_min_visits'] >= 16} for name in names[1:]])
    for name in names[1:]:
        selected = [r for r in all_rows if r['panel'] == name and r['decision_index'] in shared]
        panels[name]['common_reliable'] = dict(decisions=len(selected), mean_delta_score_pp=mean(r['delta_score_pp'] for r in selected))
        panels[name]['runtime'] = dict(chunk_seconds_sum=sum(timings[name]), chunks=len(timings[name]),
                                       seconds_per_root=sum(timings[name]) / (len(searchable) * len(plan['seeds'])))
    comparisons = []
    row_lookup = {(r['panel'], r['decision_index']): r for r in all_rows}
    main_ids = [i for i in searchable if record_map[i]['request']['observation']['phase'] == 'main_action']
    for a, b in itertools.combinations(names[1:], 2):
        pairs = [(raw[a][seed][i], raw[b][seed][i]) for seed in plan['seeds'] for i in main_ids]
        comparisons.append(dict(a=a, b=b, roots=len(main_ids), root_seeds=len(pairs),
                                seed_top1_changes=sum(x['choice'] != y['choice'] for x, y in pairs),
                                mean_js_nats=mean(js(x['visits'], y['visits']) for x, y in pairs),
                                consensus_changes=sum(row_lookup[a, i]['choice'] != row_lookup[b, i]['choice'] for i in main_ids),
                                mean_human_probability_delta_pp=100 * mean(row_lookup[b, i]['actual_probability'] - row_lookup[a, i]['actual_probability'] for i in main_ids)))
    summary = dict(complete=True, searches=1824, simulations=15876096, main_decisions=56, choice_payments=20,
                   forced_decisions=12, shared_reliable_decision_indices=sorted(shared), panels=panels, comparisons=comparisons,
                   q_reference='Primary: argmax of equal-seed mean visit shares, prior tie-break. Delta score pp = 50*(Q_human-Q_reference), paired within seed then averaged within decision then equally over decisions with Q. Never impute unvisited Q.',
                   probability='Policy: softmax prior. Search: mean normalized root visits across three seeds, not deterministic T=0 behavior probability.',
                   score='Expected game score, win=1/draw=.5/loss=0. (Q+1)/2 is NOT pure win probability when draws are possible. Model estimate only, not calibrated or rollout-verified.',
                   scope='One game, main actions and payment choices separately. Low-visit and selection biases retained and explicitly reported; reliable subset is conditional, not an unbiased estimate of full-game regret.')
    OUT.mkdir(exist_ok=True)
    write(OUT / 'summary.json', summary)
    write(OUT / 'validation.json', dict(complete_searches=1824, total_simulations=15876096,
          per_world_counts_and_value_sums_checked=True, original_decision_crosschecks=checked,
          maximum_delta_error_pp=maximum_delta_error, frozen_inputs_checked=True))
    write(OUT / 'decision_metrics.json', all_rows)
    for path, items in [(OUT / 'decision_metrics.csv', all_rows), (OUT / 'method_comparisons.csv', comparisons)]:
        fields = list(dict.fromkeys(k for row in items for k in row))
        with path.open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(items)
    manifest = dict(producer={'path': str(Path(__file__).relative_to(ROOT)), 'sha256': digest(Path(__file__))}, command=sys.argv,
                    inputs={str(p.relative_to(ROOT)): digest(p) for p in inputs},
                    protocol='Derived statistics only. Frozen search producer and raw results unchanged.')
    write(OUT / 'manifest.json', manifest)
    render(summary, all_rows)
    print(json.dumps({'complete': True, 'main': 56, 'payments': 20, 'searches': 1824, 'shared_reliable': len(shared), 'report': str(OUT / 'results.md')}, ensure_ascii=False))


def render(summary, rows):
    lines = ['# BGA #927500015：全部搜索配置统计', '',
             '第20轮原模型，8组×3种子×76个可选决策=1824次根搜索、15876096次模拟全部完成；CUDA无编译失败／fallback。主行动56步、可选支付20次，强制12次排除。仅一局，不是棋力面板。', '',
             '“实际概率”：Policy为softmax概率，搜索为根访问份额，先等权平均3个种子、再等权平均局面。实际执行T0时仍是确定性首选。', '',
             '“得分率变化”：相对各组3种子平均访问份额的首选，Δ=50×(Q实际−Q首选)，单位百分点；负数表示模型认为比其首选差。按同种子成对Q差求均值，再对可估局面等权平均。胜1／和0.5／负0；不是严格胜率，也不是独立续局验证的真实损失，不可逐步累加。', '',
             '未访问Q保留缺失，不能补0。Greedy没有逐动作Q，不能给出同口径价值差。访问首选不保证Q最高，保留少数正向变化；主表不是对Q最大动作计算非负后悔值。', '',
             '## 56步主行动：整体', '',
             '| 方法 | 实际平均概率 | 人类top1 | 人类top3 | 平均排名 | Δ得分率pp | Q可估步 | 种子变选 | 平均最大深度 |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    def number(v):
        return '—' if v is None else f'{v:+.2f}'
    for name, groups in summary['panels'].items():
        s = groups['main_action']['all']
        lines.append(f"| {NAMES[name]} | {s['mean_human_probability_pct']:.2f}% | {s['human_top1']}/56 | {s['human_top3']}/56 | {s['mean_human_rank']:.2f} | {number(s['mean_delta_score_pp'])} | {s['value_covered_roots']}/56 | {'—' if name == 'greedy' else str(s['unstable_roots'])+'/56'} | {'—' if s['mean_max_depth'] is None else format(s['mean_max_depth'],'.1f')} |")
    lines += ['', '## 主行动：双方分开', '',
              '| 方法 | 先手实际概率 | 先手一致 | 先手Δpp（覆盖） | 后手实际概率 | 后手一致 | 后手Δpp（覆盖） |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name, groups in summary['panels'].items():
        a, b = groups['main_action']['p0'], groups['main_action']['p1']
        lines.append(f"| {NAMES[name]} | {a['mean_human_probability_pct']:.2f}% | {a['human_top1']}/28 | {number(a['mean_delta_score_pp'])} ({a['value_covered_roots']}/28) | {b['mean_human_probability_pct']:.2f}% | {b['human_top1']}/28 | {number(b['mean_delta_score_pp'])} ({b['value_covered_roots']}/28) |")
    lines += ['', '## 低访问敏感性与不同参考动作', '',
              f"共同子集：8组中实际及共识首选在3种子均有Q，且各至少16访问，共{len(summary['shared_reliable_decision_indices'])}步。该筛选会排除不少最大分歧，不能据此替代全部56步的平均。", '',
              '| 方法 | 全部可估步Δpp | 各组≥16访问子集Δpp（步数） | 共同39步Δpp | 相对每种子各自首选Δpp |',
              '| --- | ---: | ---: | ---: | ---: |']
    for name, groups in summary['panels'].items():
        if name == 'greedy': continue
        s = groups['main_action']['all']
        lines.append(f"| {NAMES[name]} | {number(s['mean_delta_score_pp'])} | {number(s['reliable_mean_delta_pp'])} ({s['reliable_roots']}) | {number(groups['common_reliable']['mean_delta_score_pp'])} | {number(s['mean_delta_vs_each_seed_choice_pp'])} |")
    lines += ['', '## 有选择的支付阶段（20次）', '',
              '| 方法 | 实际平均概率 | 一致 | top3 | Δ得分率pp | Q可估 | 种子变选 |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name, groups in summary['panels'].items():
        s = groups['payment']['all']
        lines.append(f"| {NAMES[name]} | {s['mean_human_probability_pct']:.2f}% | {s['human_top1']}/20 | {s['human_top3']}/20 | {number(s['mean_delta_score_pp'])} | {s['value_covered_roots']}/20 | {'—' if name == 'greedy' else str(s['unstable_roots'])+'/20'} |")
    lines += ['', '## 种子变化、Q覆盖与运行', '',
              '| 方法 | 种子两两首选分歧 | 同组种子JS | 实际概率种子SD(pp) | 实际Q种子SD | 实际最低访问<16的步数 | 无Q的BGA步 | 搜索批次秒数 |',
              '| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |']
    for name, groups in summary['panels'].items():
        if name == 'greedy': continue
        s = groups['main_action']['all']; runtime = groups['runtime']
        lines.append(f"| {NAMES[name]} | {100*s['mean_seed_pair_disagreement']:.1f}% | {s['mean_seed_js_nats']:.4f} | {s['mean_human_probability_seed_sd_pp']:.2f} | {s['mean_human_q_seed_sd']:.4f} | {s['low_actual_visits_roots']} | {s['no_q_bga_moves']} | {runtime['chunk_seconds_sum']:.1f} |")
    lines += ['', '批次计时包含本批推理／编译和所有主行动、支付搜索，不能解释为无并发环境下纯算法速度。N4同时并行4个世界，单个主机批的推理利用率不同。', '',
              '## 所有方法两两比较（56步×3种子）', '',
              '| A | B | 同种子首选变化/168 | 共识首选变化/56 | 访问分布JS | B−A实际概率pp |',
              '| --- | --- | ---: | ---: | ---: | ---: |']
    for c in summary['comparisons']:
        lines.append(f"| {NAMES[c['a']]} | {NAMES[c['b']]} | {c['seed_top1_changes']}/168 | {c['consensus_changes']}/56 | {c['mean_js_nats']:.4f} | {c['mean_human_probability_delta_pp']:+.2f} |")
    lines += ['', '## 第50步：低访问造成的大Q差', '',
              '人类拿蓝绿黑，原Policy概率0.687%，模型首选买保留032、概率96.94%。以下Q差不能当作已确认失误：实际动作只被访问0或1次，估计主要来自单次叶子value，可能明显偏差。', '',
              '| 方法 | 人类实际概率 | 共识首选 | Q差换算Δpp | 实际最少访问/种子 | 有Q种子 |',
              '| --- | ---: | --- | ---: | ---: | ---: |']
    for r in rows:
        if r['bga_move'] == 50 and r['phase'] == 'main_action' and r['panel'] != 'greedy':
            lines.append(f"| {NAMES[r['panel']]} | {100*r['actual_probability']:.5f}% | {r['recommendation']} | {number(r['delta_score_pp'])} | {r['actual_min_visits']} | {r['value_pair_seeds']}/3 |")
    lines += ['', '## 结论与范围', '',
              '- 搜索对人类实际动作的平均访问份额约34%–44%，原Policy约29%；这比top1一致率提供更多信息，但仍不是人类动作正确率。',
              '- S提高16倍后，四类方法的人类top1一致率没有一致上升，原N1跨种子首选变化反而增加。更多单世界搜索不能消除世界偏差。',
              '- Chance高预算最稳定；固定总S的N4更稳定但每世界预算减少。这些数据描述该局面集，不证明相应方法整体棋力更好。',
              '- 全体可估步平均Q差换算约每步−2至−3.5个百分点，但受极少访问动作强烈影响；共同39步的估计差约−0.47至−1.16个百分点。不能累加为整局损失，也不能据此给人类或模型定Elo。',
              '- Chance标签中的N1只表示单共享统计池，根未知状态逐模拟重采、沿途随机事件重新采样；它不是只抽一个固定隐藏世界。CPW限制逐步开放的随机结果支持。', '',
              '逐决策完整数据：decision_metrics.csv／decision_metrics.json；全部分组统计与口径：summary.json；来源SHA：manifest.json。原逐步HTML、完整候选与所有逐世界原始搜索结果保持不变。']
    (OUT / 'results.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
