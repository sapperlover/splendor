"""Recompute matched-root gold statistics and verify the completed search matrix."""
from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path
import time

import numpy as np

from splendor.alphazero_recording import read_json, sha256, write_json
import run as experiment

ROOT, ART, RUNTIME = experiment.ROOT, experiment.ART, experiment.RUNTIME


def rank(weights, prior):
    return np.lexsort((np.arange(len(weights)), -np.asarray(prior), -np.asarray(weights)))


def metric(ref, weights, prior, actual_choice=None):
    weights = np.asarray(weights, dtype=float)
    weights /= weights.sum()
    gold = np.asarray(ref['gold'])
    extra = gold > gold.min()
    order = rank(weights, prior)
    if actual_choice is not None:
        assert actual_choice == int(order[0])
    top = order[:3]
    mass = float(weights[top].sum())
    extra_mass = float(weights[top][extra[top]].sum())
    return dict(top1_extra=int(extra[order[0]]), top1_extra_gold=int(gold[order[0]] - gold.min()),
                top3_extra_mass=extra_mass, top3_mass=mass,
                top3_extra_fraction=extra_mass / mass, extra_mass=float(weights[extra].sum()),
                minimum_outside_top3=int(int(np.argmin(gold)) not in top),
                minimum_rank=int(np.flatnonzero(order == np.argmin(gold))[0]) + 1,
                expected_extra_gold=float(weights @ (gold - gold.min())),
                top3_any_positive_extra=int((extra[top] & (weights[top] > 0)).any()),
                chosen_index=int(order[0]))


def groups(refs):
    masks = {'all': np.ones(len(refs), dtype=bool)}
    masks['nonterminal'] = np.array([not r['direct_terminal'] for r in refs])
    masks['direct_terminal'] = ~masks['nonterminal']
    for s in ('opening_first10turns', 'middle', 'late_maxscore_ge10'):
        masks[s] = np.array([r['stage'] == s for r in refs])
        masks[s + '_nonterminal'] = masks[s] & masks['nonterminal']
    masks['minimum_gold_zero'] = np.array([r['minimum_gold'] == 0 for r in refs])
    masks['minimum_gold_positive'] = ~masks['minimum_gold_zero']
    for n in (2, 3, '4plus'):
        masks[f'candidates_{n}'] = np.array([r['candidates'] >= 4 if n == '4plus' else r['candidates'] == n for r in refs])
    for lo, hi in [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 10000)]:
        masks[f'completed_turns_{lo}_{hi-1}'] = np.array([lo <= r['completed_player_turns'] < hi for r in refs])
    return {k: v for k, v in masks.items() if v.any()}


def aggregate(rows, mask):
    selected = [r for r, yes in zip(rows, mask, strict=True) if yes]
    n = len(selected)
    result = {'roots': n, 'top1_extra_count': sum(r['top1_extra'] for r in selected),
              'minimum_outside_top3_count': sum(r['minimum_outside_top3'] for r in selected)}
    for key in ('top1_extra', 'top1_extra_gold', 'top3_extra_mass', 'top3_mass', 'top3_extra_fraction',
                'extra_mass', 'minimum_outside_top3', 'minimum_rank', 'expected_extra_gold', 'top3_any_positive_extra'):
        result['mean_' + key] = float(np.mean([r[key] for r in selected]))
    result['pooled_top3_extra_fraction'] = sum(r['top3_extra_mass'] for r in selected) / sum(r['top3_mass'] for r in selected)
    return result


def main():
    started = time.perf_counter()
    plan = read_json(ART / 'plan.json'); experiment.identity_check(plan)
    refs = [json.loads(line) for line in (ART / 'root_refs.jsonl').read_text().splitlines()]
    assert len(refs) == 2000 and [r['root_id'] for r in refs] == list(range(2000))
    with np.load(RUNTIME / 'population.npz', allow_pickle=False) as z:
        population = z['game_decision']
    chosen = np.sort(np.random.default_rng(plan['sample_seed']).choice(len(population), 2000, replace=False))
    np.testing.assert_array_equal(chosen, [r['population_index'] for r in refs])
    np.testing.assert_array_equal(population[chosen], [[r['game_id'], r['decision_index']] for r in refs])
    assert len({(r['game_id'], r['decision_index']) for r in refs}) == 2000
    for p, digest in plan['selected_game_inputs'].items():
        assert sha256(ROOT / p) == digest, p
    for p, digest in read_json(ART / 'catalog_inputs.json').items():
        assert sha256(ROOT / p) == digest, p
    execution_plan = read_json(ART / 'parallel_execution.json')
    assert sha256(ART / 'parallel.py') == execution_plan['driver_sha256']
    raw = {}
    raw_hashes = {}
    times = {}
    for panel in plan['panels']:
        rows, seconds = [], 0.
        for start in range(0, 2000, plan['chunk_roots']):
            path = RUNTIME / 'search' / panel['name'] / f'{start:04d}.json'
            data = read_json(path)
            assert data['plan_sha256'] == sha256(ART / 'plan.json') and data['panel'] == panel
            raw_hashes[str(path.relative_to(ROOT))] = sha256(path)
            rows.extend(data['rows']); seconds += data['seconds']
        assert [r['root_id'] for r in rows] == list(range(2000))
        raw[panel['name']] = rows
        times[panel['name']] = seconds
    raw_hashes[str((RUNTIME / 'roots.jsonl').relative_to(ROOT))] = sha256(RUNTIME / 'roots.jsonl')
    baseline = raw['S256N4_noise0']
    noisy = raw['S256N4_noise1']
    metrics = {'greedy_noise0': [], 'greedy_noise1': []}
    world_checks = 0
    simulations = 0
    cap_hits = 0
    terminal_checks = 0
    terminal_source_values = {}
    for ref in refs:
        if ref['direct_terminal']:
            path = ROOT / plan['source_dataset'] / 'games' / f"game-{ref['game_id']:06d}.npz"
            with np.load(path, allow_pickle=False) as z:
                terminal_source_values[ref['root_id']] = int(z['value_target'][ref['decision_index']])
    for i, ref in enumerate(refs):
        metrics['greedy_noise0'].append(metric(ref, baseline[i]['prior'], baseline[i]['prior']))
        metrics['greedy_noise1'].append(metric(ref, noisy[i]['used_prior'], noisy[i]['prior']))
    for panel in plan['panels']:
        name = panel['name']; result = []
        for i, (ref, row) in enumerate(zip(refs, raw[name], strict=True)):
            p, u = row['prior'], row['used_prior']
            np.testing.assert_allclose(p, baseline[i]['prior'], atol=2e-5, rtol=2e-5)
            if panel['noise']:
                np.testing.assert_allclose(u, noisy[i]['used_prior'], atol=2e-5, rtol=2e-5)
                eta = (np.asarray(u) - .75 * np.asarray(p)) / .25
                eta_ref = (np.asarray(noisy[i]['used_prior']) - .75 * np.asarray(noisy[i]['prior'])) / .25
                np.testing.assert_allclose(eta, eta_ref, atol=2e-5, rtol=2e-5)
                assert eta.min() >= -1e-6 and np.isclose(eta.sum(), 1.)
            else:
                np.testing.assert_array_equal(p, u)
            visits = np.asarray(row['visits'], dtype=np.int64)
            worlds = np.asarray(row['per_world_visits'], dtype=np.int64)
            sums = np.asarray(row['per_world_value_sums'], dtype=float)
            assert row['sampling_seed'] == ref['sampling_seed']
            assert worlds.shape == (panel['determinizations'], ref['candidates'])
            assert len(row['world_seeds']) == panel['determinizations']
            assert row['world_seeds'][0] == baseline[i]['world_seeds'][0]
            if panel['determinizations'] == 4:
                assert row['world_seeds'] == baseline[i]['world_seeds']
            np.testing.assert_array_equal(visits, worlds.sum(axis=0))
            budget = panel['simulations'] // panel['determinizations']
            assert row['simulation_budgets'] == [budget] * panel['determinizations']
            np.testing.assert_array_equal(worlds.sum(axis=1), row['simulation_budgets'])
            assert visits.sum() == panel['simulations']
            assert np.all(np.isfinite(sums)) and np.all(np.abs(sums) <= worlds + 1e-6)
            world_checks += len(worlds); simulations += int(visits.sum())
            cap_hits += row['transition_cap_hits']
            if ref['direct_terminal']:
                # All legal payments differ only in leftover tokens, after game has ended.
                np.testing.assert_allclose(sums, worlds * terminal_source_values[i], atol=1e-8, rtol=0)
                if len(worlds) == 4:
                    np.testing.assert_array_equal(worlds, np.repeat(worlds[:1], 4, axis=0))
                terminal_checks += 1
            m = metric(ref, visits, p, row['choice'])
            used_choice = int(rank(u, p)[0])
            m.update(same_as_used_prior=int(row['choice'] == used_choice),
                     used_prior_top1_extra=int(ref['gold'][used_choice] > ref['minimum_gold']),
                     all_world_visits_identical=int(np.all(worlds == worlds[0])))
            result.append(m)
        metrics[name] = result
    assert simulations == 73728000 and world_checks == 36000 and terminal_checks == 204
    masks = groups(refs)
    table = {name: {group: aggregate(rows, mask) for group, mask in masks.items()} for name, rows in metrics.items()}
    # Paired bootstrap resamples whole sampled games, preserving within-game correlation.
    game_ids = sorted({r['game_id'] for r in refs})
    game_index = {g: i for i, g in enumerate(game_ids)}
    ids = np.array([game_index[r['game_id']] for r in refs])
    bootstrap_seed = 2026102604
    draws = np.random.default_rng(bootstrap_seed).multinomial(len(game_ids), np.full(len(game_ids), 1 / len(game_ids)), size=2000).astype(float)
    comparisons = []
    pairs = [(f'{m}_noise1', f'{m}_noise0') for m in ('greedy', 'S256N4', 'S1024N1', 'S4096N4')]
    pairs += [(f'S256N4_noise{n}', f'S1024N1_noise{n}') for n in (0, 1)]
    pairs += [(f'S4096N4_noise{n}', f'S256N4_noise{n}') for n in (0, 1)]
    pairs += [(f'{m}_noise{n}', f'greedy_noise{n}') for m in ('S256N4', 'S1024N1', 'S4096N4') for n in (0, 1)]
    for group in ('all', 'nonterminal'):
        mask = masks[group]
        denominators = draws @ np.bincount(ids, weights=mask, minlength=len(game_ids))
        for left, right in pairs:
            stats = {}
            for key in ('top1_extra', 'top3_extra_mass', 'top3_extra_fraction', 'extra_mass', 'expected_extra_gold'):
                delta = np.array([a[key] - b[key] for a, b in zip(metrics[left], metrics[right], strict=True)])
                totals = np.bincount(ids, weights=delta * mask, minlength=len(game_ids))
                distribution = (draws @ totals) / denominators
                stats[key] = dict(mean_delta=float(delta[mask].mean()),
                                  paired_game_bootstrap_95ci=np.quantile(distribution, [.025, .975]).tolist())
            comparisons.append(dict(left=left, right=right, group=group, roots=int(mask.sum()), metrics=stats))
    prior_effects = {}
    for name, rows in metrics.items():
        if name.startswith('greedy'):
            continue
        prior_effects[name] = {}
        for group in ('all', 'nonterminal', 'direct_terminal'):
            selected = [r for r, yes in zip(rows, masks[group], strict=True) if yes]
            prior_effects[name][group] = dict(roots=len(selected),
                same_as_used_prior=sum(r['same_as_used_prior'] for r in selected),
                extra_despite_used_prior_minimal=sum(r['top1_extra'] and not r['used_prior_top1_extra'] for r in selected),
                corrected_used_prior_extra=sum(not r['top1_extra'] and r['used_prior_top1_extra'] for r in selected),
                all_world_visits_identical=sum(r['all_world_visits_identical'] for r in selected))
    elapsed = read_json(RUNTIME / 'complete.json')
    assert not elapsed['inference']['compile']['compile_failures']
    assert elapsed['inference']['compile']['compile_failure_fallbacks'] == 0
    assert elapsed['inference']['compile']['eager_fallbacks'] == 0
    assert elapsed['runner_wide_batches'] == 0
    summary = dict(complete=True, roots=2000, sampled_games=len(game_ids), search_arms=6, search_roots=12000,
                   simulations=simulations, stages=dict(Counter(r['stage'] for r in refs)),
                   direct_terminal=int(masks['direct_terminal'].sum()),
                   stage_terminal_counts=dict(Counter(r['stage'] for r in refs if r['direct_terminal'])),
                   candidate_counts=dict(Counter(r['candidates'] for r in refs)),
                   minimum_gold_counts=dict(Counter(r['minimum_gold'] for r in refs)),
                   table=table, comparisons=comparisons, prior_effects=prior_effects,
                   search_seconds_by_arm=times, search_chunk_seconds_total=sum(times.values()), execution=elapsed,
                   bootstrap=dict(unit='sampled game, paired across methods', repeats=2000, seed=bootstrap_seed),
                   numpy_version=np.__version__, random_generator='numpy.random.default_rng / PCG64',
                   plan_sha256=sha256(ART / 'plan.json'), summarize_sha256=sha256(Path(__file__)),
                   limitations=['Single frozen local12 checkpoint; roots follow its local13 selfplay distribution.',
                                'One search seed and one common noise draw per root; no repeated-seed stability estimate.',
                                '34 direct-terminal roots only; rare event and subgroup rates have sampling uncertainty.',
                                'Extra gold is above legal minimum, not a proven mistake: normal token-retention tradeoffs may exist.'],
                   summarize_seconds=time.perf_counter() - started)
    csv_path = RUNTIME / 'root_metrics.csv'
    fieldnames = ['method', 'root_id', 'game_id', 'stage', 'direct_terminal'] + list(metrics['S256N4_noise0'][0])
    with csv_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for name, rows in metrics.items():
            for ref, row in zip(refs, rows, strict=True):
                writer.writerow(dict(method=name, **{k: ref[k] for k in ('root_id', 'game_id', 'stage', 'direct_terminal')}, **row))
    raw_hashes[str(csv_path.relative_to(ROOT))] = sha256(csv_path)
    write_json(ART / 'raw_hashes.json', raw_hashes)
    write_json(ART / 'summary.json', summary)
    write_json(ART / 'verification.json', dict(complete=True, inputs_and_sources_unchanged=True,
               sampled_archive_files_verified=len(plan['selected_game_inputs']), feature_arrays_checked=plan['feature_arrays_checked'],
               search_roots_verified=12000, world_budget_rows_verified=world_checks,
               terminal_action_values_verified=terminal_checks, same_noise_and_world_seeds_verified=True,
               same_unnoised_priors_verified=True, raw_chunks=len(raw_hashes) - 2,
               simulations=simulations, transition_cap_hits=cap_hits, execution=elapsed,
               raw_hash_manifest_sha256=sha256(ART / 'raw_hashes.json')))
    report(summary)
    print(json.dumps({'complete': True, 'roots': 2000, 'simulations': simulations,
                      'seconds': sum(times.values()), 'all': {k: v['all'] for k, v in table.items()}}, ensure_ascii=False), flush=True)


def pct(x):
    return f'{100*x:.3f}%'


def report(s):
    methods = ['greedy', 'S256N4', 'S1024N1', 'S4096N4']
    lines = ['# 第12轮模型：2000个多支付局面的花金对照', '',
             '六组正式搜索和两组greedy全部完成。每组2000个相同局面，共12000次根搜索、73728000次模拟。', '',
             '模型为本分支第12轮final（`round-011/checkpoints/final.pt`，SHA256 `'+experiment.MODEL_SHA+'`）；'
             '从其生成的第13轮10000盘中，均匀无放回抽取2000/89027个多方案支付根，来自1793盘。'
             '开局132、中盘1621、后期247；支付后直接终局34个。没有按模型选择筛样。', '',
             '“多花金”指超过同一局面的合法最低黄金数，不等于已证明的错误。SxNy表示每世界x次、y个世界：'
             'S256N4与S1024N1总预算均1024，S4096N4总预算16384。所有选择T=0；有噪声为25%根噪声、Dirichlet α=0.3，'
             '六组使用共同世界种子，三组有噪声搜索使用同一份逐根噪声。'
             '有噪声greedy是对完全相同的native加噪先验直接取首选，仍为零模拟。FP/PTP关闭。', '',
             '“前三多金份额”是前三候选中的多金概率／访问量占全部候选的份额；“前三内占比”先在各根前三内归一化，再跨根等权平均。'
             '候选不足三个时全部计入；并列按未加噪网络先验、候选顺序裁决。', '',
             '## 全部2000个局面', '',
             '| 方法 | 根噪声 | 多金首选（个） | 首选率 | 前三多金份额 | 前三内占比 | 全部多金份额 | 最低金掉出前三（个） |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for m in methods:
        for n in (0, 1):
            a = s['table'][f'{m}_noise{n}']['all']
            lines.append(f"| {m} | {'25%' if n else '无'} | {a['top1_extra_count']} | {pct(a['mean_top1_extra'])} | {pct(a['mean_top3_extra_mass'])} | {pct(a['mean_top3_extra_fraction'])} | {pct(a['mean_extra_mass'])} | {a['minimum_outside_top3_count']} |")
    a, b, c = [s['table'][name]['all'] for name in ('S256N4_noise0', 'S1024N1_noise0', 'S4096N4_noise0')]
    lines += ['',
              f"同总预算1024，无噪声S256N4的多金首选率为{pct(a['mean_top1_extra'])}，S1024N1为{pct(b['mean_top1_extra'])}。"
              '因此此前采集数据里高预算组较高的比例，不能直接归因于N4本身；原统计还同时改变了预算、噪声和根集合。', '',
              f"同为4世界，无噪声预算从1024增至16384，多金首选率由{pct(a['mean_top1_extra'])}变为{pct(c['mean_top1_extra'])}，"
              f"前三多金份额则由{pct(a['mean_top3_extra_mass'])}变为{pct(c['mean_top3_extra_mass'])}。"
              '首选率与分布尾部份额是不同指标，增加搜索预算不保证二者同方向变化。', '',
              '噪声的净影响、逐时期表现和剔除直接终局后的结果见下表及配对区间。'
              '终局根的已访问动作具有完全相同的终局Q，但本次没有重训或标签消融，不能据此判断这些终局标签对普通局面学习的因果影响。']
    lines += ['', '## 时期与终局', '',
              '时期口径先冻结：双方累计完成的玩家回合数<10为开局；其余根中支付前双方最高分≥10为后期，否则中盘。'
              '直接终局单列，不能仅凭“后期”替代终局；本样本各时期直接终局数为`'+json.dumps(s['stage_terminal_counts'],ensure_ascii=False)+'`。', '',
              '下表均为多金首选率，括号为根数；时期三列包含直接终局，最后两列单独区分终局。', '',
              '| 方法 | 噪声 | 开局(132) | 中盘(1621) | 后期(247) | 非直接终局(1966) | 直接终局(34) |',
              '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for m in methods:
        for n in (0, 1):
            rows = s['table'][f'{m}_noise{n}']
            cells = [pct(rows[g]['mean_top1_extra']) for g in ('opening_first10turns','middle','late_maxscore_ge10','nonterminal','direct_terminal')]
            lines.append(f"| {m} | {'25%' if n else '无'} | " + ' | '.join(cells) + ' |')
    lines += ['', '剔除直接终局后，每格为“多金首选率 / 前三多金份额”：', '',
              '| 方法 | 噪声 | 开局(132) | 中盘(1621) | 后期(213) |', '| --- | --- | ---: | ---: | ---: |']
    for m in methods:
        for n in (0, 1):
            rows = s['table'][f'{m}_noise{n}']
            cells = [pct(rows[g+'_nonterminal']['mean_top1_extra']) + ' / ' + pct(rows[g+'_nonterminal']['mean_top3_extra_mass'])
                     for g in ('opening_first10turns','middle','late_maxscore_ge10')]
            lines.append(f"| {m} | {'25%' if n else '无'} | " + ' | '.join(cells) + ' |')
    lines += ['', '## 配对差异', '',
              '差异为左减右，单位百分点；95%区间用共同的2000次整盘bootstrap，保留一盘内多支付根之间的相关性。'
              '这些区间描述当前采样分布，不能替代多种子或棋力实验。', '',
              '| 对比 | 范围 | 首选率差 | 95%区间 | 前三份额差 | 95%区间 |',
              '| --- | --- | ---: | --- | ---: | --- |']
    for c in s['comparisons']:
        if c['left'].startswith('S') and c['right'].startswith('greedy'):
            continue
        a, b = c['metrics']['top1_extra'], c['metrics']['top3_extra_mass']
        interval = lambda x: '[' + ', '.join(f'{100*v:+.3f}' for v in x) + ']'
        lines.append(f"| {c['left']} − {c['right']} | {c['group']} | {100*a['mean_delta']:+.3f} | {interval(a['paired_game_bootstrap_95ci'])} | {100*b['mean_delta']:+.3f} | {interval(b['paired_game_bootstrap_95ci'])} |")
    lines += ['', '## 执行和复核', '',
              f"正式搜索从首次启动到完成的墙钟时间{s['execution']['seconds']/60:.2f}分钟；分块工作时间合计{s['search_chunk_seconds_total']/60:.2f}分钟（并发块时间重叠）；每组分块时间（秒）：`{json.dumps({k:round(v,2) for k,v in s['search_seconds_by_arm'].items()})}`。"
              '先单进程完成低预算组和一部分高预算块，再从已完成的原子分块切换到两个独立worker；每块仍为同样64根，候选和种子不变，分块互不重叠。'
              '每worker capacity256、双slot、batch buckets64/128/256、candidate buckets16/32/48/64，'
              'CUDA FP32/native-cudagraph、Torch单线程；不是整盘对战的评测A入口。'
              '无compile失败、无eager回退、无超桶。详细设备、命令、吞吐和资源见runtime execution/live文件。', '',
              '每个根的安全DecisionRequest由checked引擎从原始轨迹重放，逐字段对照归档特征；搜索不接收真实隐藏牌堆或未来结果。'
              '全部36000个世界预算行、根与逐世界访问量、共同世界种子、共同噪声、未加噪先验已校验。'
              '34个直接终局根×6组的每个已访问动作Q均等于相应终局回报；四世界访问数组在这些根逐世界相同。', '',
              '局限：每根只有一组共同搜索种子和噪声，没有多种子稳定性实验；直接终局只有34根，罕见事件及小分组比例不宜外推。'
              '本实验衡量支付分布，未评估胜率，也未验证某个非最低花金动作的实际优劣。', '',
              '## 文件、执行与统计复算', '',
              '- `plan.json`：模型、配置、种子、源码与输入身份；`root_refs.jsonl`：固定抽样引用和分组。',
              '- `parallel_execution.json`：双进程续跑的命令、代码SHA与不相交分块规则；原串行会话停在已保存块后，未保存块重新完整计算。',
              '- `summary.json`：完整分组、配对差异、先验与搜索关系；`verification.json`和`raw_hashes.json`：核验与原件SHA。',
              '- `smoke.json`：总预算32的工程验证，与正式结果分开。',
              '- `runtime/'+ART.name+'/roots.jsonl`、`search/`：安全请求和逐根／逐世界原始结果；`root_metrics.csv`：八组逐根统计。重型原件受现有runtime忽略规则覆盖。', '',
              '本次执行过的命令如下（仓库根目录、conda splendor、CUDA）。`run.py run`会按plan SHA跳过已存在的正式分块。'
              '已完成的矩阵只需最后一条`summarize.py`复算统计；不要再次启动搜索入口覆盖已完成会话的耗时记录。', '', '```bash',
              f'conda run --no-capture-output -n splendor env PYTHONPATH=src python artifacts/{ART.name}/run.py prepare',
              f'conda run --no-capture-output -n splendor env PYTHONPATH=src python artifacts/{ART.name}/run.py smoke',
              f'conda run --no-capture-output -n splendor env PYTHONPATH=src python artifacts/{ART.name}/run.py run',
              '# 本次在上一个命令的串行阶段中断后，按原plan继续双进程分块：',
              f'conda run --no-capture-output -n splendor env PYTHONPATH=src python artifacts/{ART.name}/parallel.py',
              f'conda run --no-capture-output -n splendor env PYTHONPATH=src python artifacts/{ART.name}/summarize.py', '```', '']
    (ART / 'README.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
