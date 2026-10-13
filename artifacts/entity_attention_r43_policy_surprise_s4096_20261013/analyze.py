"""Verify the matched draws and summarize unweighted/weighted fixed labels."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from splendor.alphazero_checkpoint import load_alphazero_checkpoint
from splendor.alphazero_recording import read_json, sha256, write_json
from splendor.paired_statistics import paired_bootstrap_difference, serialized_pair_scores

ART = Path(__file__).resolve().parent
ROOT = ART.parents[1]
RUN = ROOT / 'runs' / ART.name


def references(branch):
    batches = {}
    for path in sorted((RUN / branch / 'usage').glob('*.bin')):
        raw = path.read_bytes()
        rows = np.frombuffer(raw[:len(raw)//32*32], dtype='<i8').reshape(-1, 4)
        for step in np.unique(rows[:, 0]):
            block = rows[rows[:, 0] == step]
            if len(block) != 256:
                continue
            if int(step) in batches and not np.array_equal(batches[int(step)], block):
                raise ValueError('different resumed draws')
            batches[int(step)] = block
    if sorted(batches) != list(range(1, 1001)):
        raise ValueError('incomplete sampling ledger')
    return np.concatenate([batches[s] for s in sorted(batches)])


def compare_games(before, after):
    keys = sorted(k for k in before if k.endswith('/all'))
    if sorted(k for k in after if k.endswith('/all')) != keys:
        raise ValueError('different validation games')
    rng = np.random.default_rng(34350000)
    selected = rng.integers(0, len(keys), size=(10000, len(keys)))
    result = {'unit': 'whole validation game', 'games': len(keys), 'resamples': 10000, 'seed': 34350000}
    for name, numerator, denominator in (('policy_kl', 'kl', 'policy_weight'),
            ('value_mse', 'sq_error', 'positions'), ('surprise_policy_kl', 'surprise_kl', 'surprise_weight')):
        weight = np.array([before[k][denominator] for k in keys])
        if not np.array_equal(weight, [after[k][denominator] for k in keys]):
            raise ValueError('different metric denominators')
        diff = np.array([after[k][numerator]-before[k][numerator] for k in keys])
        estimates = diff[selected].sum(1)/weight[selected].sum(1)
        result[name] = {'delta': float(diff.sum()/weight.sum()),
                        'ci95': np.quantile(estimates, [.025, .975]).tolist()}
    return result


def main():
    plan = read_json(ART / 'plan.json')
    data = read_json(ART / 'dataset.json')['games']
    a, b = references('control'), references('surprise')
    if not np.array_equal(a, b):
        raise ValueError('control/treatment draws differ')
    train_keys = {(g['round_index'], g['game_id']) for g in data if g['split'] == 'train'}
    if any(tuple(r[1:3]) not in train_keys for r in a):
        raise ValueError('validation game sampled for training')
    np.savez_compressed(ART / 'matched_training_references.npz', columns=np.array(['step','round','game','decision']), references=a)
    states = [load_alphazero_checkpoint(RUN / branch / 'checkpoints/final.pt') for branch in ('control', 'surprise')]
    if states[0].sampling_rng_state != states[1].sampling_rng_state:
        raise ValueError('different final sampling RNG')
    verification = {'same_256_positions_for_both_heads_and_branches': True,
        'steps': 1000, 'draws': len(a), 'unique_positions': len(np.unique(a[:, 1:], axis=0)),
        'reference_sha256': hashlib.sha256(a.tobytes()).hexdigest(),
        'same_final_sampling_rng': True, 'validation_excluded': True,
        'draws_by_absolute_generation': {str(int(g)+1): int((a[:, 1] == g).sum()) for g in np.unique(a[:, 1])},
        'checkpoints': {branch: {'path': str((RUN / branch / 'checkpoints/final.pt').relative_to(ROOT)),
                                'sha256': sha256(RUN / branch / 'checkpoints/final.pt')}
                        for branch in ('control', 'surprise')}}
    write_json(ART / 'verification.json', verification)
    curves, raw = [], {}
    for branch in ('control', 'surprise'):
        rows = [json.loads(line) for line in (RUN / branch / 'training_metrics.jsonl').read_text().splitlines()]
        train = [r for r in rows if r['event'] == 'train_step']
        if [r['round_train_step'] for r in train] != list(range(1, 1001)):
            raise ValueError('training metrics incomplete')
        compact = [{k: r[k] for k in ('round_train_step', 'total_train_step', 'policy_loss', 'policy_kl',
                   'value_loss', 'total_loss', 'grad_norm', 'grad_clipped', 'reference_sha256', 'target_entropy')}
                   | ({'policy_surprise_weighting': r['policy_surprise_weighting']} if 'policy_surprise_weighting' in r else {})
                   for r in train]
        (ART / branch / 'learning_curve.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in compact))
        for step in plan['config']['fitting_steps']:
            raw[branch, step] = read_json(ART / branch / 'fitting' / f'step-{step:09d}.json')
            row = {'branch': branch, 'step': step}
            for split, field in (('train','online_train_fixed'), ('validation','online_validation_full')):
                metrics = raw[branch, step][field]['metrics']
                for name in ('policy_ce', 'policy_kl', 'value_mse', 'surprise_policy_ce', 'surprise_policy_kl'):
                    count = 'decision_count' if name == 'value_mse' else 'policy_samples'
                    weights = {g: sum(r[count] for r in data if r['round_index'] == g and r['split'] == split)
                               for g in plan['config']['source_round_indices']}
                    row[f'{split}_{name}'] = sum(metrics[f'g{g+1:03d}/{split}/all'][name]*n for g,n in weights.items())/sum(weights.values())
                row[f'{split}_joint_loss'] = row[f'{split}_policy_ce'] + row[f'{split}_value_mse']
                row[f'{split}_surprise_joint_loss'] = row[f'{split}_surprise_policy_ce'] + row[f'{split}_value_mse']
            curves.append(row)
    comparisons = {}
    for branch in ('control', 'surprise'):
        comparisons[branch+'_minus_R43'] = compare_games(raw['control', 0]['online_validation_full']['per_game'],
                                                         raw[branch, 1000]['online_validation_full']['per_game'])
    comparisons['surprise_minus_control'] = compare_games(raw['control', 1000]['online_validation_full']['per_game'],
                                                          raw['surprise', 1000]['online_validation_full']['per_game'])
    evaluations = {'R43': read_json(ROOT / plan['config']['greedy_reference'])}
    evaluations.update({branch: read_json(ART / branch / 'greedy.json') for branch in ('control', 'surprise')})
    greedy = {name: {'score': r['win_rate_score'], 'ci95': r['paired_bootstrap'],
                    'wins': r['a_wins'], 'draws': r['draws'], 'losses': r['a_losses']}
              for name,r in evaluations.items()}
    greedy_differences = {}
    for left, right in (('control','R43'), ('surprise','R43'), ('surprise','control')):
        ls, lv = serialized_pair_scores(evaluations[left])
        rs, rv = serialized_pair_scores(evaluations[right])
        greedy_differences[left+'_minus_'+right] = paired_bootstrap_difference(ls, lv, rs, rv,
            resamples=10000, seed=34360000).to_dict()
    result = {'status': 'complete', 'scope': plan['config']['scope'], 'curves': curves,
              'validation_differences': comparisons, 'greedy': greedy, 'greedy_differences': greedy_differences,
              'verification': verification}
    write_json(ART / 'summary.json', result)
    lines = ['# R43 固定 4k S4096：Policy Surprise Weighting 对照', '',
        '两条分支都从原 R43 出发，继承 AdamW、Torch RNG、采样 RNG。只用既有第44–47代 S4096/N1 的4000局；原整局划分不变。联合更新1000步、B256、lr2e-5、统一clip1，两个头与两条分支逐步使用完全相同的256个局面。', '',
        'surprise=KL(归档搜索策略∥归档加噪前教师策略)，每局 searchable policy 的权重为 0.5+0.5×surprise/本局均值；value权重不变。对照不加权。统计中的普通CE/KL/MSE都不加权；surprise列对两条分支使用同一固定加权指标。Train为每代2048个既有固定子集、按原池局面数加权；Validation覆盖全部原留出整局。', '',
        '| 分支 | 步数 | Train KL | 留出 KL | Train MSE | 留出 MSE | 留出 CE+MSE | 留出 surprise KL |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in curves:
        lines.append(f"| {row['branch']} | {row['step']} | {row['train_policy_kl']:.6f} | {row['validation_policy_kl']:.6f} | {row['train_value_mse']:.6f} | {row['validation_value_mse']:.6f} | {row['validation_joint_loss']:.6f} | {row['validation_surprise_policy_kl']:.6f} |")
    lines += ['', '终点留出差值及95%整局配对bootstrap区间；负值为改善。', '',
        '| 对比 | ΔKL | ΔMSE | Δsurprise KL |', '| --- | --- | --- | --- |']
    for name, comp in comparisons.items():
        cells = [f"{comp[m]['delta']:+.6f} [{comp[m]['ci95'][0]:+.6f}, {comp[m]['ci95'][1]:+.6f}]"
                 for m in ('policy_kl','value_mse','surprise_policy_kl')]
        lines.append(f"| {name} | {' | '.join(cells)} |")
    lines += ['', 'Greedy 对固定 P4V4，200对换先后手开局；单训练seed、复用DEV面板。它只测试policy出招，不能验证带搜索棋力或value的棋力贡献。差值是同一对手面板上的得分差，不是分支直接互战胜率。', '',
        '| 模型 | 胜/和/负 | 得分率 |', '| --- | --- | ---: |']
    for name, r in greedy.items():
        lines.append(f"| {name} | {r['wins']}/{r['draws']}/{r['losses']} | {100*r['score']:.3f}% |")
    for name, d in greedy_differences.items():
        lines += ['', f"{name}：{100*d['estimate']:+.3f}pp，95% CI [{100*d['lower']:+.3f}, {100*d['upper']:+.3f}]。"]
    lines += ['', '配置、代码身份、数据来源、种子、逐步抽样核对、学习曲线、逐局验证统计与原始对战摘要均在本目录。训练checkpoint在runs同名目录。两个新value辅助头本次未接入。', '']
    (ART / 'results.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'status': 'complete', 'greedy': greedy, 'validation_differences': comparisons}), flush=True)


if __name__ == '__main__':
    main()
