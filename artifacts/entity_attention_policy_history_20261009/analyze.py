"""Read completed nodes and paired, whole-game holdout differences; no GPU."""
from pathlib import Path
import json

import numpy as np

from splendor.alphazero_recording import read_json, write_json, sha256

ART = Path(__file__).resolve().parent
ROOT = ART.parents[1]
RUNTIME = ROOT / 'runtime' / ART.name
NODES = (50, 200, 500)


def game_difference(key, left, right, step, group, *, right_step=None):
    a = read_json(RUNTIME / key / left / f'holdout_games_{step:06d}.json')
    b = read_json(RUNTIME / key / right / f'holdout_games_{step if right_step is None else right_step:06d}.json')
    a = {(x['round'], x['game_id']): x for x in a}
    b = {(x['round'], x['game_id']): x for x in b}
    assert a.keys() == b.keys()
    latest = max(r for r, _ in a)
    keys = sorted(k for k in a if group == 'recent_four' or k[0] == latest)
    weights = np.asarray([a[k]['policy_weight_sum'] for k in keys], dtype=np.float64)
    assert all(a[k]['policy_weight_sum'] == b[k]['policy_weight_sum'] for k in keys)
    delta = np.asarray([a[k]['policy_ce_sum']-b[k]['policy_ce_sum'] for k in keys], dtype=np.float64)
    return delta, weights


def interval(cases, seed=28400000, resamples=10000):
    rng = np.random.default_rng(seed)
    estimates, bootstrap = [], []
    for delta, weights in cases:
        estimates.append(float(delta.sum()/weights.sum()))
        chosen = rng.integers(0, len(delta), size=(resamples, len(delta)))
        bootstrap.append(delta[chosen].sum(axis=1)/weights[chosen].sum(axis=1))
    distribution = np.mean(bootstrap, axis=0)
    lo, hi = np.quantile(distribution, [.025, .975])
    return {'estimate': float(np.mean(estimates)), 'lower': float(lo), 'upper': float(hi),
            'cases': len(cases), 'holdout_games': sum(len(x[0]) for x in cases), 'resamples': resamples,
            'seed': seed, 'unit': 'paired whole games, stratified within each fixed checkpoint case; '
                                  'equal case means, conditional on trained trajectories'}


def main():
    spec = read_json(ART/'specification.json')
    result = {'status': 'complete', 'specification_sha256': sha256(ART/'specification.json'),
              'analysis_script_sha256': sha256(Path(__file__)), 'cases': {}, 'pooled_fixed_cases': {},
              'pooled_changes_from_start': {},
              'scope': 'KL and MSE only; no strength inference; exploratory multiple nodes/arms, '
                       'three checkpoints on one training trajectory, not independent training seeds'}
    for key, subject in spec['subjects'].items():
        gradients = read_json(ART/f'gradients_{key}.json')
        case = {'round': subject['round'], 'start_step': subject['start_step'],
                'gradient_cosines': {g: v['cosine'] for g,v in gradients['gradient_latest_vs_previous'].items()},
                'AdamW_update_cosines_with_latest_descent': {
                    a: gradients['arms'][a]['direction_by_group']['policy_all']['update_vs_negative_latest_gradient']['cosine']
                    for a in 'ABC'}, 'arms': {}, 'paired_holdout_differences': {}}
        for arm in 'ABC':
            points = []
            baseline = read_json(ART/'nodes'/f'{key}_{arm}_000000.json')
            for step in (0,)+NODES:
                p = ART/'nodes'/f'{key}_{arm}_{step:06d}.json'
                if not p.exists():
                    result['status'] = 'incomplete'; continue
                data = read_json(p)
                point = {'step': step}
                for group in ('latest', 'recent_four'):
                    point[group] = {}
                    for split in ('train_fixed_subset', 'holdout_full'):
                        point[group][split] = {m: data[group][split][m] for m in ('policy_kl', 'value_loss')}
                        point[group][split]['KL_change_from_start'] = data[group][split]['policy_kl']-baseline[group][split]['policy_kl']
                points.append(point)
            case['arms'][arm] = points
        for step in NODES:
            case['paired_holdout_differences'][str(step)] = {}
            for left, right in [('B', 'A'), ('C', 'A'), ('C', 'B')]:
                if not (ART/'nodes'/f'{key}_{left}_{step:06d}.json').exists() or not (ART/'nodes'/f'{key}_{right}_{step:06d}.json').exists():
                    continue
                case['paired_holdout_differences'][str(step)][f'{left}-{right}'] = {
                    group: interval([game_difference(key,left,right,step,group)])
                    for group in ('latest', 'recent_four')}
        result['cases'][key] = case
    if result['status'] == 'complete':
        for step in NODES:
            result['pooled_fixed_cases'][str(step)] = {}
            result['pooled_changes_from_start'][str(step)] = {
                arm: {group: interval([game_difference(key,arm,arm,step,group,right_step=0)
                                      for key in spec['subjects']])
                      for group in ('latest', 'recent_four')} for arm in 'ABC'}
            for left, right in [('B', 'A'), ('C', 'A'), ('C', 'B')]:
                result['pooled_fixed_cases'][str(step)][f'{left}-{right}'] = {
                    group: interval([game_difference(key,left,right,step,group) for key in spec['subjects']])
                    for group in ('latest', 'recent_four')}
    write_json(ART/'analysis.json', result)
    print(json.dumps({'status': result['status'], 'output': str(ART/'analysis.json')},ensure_ascii=False))


if __name__ == '__main__':
    main()
