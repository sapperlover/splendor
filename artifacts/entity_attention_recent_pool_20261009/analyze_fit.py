"""Review completed fixed-pool training, with paired whole-game holdout deltas."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from splendor.alphazero_recording import read_json, sha256, write_json

ART=Path(__file__).resolve().parent


def paired_change(before, after, *, numerator, denominator, seed):
    keys=sorted(k for k in before if k.endswith('/all'))
    if set(keys) != {k for k in after if k.endswith('/all')}:
        raise ValueError('holdout games changed')
    den=np.asarray([before[k][denominator] for k in keys],dtype=np.float64)
    if not np.array_equal(den,np.asarray([after[k][denominator] for k in keys])):
        raise ValueError('holdout weights changed')
    delta=np.asarray([after[k][numerator]-before[k][numerator] for k in keys],dtype=np.float64)
    rng=np.random.default_rng(seed)
    boot=[]
    for start in range(0,10000,128):
        indices=rng.integers(0,len(keys),size=(min(128,10000-start),len(keys)))
        boot.extend((delta[indices].sum(-1)/den[indices].sum(-1)).tolist())
    return {'estimate':float(delta.sum()/den.sum()),'lower':float(np.percentile(boot,2.5)),
        'upper':float(np.percentile(boot,97.5)),'games':len(keys),'seed':seed,'resamples':10000,
        'unit':'one complete original validation game; paired ratio of summed errors to summed weights'}


def main():
    summary=read_json(ART/'training_summary.json')
    if summary['status'] != 'complete':
        raise ValueError('training incomplete')
    steps=[34398,68796,103194]
    full={s:read_json(ART/f'full_holdout_step_{s:09d}.json') for s in steps}
    result={'protocol':'recent-pool-fit-review-v1','training_status':'complete','final_step':103194,
        'fixed_subset_best_value':summary['best_validation_mse'],
        'fixed_subset_best_policy':summary['best_validation_kl'],
        'full_holdout':{str(s):full[s]['evaluation']['pool_weighted']['validation'] for s in steps},
        'stage_metrics':{},'source_changes_4_to_12':{},'paired_changes':{},
        'scope':'stored original teacher labels; repeated DEV holdout, one initialization/training seed; bootstrap does not cover training-seed variability or adaptive checkpoint selection',
        'inputs':{str(p):sha256(p) for p in [ART/'training_summary.json',*[ART/f'full_holdout_step_{s:09d}.json' for s in steps]]}}
    fields=['positions','policy_weight','policy_kl','value_mse','value_zero_mse','value_bias','value_correlation','teacher_top1_agreement']
    for s in steps:
        m=full[s]['evaluation']['metrics']
        result['stage_metrics'][str(s)]={stage:{k:m[f'__all_observed__/validation/{stage}'][k] for k in fields}
            for stage in ['all','early_mid','late']}
    first,last=[full[s]['evaluation']['metrics'] for s in [34398,103194]]
    for key in first:
        if not key.endswith('/validation/all') or key.startswith('__'):
            continue
        result['source_changes_4_to_12'][key]={
            'policy_kl_before':first[key]['policy_kl'],'policy_kl_after':last[key]['policy_kl'],
            'delta_policy_kl':last[key]['policy_kl']-first[key]['policy_kl'],
            'value_mse_before':first[key]['value_mse'],'value_mse_after':last[key]['value_mse'],
            'delta_value_mse':last[key]['value_mse']-first[key]['value_mse']}
    for i,(a,b) in enumerate([(34398,68796),(68796,103194),(34398,103194)]):
        before,after=[full[s]['evaluation']['per_game'] for s in [a,b]]
        result['paired_changes'][f'{b}_minus_{a}']={
            'policy_kl':paired_change(before,after,numerator='kl',denominator='policy_weight',seed=30300000+2*i),
            'value_mse':paired_change(before,after,numerator='sq_error',denominator='positions',seed=30300001+2*i)}
    result['interpretation']={
        'policy':'Heldout stored KL continues to decrease with smaller late gains; train/validation policy gaps do not exhibit value-like divergence. LR also drops 4x at exposure 8, so the curve alone cannot establish an architecture/noise fitting limit.',
        'value':'Train MSE continues downward while full holdout MSE rises across all seven sources from exposure 4 to 12, with worse early/mid and late correlations: clear overfitting of terminal-value targets in this fixed-pool experiment.',
        'stopping':'Exposure 4 is best for value among the three full milestone evaluations; subset best is 31000 steps. Neither is an established optimal step or strength-best checkpoint.',
        'strength':'One predeclared final-103194 S1024/N1 panel versus theta5000 is being measured separately; this single panel cannot identify the best training duration or isolate critic damage.'}
    write_json(ART/'fit_review.json',result)
    print(json.dumps({'full_holdout':result['full_holdout'],'paired_changes':result['paired_changes']},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
