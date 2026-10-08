"""CPU-only per-position holdout outputs and game-cluster uncertainty; no games/training."""
from pathlib import Path
import gc
import hashlib
import json

import numpy as np
import torch

from splendor.alphazero_recording import read_json, write_json, sha256
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.offline_train import make_index
from splendor.selfplay_data import collate_selfplay_samples
from experiment import ART, ARMS, INITIAL, INITIAL_SHA, NODES, check_source, checkpoint, record


def clustered_ci(rows, deltas):
    keys = sorted({r['game_key'] for r in rows})
    mapping = {key:i for i,key in enumerate(keys)}
    index = np.array([mapping[r['game_key']] for r in rows])
    numerator = np.bincount(index, weights=deltas, minlength=len(keys))
    denominator = np.bincount(index, minlength=len(keys))
    rng = np.random.default_rng(16860000)
    draws = rng.integers(0,len(keys),size=(10000,len(keys)))
    samples = numerator[draws].sum(axis=1) / denominator[draws].sum(axis=1)
    return {'difference':float(np.mean(deltas)), 'lower':float(np.quantile(samples,.025)),
            'upper':float(np.quantile(samples,.975)), 'games':len(keys), 'positions':len(rows),
            'bootstrap_replicates':10000, 'bootstrap_seed':16860000}


def own_holdout(arm):
    data = read_json(ART / f'dataset_{arm}.json')
    index = make_index(data, 1024**3)
    games = index.split_games('validation')
    samples = [index.cache.sample(g,i) for g in games for i in range(g.decision_count)]
    predictions = {}
    for step in NODES:
        model = load_frozen_actor_critic_checkpoint(checkpoint(arm,step),device='cpu')
        output = []
        with torch.inference_mode():
            for start in range(0,len(samples),256):
                subset = samples[start:start+256]
                batch = collate_selfplay_samples(subset)
                logits, values = model(batch.decisions)
                raw_logits = logits.cpu().numpy().astype(np.float64)
                values = values.cpu().numpy().astype(np.float64)
                for row, sample, value in zip(raw_logits,subset,values):
                    arrays, meta, i = sample.arrays, sample.metadata, sample.index
                    begin,end = map(int,arrays['candidate_offsets'][i:i+2])
                    pi = arrays['policy_target'][begin:end].astype(np.float64)
                    logit = row[:end-begin]
                    logp = logit - np.logaddexp.reduce(logit)
                    nz=pi>0
                    diag = meta['search_diagnostics'][i]
                    z=int(arrays['value_target'][i])
                    output.append({'game_key':f"{meta['run_id']}/{meta['game_id']}",
                        'game_id':meta['game_id'],'decision_index':i,'phase':int(arrays['phase'][i]),
                        'max_visible_score':diag['max_visible_score'],'completed_player_turns':diag['completed_player_turns'],
                        'policy_weight':float(arrays['policy_weight'][i]),'value':float(value),'value_target':z,
                        'value_sq_error':float((value-z)**2),'search_clipped_value_sq_error':float((np.clip(value,-1,1)-z)**2),
                        'value_sign_correct':bool(value*z>0),'policy_kl':float(np.sum(pi[nz]*(np.log(pi[nz])-logp[nz]))),
                        'network_chosen_index':int(logit.argmax()),'target_chosen_index':int(pi.argmax())})
        predictions[step] = output
        del model
        gc.collect()
    differences = {}
    strata = {}
    for step in NODES:
        rows=predictions[step]
        strata[str(step)]={}
        for stage,lo,hi in (('all',0,100),('score_0_4',0,5),('score_5_9',5,10),('score_10_plus',10,100)):
            chosen=[r for r in rows if lo<=r['max_visible_score']<hi]
            policy=[r for r in chosen if r['policy_weight']]
            decisive=[r for r in chosen if r['value_target']]
            strata[str(step)][stage]={'positions':len(chosen),'policy_positions':len(policy),
                'value_mse':float(np.mean([r['value_sq_error'] for r in chosen])),
                'policy_kl':float(np.mean([r['policy_kl'] for r in policy])) if policy else None,
                'value_sign_accuracy':float(np.mean([r['value_sign_correct'] for r in decisive])),
                'value_outside_payoff_fraction':float(np.mean([abs(r['value'])>1 for r in chosen]))}
        if step:
            base=predictions[0]
            assert [(r['game_id'],r['decision_index']) for r in rows]==[(r['game_id'],r['decision_index']) for r in base]
            policy_idx=[i for i,r in enumerate(rows) if r['policy_weight']]
            differences[str(step)]={
                'value_mse_after_minus_before':clustered_ci(rows,np.array([r['value_sq_error']-b['value_sq_error'] for r,b in zip(rows,base)])),
                'policy_kl_after_minus_before':clustered_ci([rows[i] for i in policy_idx],
                    np.array([rows[i]['policy_kl']-base[i]['policy_kl'] for i in policy_idx]))}
    saved={'arm':arm,'device':'cpu','predictions':{str(k):v for k,v in predictions.items()},
           'by_visible_score':strata,'paired_within_arm_differences':differences,
           'inputs':[record(checkpoint(arm,s)) for s in NODES]+[record(ART / f'dataset_{arm}.json')],
           'script':record(Path(__file__))}
    write_json(ART / f'holdout_predictions_{arm}.json',saved)
    return saved


if __name__=='__main__':
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    check_source()
    output={'protocol':'chance-training-probe-holdout-diagnostics-v1','own_holdouts':{},'common_differences':{}}
    for arm in ARMS:
        result=own_holdout(arm)
        output['own_holdouts'][arm]={'by_visible_score':result['by_visible_score'],
                                   'within_arm_differences':result['paired_within_arm_differences'],
                                   'predictions':record(ART / f'holdout_predictions_{arm}.json')}
    training={a:read_json(ART / f'train_{a}.json') for a in ARMS}
    for step in (100,300,500):
        base=next(x for x in training['native']['curve'] if x['step']==step)['common']['per_root']
        for arm in ('chance','cpw'):
            rows=next(x for x in training[arm]['curve'] if x['step']==step)['common']['per_root']
            assert [r['root_id'] for r in rows]==[r['root_id'] for r in base]
            output['common_differences'][f'{arm}_{step}_minus_native']={
                'value_mse':clustered_ci(rows,np.array([r['value_sq_error']-b['value_sq_error'] for r,b in zip(rows,base)])),
                **{f'policy_kl_to_{teacher}_frozen_mean':clustered_ci(rows,np.array([
                    r['policy_kl_to_frozen_mean_targets'][teacher]-b['policy_kl_to_frozen_mean_targets'][teacher]
                    for r,b in zip(rows,base)])) for teacher in ARMS}}
    write_json(ART / 'holdout_diagnostics.json',output)
    print(json.dumps({'passed':True,'arms':list(ARMS),'observations':list(NODES)},ensure_ascii=False))
