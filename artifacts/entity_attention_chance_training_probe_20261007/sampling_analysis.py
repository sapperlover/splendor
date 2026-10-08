"""Finite-data coverage and effective targets, recomputed from recorded raw ledgers."""
from collections import Counter
from pathlib import Path
import json

import numpy as np
from splendor.alphazero_recording import read_json, write_json
from splendor.selfplay_data import load_game
from experiment import ART, ARMS, check_source, record

if __name__=='__main__':
    check_source()
    results={}
    for arm in ARMS:
        data=read_json(ART/f'dataset_{arm}.json')
        eligible=set();positions=set();roots=[]
        for game in data['games']:
            arrays,meta=load_game(game['path'])
            if game['split']=='train':
                positions.update((game['game_id'],i) for i in range(game['decision_count']))
                eligible.update((game['game_id'],int(i)) for i in np.flatnonzero(arrays['policy_weight']))
            roots.extend(r for r in meta['search_diagnostics'] if r['stats']['simulations'])
        ledger=np.load(read_json(ART/f'train_{arm}.json')['batch_ledger']['path'])
        metrics=[json.loads(x) for x in (ART/f'metrics_{arm}.jsonl').read_text().splitlines()]
        coverage={}
        for step in (100,300,500):
            coverage[str(step)]={}
            for head,name in enumerate(('policy','value')):
                refs=[tuple(map(int,x)) for x in ledger[:step,head].reshape(-1,2)]
                support=eligible if name=='policy' else positions
                effective=[r for r in refs if r in support]
                assert all(r in positions for r in refs)
                used=Counter(effective)
                coverage[str(step)][name]={
                    'draws':len(refs),'effective_draws':len(effective),'eligible_positions':len(support),
                    'distinct_effective_positions':len(used),'coverage_fraction':len(used)/len(support),
                    'effective_draws_per_eligible_position':len(effective)/len(support),
                    'clipped_steps':sum(x[name+'_grad_clipped'] for x in metrics[:step]),
                    'mean_unclipped_grad_norm':float(np.mean([x[name+'_grad_norm'] for x in metrics[:step]]))}
        strata={}
        for label,lo,hi in (('all',0,100),('score_0_4',0,5),('score_5_9',5,10),('score_10_plus',10,100)):
            group=[r for r in roots if lo<=r['max_visible_score']<hi]
            sims=sum(r['stats']['simulations'] for r in group)
            strata[label]={'roots':len(group),'simulations':sims,
                'mean_depth':sum(r['stats']['tree_depth_sum'] for r in group)/sims,
                'terminal_fraction':sum(r['stats']['terminal_hits'] for r in group)/sims}
            if arm!='native':
                keys=('chance_draws','root_hidden_samples','information_set_hits')
                if arm=='cpw':keys+=('chance_pw_fresh_samples','chance_pw_reuses')
                strata[label].update({k:sum(r['chance'][k] for r in group) for k in keys})
        results[arm]={'train_positions':len(positions),'train_policy_positions':len(eligible),
                      'coverage':coverage,'collection_search_strata':strata}
    output={'protocol':'chance-training-probe-sampling-analysis-v1','arms':results,'script':record(Path(__file__))}
    write_json(ART/'sampling_analysis.json',output)
    print(json.dumps({a:{'coverage_100':r['coverage']['100'],'search':r['collection_search_strata']['all']} for a,r in results.items()}))
