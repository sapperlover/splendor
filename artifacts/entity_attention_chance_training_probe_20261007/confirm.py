"""Independent game-seed confirmation of the screened primary100 CPW/control pair."""
from pathlib import Path
import json
import subprocess
import sys
import numpy as np
from splendor.alphazero_recording import read_json,write_json
from experiment import ART,check_source,record

if __name__=='__main__':
    spec=check_source()
    assert read_json(ART/'analysis.json')['complete_main_panels']
    panels=spec['confirmation_panels']
    results={}
    for panel in panels:
        name=panel['id'];path=ART/f'evaluations/{name}.json'
        if not path.exists() or not read_json(path)['passed']:
            subprocess.run([sys.executable,'-u',str(ART/'experiment.py'),'evaluate','--case',name],check=True)
        result=read_json(path);assert result['passed'] and result['valid_games']==200
        results[panel['arm']]=result
    payoff={'a_win':1.,'a_loss':0.,'draw':.5}
    def scores(result):
        return np.array([np.mean([payoff[g['outcome_for_a']] for g in p['games']]) for p in result['pairs']])
    assert [p['game_seed'] for p in results['cpw']['pairs']]==[p['game_seed'] for p in results['native']['pairs']]
    delta=scores(results['cpw'])-scores(results['native'])
    rng=np.random.default_rng(16890000)
    values=delta[rng.integers(0,100,size=(10000,100))].mean(axis=1)
    output={'protocol':'chance-training-probe-confirmation-v1','selection':spec['confirmation_selection'],
        'difference_cpw_minus_native':{'difference':float(delta.mean()),'lower':float(np.quantile(values,.025)),
            'upper':float(np.quantile(values,.975)),'pairs':100,'confidence':.95,'resamples':10000,'seed':16890000},
        'results':{a:{k:r[k] for k in ('win_rate_score','paired_bootstrap','valid_games','a_wins','draws','a_losses','elapsed_seconds_this_invocation')} for a,r in results.items()},
        'scope':'Independent evaluation game/policy seeds only; same frozen student weights and training seed. Neither new data nor model updates; compare relative learning effect, not replicated training or proof of absolute gain.',
        'inputs':[record(ART/f"evaluations/{p['id']}.json") for p in panels],'script':record(Path(__file__))}
    write_json(ART/'confirmation.json',output)
    print(json.dumps(output,ensure_ascii=False),flush=True)
