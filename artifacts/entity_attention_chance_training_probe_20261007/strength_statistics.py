"""Paired before/after and update-length comparisons from completed DEV raw games."""
from pathlib import Path
import json
from splendor.alphazero_recording import read_json, write_json
from experiment import ART, ARMS, check_source, bootstrap_difference, record

if __name__=='__main__':
    spec=check_source()
    results={p['id']:read_json(ART/f"evaluations/{p['id']}.json") for p in spec['panels']}
    assert all(r['passed'] and r['valid_games']==100 for r in results.values())
    comparisons={}
    for arm in ARMS:
        for step in (100,500):
            name=f'{arm}_{step}_native'
            comparisons[f'{name}_minus_initial_native']=bootstrap_difference(results[name],results['initial_native'])
        for mode in ('greedy','native'):
            before=f'{arm}_100_{mode}';after=f'{arm}_500_{mode}'
            comparisons[f'{after}_minus_{before}']=bootstrap_difference(results[after],results[before])
    output={'protocol':'chance-training-probe-strength-statistics-v1','seed':16860000,
            'resamples':10000,'confidence':.95,'paired_differences':comparisons,
            'scope':'DEV50 matched game seeds and swapped seats. An initial-native self-match estimate is recorded rather than assumed exactly .5. Exploratory comparisons, no multiplicity correction or training-seed replication.',
            'inputs':[record(ART/f"evaluations/{p['id']}.json") for p in spec['panels']],
            'script':record(Path(__file__))}
    write_json(ART/'strength_statistics.json',output)
    print(json.dumps(comparisons))
