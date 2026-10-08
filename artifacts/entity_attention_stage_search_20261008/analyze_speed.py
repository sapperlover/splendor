"""Compare bounded execution measurements without treating smoke scores as strength."""
from pathlib import Path
import statistics

import numpy as np

from splendor.alphazero_recording import read_json, write_json

ART = Path(__file__).resolve().parent / 'speed'


def main():
    cases = {name: read_json(ART / f'{name}.json') for name in ['eager64', 'eager200', 'graph200']}
    baseline = cases['eager64']['windows'][1]['records']
    summary = {'scope': 'Same 200 frozen roots, one frozen16500 model, S1024/N1; '
                        'first window includes cold startup; median of windows1..3 is warm. '
                        'This is an execution benchmark, not a complete 500-pair time estimate.',
               'cases': {}, 'gpu_smoke': {}}
    for name, case in cases.items():
        if any(case[key] != cases['eager64'][key] for key in
               ['checkpoint_sha256', 'bank_sha256', 'code_hashes', 'benchmark_sha256',
                'sampling_seed', 'search_config']):
            raise ValueError('benchmark inputs/code differ')
        windows = case['windows']
        actual = windows[1]['records']
        if [r['bank_index'] for r in baseline] != [r['bank_index'] for r in actual]:
            raise ValueError('benchmark roots differ')
        changed = sum(a['chosen_candidate_index'] != b['chosen_candidate_index']
                      for a, b in zip(actual, baseline))
        summary['cases'][name] = {
            'cold_seconds': windows[0]['wall_seconds'],
            'warm_median_seconds': statistics.median(w['wall_seconds'] for w in windows[1:]),
            'warm_median_simulations_per_second': statistics.median(w['simulations_per_second'] for w in windows[1:]),
            'changed_actions_vs_eager64': changed, 'roots': len(actual),
            'maximum_prior_absolute_error': max(abs(x - y) for a, b in zip(actual, baseline)
                                               for x, y in zip(a['prior'], b['prior'])),
            'mean_target_total_variation': float(np.mean([
                sum(abs(x - y) for x, y in zip(a['target'], b['target'])) / 2
                for a, b in zip(actual, baseline)])),
            'warm_memory_samples': [w['memory'] for w in windows[1:]],
            'active_jobs_after_windows': [w['active_jobs'] for w in windows],
            'after_last_batch_release': case['after_last_batch_release'],
            'inference_stats': case['inference_stats']}
    for name, row in summary['cases'].items():
        row['warm_speedup_vs_eager64'] = summary['cases']['eager64']['warm_median_seconds'] / row['warm_median_seconds']
    for mode in ['late_only', 'early_only']:
        if not all((ART / f'{mode}_{execution}.json').exists() for execution in ['eager', 'fast']):
            continue
        panels = {execution: read_json(ART / f'{mode}_{execution}.json') for execution in ['eager', 'fast']}
        rows = {execution: {pair['pair_id']: sorted(pair['games'], key=lambda r: r['game_in_pair'])
                            for pair in panel['pairs']}
                for execution, panel in panels.items()}
        digest_matches = sum(a['final_state_digest'] == b['final_state_digest']
            for pair in rows['eager'] for a, b in zip(rows['eager'][pair], rows['fast'][pair]))
        summary['gpu_smoke'][mode] = {'games': panels['fast']['completed_games'],
            'passed': all(p['passed'] for p in panels.values()),
            'identical_final_state_digests': digest_matches,
            'timing_and_memory': {execution: read_json(ART / f'{mode}_{execution}.memory.json')
                                  for execution in panels},
            'scope': '8 pairs per execution, independent diagnostic seeds; too small for strength inference '
                     'and does not fill 200 active slots. Cold compilation is included.'}
    write_json(ART / 'analysis.json', summary)
    print({name: {key: row[key] for key in ['cold_seconds', 'warm_median_seconds',
                                         'warm_speedup_vs_eager64', 'changed_actions_vs_eager64']}
           for name, row in summary['cases'].items()})


if __name__ == '__main__':
    main()
