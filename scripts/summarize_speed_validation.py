"""Aggregate completed speed experiments without rerunning their workloads."""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def metadata(path):
    with zipfile.ZipFile(path) as archive:
        with archive.open('metadata_json.npy') as stream:
            data = np.lib.format.read_array(stream, allow_pickle=False)
    return json.loads(data.tobytes())


def resources(record):
    data = read(record['resources'])
    result = {'sample_interval_seconds': 5, 'samples': len(data),
        'max_tree_rss_bytes': max(row['tree_rss_bytes'] for row in data),
        'max_tree_pss_bytes': max(row['tree_pss_bytes'] for row in data),
        'tree_rss_note': 'sum includes shared pages repeatedly; PSS apportions shared mappings',
        'gpu_device_note': 'device total includes graphics, context, reserved and allocated memory'}
    gpu = [list(map(float, row['gpu'].split(','))) for row in data if row['gpu']]
    if gpu:
        names = ('utilization_percent', 'memory_mib', 'temperature_c', 'power_w', 'sm_mhz', 'memory_mhz')
        result['gpu'] = {name: {'mean': float(np.mean([row[i] for row in gpu])),
                               'max': max(row[i] for row in gpu), 'min': min(row[i] for row in gpu)}
                         for i, name in enumerate(names)}
    return result


def collection(record):
    assert len(record.get('sessions', [record])) == 1, 'aggregate resumed sessions before ranking their total work'
    result = record['result']
    summary, status = result['summary'], result['status']
    assert summary['completed_games'] == 1000 and summary['missing_games'] == 0
    assert summary['exceptions'] == summary['engineering_truncations'] == 0
    assert result['checked_replay']['completed_games'] == 1000
    assert summary['statistics_complete']
    wall = result['main_seconds']
    owner, infer = status['gpu_owner'], status['gpu_owner']['inference']
    phases = {key: owner[key] for key in ('owner_wait_input_seconds', 'owner_wait_coalesce_seconds',
                                        'owner_gather_seconds', 'owner_dispatch_seconds')}
    phases.update({key: infer[key] for key in ('forward_seconds', 'h2d_seconds', 'd2h_seconds', 'wait_seconds')})
    phases['initialization_validation_and_other'] = wall - sum(phases.values())
    assert phases['initialization_validation_and_other'] >= 0
    rows = infer['logical_model_positions']
    assert rows == sum(int(width_rows) for width_rows in owner['width_histogram'].values())
    assert rows == status['native_batch_rows']
    assert infer['physical_model_batches'] == sum(infer['batch_size_histogram'].values())
    physical_positions = sum(int(k) * v for k, v in owner['width_histogram'].items())
    milestones = result['milestones']
    thresholds = {str(n): next(m['seconds'] for m in milestones if m['completed_games'] >= n)
                  for n in (500, 900, 990, 1000)}
    last = {str(n): {'seconds_until_last_commit': thresholds['1000'] - next(
        m['seconds'] for m in milestones if m['completed_games'] >= 1000 - n),
        'rows': result['aggregate_inference'][f'last:{n}']['rows'],
        'batches': result['aggregate_inference'][f'last:{n}']['batches'],
        'inference_wall_seconds': result['aggregate_inference'][f'last:{n}']['inference_wall_seconds']}
        for n in (100, 10, 8, 1)}
    source = Path(record['command'][record['command'].index('--run') + 1])
    games = [metadata(path) for path in sorted((source / 'games').glob('*.npz'))]
    lengths = [g['completed_player_turns'] for g in games]
    latency = status['decision_latencies_ms']
    data = {'main_seconds': wall, 'checked_replay_seconds': result['checked_replay_seconds'],
        'delivery_seconds': result['delivery_seconds'], 'games': 1000,
        'games_per_minute': 60000 / wall, 'policy_samples': summary['policy_samples'],
        'samples': summary['samples'], 'simulations': summary['simulations'], 'physical_rows': rows,
        'policy_samples_per_second': summary['policy_samples'] / wall,
        'samples_per_second': summary['samples'] / wall, 'simulations_per_second': summary['simulations'] / wall,
        'physical_rows_per_second': rows / wall, 'batch_mean': infer['mean_batch_size'],
        'batch_fill_fraction': infer['mean_batch_size'] / 500,
        'data_bytes': summary['data_bytes'], 'owner_exclusive_seconds': phases,
        'owner_exclusive_percent': {k: 100 * v / wall for k, v in phases.items()},
        'completion_seconds': thresholds, 'last_games_nested': last,
        'length': {'mean': float(np.mean(lengths)), 'median': float(np.median(lengths)),
                   'p90': float(np.percentile(lengths, 90)), 'p99': float(np.percentile(lengths, 99)), 'max': max(lengths)},
        'decision_latency_ms_including_queue': {str(p): float(np.percentile(latency, p)) for p in (50, 90, 99)},
        'physical_candidate_positions': physical_positions,
        'owner_padding_positions': owner['padding_positions'],
        'descriptor_candidate_positions': physical_positions - owner['padding_positions'],
        'width_rows': owner['width_histogram'], 'compile': infer['compile'], 'resources': resources(record),
        'width_inference': {k: v for k, v in result['aggregate_inference'].items() if k.startswith('width:')},
        'owner_initialization_seconds': owner['owner_initialization_seconds'],
        'owner_cpu_seconds': owner['owner_cpu_seconds'],
        'worker_cpu_seconds': sum(w['cpu_seconds'] for w in owner['workers']),
        'worker_result_wait_seconds': sum(w['wait_result_seconds'] for w in owner['workers']),
        'worker_queue_backpressure_seconds': sum(w['queue_backpressure_seconds'] for w in owner['workers']),
        'worker_overlapping_seconds': {k: summary[k] for k in ('real_rule_seconds', 'root_encoding_seconds',
            'search_rule_seconds', 'native_compute_seconds', 'native_fill_seconds', 'write_seconds')},
        'torch_gpu_allocator': result.get('torch_gpu_allocator', {'max_allocated_bytes': owner['gpu_peak_bytes'],
            'reserved_note': 'not captured by the earlier baseline collection harness'}),
        'game_statistics': [{'game_id': g['game_id'], 'seed': g['game_seed'], 'seat': g['first_player'],
                            'length': g['completed_player_turns'], 'decisions': g['decision_count'],
                            'policy_samples': g['policy_samples'], 'collection_seconds': g['collection_seconds'],
                            'winner_ids': g['terminal_result']['winner_ids'],
                            'terminal_reason': g['terminal_result']['reason'],
                            'trace_sha256': hashlib.sha256(json.dumps(g['trace'], sort_keys=True).encode()).hexdigest()}
                           for g in games]}
    if owner.get('worker_row_statistics'):
        actual = sum(int(width) * count for w in owner['worker_row_statistics']
            for stage in w['actual_width_rows_by_root_stage'].values() for width, count in stage.items())
        actual_rows = sum(count for w in owner['worker_row_statistics']
            for stage in w['actual_width_rows_by_root_stage'].values() for count in stage.values())
        assert actual_rows == rows
        descriptor = sum(int(width) * count for w in owner['worker_row_statistics']
                         for width, count in w['descriptor_width_rows'].items())
        assert descriptor == data['descriptor_candidate_positions']
        root_stages = {}
        for stage in ('opening', 'middle', 'late'):
            widths = {}
            for worker in owner['worker_row_statistics']:
                for width, count in worker['actual_width_rows_by_root_stage'].get(stage, {}).items():
                    widths[width] = widths.get(width, 0) + count
            root_stages[stage] = {'actual_width_rows': widths, 'rows': sum(widths.values()),
                'candidate_positions': sum(int(w) * n for w, n in widths.items())}
        data.update(actual_candidate_positions=actual,
                    worker_padding_positions=data['descriptor_candidate_positions'] - actual,
                    total_padding_fraction=1 - actual / physical_positions,
                    root_stage_statistics=root_stages,
                    batch_trigger_detail=owner['batch_trigger_detail'],
                    tail_intervals_exclusive=owner['tail_intervals_exclusive'])
    return data


def learner(record):
    data = record['result']
    assert data['measured_steps'] == 128 and data['warmup_steps'] == 8
    chunks = []
    for start in range(0, 128, 32):
        steps = data['steps'][start:start + 32]
        sums = {}
        for step in steps:
            for key, value in step['host_wall_seconds'].items():
                sums[key] = sums.get(key, 0.) + value
        chunks.append({'first_measured_step': start + 1, 'wall_seconds': sum(s['wall_seconds'] for s in steps),
            'load_seconds': sum(s['decode_seconds'] for s in steps),
            'hits': sum(s['cache_delta'].get('hits', 0) for s in steps),
            'misses': sum(s['cache_delta'].get('misses', 0) for s in steps), 'host_seconds': sums})
    return {'main_seconds': data['main_seconds'], 'warmup_seconds': data['warmup_seconds'],
            'measured_seconds': data['measured_wall_seconds'], 'steps_per_second': 128 / data['measured_wall_seconds'],
            'last32_seconds': chunks[-1]['wall_seconds'], 'chunks': chunks,
            'exclusive_host_seconds': data['exclusive_host_wall_seconds'],
            'preparation_seconds': {k: data[k] for k in ('index_seconds', 'validation_prepare_seconds',
                                                       'initialization_seconds')},
            'checkpoint_and_validation': data['checkpoint_validation'],
            'torch_gpu_allocator': data['torch_gpu_allocator'],
            'cache_statistics_including_preparation': data['production_cache_stats'],
            'reference_sequence_sha256': data['reference_sequence_sha256'], 'resources': resources(record)}


def evaluation(record):
    assert len(record.get('sessions', [record])) == 1, 'aggregate resumed sessions before ranking their total work'
    data, summary = record['result'], record['result']['summary']
    assert summary['passed'] and summary['completed_pairs'] == 50 and summary['completed_games'] == 100
    work = sum(summary[k]['simulations'] for k in ('policy_a_stats', 'policy_b_stats'))
    games = [game for pair in summary['pairs'] for game in pair['games']]
    assert len(games) == 100
    result = {'main_seconds': data['main_seconds'], 'pairs_per_minute': 3000 / data['main_seconds'],
            'games_per_minute': 6000 / data['main_seconds'], 'simulations': work,
            'simulations_per_second': work / data['main_seconds'],
            'a_wins': summary['a_wins'], 'draws': summary['draws'], 'a_losses': summary['a_losses'],
            'score_rate': summary['win_rate_score'], 'a_first': summary['a_first'], 'a_second': summary['a_second'],
            'mean_decisions': summary['mean_decisions'], 'mean_turns': summary['mean_completed_player_turns'],
            'policy_cycle_draws': summary['policy_cycle_draws'], 'paired_bootstrap': summary['paired_bootstrap'],
            'policy_a_stats': summary['policy_a_stats'], 'policy_b_stats': summary['policy_b_stats'],
            'execution_stats': summary.get('execution_stats'), 'resources': resources(record),
            'torch_gpu_allocator': data.get('torch_gpu_allocator'),
            'game_statistics': [{key: g[key] for key in ('pair_id', 'game_in_pair', 'game_seed', 'policy_seeds',
                'first_player', 'policy_names', 'outcome_for_a', 'status', 'completed_player_turns',
                'decision_count', 'final_state_digest')} for g in games]}
    execution = summary.get('execution_stats')
    if execution is not None:
        phases = {key: execution[key] for key in ('model_creation_seconds', 'checked_rule_seconds', 'sidecar_seconds')}
        for model in execution['models'].values():
            for key in ('root_start_seconds', 'batch_non_inference_seconds'):
                phases[key] = phases.get(key, 0.) + model[key]
            for key in ('forward_seconds', 'h2d_seconds', 'd2h_seconds', 'wait_seconds'):
                phases[key] = phases.get(key, 0.) + model['inference'][key]
        phases['other_and_summary'] = data['main_seconds'] - sum(phases.values())
        assert phases['other_and_summary'] >= 0
        result['owner_exclusive_seconds'] = phases
        result['owner_exclusive_percent'] = {key: 100 * value / data['main_seconds'] for key, value in phases.items()}
    else:
        stats = [summary[k] for k in ('policy_a_stats', 'policy_b_stats')]
        phases = {key: sum(s[key] for s in stats) for key in ('core_forward_seconds', 'input_tensor_seconds',
                                                            'inference_other_seconds')}
        search = sum(s['total_seconds'] for s in stats)
        phases['search_except_inference'] = search - sum(phases.values())
        assert phases['search_except_inference'] >= 0
        result['worker_search_exclusive_seconds_sum'] = phases
        result['worker_search_exclusive_percent'] = {key: 100 * value / search for key, value in phases.items()}
        result['batch_histogram'] = {'1': sum(s['physical_batches'] for s in stats)}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('artifacts/end_to_end_speed_2026-10-03'))
    args = parser.parse_args()
    cases = [('collection_baseline_low', collection), ('collection_optimized', collection),
             ('learner_256', learner), ('learner_2048', learner),
             ('evaluation_baseline', evaluation), ('evaluation_optimized', evaluation)]
    result = {name: process(read(args.root / (name + '.json'))) for name, process in cases}
    comparisons = {}
    for label, base, opt in [('collection', 'collection_baseline_low', 'collection_optimized'),
                             ('learner', 'learner_256', 'learner_2048'),
                             ('evaluation', 'evaluation_baseline', 'evaluation_optimized')]:
        a, b = result[base]['main_seconds'], result[opt]['main_seconds']
        comparisons[label] = {'speedup': a / b, 'time_reduction_fraction': 1 - b / a,
                              'throughput_increase_fraction': a / b - 1}
    result['comparisons'] = comparisons
    for label, base, opt, field in [('collection', 'collection_baseline_low', 'collection_optimized', 'simulations'),
                                   ('evaluation', 'evaluation_baseline', 'evaluation_optimized', 'simulations')]:
        a, b = result[base], result[opt]
        comparisons[label]['work_ratio_optimized_over_baseline'] = b[field] / a[field]
        comparisons[label]['work_normalized_speedup'] = (b[field] / b['main_seconds']) / (a[field] / a['main_seconds'])
    for label, key in [('collection_behavior', 'game_statistics')]:
        a, b = result['collection_baseline_low'][key], result['collection_optimized'][key]
        assert [g['seed'] for g in a] == [g['seed'] for g in b]
        result[label] = {'same_trace_games': sum(x['trace_sha256'] == y['trace_sha256'] for x, y in zip(a, b)),
                         'same_winner_games': sum(x['winner_ids'] == y['winner_ids'] for x, y in zip(a, b)),
                         'same_length_games': sum(x['length'] == y['length'] for x, y in zip(a, b))}
    a, b = result['evaluation_baseline']['game_statistics'], result['evaluation_optimized']['game_statistics']
    identity = ('pair_id', 'game_in_pair', 'game_seed', 'policy_seeds', 'first_player', 'policy_names')
    assert [[g[k] for k in identity] for g in a] == [[g[k] for k in identity] for g in b]
    result['evaluation_behavior'] = {key: sum(x[key] == y[key] for x, y in zip(a, b))
        for key in ('final_state_digest', 'outcome_for_a', 'completed_player_turns', 'decision_count')}
    (args.root / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(comparisons, indent=2))


if __name__ == '__main__':
    main()
