"""Recompute stage/full-search scores and descriptive costs from paired raw games."""
from collections import Counter
import hashlib
from pathlib import Path

import numpy as np

from splendor.alphazero_recording import read_json, sha256, write_json
from splendor.paired_statistics import outcome_score, paired_bootstrap_mean, serialized_pair_scores

ART = Path(__file__).resolve().parent
PLAN = read_json(ART / 'plan_full.json')


def panel(mode, path):
    summary = read_json(path)
    seeds, scores = serialized_pair_scores(summary)
    if len(seeds) != PLAN['pairs'] or summary['valid_games'] != 2 * PLAN['pairs']:
        raise ValueError('incomplete panel')
    if any(identity['sha256'] != PLAN['checkpoint_sha256']
           for identity in summary['checkpoint_identity'].values()):
        raise ValueError('comparison weights differ')
    for key in ['search_config', 'inference']:
        expected = read_json(ART / 'fast/evaluations/late_only.json')['config']['policy_a'][key]
        if summary['config']['policy_a'][key] != expected:
            raise ValueError(f'comparison {key} differs')
    corpus = hashlib.sha256()
    games = []
    raw_root = Path(summary['game_log_root'])
    import json
    for pair in summary['pairs']:
        raw_path = raw_root / f"pair-{pair['pair_id']:07d}.jsonl"
        raw = raw_path.read_bytes()
        corpus.update(raw_path.name.encode() + b'\0' + raw)
        records = sorted([json.loads(line) for line in raw.splitlines() if line],
                         key=lambda g: g['game_in_pair'])
        if records != sorted(pair['games'], key=lambda g: g['game_in_pair']):
            raise ValueError('summary differs from durable raw pair records')
        games.extend(records)
    outcomes = Counter(game['outcome_for_a'] for game in games)
    if (outcomes['a_win'], outcomes['draw'], outcomes['a_loss']) != (
            summary['a_wins'], summary['draws'], summary['a_losses']):
        raise ValueError('summary outcomes differ')
    stats = summary['policy_a_stats']
    for key, value in stats.items():
        if isinstance(value, int) and sum(g['policy_a_stats'][key] for g in games) != value:
            raise ValueError(f'aggregated policy stats differ: {key}')
    if stats['simulations'] != 1024 * stats['searchable_decisions']:
        raise ValueError('S1024 budget differs')
    if stats['early_simulations'] + stats['late_simulations'] != stats['simulations']:
        raise ValueError('phase simulation counts differ')
    if mode == 'full_search':
        if stats['early_greedy_decisions'] + stats['late_greedy_decisions'] != 0:
            raise ValueError('full search skipped roots')
    elif stats[f"{'early' if mode == 'late_only' else 'late'}_simulations"]:
        raise ValueError('search ran outside its assigned stage')
    if any(g['policy_b_stats'] for g in games):
        raise ValueError('greedy opponent searched')
    bootstrap = paired_bootstrap_mean(scores, seed=PLAN['bootstrap_seed'],
                                      resamples=PLAN['bootstrap_resamples']).to_dict()
    if bootstrap != summary['paired_bootstrap']:
        raise ValueError('bootstrap interval differs')
    zero = [g for g in games if g['policy_a_stats']['simulations']
            and not g['policy_a_stats']['terminal_hits']]
    root_counts = [g['policy_a_stats']['searchable_decisions'] for g in games]
    metrics = {'score': float(scores.mean()), 'ci95': bootstrap, 'outcomes': dict(outcomes),
        'natural_terminal_games': sum(g['status'] == 'natural_terminal' for g in games),
        'cycle_draws': summary['policy_cycle_draws'],
        'a_first': summary['a_first'], 'a_second': summary['a_second'],
        'searched_roots': stats['searchable_decisions'],
        'searched_roots_per_game': stats['searchable_decisions'] / len(games),
        'searched_roots_quantiles': dict(zip(['min', 'p25', 'median', 'p75', 'p90', 'max'],
                                            map(float, np.percentile(root_counts, [0, 25, 50, 75, 90, 100])))),
        'simulations': stats['simulations'], 'model_positions': stats['model_positions'],
        'terminal_hits': stats['terminal_hits'],
        'terminal_fraction': stats['terminal_hits'] / stats['simulations'],
        'mean_tree_depth': stats['tree_depth_sum'] / stats['simulations'],
        'changed_actions': stats['changed_action_count'],
        'changed_action_fraction': stats['changed_action_count'] / stats['searchable_decisions'],
        'mean_completed_turns': summary['mean_completed_player_turns'],
        'late_entry_mean_completed_turns': (stats['late_entry_completed_turns_sum'] / stats['late_entry_count']),
        'mean_final_score_margin': float(np.mean([g['scores'][0] - g['scores'][1] for g in games])),
        'wall_seconds_this_invocation': summary['elapsed_seconds_this_invocation'],
        'zero_terminal_hit_searched_games': {'games': len(zero),
            'outcomes': dict(Counter(g['outcome_for_a'] for g in zero)),
            'score': float(np.mean([outcome_score(g['outcome_for_a']) for g in zero])) if zero else None,
            'scope': 'Post-selected descriptive subset; not a causal estimate or randomized ablation.'},
        'phase_stats': {phase: {'search_decisions': stats[f'{phase}_search_decisions'],
            'greedy_decisions': stats[f'{phase}_greedy_decisions'],
            'simulations': stats[f'{phase}_simulations'], 'terminal_hits': stats[f'{phase}_terminal_hits'],
            'terminal_fraction': (stats[f'{phase}_terminal_hits'] / stats[f'{phase}_simulations']
                                  if stats[f'{phase}_simulations'] else None),
            'mean_tree_depth': (stats[f'{phase}_tree_depth_sum'] / stats[f'{phase}_simulations']
                               if stats[f'{phase}_simulations'] else None)} for phase in ['early', 'late']},
        'verification': {'raw_games': len(games), 'raw_pair_count': len(summary['pairs']),
            'ordered_pair_corpus_sha256': corpus.hexdigest(), 'summary_sha256': sha256(path),
            'integer_stats_and_outcomes_match_raw': True, 'paired_ci_recomputed': True}}
    return metrics, seeds, scores, {(g['game_seed'], g['game_in_pair']): g for g in games}


def main():
    paths = {'late_only': ART / 'fast/evaluations/late_only.json',
             'early_only': ART / 'fast/evaluations/early_only.json',
             'full_search': ART / 'full/evaluation.json'}
    panels, seed_vectors, values, games, pending = {}, {}, {}, {}, {}
    for mode, path in paths.items():
        if not path.exists():
            pending[mode] = 'not started'
            continue
        raw = read_json(path)
        if raw['completed_pairs'] != PLAN['pairs']:
            pending[mode] = {'completed_pairs': raw['completed_pairs'], 'pairs': PLAN['pairs']}
            continue
        panels[mode], seed_vectors[mode], values[mode], games[mode] = panel(mode, path)
    comparisons = {}
    for left, right, offset in [('early_only', 'late_only', 1),
                                 ('full_search', 'late_only', 2), ('full_search', 'early_only', 3)]:
        if left in panels and right in panels:
            if seed_vectors[left] != seed_vectors[right]:
                raise ValueError('comparison game seeds differ')
            comparisons[f'{left}_minus_{right}'] = paired_bootstrap_mean(
                values[left] - values[right], seed=PLAN['bootstrap_seed'] + offset,
                resamples=PLAN['bootstrap_resamples']).to_dict()
    joint = Counter((games['late_only'][k]['outcome_for_a'], games['early_only'][k]['outcome_for_a'])
                    for k in games['late_only'])
    result = {'status': 'complete' if len(panels) == 3 else 'stage_complete_full_pending',
        'analysis_script_sha256': sha256(Path(__file__)),
        'scope': 'Frozen16500, same-weight greedy, shared500paired seeds. Different strategies change '
                 'trajectories; stage statistics and post-selected subsets are descriptive, not exact-terminal causality.',
        'panels': panels, 'comparisons': comparisons, 'pending': pending,
        'stage_joint_outcomes': [{'late': late, 'early': early, 'games': count}
                                 for (late, early), count in sorted(joint.items())],
        'stage_opposite_win_loss_games': joint['a_win', 'a_loss'] + joint['a_loss', 'a_win'],
        'stage_identical_final_state_digests': sum(games['late_only'][k]['final_state_digest'] ==
            games['early_only'][k]['final_state_digest'] for k in games['late_only']),
        'early_over_late_cost': {key: panels['early_only'][key] / panels['late_only'][key]
                               for key in ['simulations', 'model_positions', 'wall_seconds_this_invocation']}}
    if 'full_search' in panels:
        result['full_over_stage_cost'] = {mode: {key: panels['full_search'][key] / panels[mode][key]
            for key in ['simulations', 'model_positions', 'wall_seconds_this_invocation']}
            for mode in ['late_only', 'early_only']}
        fields = ['early_decisions', 'early_search_decisions', 'early_simulations',
                  'late_entry_count', 'late_entry_completed_turns_sum']
        result['full_early_phase_gate_counter_matches'] = {
            'games': sum(all(games['full_search'][k]['policy_a_stats'][field] ==
                             games['early_only'][k]['policy_a_stats'][field] for field in fields)
                         for k in games['full_search']),
            'total_games': len(games['full_search']), 'fields': fields,
            'scope': 'Matching phase counts/entry times; does not prove bit-identical actions or trajectories.'}
        memory_path = ART.parents[1] / 'runtime/entity_attention_stage_search_20261008/full/memory.jsonl'
        import json
        memory = [json.loads(line) for line in memory_path.read_text().splitlines()]
        result['full_memory_samples'] = memory
        result['full_memory_scope'] = 'Sampled every25pairs; not a continuous peak measurement. '
        result['full_memory_scope'] += 'Tail static graph shapes increase reserved allocator cache; no RAM cap.'
    write_json(ART / 'results_review.json', result)
    print({mode: {'score': row['score'], 'ci95': row['ci95'],
                  'wall_seconds': row['wall_seconds_this_invocation']} for mode, row in panels.items()})


if __name__ == '__main__':
    main()
