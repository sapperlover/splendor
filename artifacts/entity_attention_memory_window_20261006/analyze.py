"""Bounded CPU analysis of existing data; never collect games or update weights."""
from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from splendor.alphazero_recording import read_json, sha256, source_identity, write_json
from splendor.alphazero_checkpoint import load_alphazero_checkpoint
from splendor.alphazero_replay import PositionPool
from splendor.alphazero_train import AlphazeroLearner
from splendor.hybrid_train import HybridConfig
from splendor.offline_train import make_index
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.paired_statistics import paired_bootstrap_difference, serialized_pair_scores
from splendor.rng import DeterministicRng

ROOT = Path(__file__).resolve().parents[2]
ART = Path(__file__).resolve().parent
RUN = ROOT / 'runs/entity_attention_hybrid_s1024_n1_from_round9_seed1'
INPUTS = {}


def record(path):
    path = path.resolve()
    INPUTS[str(path)] = {'path': str(path), 'sha256': sha256(path), 'bytes': path.stat().st_size}
    return path


def read(path):
    return read_json(record(path))


def references(stage, source, quota, steps=100):
    batches = {}
    for path in sorted((stage / 'usage' / source).glob('*.bin')):
        meta = read(path.with_suffix('.json'))
        assert meta['batch_size'] == quota
        raw = np.fromfile(record(path), dtype='<i8')
        assert len(raw) % 4 == 0
        rows = raw.reshape(-1, 4)
        for step in np.unique(rows[:, 0]):
            if step > steps:
                continue
            block = rows[rows[:, 0] == step]
            assert len(block) == quota
            if int(step) in batches:
                assert np.array_equal(batches[int(step)], block)
            batches[int(step)] = block
    assert set(batches) == set(range(1, steps + 1))
    return np.concatenate([batches[s] for s in range(1, steps + 1)])[:, 1:]


def validation_on_refs(learner, pool, refs):
    sums = Counter()
    fields = ('policy_ce_sum', 'policy_weight_sum', 'target_entropy_sum',
              'value_sq_sum', 'value_weight_sum', 'positions')
    games = {game.key: game for game in pool.games}
    for start in range(0, len(refs), 64):
        chunk = refs[start:start + 64]
        samples = [pool._cache.sample(games[(r.round_index, r.game_id)], r.decision_index) for r in chunk]
        metrics = learner.evaluate_samples(samples)
        sums.update({field: metrics[field] for field in fields})
    return {**sums,
            'policy_kl': (sums['policy_ce_sum'] - sums['target_entropy_sum']) / sums['policy_weight_sum'],
            'value_mse': sums['value_sq_sum'] / sums['value_weight_sum']}


def main():
    torch.set_num_threads(1)
    cfg = HybridConfig.from_dict(read(RUN / 'config.json'))
    offset = read(RUN / 'hybrid_state.json')['online_round_offset']
    rounds, cohort16, teacher_steps = [], [], {}
    for human in range(1, 31):
        stage = RUN / 'rounds' / f'round-{human - 1:03d}'
        snap = read(stage / 'replay_snapshot.json')
        training = [json.loads(line) for line in record(stage / 'training_metrics.jsonl').read_text().splitlines()]
        training = [row for row in training if row['event'] == 'train_step']
        # Last committed sequence, excluding duplicated session attempts.
        by_step = {row['round_train_step']: row for row in training}
        assert set(by_step) == set(range(1, 101))
        training = list(by_step.values())
        rows = references(stage, 'online', snap['quotas'][1])
        current_round = snap['round_index']
        current = rows[rows[:, 0] == current_round]
        positions = sum(g['decision_count'] for g in snap['games'] if g['split'] == 'train')
        new_positions = sum(g['decision_count'] for g in snap['games']
                            if g['split'] == 'train' and g['round_index'] == current_round)
        before, after = read(stage / 'validation_before.json'), read(stage / 'validation_after.json')
        teacher_counts = Counter(map(int, rows[:, 0]))
        teachers = {}
        for game in snap['games']:
            data_round = game['round_index']
            if data_round not in teachers:
                manifest = read(Path(game['path']).parent.parent / 'manifest.json')
                identity = manifest['checkpoint_identity']
                if identity['sha256'] not in teacher_steps:
                    checkpoint = record(Path(identity['path']))
                    assert INPUTS[str(checkpoint)]['sha256'] == identity['sha256']
                    teacher_steps[identity['sha256']] = load_alphazero_checkpoint(checkpoint).total_train_step
                teachers[data_round] = {**identity, 'total_train_step': teacher_steps[identity['sha256']]}
        generated_step = snap['teacher']['total_train_step']
        draws_weighted_age = sum(teacher_counts[r] *
                                 (generated_step - teachers[r]['total_train_step']) for r in teacher_counts) / len(rows)
        collection = read(stage / 'selfplay/summary.json')
        rounds.append({
            'round': human, 'step_before': generated_step, 'step_after': after['total_train_step'],
            'quotas': snap['quotas'], 'active_games': len(snap['games']), 'online_train_positions': positions,
            'current_train_positions': new_positions, 'online_draws': len(rows),
            'current_draws': len(current), 'current_unique_positions': len(np.unique(current, axis=0)),
            'current_share_of_batch': len(current) / (100 * cfg.offline.learner.batch_size),
            'current_share_of_online': len(current) / len(rows),
            'current_training_coverage': len(np.unique(current, axis=0)) / new_positions,
            'online_repeat_draw_fraction': 1 - len(np.unique(rows, axis=0)) / len(rows),
            'draws_by_data_round': dict(sorted(teacher_counts.items())),
            'draws_weighted_teacher_age_steps': draws_weighted_age,
            'teacher_identities': teachers, 'online_cache': training[-1]['replay_cache'],
            'reported_peak_rss_kib': collection['cpu_peak_rss_kib'],
            'training_policy_kl': sum(r['policy_ce_sum'] - r['target_entropy_sum'] for r in training)
                                  / sum(r['policy_weight_sum'] for r in training),
            'training_value_mse': sum(r['value_sq_sum'] for r in training)
                                 / sum(r['value_weight_sum'] for r in training),
            'validation_before': before, 'validation_after': after,
            'current_delta': {k: after['current'][k] - before['current'][k]
                              for k in ('policy_kl', 'value_loss')}})
        cohort16.append(rows[rows[:, 0] == offset + 15])
    lifetime = np.concatenate(cohort16)
    cohort_positions = rounds[15]['current_train_positions']
    unique, counts = np.unique(lifetime, axis=0, return_counts=True)
    scores = []
    latest = read(RUN / 'rounds/round-029/evaluation/mcts_vs_theta5000.json')
    latest_seeds, latest_scores = serialized_pair_scores(latest)
    for human in (5, 10, 15, 20, 25, 30):
        evaluation = read(RUN / 'rounds' / f'round-{human - 1:03d}' / 'evaluation/mcts_vs_theta5000.json')
        seeds, values = serialized_pair_scores(evaluation)
        assert evaluation['checkpoint_identity']['policy_b'] == latest['checkpoint_identity']['policy_b']
        scores.append({'round': human, 'score': float(values.mean()),
                       'r30_minus': paired_bootstrap_difference(latest_seeds, latest_scores, seeds, values,
                                                               resamples=10000, seed=14720000).to_dict()})
    # Distinguish old-policy fit from current-policy fit using the same sample
    # references at the actual round-30 boundary. Train and holdout are separate.
    stage = RUN / 'rounds/round-029'
    snapshot = read(stage / 'replay_snapshot.json')
    index = make_index({'games': snapshot['games']}, 128 * 1024**2)
    groups = []
    for label, lo, hi in [('newest', 30, 30), ('recent', 26, 29), ('older', 16, 25)]:
        for split in ('train', 'validation'):
            pool = PositionPool([g for g in index.games if g.split == split
                                 and lo <= g.round_index - offset + 1 <= hi], index.cache, label)
            rng = DeterministicRng(14650000 + lo + (100 if split == 'validation' else 0))
            refs = [pool.sample(rng)[1] for _ in range(512)]
            groups.append((label, split, pool, refs))
    cohort_metrics = []
    for checkpoint in ('initial.pt', 'final.pt'):
        model = load_frozen_actor_critic_checkpoint(record(stage / 'checkpoints' / checkpoint), device='cpu')
        learner = AlphazeroLearner(model, replace(cfg.offline.learner, device='cpu'),
                                   sampling_rng=DeterministicRng(0))
        for label, split, pool, refs in groups:
            cohort_metrics.append({'checkpoint': checkpoint, 'cohort': label, 'split': split,
                                   'sample_seed': 14650000 + {'newest': 30, 'recent': 26, 'older': 16}[label]
                                                  + (100 if split == 'validation' else 0),
                                   'pool_positions': len(pool), 'sample_count': len(refs),
                                   **validation_on_refs(learner, pool, refs)})
        del learner, model
    result = {'run': str(RUN), 'rounds': rounds, 'scores': scores,
              'fully_observed_cohort16': {'positions': cohort_positions, 'draws': len(lifetime),
                  'unique_positions': len(unique), 'coverage': len(unique) / cohort_positions,
                  'mean_lifetime_uses': len(lifetime) / cohort_positions, 'max_uses': int(counts.max())},
              'cohort_metrics_round30': cohort_metrics,
              'optimizer_updates': 0, 'new_games_collected': 0,
              'limitations': 'One training seed; fixed opponent/panel; no window or sampling ablation. '
                             'Cohort sample losses do not establish causality or playing strength.'}
    write_json(ART / 'analysis.json', result)
    write_json(ART / 'manifest.json', {'command': sys.argv, 'source': source_identity(ROOT),
                                      'torch': torch.__version__, 'device': 'cpu', 'threads': 1,
                                      'inputs': list(INPUTS.values()), 'script_sha256': sha256(Path(__file__)),
                                      'optimizer_updates': 0, 'new_games_collected': 0})
    print(json.dumps({'latest': {k: rounds[-1][k] for k in
                     ('current_draws', 'current_share_of_batch', 'current_training_coverage',
                      'draws_weighted_teacher_age_steps')}, 'cohort16': result['fully_observed_cohort16'],
                     'cohort_metrics': cohort_metrics}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
