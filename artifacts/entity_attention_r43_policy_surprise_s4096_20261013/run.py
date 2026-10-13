"""Matched R43 +1000-step fixed-4k S4096 policy surprise experiment."""
from __future__ import annotations

import argparse
import fcntl
import json
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from splendor.alphazero_checkpoint import load_alphazero_checkpoint
from splendor.alphazero_recording import ReferenceWriter, environment_identity, read_json, sha256, source_identity, write_json
from splendor.alphazero_replay import PositionPool
from splendor.alphazero_train import AlphazeroLearner, LearnerConfig, run_training_round
from splendor.compact_game_cache import CompactGameCache
from splendor.evaluation import FixedEvaluationConfig, run_fixed_evaluation
from splendor.offline_train import _digest, make_index
from splendor.policy_surprise import PolicySurpriseConfig, game_policy_surprise
from splendor.pool_fitting import RoundFitting

ART = Path(__file__).resolve().parent
ROOT = ART.parents[1]
RUN, RUNTIME = ROOT / 'runs' / ART.name, ROOT / 'runtime' / ART.name
CONFIG = ROOT / 'configs/entity_attention_v1/r43_policy_surprise_s4096_fixed4k_20261013.json'


def identity(path):
    path = Path(path)
    return {'path': str(path.relative_to(ROOT)), 'sha256': sha256(path)}


def freeze(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f'frozen experiment changed: {path}')
    else:
        write_json(path, value)


def prepare():
    cfg = read_json(CONFIG)
    for name in ('initial_checkpoint', 'source_snapshot'):
        if sha256(ROOT / cfg[name]) != cfg[name + '_sha256']:
            raise ValueError(f'changed {name}')
    records = [g for g in read_json(ROOT / cfg['source_snapshot'])['games']
               if g['round_index'] in cfg['source_round_indices']]
    if len(records) != 4000 or Counter(g['round_index'] for g in records) != dict.fromkeys(cfg['source_round_indices'], 1000):
        raise ValueError('expected exactly 4x1000 S4096 games')
    generations = []
    for r in cfg['source_round_indices']:
        games = [g for g in records if g['round_index'] == r]
        manifest = Path(games[0]['path']).parents[1] / 'manifest.json'
        raw = read_json(manifest)
        search = raw['config']['search_config']
        if (search['simulations'], search['determinizations']) != (4096, 1):
            raise ValueError('source search budget differs')
        generations.append({'absolute_generation': r+1, 'manifest': identity(manifest),
            'teacher': raw['checkpoint_identity'], 'splits': dict(Counter(g['split'] for g in games)),
            'positions': {s: sum(g['decision_count'] for g in games if g['split'] == s)
                          for s in ('train', 'validation')}})
    freeze(ART / 'dataset.json', {'games': records, 'digest': _digest(records),
                                 'source': identity(ROOT / cfg['source_snapshot'])})
    payload = load_alphazero_checkpoint(ROOT / cfg['initial_checkpoint'])
    learner = LearnerConfig(**payload.learner_config)
    if (payload.total_train_step, learner.learning_rate, learner.batch_size, learner.max_grad_norm,
            learner.policy_loss_weight, learner.value_loss_weight) != (72898, 2e-5, 256, 1., 1., 1.):
        raise ValueError('R43 learner settings differ')
    refs = [r for r in read_json(ROOT / cfg['source_train_references'])['references']
            if r['round_index'] in cfg['source_round_indices']]
    if len(refs) != 8192:
        raise ValueError('expected existing 2048 Train references per generation')
    plan = {'config': cfg, 'config_identity': identity(CONFIG), 'learner': asdict(learner),
            'initial_round_index': payload.round_index, 'experiment_round_index': payload.round_index+1,
            'initial_total_step': payload.total_train_step, 'initial_sampling_rng': asdict(payload.sampling_rng_state),
            'dataset': identity(ART / 'dataset.json'), 'generations': generations,
            'train_references_source': identity(ROOT / cfg['source_train_references']),
            'greedy_reference': identity(ROOT / cfg['greedy_reference']),
            'runner': identity(Path(__file__)), 'branch_identities': {}}
    for branch, weighting in cfg['branches'].items():
        replay = {'digest': _digest({'dataset': _digest(records), 'weighting': weighting}),
                  'dataset': str(ART / 'dataset.json'), 'fixed_pool_games': 4000,
                  'training': 'joint_same_batch_original_AdamW_clip1'}
        if weighting is not None:
            replay['policy_surprise_weighting'] = weighting
        plan['branch_identities'][branch] = replay
    freeze(ART / 'plan.json', plan)
    freeze(ART / 'common/train_references.json', {'references': refs, 'digest': _digest(refs)})
    RUN.mkdir(parents=True, exist_ok=True)
    RUNTIME.mkdir(parents=True, exist_ok=True)
    return plan


def fitting(index, stage, plan):
    return RoundFitting({'steps': plan['config']['fitting_steps'], 'seed': 34300000,
                         'train_positions_per_generation': 2048,
                         'policy_surprise_weighting': plan['config']['branches']['surprise']},
                        RUN, stage, None, index, None,
                        inherited_references=read_json(ART / 'common/train_references.json')['references'])


def train(branch, plan):
    cfg = plan['config']
    config = LearnerConfig(**plan['learner'])
    spec = plan['branch_identities'][branch]
    initial = ROOT / cfg['initial_checkpoint']
    cps = RUN / branch / 'checkpoints'
    saved = sorted(cps.glob('step-*.pt'))
    start = saved[-1] if saved else cps / 'initial.pt'
    if not start.exists():
        payload = load_alphazero_checkpoint(initial)
        learner = AlphazeroLearner.resume(initial, config, replay_identity=payload.replay_identity,
            round_index=payload.round_index, round_steps_target=payload.round_steps_target)
        learner.round_index, learner.round_train_step = plan['experiment_round_index'], 0
        learner.round_steps_target = cfg['updates']
        learner.replay_identity = spec
        learner.parent_checkpoint = identity(initial)
        learner.save(start, metadata={'experiment': ART.name, 'branch': branch})
        del learner
    index = make_index(read_json(ART / 'dataset.json'), cfg['cache_bytes'])
    index.cache = CompactGameCache(cfg['cache_bytes'])
    common = fitting(index, ART / 'common', plan)
    zero_path = ART / 'common/fitting/step-000000000.json'
    if not zero_path.exists():
        zero = AlphazeroLearner.resume(start, config, replay_identity=spec,
            round_index=plan['experiment_round_index'], round_steps_target=cfg['updates'])
        if zero.round_train_step:
            raise ValueError('missing common zero-step diagnostic')
        common(zero, start)
        del zero
    diagnostic = fitting(index, ART / branch, plan)
    branch_initial = cps / 'initial.pt'
    freeze(ART / branch / 'fitting/step-000000000.json', {
        **read_json(zero_path), 'checkpoint': str(branch_initial),
        'checkpoint_sha256': sha256(branch_initial), 'shared_zero_step': identity(zero_path)})
    # All diagnostics precede this resume, preserving R43's Torch/sampling RNG.
    learner = AlphazeroLearner.resume(start, config, replay_identity=spec,
        round_index=plan['experiment_round_index'], round_steps_target=cfg['updates'])
    pool = PositionPool(index.split_games('train'), index.cache, 'fixed4k')
    weighting = cfg['branches'][branch]
    writer = ReferenceWriter(RUN / branch / 'usage', str(time.time_ns()), config.batch_size, spec)
    try:
        stats = run_training_round(learner, steps_target=cfg['updates'], current_pool=pool,
            history_pool=None, validation_samples=(), validation_games=0, checkpoint_dir=cps,
            metrics_path=RUN / branch / 'training_metrics.jsonl', reference_writer=writer,
            checkpoint_callback=diagnostic,
            policy_surprise_weighting=PolicySurpriseConfig(**weighting) if weighting is not None else None,
            progress_callback=lambda row: print(json.dumps({'event': 'train', 'branch': branch,
                'step': row['round_train_step'], 'loss': row['total_loss'],
                'policy_kl': row['policy_kl'], 'value_mse': row['value_loss']}), flush=True)
                if row['round_train_step'] % 100 == 0 else None)
    finally:
        writer.close()
    shutil.copyfile(cps / 'step-000001000.pt', cps / 'final.pt')
    write_json(ART / branch / 'training_summary.json', {**stats, 'checkpoint': identity(cps / 'final.pt')})


def arena(branch, plan):
    template = read_json(ROOT / plan['config']['greedy_reference'])['config']
    config = {**template, 'policy_a': {'kind': 'greedy_checkpoint', 'name': branch,
              'checkpoint': str(RUN / branch / 'checkpoints/final.pt')}}
    freeze(ART / branch / 'greedy_config.json', config)
    path = ART / branch / 'greedy.json'
    run_fixed_evaluation(FixedEvaluationConfig.from_dict(config), path,
        resume=path.exists() or path.with_suffix('.json.games').exists(), progress_every=50,
        progress_callback=lambda done, total: print(json.dumps({'event': 'greedy', 'branch': branch,
                                                               'pairs': done}), flush=True),
        failure_dir=RUNTIME / branch / 'failures')


def weight_statistics(plan):
    path = ART / 'weight_statistics.json'
    if path.exists():
        return
    rows = []
    raw_surprise, raw_weights = [], []
    for game in read_json(ART / 'dataset.json')['games']:
        if game['split'] != 'train':
            continue
        with np.load(game['path'], allow_pickle=False) as data:
            arrays = {k: data[k] for k in ('candidate_offsets', 'policy_target', 'root_network_prior', 'policy_weight')}
        surprise, weights = game_policy_surprise(arrays, PolicySurpriseConfig(**plan['config']['branches']['surprise']))
        active = arrays['policy_weight'] > 0
        raw_surprise.extend(surprise[active])
        raw_weights.extend(weights[active])
        rows.append({'round_index': game['round_index'], 'game_id': game['game_id'],
                     'policy_positions': int(active.sum()), 'mean_surprise': float(surprise[active].mean()),
                     'weight_sum': float(weights.sum()), 'original_weight_sum': float(arrays['policy_weight'].sum())})
    s, w = np.asarray(raw_surprise), np.asarray(raw_weights, dtype=np.float64)
    order = np.argsort(s)
    write_json(path, {'normalization': 'per Train game over original searchable policy weights',
        'positions': len(w), 'games': len(rows), 'mean_surprise': float(s.mean()), 'mean_weight': float(w.mean()),
        'quantile_levels': [0, .1, .5, .9, .99, 1],
        'surprise_quantiles': np.quantile(s, [0, .1, .5, .9, .99, 1]).tolist(),
        'weight_quantiles': np.quantile(w, [0, .1, .5, .9, .99, 1]).tolist(),
        'effective_policy_positions': float(w.sum()**2 / np.dot(w, w)),
        'highest_surprise_10pct_weight_fraction': float(w[order[-int(.1*len(w)):]].sum()/w.sum()),
        'per_game': rows})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'train', 'arena'), nargs='?', default='run')
    parser.add_argument('--branch', choices=('control', 'surprise'))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    plan = prepare()
    if args.action == 'prepare':
        weight_statistics(plan)
        return
    if args.action in ('train', 'arena'):
        if args.branch is None:
            parser.error('--branch required')
        if args.action == 'train' and not torch.cuda.is_available():
            raise RuntimeError('CUDA required for this experiment')
        (train if args.action == 'train' else arena)(args.branch, plan)
        return
    with (RUNTIME / 'run.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        session = str(time.time_ns())
        source = source_identity(ROOT, RUNTIME / session)
        write_json(ART / 'execution_sessions' / f'{session}.json', {'source': source,
            'environment': environment_identity(), 'argv': sys.argv, 'plan': identity(ART / 'plan.json'),
            'time_unix': time.time()})
        weight_statistics(plan)
        for branch in plan['config']['branches']:
            for action in ('train', 'arena'):
                marker = ART / branch / ('training_summary.json' if action == 'train' else 'greedy.json')
                if marker.exists() and (action == 'train' or read_json(marker)['passed']):
                    continue
                write_json(RUNTIME / 'live.json', {'branch': branch, 'action': action, 'status': 'running'})
                subprocess.run([sys.executable, '-u', str(Path(__file__)), action, '--branch', branch],
                               cwd=ROOT, check=True)
        subprocess.run([sys.executable, str(ART / 'analyze.py')], cwd=ROOT, check=True)
        write_json(RUNTIME / 'live.json', {'status': 'complete'})


if __name__ == '__main__':
    main()
