"""Fixed M4 replay: policy sampling A/B/C and inherited-AdamW directions.

Formal collection/training files are read only. Each job is a fresh subprocess;
checkpoints and batch plans live in ignored runs/runtime, small results here.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from splendor.alphazero_checkpoint import load_alphazero_checkpoint, move_optimizer_state, model_state_sha256
from splendor.alphazero_recording import read_json, write_json, sha256, environment_identity, source_identity
from splendor.alphazero_replay import PositionPool, ReplayReference
from splendor.alphazero_train import AlphazeroLearner, LearnerConfig, _torch_batch, alphazero_loss_sums, label_statistics, metric_ratios
from splendor.offline_train import make_index
from splendor.policy_averaging import AveragedTargetStore
from splendor.rng import DeterministicRng, RngState
from splendor.selfplay import derive_seed
from splendor.selfplay_data import collate_selfplay_samples

ROOT = Path(__file__).resolve().parents[2]
ART = Path(__file__).resolve().parent
IDENTIFIER = ART.name
RUNTIME = ROOT / 'runtime' / IDENTIFIER
OUT = ROOT / 'runs' / IDENTIFIER
SOURCE = ROOT / 'runs/entity_attention_avg4_behavior_20261009/seed1_avg4'
ROUNDS = (4, 8, 12)
ARMS = ('A', 'B', 'C')
NODES = (0, 50, 200, 500)
STEPS = 500
CACHE_BYTES = 2 * 1024**3
TRAIN_BANK = 4096  # Fixed diagnostic subset per teacher, not a training restriction.
GRAD_REPEATS = 8
GRAD_POSITIONS = 1024
SUM_FIELDS = ('policy_ce_sum', 'policy_weight_sum', 'target_entropy_sum', 'entropy_sum',
              'value_sq_sum', 'value_weight_sum', 'value_target_sum', 'value_target_sq_sum',
              'positions', 'policy_positions', 'value_positions')


def emit(event, **values):
    print(json.dumps({'event': event, **values}, ensure_ascii=False), flush=True)


def record(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha256(path)}


def npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as handle:
        np.savez_compressed(handle, **arrays)
    tmp.replace(path)


def reference_draw(pool, rng, count):
    """Exactly PositionPool.sample's mapping/RNG, without decoding an archive."""
    rows = []
    for _ in range(count):
        position = rng.randbelow(len(pool))
        ordinal = int(pool._game_of_position[position])
        game = pool.games[ordinal]
        rows.append((game.round_index, game.game_id, int(position - pool._starts[ordinal])))
    return np.asarray(rows, dtype=np.int64)


def pools(index, recent):
    train = index.split_games('train')
    selected = set(recent)
    return {name: PositionPool([g for g in train if predicate(g)], index.cache, name)
            for name, predicate in {
                'recent': lambda g: g.round_index in selected,
                'history': lambda g: g.round_index not in selected,
                'latest': lambda g: g.round_index == recent[-1],
                'previous': lambda g: g.round_index in selected - {recent[-1]},
            }.items()}


def batch_plans(bucket, state, seed, steps=STEPS):
    rng = DeterministicRng.restore(state)
    b_rng, c_rng = DeterministicRng(seed), DeterministicRng(seed)
    result = {f'policy_{a}': [] for a in ARMS} | {'value': [], 'rng_state': [], 'rng_draws': []}
    for _ in range(steps):
        result['policy_A'].append(reference_draw(bucket['recent'], rng, 256))
        result['value'].append(np.concatenate((reference_draw(bucket['recent'], rng, 128),
                                               reference_draw(bucket['history'], rng, 128))))
        result['policy_B'].append(np.concatenate((reference_draw(bucket['latest'], b_rng, 128),
                                                  reference_draw(bucket['previous'], b_rng, 128))))
        result['policy_C'].append(reference_draw(bucket['latest'], c_rng, 256))
        result['rng_state'].append(rng.snapshot().state)
        result['rng_draws'].append(rng.snapshot().draws)
    return {k: np.asarray(v, dtype=np.uint64 if k.startswith('rng_') else np.int64)
            for k, v in result.items()}


def ledger_prefix(stage, plan):
    for head, expected in [('policy_recent', plan['policy_A'][:100]),
                           ('value_recent', plan['value'][:100, :128]),
                           ('value_history', plan['value'][:100, 128:])]:
        bins = list((stage / 'usage' / head).glob('*.bin'))
        if len(bins) != 1:
            raise ValueError(f'exactly one committed source ledger required: {head}')
        raw = np.frombuffer(bins[0].read_bytes(), dtype='<i8').reshape(100, expected.shape[1], 4)
        if not np.array_equal(raw[:, :, 1:], expected):
            raise ValueError(f'A formal sampling prefix differs: {stage}/{head}')


def prepare():
    config = read_json(SOURCE / 'config.json')
    spec = {'protocol': 'fixed-m4-policy-history-v1', 'rounds': list(ROUNDS), 'arms': {
        'A': 'original uniform positions in recent four rounds',
        'B': '128 latest + 128 previous three, uniform positions within each bucket',
        'C': '256 latest; value recent/history unchanged'}, 'nodes': list(NODES),
        'steps': STEPS, 'fixed_train_positions_per_teacher': TRAIN_BANK,
        'gradient_repeats': GRAD_REPEATS, 'gradient_positions_per_repeat_per_bucket': GRAD_POSITIONS,
        'source_config': record(SOURCE / 'config.json'), 'code': {}, 'subjects': {}}
    for file in ('alphazero_train.py', 'alphazero_replay.py', 'alphazero_checkpoint.py', 'offline_train.py',
                 'policy_averaging.py', 'actor_critic.py', 'entity_attention.py', 'selfplay_data.py', 'rng.py'):
        spec['code'][file] = sha256(ROOT / 'src/splendor' / file)
    spec['code']['run.py'] = sha256(Path(__file__))
    for n in ROUNDS:
        key = f'r{n:03d}'
        stage = SOURCE / 'rounds' / f'round-{n-1:03d}'
        snapshot = read_json(stage / 'replay_snapshot.json')
        recent = snapshot['split_training']['recent_round_indices']
        start = stage / 'checkpoints/initial.pt'
        payload = load_alphazero_checkpoint(start, device='cpu')
        if payload.round_train_step != 0 or payload.learner_config != config['offline']['learner']:
            raise ValueError('requires the original untrained round checkpoint/config')
        if len(recent) != 4 or any(sum(g['round_index'] == r for g in snapshot['games']) != 200 for r in recent):
            raise ValueError('requires four complete 200-game teacher generations')
        folder = RUNTIME / key
        folder.mkdir(parents=True, exist_ok=True)
        index = make_index(snapshot, CACHE_BYTES)
        bucket = pools(index, recent)
        seed = derive_seed(28100000, n, 0, 'policy_sampling')
        path = folder / 'batch_plans.npz'
        if not path.exists():
            plan = batch_plans(bucket, payload.sampling_rng_state, seed)
            ledger_prefix(stage, plan)
            # Fixed without-replacement diagnostic train banks per generation.
            banks = {}
            for r in recent:
                sub = PositionPool([g for g in bucket['recent'].games if g.round_index == r], index.cache, 'train')
                rng = np.random.default_rng(derive_seed(28200000, n, r, 'fit_bank'))
                positions = np.sort(rng.choice(len(sub), size=min(TRAIN_BANK, len(sub)), replace=False))
                rows = []
                for i in positions:
                    ordinal = int(sub._game_of_position[i]); game = sub.games[ordinal]
                    rows.append((r, game.game_id, int(i - sub._starts[ordinal])))
                banks[f'train_{r}'] = np.asarray(rows, dtype=np.int64)
            # Independent gradient samples from train games only.
            grad_rng = DeterministicRng(derive_seed(28300000, n, 0, 'gradient'))
            for name in ('latest', 'previous'):
                banks[f'gradient_{name}'] = np.stack([
                    reference_draw(bucket[name], grad_rng, GRAD_POSITIONS) for _ in range(GRAD_REPEATS)])
            npz(path, **plan, **banks)
        with np.load(path, allow_pickle=False) as z:
            ledger_prefix(stage, z)
        labels = {run: info for run, info in snapshot['policy_labels'].items()
                  if any(g.run_id == run for g in bucket['recent'].games)}
        for info in labels.values():
            if sha256(Path(info['path'])) != info['sha256']:
                raise ValueError('frozen M4 manifest changed')
        games = [{**g, 'size': Path(g['path']).stat().st_size,
                  'mtime_ns': Path(g['path']).stat().st_mtime_ns} for g in snapshot['games']]
        dataset = folder / 'dataset.json'
        write_json(dataset, {'games': games}) if not dataset.exists() else None
        if read_json(dataset) != {'games': games}:
            raise ValueError('source archive stat changed since preparation')
        spec['subjects'][key] = {'round': n, 'start': record(start), 'start_step': payload.total_train_step,
            'formal_final': record(stage / 'checkpoints/final.pt'), 'snapshot': record(stage / 'replay_snapshot.json'),
            'dataset': record(dataset), 'batch_plans': record(path), 'sampling_seed_B_C': seed,
            'recent_rounds': recent, 'labels': labels,
            'pool_positions': {k: len(v) for k, v in bucket.items()},
            'source_learner_config': payload.learner_config, 'split_training': config['split_training'],
            'validation_before': record(stage / 'validation_before.json')}
    path = ART / 'specification.json'
    if path.exists() and read_json(path) != spec:
        raise ValueError('frozen experiment/source/code changed; use a new experiment')
    if not path.exists():
        write_json(path, spec)
        write_json(RUNTIME / 'source/identity.json', source_identity(ROOT, RUNTIME / 'source'))
    emit('prepared', rounds=ROUNDS, source_formal_sampling_prefix='exact', nodes=NODES)
    return spec


def context(subject):
    data = read_json(Path(subject['dataset']['path']))
    for g in data['games']:
        stat = Path(g['path']).stat()
        if (stat.st_size, stat.st_mtime_ns) != (g['size'], g['mtime_ns']):
            raise ValueError(f'source game changed: {g["path"]}')
    index = make_index(data, CACHE_BYTES)
    lookup = {g.key: g for g in index.games}
    recent = set(subject['recent_rounds'])
    required = {g.run_id for g in index.games if g.round_index in recent}
    store = AveragedTargetStore([g for g in index.games if g.round_index in recent], required_run_ids=required)
    if store.manifests != subject['labels']:
        raise ValueError('M4 manifest no longer matches frozen subject')
    with np.load(subject['batch_plans']['path'], allow_pickle=False) as z:
        plan = {k: z[k].copy() for k in z.files}
    return index, lookup, store, plan


def resolve(index, lookup, refs, store=None):
    samples, references = [], []
    for r, game_id, decision in refs:
        game = lookup[int(r), int(game_id)]
        sample = index.cache.sample(game, int(decision))
        samples.append(store.transform(sample) if store is not None else sample)
        references.append(ReplayReference(game.round_index, game.run_id, game.game_id, int(decision)))
    return samples, references


def restore(path, device, identity, *, require_identity=False):
    payload = load_alphazero_checkpoint(path, device=device)
    if require_identity and payload.replay_identity != identity:
        raise ValueError('resume identity changed')
    config = replace(LearnerConfig(**payload.learner_config), device=device)
    learner = AlphazeroLearner(payload.build_model(device=device), config,
                             sampling_rng=DeterministicRng.restore(payload.sampling_rng_state))
    learner.optimizer.load_state_dict(payload.optimizer_state_dict)
    move_optimizer_state(learner.optimizer, learner.device)
    learner.round_index, learner.round_train_step = payload.round_index, payload.round_train_step
    learner.total_train_step = payload.total_train_step
    learner.round_steps_target = STEPS
    learner.parent_checkpoint = dict(payload.parent_checkpoint)
    learner.replay_identity = identity
    torch.set_rng_state(payload.torch_cpu_rng_state.cpu())
    if device == 'cuda' and payload.torch_cuda_rng_states:
        torch.cuda.set_rng_state_all([s.cpu() for s in payload.torch_cuda_rng_states])
    return learner


def aggregate(rows):
    sums = Counter()
    for metrics in rows:
        sums.update({k: metrics[k] for k in SUM_FIELDS})
    return {'available': bool(sums['positions']), **sums, **metric_ratios(sums)}


def evaluate_refs(learner, index, lookup, store, refs):
    rows = []
    for start in range(0, len(refs), 256):
        samples, _ = resolve(index, lookup, refs[start:start+256], store)
        rows.append(learner.evaluate_samples(samples))
    return aggregate(rows)


def node(learner, subject, key, arm, step, ctx):
    baseline = ART / 'nodes' / f'{key}_A_000000.json'
    if step == 0 and arm != 'A' and baseline.exists():
        result = {**read_json(baseline), 'arm': arm}
        write_json(ART / 'nodes' / f'{key}_{arm}_{step:06d}.json', result)
        write_json(RUNTIME / key / arm / 'holdout_games_000000.json',
                   read_json(RUNTIME / key / 'A/holdout_games_000000.json'))
        emit('node_baseline_reused', round=subject['round'], arm=arm)
        return result
    index, lookup, store, plan = ctx
    by_teacher, raw = {}, []
    for r in subject['recent_rounds']:
        train = evaluate_refs(learner, index, lookup, store, plan[f'train_{r}'])
        validation = []
        for game in index.split_games('validation'):
            if game.round_index != r:
                continue
            refs = np.asarray([(r, game.game_id, i) for i in range(game.decision_count)])
            metrics = evaluate_refs(learner, index, lookup, store, refs)
            validation.append(metrics)
            raw.append({'round': r, 'game_id': game.game_id, **{k: metrics[k] for k in SUM_FIELDS}})
        by_teacher[str(r)] = {'train_fixed_subset': train, 'holdout_full': aggregate(validation),
                              'holdout_games': len(validation)}
    recent = {name: aggregate([v[name] for v in by_teacher.values()])
              for name in ('train_fixed_subset', 'holdout_full')}
    result = {'round': subject['round'], 'arm': arm, 'step': step, 'total_train_step': learner.total_train_step,
              'by_teacher': by_teacher, 'recent_four': recent,
              'latest': by_teacher[str(subject['recent_rounds'][-1])]}
    if step == 0:
        expected = read_json(Path(subject['validation_before']['path']))
        for group, got in [('current', result['latest']['holdout_full']), ('recent', recent['holdout_full'])]:
            for metric in ('policy_kl', 'value_loss'):
                if abs(got[metric] - expected[group][metric]) > 2e-6:
                    raise ValueError(f'baseline differs from formal {group}/{metric}')
    write_json(RUNTIME / key / arm / f'holdout_games_{step:06d}.json', raw)
    write_json(ART / 'nodes' / f'{key}_{arm}_{step:06d}.json', result)
    emit('node', round=subject['round'], arm=arm, step=step,
         latest_train_KL=result['latest']['train_fixed_subset']['policy_kl'],
         latest_holdout_KL=result['latest']['holdout_full']['policy_kl'],
         recent_train_KL=recent['train_fixed_subset']['policy_kl'], recent_holdout_KL=recent['holdout_full']['policy_kl'])
    return result


def policy_parameters(model):
    excluded = {id(p) for p in model.critic.parameters()}
    return [(name, p) for name, p in model.named_parameters() if id(p) not in excluded]


def vector(parameters, grad=False):
    return np.concatenate([(p.grad if grad and p.grad is not None else torch.zeros_like(p) if grad else p)
                           .detach().cpu().reshape(-1).numpy().copy() for _, p in parameters])


def geometry(left, right):
    a, b = left.astype(np.float64), right.astype(np.float64)
    dot = float(a @ b); an, bn = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return {'dot': dot, 'cosine': dot / (an * bn) if an and bn else None,
            'left_norm': an, 'right_norm': bn}


def policy_gradient(learner, index, lookup, store, refs):
    learner.model.train()
    learner.optimizer.zero_grad(set_to_none=True)
    weight, ce, entropy = 0., 0., 0.
    for start in range(0, len(refs), 256):
        samples, _ = resolve(index, lookup, refs[start:start+256], store)
        batch = collate_selfplay_samples(samples); tensors = _torch_batch(batch, learner.device)
        prepared = learner.model._prepare(batch.decisions)
        logits = learner.model._actor_logits(prepared, learner.model.encode_state(prepared))
        terms = alphazero_loss_sums(logits, torch.zeros_like(tensors['value_targets']), **tensors)
        terms.policy_ce_sum.backward()
        weight += float(terms.policy_weight_sum); ce += float(terms.policy_ce_sum.detach())
        entropy += label_statistics(batch)['target_entropy_sum']
    if weight == 0:
        raise ValueError('gradient bank has no weighted policy positions')
    result = vector(policy_parameters(learner.model), grad=True) / weight
    learner.optimizer.zero_grad(set_to_none=True)
    return result, {'policy_kl': (ce-entropy)/weight, 'policy_weight_sum': weight}


def parameter_groups(parameters):
    heads = ('action_encoder.', 'query_norm.', 'memory_norm.', 'cross_attention.', 'actor.')
    flags = np.concatenate([np.full(p.numel(), name.startswith(heads), dtype=np.bool_) for name, p in parameters])
    return {'policy_all': np.ones(len(flags), dtype=np.bool_), 'encoder': ~flags, 'policy_head': flags}


def gradients(subject, key, device, identity):
    path = ART / f'gradients_{key}.json'
    if path.exists():
        return
    ctx = context(subject); index, lookup, store, plan = ctx
    learner = restore(subject['start']['path'], device, identity)
    all_refs = {name: plan[f'gradient_{name}'].reshape(-1, 3) for name in ('latest', 'previous')}
    means, before, repeats = {}, {}, []
    for name in all_refs:
        means[name], before[name] = policy_gradient(learner, index, lookup, store, all_refs[name])
    groups = parameter_groups(policy_parameters(learner.model))
    for i in range(GRAD_REPEATS):
        a, _ = policy_gradient(learner, index, lookup, store, plan['gradient_latest'][i])
        b, _ = policy_gradient(learner, index, lookup, store, plan['gradient_previous'][i])
        repeats.append({name: geometry(a[mask], b[mask]) for name, mask in groups.items()})
    report = {'round': subject['round'], 'checkpoint': subject['start'], 'gradient_before_KL': before,
              'gradient_latest_vs_previous': {name: geometry(means['latest'][mask], means['previous'][mask])
                                               for name, mask in groups.items()}, 'independent_minibatch_repeats': repeats,
              'definition': 'raw policy CE gradients; KL has identical gradient. update=theta_after-theta_before. '
                            'cos(update,-gradient)>0 or dot(gradient,update)<0 predicts descent locally; '
                            'actual AdamW includes inherited moments, clipping and weight decay.', 'arms': {}}
    npz(RUNTIME / key / 'gradient_vectors.npz', **means)
    for arm in ARMS:
        learner = restore(subject['start']['path'], device, identity)
        p_samples, p_refs = resolve(index, lookup, plan[f'policy_{arm}'][0], store)
        v_samples, v_refs = resolve(index, lookup, plan['value'][0])
        batch_gradient, _ = policy_gradient(learner, index, lookup, store, plan[f'policy_{arm}'][0])
        parameters = policy_parameters(learner.model); theta = vector(parameters)
        metrics = learner.train_split_step(collate_selfplay_samples(p_samples), collate_selfplay_samples(v_samples),
            references=p_refs+v_refs, policy_max_grad_norm=subject['split_training']['policy_max_grad_norm'],
            value_max_grad_norm=subject['split_training']['value_max_grad_norm'])
        delta = vector(parameters)-theta
        report['arms'][arm] = {'first_step': metrics, 'direction_by_group': {}, 'measured_KL_change': {}}
        for name, mask in groups.items():
            report['arms'][arm]['direction_by_group'][name] = {
                'update_vs_negative_batch_gradient': geometry(delta[mask], -batch_gradient[mask]),
                **{f'update_vs_negative_{g}_gradient': geometry(delta[mask], -means[g][mask]) for g in means},
                **{f'first_order_{g}_KL_change': geometry(means[g][mask], delta[mask])['dot'] for g in means}}
        for name, refs in all_refs.items():
            after = evaluate_refs(learner, index, lookup, store, refs)['policy_kl']
            report['arms'][arm]['measured_KL_change'][name] = {'before': before[name]['policy_kl'],
                                                           'after': after, 'delta': after-before[name]['policy_kl']}
        npz(RUNTIME / key / f'actual_update_{arm}.npz', delta=delta, batch_gradient=batch_gradient)
        emit('gradient_direction', round=subject['round'], arm=arm,
             directions=report['arms'][arm]['direction_by_group']['policy_all'],
             measured=report['arms'][arm]['measured_KL_change'])
    write_json(path, report)


def train(subject, key, arm, device, identity, max_steps=None):
    folder = OUT / key / arm
    cps = folder / 'checkpoints'; cps.mkdir(parents=True, exist_ok=True)
    latest = sorted(cps.glob('step-*.pt'))
    saved = latest[-1] if latest else Path(subject['start']['path'])
    learner = restore(saved, device, identity, require_identity=bool(latest))
    ctx = context(subject); index, lookup, store, plan = ctx
    metrics_file = folder / 'metrics.jsonl'
    committed = learner.round_train_step
    if metrics_file.exists():
        rows = [json.loads(line) for line in metrics_file.read_text().splitlines()]
        metrics_file.write_text(''.join(json.dumps(row)+'\n' for row in rows if row['round_train_step'] <= committed))
    for step in NODES:
        if step == committed and not (ART / 'nodes' / f'{key}_{arm}_{step:06d}.json').exists():
            node(learner, subject, key, arm, step, ctx)
    limit = STEPS if max_steps is None else min(STEPS, committed+max_steps)
    for i in range(committed, limit):
        p_samples, p_refs = resolve(index, lookup, plan[f'policy_{arm}'][i], store)
        v_samples, v_refs = resolve(index, lookup, plan['value'][i])
        metrics = learner.train_split_step(collate_selfplay_samples(p_samples), collate_selfplay_samples(v_samples),
            references=p_refs+v_refs, policy_max_grad_norm=subject['split_training']['policy_max_grad_norm'],
            value_max_grad_norm=subject['split_training']['value_max_grad_norm'])
        learner.sampling_rng = DeterministicRng.restore(RngState('splitmix64-v1', int(plan['rng_state'][i]), int(plan['rng_draws'][i])))
        with metrics_file.open('a') as handle:
            handle.write(json.dumps(metrics)+'\n')
        step = learner.round_train_step
        if step % 25 == 0 or step == limit:
            checkpoint = cps / f'step-{step:06d}.pt'
            learner.save(checkpoint, metadata={'experiment': IDENTIFIER, 'arm': arm, 'source_round': subject['round']})
            if step % 50 == 0 or step == limit:
                emit('training_progress', round=subject['round'], arm=arm, step=step, KL=metrics['policy_kl'])
        if arm == 'A' and step == 100:
            formal = load_alphazero_checkpoint(subject['formal_final']['path'], device='cpu')
            if model_state_sha256(learner.model.state_dict()) != formal.model_sha256:
                raise ValueError('A@100 does not exactly reproduce the formal model')
            got = learner.optimizer.state_dict()
            if got['param_groups'] != formal.optimizer_state_dict['param_groups']:
                raise ValueError('formal AdamW groups differ')
            for ordinal, state in got['state'].items():
                for name, value in state.items():
                    other = formal.optimizer_state_dict['state'][ordinal][name]
                    equal = torch.equal(value.cpu(), other.cpu()) if isinstance(value, torch.Tensor) else value == other
                    if not equal:
                        raise ValueError('A@100 does not reproduce formal AdamW')
            write_json(ART / f'reproduction_{key}.json', {'model_and_AdamW': 'bitwise_exact',
                'model_sha256': formal.model_sha256, 'formal': subject['formal_final']})
        if step in NODES:
            node(learner, subject, key, arm, step, ctx)
    emit('training_job_finished', round=subject['round'], arm=arm, step=learner.round_train_step)


def summarize(spec):
    result = {'protocol': spec['protocol'], 'status': 'complete', 'subjects': {},
              'scope': 'fixed-data supervised diagnostic, not strength or online training; one batch seed per subject'}
    for key in spec['subjects']:
        result['subjects'][key] = {'arms': {}, 'gradients': None}
        gp = ART / f'gradients_{key}.json'
        if gp.exists():
            result['subjects'][key]['gradients'] = read_json(gp)
        else:
            result['status'] = 'incomplete'
        for arm in ARMS:
            points = []
            for step in NODES:
                path = ART / 'nodes' / f'{key}_{arm}_{step:06d}.json'
                if path.exists():
                    points.append(read_json(path))
                else:
                    result['status'] = 'incomplete'
            result['subjects'][key]['arms'][arm] = points
    write_json(ART / 'results.json', result)
    emit('summarized', status=result['status'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--worker', choices=['gradients', 'train', 'summarize'])
    parser.add_argument('--key', choices=[f'r{n:03d}' for n in ROUNDS])
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--max-steps-this-job', type=int)
    args = parser.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    if args.worker:
        spec = read_json(ART / 'specification.json')
        for name, expected in spec['code'].items():
            path = Path(__file__) if name == 'run.py' else ROOT / 'src/splendor' / name
            if sha256(path) != expected:
                raise ValueError(f'execution source changed: {path}')
        if args.worker == 'summarize':
            summarize(spec); return
        subject = spec['subjects'][args.key]
        for field in ('start', 'snapshot', 'dataset', 'batch_plans'):
            if sha256(Path(subject[field]['path'])) != subject[field]['sha256']:
                raise ValueError(f'frozen input changed: {field}')
        identity = {'digest': sha256(ART / 'specification.json'), 'subject': args.key,
                    'arm': args.arm if args.worker == 'train' else 'diagnostic', 'device': args.device}
        if args.worker == 'gradients':
            gradients(subject, args.key, args.device, identity)
        else:
            train(subject, args.key, args.arm, args.device, identity, args.max_steps_this_job)
        return
    RUNTIME.mkdir(parents=True, exist_ok=True)
    with (RUNTIME / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        spec = prepare()
        if args.prepare_only:
            return
        write_json(ART / 'environment.json', {'environment': environment_identity(), 'command': sys.argv,
                                            'device': args.device, 'started_at_unix': time.time()})
        # Run the latest case first; others are non-overlapping four-teacher windows.
        for key in reversed(list(spec['subjects'])):
            for kind, arm in [('gradients', None)]+[('train', a) for a in ARMS]:
                command = [sys.executable, '-u', str(Path(__file__)), '--worker', kind, '--key', key, '--device', args.device]
                if arm:
                    command += ['--arm', arm]
                subprocess.run(command, cwd=ROOT, check=True)
        summarize(spec)


if __name__ == '__main__':
    main()
