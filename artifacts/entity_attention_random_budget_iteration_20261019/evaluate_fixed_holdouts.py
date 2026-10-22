"""Read-only network evaluation on two complete, frozen validation cohorts."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from splendor.actor_critic import TorchDecisionBatch, TorchStateBatch
from splendor.alphazero_recording import environment_identity, source_identity, write_json
from splendor.alphazero_train import AlphazeroLearner, LearnerConfig
from splendor.features import FEATURE_SCHEMA_VERSION
from splendor.offline_train import _digest
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.pool_fitting import ratios
from splendor.selfplay_data import STATE_FIELDS, SelfPlaySample, collate_selfplay_samples

STEPS = (0, 100, 250, 500, 1000)
COLUMNS = ('positions', 'policy_weight', 'ce', 'kl', 'target_entropy', 'entropy', 'top1',
           'sq_error', 'pred', 'pred_sq', 'target', 'target_sq', 'product')
GROUPS = (('models06_11_data01_06', range(1, 7), range(6, 12)),
          ('models12_17_data06_11', range(6, 12), range(12, 18)))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def emit(**row):
    print(json.dumps(row, ensure_ascii=False), flush=True)


def tensor_state(fields, selected):
    return TorchStateBatch(feature_schema_version=FEATURE_SCHEMA_VERSION,
        **{name: value[selected] for name, value in fields.items()})


def summarize(sums):
    out = ratios(dict(zip(COLUMNS, map(float, sums))))
    n = out['positions']
    out['value_prediction_mean'] = out['pred']/n
    out['value_prediction_std'] = math.sqrt(max(0., out['pred_sq']/n-(out['pred']/n)**2))
    out['value_target_mean'] = out['target']/n
    out['value_target_std'] = math.sqrt(max(0., out['target_sq']/n-(out['target']/n)**2))
    out['kl_plus_mse'] = out['policy_kl']+out['value_mse']
    return out


class FixedPanel:
    """One GPU copy of fixed states, plus tightly padded positive-policy rows."""

    def __init__(self, records, device, batch_size):
        self.records, self.device, self.batch_size = records, device, batch_size
        fields = {name: [] for name in STATE_FIELDS}
        features, targets, lengths, policy_ids = [], [], [], []
        values, weights, game_ids, low_budget = [], [], [], []
        self.verification_samples = []
        self.verification_ids = []
        offset = 0
        needed = (*STATE_FIELDS, 'candidate_features', 'candidate_offsets', 'policy_target',
                  'policy_weight', 'value_target', 'value_weight', 'low_budget', 'metadata_json',
                  'behavior_policy', 'decision_index', 'player_id')
        for game_index, record in enumerate(records):
            path = Path(record['path'])
            stat = path.stat()
            if (stat.st_size, stat.st_mtime_ns) != (record['size'], record['mtime_ns']):
                raise ValueError(f'archive identity changed: {path}')
            with np.load(path, allow_pickle=False) as archive:
                arrays = {name: archive[name] for name in needed}
            metadata = json.loads(arrays.pop('metadata_json').tobytes().decode())
            if any(metadata[k] != record[k] for k in ('run_id', 'round_index', 'game_id', 'split')):
                raise ValueError(f'archive metadata changed: {path}')
            if metadata['split'] != 'validation' or metadata['feature_schema_version'] != FEATURE_SCHEMA_VERSION:
                raise ValueError('only original whole-game validation data may enter panel')
            if not np.all(arrays['value_weight'] == 1):
                raise ValueError('this experiment requires the original unit value weights')
            for name in STATE_FIELDS:
                fields[name].append(arrays[name])
            n = len(arrays['value_target'])
            if n != record['decision_count']:
                raise ValueError('archive decision count changed')
            positive = arrays['policy_weight'] > 0
            if np.count_nonzero(positive) != record['policy_samples']:
                raise ValueError('archive policy count changed')
            lens = np.diff(arrays['candidate_offsets'])
            candidates = np.repeat(positive, lens)
            features.append(arrays['candidate_features'][candidates])
            targets.append(arrays['policy_target'][candidates])
            lengths.append(lens[positive])
            policy_ids.append(offset+np.flatnonzero(positive))
            weights.append(arrays['policy_weight'][positive])
            values.append(arrays['value_target'])
            game_ids.append(np.full(n, game_index, dtype=np.int32))
            low_budget.append(arrays['low_budget'])
            if game_index < 6:
                indices = np.unique(np.r_[np.linspace(0, n-1, 8, dtype=int), np.flatnonzero(positive)[:2]])
                self.verification_samples.extend(SelfPlaySample(arrays, metadata, int(i)) for i in indices)
                self.verification_ids.extend(offset+indices)
            offset += n
            if (game_index+1) % 400 == 0:
                emit(event='panel_loading', games=game_index+1, total_games=len(records))
        self.fields = {name: torch.as_tensor(np.concatenate(parts), device=device)
                       for name, parts in fields.items()}
        self.y = np.concatenate(values).astype(np.float64)
        self.game_ids = np.concatenate(game_ids)
        self.low_budget = np.concatenate(low_budget)
        self.policy_ids = np.concatenate(policy_ids)
        self.high_policy = np.zeros(len(self.y), dtype=np.bool_)
        self.high_policy[self.policy_ids] = True
        self.pw = np.concatenate(weights).astype(np.float64)
        self.policy_game_ids = self.game_ids[self.policy_ids]
        lens = np.concatenate(lengths)
        candidate_offsets = np.r_[0, np.cumsum(lens)]
        flat_features, flat_targets = np.concatenate(features), np.concatenate(targets)
        log_targets = np.zeros(flat_targets.shape, dtype=np.float64)
        np.log(flat_targets, out=log_targets, where=flat_targets > 0)
        self.target_entropy = np.add.reduceat(-flat_targets*log_targets, candidate_offsets[:-1])
        self.policy_batches = []
        order = np.argsort(lens, kind='stable')
        for start in range(0, len(order), batch_size):
            ids = order[start:start+batch_size]
            width = int(lens[ids].max())
            mask = np.arange(width)[None, :] < lens[ids, None]
            gather = np.where(mask, candidate_offsets[ids, None]+np.arange(width)[None, :], 0)
            candidates = np.where(mask[..., None], flat_features[gather], 0)
            target = torch.as_tensor(np.where(mask, flat_targets[gather], 0), device=device)
            states = tensor_state(self.fields, torch.as_tensor(self.policy_ids[ids], device=device))
            prepared = TorchDecisionBatch(feature_schema_version=FEATURE_SCHEMA_VERSION,
                **{name: getattr(states, name) for name in STATE_FIELDS},
                candidate_features=torch.as_tensor(candidates, device=device),
                candidate_mask=torch.as_tensor(mask, device=device))
            self.policy_batches.append((torch.as_tensor(ids, device=device), prepared, target))
        self.value_batches = [(slice(start, min(start+batch_size, len(self.y))),
            tensor_state(self.fields, slice(start, start+batch_size)))
            for start in range(0, len(self.y), batch_size)]
        if device.type == 'cuda':
            torch.cuda.synchronize()

    def evaluate(self, model):
        tick = time.perf_counter()
        pred = torch.empty(len(self.y), device=self.device)
        policy = torch.empty((len(self.policy_ids), 3), device=self.device)
        with torch.inference_mode():
            for selected, batch in self.value_batches:
                pred[selected] = model.value_from_state(batch)
            for ids, batch, target in self.policy_batches:
                logits = model.policy_logits(batch)
                lp = torch.log_softmax(logits.masked_fill(~batch.candidate_mask, -torch.inf), dim=-1)
                lp = torch.where(batch.candidate_mask, lp, 0.)
                probabilities = torch.where(batch.candidate_mask, lp.exp(), 0.)
                ce = -(target*lp).sum(-1)
                entropy = -(probabilities*lp).sum(-1)
                best = logits.masked_fill(~batch.candidate_mask, -torch.inf).argmax(-1)
                correct = target.gather(1, best[:, None]).squeeze(1) >= target.max(-1).values-1e-7
                policy[ids] = torch.stack((ce, entropy, correct.float()), dim=-1)
        predictions = pred.cpu().numpy().astype(np.float64)
        policies = policy.cpu().numpy().astype(np.float64)
        if not np.isfinite(predictions).all() or not np.isfinite(policies).all():
            raise ValueError('nonfinite fixed-panel predictions')
        per_game = self.statistics(predictions, policies)
        return {'metrics': summarize(per_game.sum(axis=0)),
                'value_by_budget': {label: self.value_stats(predictions, selected) for label,selected in
                    (('low', self.low_budget), ('high_policy', self.high_policy),
                     ('forced_or_other', ~(self.low_budget | self.high_policy)))},
                'seconds': time.perf_counter()-tick}, per_game, predictions, policies

    def value_stats(self, predictions, selected):
        y, p = self.y[selected], predictions[selected]
        return {'positions': len(y), 'mse': float(np.mean((p-y)**2)),
                'prediction_std': float(p.std()), 'bias': float((p-y).mean())}

    def statistics(self, predictions, policies):
        out = np.zeros((len(self.records), len(COLUMNS)), dtype=np.float64)
        all_values = {'positions': np.ones(len(self.y)), 'sq_error': (predictions-self.y)**2,
            'pred': predictions, 'pred_sq': predictions**2, 'target': self.y,
            'target_sq': self.y**2, 'product': predictions*self.y}
        p_values = {'policy_weight': self.pw, 'ce': policies[:, 0]*self.pw,
            'kl': (policies[:, 0]-self.target_entropy)*self.pw,
            'target_entropy': self.target_entropy*self.pw, 'entropy': policies[:, 1]*self.pw,
            'top1': policies[:, 2]*self.pw}
        for name, values in all_values.items():
            out[:, COLUMNS.index(name)] = np.bincount(self.game_ids, values, minlength=len(out))
        for name, values in p_values.items():
            out[:, COLUMNS.index(name)] = np.bincount(self.policy_game_ids, values, minlength=len(out))
        return out

    def verify(self, model, predictions, policies):
        # Compare optimized state-only/positive-policy paths with the existing
        # joint learner loss implementation on the exact same real samples.
        learner = SimpleNamespace(model=model, device=self.device,
            config=LearnerConfig(device=str(self.device), validation_batch_size=256))
        expected = AlphazeroLearner.evaluate_samples(learner, self.verification_samples)
        ids = np.asarray(self.verification_ids)
        pids = np.flatnonzero(np.isin(self.policy_ids, ids))
        actual_mse = float(np.mean((predictions[ids]-self.y[ids])**2))
        actual_ce = float(np.sum(policies[pids, 0]*self.pw[pids])/self.pw[pids].sum())
        actual_kl = float(np.sum((policies[pids, 0]-self.target_entropy[pids])*self.pw[pids])/self.pw[pids].sum())
        differences = {'value_mse': actual_mse-expected['value_loss'],
            'policy_ce': actual_ce-expected['policy_loss'], 'policy_kl': actual_kl-expected['policy_kl']}
        if max(map(abs, differences.values())) > 5e-5:
            raise AssertionError(f'optimized panel disagrees with learner: {differences}')
        return {'positions': len(ids), 'policy_positions': len(pids), 'differences': differences,
                'reference': 'AlphazeroLearner.evaluate_samples, joint forward'}


def freeze_plan(run, output, device, batch_size, smoke):
    groups = []
    state = json.loads((run/'state.json').read_text())
    if len(state['completed_rounds']) < 17:
        raise ValueError('requires completed iterations1-17')
    for name, source_iterations, model_iterations in GROUPS:
        records, sources = [], []
        for iteration in source_iterations:
            path = run/'rounds'/f'round-{iteration-1:03d}'/'collection_index.json'
            sources.append({'path': str(path.resolve()), 'sha256': sha(path)})
            records.extend({**g, 'source_iteration': iteration} for g in json.loads(path.read_text())['games']
                           if g['split'] == 'validation')
        keys = {(g['run_id'], g['round_index'], g['game_id']) for g in records}
        if len(keys) != len(records):
            raise ValueError('duplicate validation game')
        for iteration in range(1,18):
            path = run/'rounds'/f'round-{iteration-1:03d}'/'replay_snapshot.json'
            sources.append({'path':str(path.resolve()), 'sha256':sha(path)})
            for game in json.loads(path.read_text())['games']:
                if game['split'] == 'train' and (game['run_id'], game['round_index'], game['game_id']) in keys:
                    raise ValueError('validation game leaked into training')
        if smoke:
            records = records[:2]
        for game in records:
            game['archive_sha256'] = sha(game['path'])
        checkpoints = []
        for iteration in model_iterations:
            for step in STEPS:
                path = run/'rounds'/f'round-{iteration-1:03d}'/'checkpoints'/(
                    'initial.pt' if step == 0 else f'step-{step:09d}.pt')
                checkpoints.append({'iteration':iteration, 'step':step,
                    'total_train_step':37423+(iteration-1)*1000+step,
                    'path':str(path.resolve()),'sha256':sha(path)})
        if smoke:
            checkpoints = checkpoints[:1]
        group = {'name':name, 'source_iterations':list(source_iterations),
            'model_iterations':list(model_iterations), 'games':records, 'sources':sources,
            'value_positions':sum(g['decision_count'] for g in records),
            'policy_positions':sum(g['policy_samples'] for g in records),
            'checkpoints':checkpoints}
        group['digest'] = _digest(group)
        groups.append(group)
    plan = {'protocol':'six-round-fixed-holdout-v1', 'run':str(run.resolve()),
        'device':device, 'batch_size':batch_size, 'smoke':smoke, 'groups':groups,
        'scope':'Full original whole-game validation; fixed original labels; read-only forward evaluation. First cohort includes iteration6 data collected before iteration6 training. No strength games.'}
    plan['digest'] = _digest(plan)
    path = output/'plan.json'
    if path.exists():
        if json.loads(path.read_text()) != plan:
            raise ValueError('frozen evaluation plan changed')
    else:
        write_json(path, plan)
    return plan


def weights_sha(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path,
        default=Path('runs/entity_attention_37423_random_budget_p6k_v24k_20261019'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu','cuda'), default='cuda')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    plan = freeze_plan(args.run, output, args.device, args.batch_size, args.smoke)
    if not (output/'provenance.json').exists():
        identity = source_identity(Path.cwd(), output/'source')
        (output/'source'/'experiment.py').write_bytes(Path(__file__).read_bytes())
        write_json(output/'provenance.json', {'source':identity, 'environment':environment_identity(),
            'script_sha256':sha(__file__), 'command':sys.argv, 'plan_digest':plan['digest']})
    elif json.loads((output/'provenance.json').read_text())['script_sha256'] != sha(__file__):
        raise ValueError('resume requires the identical frozen evaluation script')
    started = time.perf_counter()
    for group in plan['groups']:
        root = output/group['name']
        root.mkdir(parents=True, exist_ok=True)
        emit(event='panel_start', group=group['name'], games=len(group['games']),
             policy_positions=group['policy_positions'], value_positions=group['value_positions'])
        panel = FixedPanel(group['games'], device, args.batch_size)
        if len(panel.y) != group['value_positions'] or len(panel.policy_ids) != group['policy_positions']:
            raise ValueError('panel population differs from frozen plan')
        verified = False
        for checkpoint in group['checkpoints']:
            model = load_frozen_actor_critic_checkpoint(Path(checkpoint['path']), device=device)
            key = weights_sha(model)
            metrics_path = root/f'weights-{key}.json'
            if metrics_path.exists():
                result = json.loads(metrics_path.read_text())
                if result['panel_digest'] != group['digest']:
                    raise ValueError('cached metrics panel changed')
                reused = True
            else:
                result, per_game, predictions, policies = panel.evaluate(model)
                if not verified:
                    result['reference_verification'] = panel.verify(model, predictions, policies)
                    verified = True
                result.update(panel_digest=group['digest'], weights_sha256=key,
                    checkpoint=checkpoint, columns=COLUMNS)
                np.savez_compressed(root/f'weights-{key}.npz', per_game=per_game)
                write_json(metrics_path, result)
                reused = False
            write_json(root/f'round-{checkpoint["iteration"]:03d}-step-{checkpoint["step"]:04d}.json',
                {'checkpoint':checkpoint, 'weights_sha256':key, 'panel_digest':group['digest'],
                 'metrics_file':str(metrics_path), 'metrics':result['metrics'], 'reused_identical_weights':reused})
            emit(event='checkpoint_evaluated', group=group['name'], iteration=checkpoint['iteration'],
                step=checkpoint['step'], reused=reused, seconds=result['seconds'],
                kl=result['metrics']['policy_kl'], mse=result['metrics']['value_mse'],
                correlation=result['metrics']['value_correlation'])
            del model
        del panel
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    write_json(output/'complete.json', {'plan_digest':plan['digest'], 'seconds':time.perf_counter()-started,
        'checkpoints':sum(len(g['checkpoints']) for g in plan['groups']), 'smoke':args.smoke})
    emit(event='complete', output=str(output), seconds=time.perf_counter()-started)


if __name__ == '__main__':
    main()
