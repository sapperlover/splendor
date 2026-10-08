"""Freeze the complete policy/encoder and replay B2's 50 recorded value batches."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timezone
import fcntl
import gc
import json
from pathlib import Path
import resource
import time

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from splendor.alphazero_checkpoint import (load_alphazero_checkpoint, model_state_sha256,
                                          move_optimizer_state, save_alphazero_checkpoint)
from splendor.alphazero_recording import (environment_identity, read_json, sha256,
                                         source_identity, write_json)
from splendor.alphazero_train import AlphazeroLearner, LearnerConfig, _torch_batch, metric_ratios
from splendor.offline_train import make_index
from splendor.rng import DeterministicRng
from splendor.selfplay_data import collate_selfplay_samples

ROOT = Path(__file__).resolve().parents[2]
ART = Path(__file__).resolve().parent
RUN = ROOT / 'runs/entity_attention_iteration_trend_seed1_20261008/seed1_steps50'
STAGE = RUN / 'rounds/round-001'
START = RUN / 'rounds/round-000/checkpoints/final.pt'
CONTROL = STAGE / 'checkpoints/final.pt'
SNAPSHOT = STAGE / 'replay_snapshot.json'
OUT = ROOT / 'runs/entity_attention_critic_only_replay_20261008'
RUNTIME = ROOT / 'runtime/entity_attention_critic_only_replay_20261008'
SUM_FIELDS = ('policy_ce_sum', 'policy_weight_sum', 'value_sq_sum', 'value_weight_sum',
              'entropy_sum', 'positions', 'target_entropy_sum', 'policy_positions',
              'value_positions', 'value_target_sum', 'value_target_sq_sum')
BASE_COLUMNS = ('round_index', 'game_id', 'validation', 'positions', 'policy_weight_sum',
                'value_weight_sum', 'policy_ce_16550', 'value_sq_16550',
                'policy_ce_16600', 'value_sq_16600', 'archive_size', 'archive_mtime_ns')
CAND_COLUMNS = ('round_index', 'game_id', 'validation', 'positions', 'policy_weight_sum',
                'value_weight_sum', 'policy_ce_sum', 'value_sq_sum')


def record(path):
    return {'path': str(path.resolve()), 'sha256': sha256(path), 'bytes': path.stat().st_size}


def emit(**kwargs):
    print(json.dumps(kwargs, ensure_ascii=False), flush=True)


def combine(rows):
    sums = Counter()
    for row in rows:
        sums.update({k: row[k] for k in SUM_FIELDS})
    return {**sums, **metric_ratios(sums)}


def load_ledgers():
    ledgers, inputs = {}, []
    for head in ('value_recent', 'value_history', 'policy_recent'):
        paths = sorted((STAGE / 'usage' / head).glob('*.bin'))
        assert len(paths) == 1, f'Expected one committed ledger: {head}'
        path = paths[0]
        quota = 256 if head == 'policy_recent' else 128
        rows = np.frombuffer(path.read_bytes(), dtype='<i8').reshape(-1, 4)
        assert np.array_equal(rows[:, 0], np.repeat(np.arange(1, 51), quota))
        ledgers[head] = rows.reshape(50, quota, 4)[:, :, 1:]
        inputs += [record(path), record(path.with_suffix('.json')), record(path.with_suffix('.steps.jsonl'))]
    return ledgers, inputs


def prepare(baseline_path):
    path = ART / 'specification.json'
    if path.exists():
        return read_json(path)
    assert baseline_path is not None, 'Initial preparation requires --baseline-matrix'
    data = read_json(baseline_path)
    assert data['checkpoints']['16550'] == record(START)
    assert data['checkpoints']['16600'] == record(CONTROL)
    assert data['snapshot'] == record(SNAPSHOT)
    source = source_identity(ROOT, RUNTIME / 'source')
    baseline_source = read_json(baseline_path.parent / 'manifest.json')['source']
    assert source['source_sha256'] == baseline_source['source_sha256']
    snapshot = read_json(SNAPSHOT)
    games = {(g['round_index'], g['game_id']): g for g in snapshot['games']}
    rows, cohorts = [], []
    for c in data['cohorts']:
        entry = {k: c[k] for k in ('round_index', 'label', 'teacher')}
        entry['metrics'] = {step: {split: c['checkpoints'][step][split]['metrics']
                                  for split in ('train', 'validation')}
                            for step in ('16550', '16600')}
        cohorts.append(entry)
        for split in ('train', 'validation'):
            before = c['checkpoints']['16550'][split]['per_game']
            control = c['checkpoints']['16600'][split]['per_game']
            assert [(g['round_index'], g['game_id']) for g in before] == [(g['round_index'], g['game_id']) for g in control]
            for a, b in zip(before, control):
                g = games[(a['round_index'], a['game_id'])]
                stat = Path(g['path']).stat()
                assert (stat.st_size, stat.st_mtime_ns) == (a['archive_size'], a['archive_mtime_ns'])
                rows.append([a['round_index'], a['game_id'], int(split == 'validation'),
                             a['positions'], a['policy_weight_sum'], a['value_weight_sum'],
                             a['policy_ce_sum'], a['value_sq_sum'], b['policy_ce_sum'], b['value_sq_sum'],
                             a['archive_size'], a['archive_mtime_ns']])
    baseline = {'columns': BASE_COLUMNS, 'rows': sorted(rows, key=lambda r: (r[0], r[1])), 'cohorts': cohorts,
                'original_input': record(baseline_path), 'original_verification': data['verification']}
    write_json(ART / 'baseline.json', baseline)
    _, ledger_inputs = load_ledgers()
    start_payload = load_alphazero_checkpoint(START, device='cpu')
    spec = {'protocol': 'critic-only-fixed-encoder-formal-ledger-replay-v1',
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'start': record(START), 'control': record(CONTROL), 'snapshot': record(SNAPSHOT),
            'baseline': record(ART / 'baseline.json'), 'ledger_inputs': ledger_inputs,
            'script': record(Path(__file__)), 'source': source, 'environment': environment_identity(),
            'steps': 50, 'value_recent': 128, 'value_history': 128,
            'learning_rate': start_payload.optimizer_config['learning_rate'],
            'weight_decay': start_payload.optimizer_config['weight_decay'], 'value_max_grad_norm': 1.0,
            'start_learner_config': start_payload.learner_config,
            'device': 'cpu', 'dtype': 'float32', 'bootstrap_replicates': 10000,
            'bootstrap_seed': 2026100830,
            'scope': 'Only critic parameters update. Complete encoder/policy and their AdamW states freeze. '
                     'Exact 50 committed value batches, unchanged terminal labels, no collection or games.',
            'command': 'conda run --no-capture-output -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 '
                       'python -u artifacts/entity_attention_critic_only_replay_20261008/run.py --resume'}
    write_json(path, spec)
    return spec


def restore(path, spec):
    payload = load_alphazero_checkpoint(path, device='cpu')
    config = replace(LearnerConfig(**spec['start_learner_config']), device='cpu', validation_batch_size=256)
    learner = AlphazeroLearner(payload.build_model(device='cpu'), config,
                               sampling_rng=DeterministicRng.restore(payload.sampling_rng_state))
    learner.optimizer.load_state_dict(payload.optimizer_state_dict)
    move_optimizer_state(learner.optimizer, learner.device)
    learner.round_index = read_json(SNAPSHOT)['round_index'] + 1
    learner.round_train_step = max(0, payload.total_train_step - 16550)
    learner.total_train_step = payload.total_train_step
    learner.round_steps_target = 50
    learner.parent_checkpoint = dict(spec['start'])
    learner.replay_identity = {'digest': read_json(SNAPSHOT)['digest'], 'dataset_path': str(SNAPSHOT)}
    torch.set_rng_state(payload.torch_cpu_rng_state.cpu())
    return learner


def frozen_hash(model):
    return model_state_sha256({k: v for k, v in model.state_dict().items() if not k.startswith('critic.')})


def freeze(learner):
    for name, param in learner.model.named_parameters():
        param.requires_grad_(name.startswith('critic.'))


def value_step(learner, batch):
    learner.model.train()
    tensors = _torch_batch(batch, learner.device)
    with torch.no_grad():
        state = learner.model.encode_state(learner.model._prepare(batch.decisions))
    values = learner.model.critic((state[0].detach(), state[1])).squeeze(-1)
    numerator = ((values - tensors['value_targets']) ** 2 * tensors['value_weights']).sum()
    denominator = tensors['value_weights'].sum()
    loss = numerator / denominator.clamp_min(torch.finfo(torch.float32).tiny)
    assert torch.isfinite(loss)
    learner.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = float(clip_grad_norm_(learner.model.critic.parameters(), 1.0))
    assert np.isfinite(norm)
    assert all(p.grad is None for n, p in learner.model.named_parameters() if not n.startswith('critic.'))
    learner.optimizer.step()
    learner.round_train_step += 1
    learner.total_train_step += 1
    return {'step': learner.round_train_step, 'total_train_step': learner.total_train_step,
            'value_loss_before_update_on_batch': float(loss), 'value_weight_sum': float(denominator),
            'value_grad_norm': norm, 'value_grad_clipped': norm > 1.0}


def samples(index, lookup, refs):
    result = []
    for r, gid, decision in refs:
        game = lookup[(int(r), int(gid))]
        assert game.split == 'train'
        assert 0 <= int(decision) < game.decision_count
        result.append(index.cache.sample(game, int(decision)))
    return result


def smoke(spec, ledger, index, lookup):
    a, b = restore(START, spec), restore(START, spec)
    freeze(a)
    before = frozen_hash(a.model)
    vrefs = np.concatenate((ledger['value_recent'][0], ledger['value_history'][0]))
    value_batch = collate_selfplay_samples(samples(index, lookup, vrefs))
    row = value_step(a, value_batch)
    policy_batch = collate_selfplay_samples(samples(index, lookup, ledger['policy_recent'][0]))
    b.train_split_step(policy_batch, value_batch, references=(), policy_max_grad_norm=1.0, value_max_grad_norm=1.0)
    difference = max(float((v-b.model.critic.state_dict()[k]).abs().max()) for k, v in a.model.critic.state_dict().items())
    assert difference == 0.0
    assert frozen_hash(a.model) == before
    original = [json.loads(line) for line in (STAGE / 'training_metrics.jsonl').read_text().splitlines()]
    logged = next(r for r in original if r.get('event') == 'train_step' and r['round_train_step'] == 1)
    assert abs(row['value_loss_before_update_on_batch'] - logged['value_loss']) < 1e-5
    out = {'one_step_critic_matches_existing_split_call_exactly': True,
           'critic_parameter_max_abs_difference': difference, 'frozen_parameters_unchanged': True,
           'first_batch_cpu_minus_original_value_loss': row['value_loss_before_update_on_batch']-logged['value_loss']}
    write_json(ART / 'smoke.json', out)
    emit(event='smoke_passed', **out)
    del a, b, value_batch, policy_batch
    gc.collect()


def save(learner, metrics, spec):
    step = learner.round_train_step
    path = OUT / f'checkpoints/step-{step:06d}.pt'
    identity = save_alphazero_checkpoint(path, model=learner.model, optimizer=learner.optimizer,
        learner_config=asdict(learner.config) | {'policy_loss_weight': 0.0},
        round_index=learner.round_index, round_train_step=step, total_train_step=learner.total_train_step,
        round_steps_target=50, parent_checkpoint=spec['start'], replay_identity=learner.replay_identity,
        sampling_rng_state=learner.sampling_rng.snapshot(),
        metadata={'experiment': spec['protocol'], 'training_mode': 'critic_only_complete_encoder_frozen',
                  'diagnostic_only_not_formal_resume': True})
    write_json(OUT / 'state.json', {'step': step, 'checkpoint': identity, 'training_metrics': metrics})


def train(spec, ledger, resume):
    state_path = OUT / 'state.json'
    state = read_json(state_path) if state_path.exists() else None
    if state:
        assert resume, 'Existing training output requires --resume'
    learner = restore(Path(state['checkpoint']['path']) if state else START, spec)
    metrics = state['training_metrics'] if state else []
    freeze(learner)
    initial = load_alphazero_checkpoint(START, device='cpu')
    before = model_state_sha256({k: v for k, v in initial.model_state_dict.items() if not k.startswith('critic.')})
    index = make_index(read_json(SNAPSHOT), 2 * 1024**3)
    lookup = {g.key: g for g in index.games}
    if not (ART / 'smoke.json').exists():
        smoke(spec, ledger, index, lookup)
    if not state:
        save(learner, metrics, spec)
    for step in range(learner.round_train_step + 1, 51):
        refs = np.concatenate((ledger['value_recent'][step-1], ledger['value_history'][step-1]))
        batch = collate_selfplay_samples(samples(index, lookup, refs))
        row = value_step(learner, batch)
        del batch
        row['references_sha256'] = __import__('hashlib').sha256(refs.astype('<i8').tobytes()).hexdigest()
        metrics.append(row)
        if step % 10 == 0:
            emit(event='training', **row)
        if step in (25, 50):
            assert frozen_hash(learner.model) == before
            save(learner, metrics, spec)
    assert frozen_hash(learner.model) == before
    original_states = initial.optimizer_state_dict['state']
    actual_states = learner.optimizer.state_dict()['state']
    for i, (name, _) in enumerate(learner.model.named_parameters()):
        if name.startswith('critic.'):
            continue
        assert original_states[i].keys() == actual_states[i].keys()
        for key, value in original_states[i].items():
            assert torch.equal(value, actual_states[i][key]) if isinstance(value, torch.Tensor) else value == actual_states[i][key]
    checks = {'complete_encoder_and_policy_parameters_unchanged': True,
              'noncritic_adamw_states_unchanged': True,
              'critic_parameter_max_abs_change': max(float((v-initial.model_state_dict[k]).abs().max())
                                                    for k, v in learner.model.state_dict().items() if k.startswith('critic.')),
              'value_batches_replayed': len(metrics), 'value_positions_drawn': len(metrics)*256,
              'value_gradient_mean': float(np.mean([m['value_grad_norm'] for m in metrics])),
              'value_clipped_steps': sum(m['value_grad_clipped'] for m in metrics), 'training_cache': index.cache.stats()}
    del initial, index, lookup
    gc.collect()
    write_json(ART / 'training.json', {'metrics': metrics, 'verification': checks,
                                    'checkpoint': record(OUT / 'checkpoints/step-000050.pt')})
    return learner, checks


def paired_ci(rows, numerator, denominator, seed):
    rng = np.random.default_rng(seed)
    strata = [np.where(rows[:, 0] == r)[0] for r in np.unique(rows[:, 0])]
    estimates = np.empty(10000)
    for start in range(0, 10000, 256):
        size = min(256, 10000-start)
        sums = np.zeros(size)
        weights = np.zeros(size)
        for group in strata:
            ids = group[rng.integers(0, len(group), size=(size, len(group)))]
            sums += numerator[ids].sum(axis=1)
            weights += denominator[ids].sum(axis=1)
        estimates[start:start+size] = sums/weights
    return {'difference': float(numerator.sum()/denominator.sum()),
            'lower': float(np.quantile(estimates, .025)), 'upper': float(np.quantile(estimates, .975)),
            'games': len(rows), 'bootstrap_replicates': 10000, 'bootstrap_seed': seed,
            'paired_whole_games': True, 'stratified_by_data_round': True}


def evaluate(learner, spec, checks):
    baseline = read_json(ART / 'baseline.json')
    prior = {(r[0], r[1]): r for r in baseline['rows']}
    index = make_index(read_json(SNAPSHOT), 64 * 1024**2)
    output, candidate_rows = [], []
    for c in baseline['cohorts']:
        entry = {k: c[k] for k in ('round_index', 'label', 'teacher')}
        entry['metrics'] = {**c['metrics'], 'critic_only': {}}
        for split in ('train', 'validation'):
            path = RUNTIME / f'evaluation/round-{c["round_index"]:03d}-{split}.json'
            if path.exists():
                saved = read_json(path)
                assert saved['checkpoint_sha256'] == sha256(OUT / 'checkpoints/step-000050.pt')
            else:
                per_game = []
                for game in index.games:
                    if game.round_index != c['round_index'] or game.split != split:
                        continue
                    a = prior[game.key]
                    stat = game.path.stat()
                    assert (stat.st_size, stat.st_mtime_ns) == (a[10], a[11])
                    metrics = learner.evaluate_samples([index.cache.sample(game, i) for i in range(game.decision_count)])
                    assert metrics['positions'] == a[3]
                    assert metrics['value_weight_sum'] == a[5]
                    assert metrics['policy_weight_sum'] == a[4]
                    assert abs(metrics['policy_ce_sum']-a[6]) < 1e-4
                    per_game.append({**metrics, 'round_index': game.round_index, 'game_id': game.game_id,
                                     'validation': int(split == 'validation')})
                saved = {'checkpoint_sha256': sha256(OUT / 'checkpoints/step-000050.pt'),
                         'metrics': combine(per_game), 'per_game': per_game}
                write_json(path, saved)
            entry['metrics']['critic_only'][split] = saved['metrics']
            for g in saved['per_game']:
                candidate_rows.append([g[k] for k in CAND_COLUMNS])
            emit(event='evaluation', cohort=c['label'], split=split, value_mse=saved['metrics']['value_loss'])
        output.append(entry)
    candidate_rows.sort(key=lambda r: (r[0], r[1]))
    before = np.array(baseline['rows'], dtype=np.float64)
    after = np.array(candidate_rows, dtype=np.float64)
    assert np.array_equal(before[:, :6], after[:, :6])
    def group(indices, seed):
        a, b = before[indices], after[indices]
        weights = a[:, 5]
        return {'games': len(a), 'positions': int(a[:, 3].sum()),
                'MSE': {'16550': float(a[:, 7].sum()/weights.sum()),
                        '16600_control': float(a[:, 9].sum()/weights.sum()),
                        'critic_only': float(b[:, 7].sum()/weights.sum())},
                'critic_only_minus_start': paired_ci(a, b[:, 7]-a[:, 7], weights, seed),
                'control_minus_start': paired_ci(a, a[:, 9]-a[:, 7], weights, seed),
                'critic_only_minus_control': paired_ci(a, b[:, 7]-a[:, 9], weights, seed)}
    aggregates = {}
    for label, condition in (('all', np.ones(len(before), dtype=bool)), ('recent_77_80', before[:, 0]>=77),
                             ('history_66_76', before[:, 0]<77), ('new_B2_80', before[:, 0]==80)):
        aggregates[label] = {split: group(np.where(condition & (before[:, 2]==val))[0],
                                         spec['bootstrap_seed']+(1000 if val == 0 else 0))
                             for split, val in (('train', 0), ('validation', 1))}
    for c in output:
        c['changes'] = {split: group(np.where((before[:, 0]==c['round_index']) & (before[:, 2]==val))[0],
                                    spec['bootstrap_seed']+c['round_index']+(1000 if val == 0 else 0))
                        for split, val in (('train', 0), ('validation', 1))}
    checks.update({'policy_loss_unchanged_on_all_3000_games': True, 'evaluated_games': len(after),
                   'cpu_peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    result = {'protocol': spec['protocol'], 'specification': record(ART / 'specification.json'),
              'cohorts': output, 'aggregates': aggregates, 'verification': checks,
              'per_game_columns': CAND_COLUMNS, 'per_game_rows': candidate_rows,
              'checkpoint': record(OUT / 'checkpoints/step-000050.pt')}
    write_json(ART / 'results.json', result)
    lines = ['# 固定encoder，只练critic：B2的50步账本回放', '',
             '从16550检查点保留AdamW状态，原样重放B2的50步value批次：每步128近期＋128历史位置，lr/weight_decay=1e-4，critic独立clip1。encoder与policy全部冻结，参数梯度为None；诊断分支独立保存，没有推进正式训练。', '',
             '全量评估固定15代、3000局、原始整局train/validation。起点16550与正式16600对照复用此前已核验的完整前向统计；本次补算critic-only。MSE越小越好，Δ为后−前。CPU FP32；无搜索或棋力对局。', '',
             '| 范围 | 划分 | 起点16550 | 正式16600 | critic-only 50步 | Δcritic-only | Δ正式对照 | critic-only变化95%区间 |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for label, splits in aggregates.items():
        for split, g in splits.items():
            ci, m = g['critic_only_minus_start'], g['MSE']
            lines.append(f"| {label} | {split} | {m['16550']:.6f} | {m['16600_control']:.6f} | {m['critic_only']:.6f} | {ci['difference']:+.6f} | {g['control_minus_start']['difference']:+.6f} | [{ci['lower']:+.6f},{ci['upper']:+.6f}] |")
    lines += ['', '| 教师代际 | 训练MSE 起点/critic-only/正式 | 留出MSE 起点/critic-only/正式 | critic-only Δ训练/留出 |',
              '|---|---:|---:|---:|']
    for c in output:
        entries = []
        for split in ('train', 'validation'):
            m = c['changes'][split]['MSE']
            entries.append(f"{m['16550']:.6f}/{m['critic_only']:.6f}/{m['16600_control']:.6f}")
        ds = [c['changes'][s]['critic_only_minus_start']['difference'] for s in ('train', 'validation')]
        lines.append(f"| {c['label']} | {' | '.join(entries)} | {ds[0]:+.6f}/{ds[1]:+.6f} |")
    lines += ['', '配对区间使用10000次整局bootstrap，聚合时按教师代际分层。仅反映这一冻结数据集与训练轨迹，没有多重比较校正或跨训练种子验证。完整逐局分子分母与critic-only对正式分支的差值区间见results.json。', '',
              '核验：首步critic更新与现行train_split_step的首步critic参数逐张量一致；50步后encoder/policy参数和其AdamW状态不变，critic参数实际改变；全部3000局的policy CE保持起点值。模型是诊断checkpoint，实际policy_loss_weight=0，不作为正式hybrid续训入口。', '',
              '## 命令与保存位置', '', '```bash', spec['command'], '```', '',
              '训练模型和可恢复状态：runs/entity_attention_critic_only_replay_20261008/；逐代可恢复前向分片：runtime/entity_attention_critic_only_replay_20261008/evaluation/。baseline.json保留本次固定起点/对照的逐局统计，运行不再依赖/tmp；训练引用、代码身份和环境在specification.json，训练指标在training.json。', '']
    (ART / 'README.md').write_text('\n'.join(lines), encoding='utf-8')
    emit(event='complete', aggregates=aggregates, verification=checks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-matrix', type=Path)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    OUT.mkdir(parents=True, exist_ok=True)
    RUNTIME.mkdir(parents=True, exist_ok=True)
    with (OUT / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        spec = prepare(args.baseline_matrix)
        for item in [spec['start'], spec['control'], spec['snapshot'], spec['baseline'], spec['script'], *spec['ledger_inputs']]:
            assert sha256(Path(item['path'])) == item['sha256'], f"Input changed: {item['path']}"
        assert source_identity(ROOT)['source_sha256'] == spec['source']['source_sha256']
        if (ART / 'results.json').exists():
            assert args.resume, 'Completed output requires --resume'
            emit(event='already_complete', results=str(ART / 'results.json'))
            return
        ledger, _ = load_ledgers()
        learner, checks = train(spec, ledger, args.resume)
        evaluate(learner, spec, checks)


if __name__ == '__main__':
    main()
