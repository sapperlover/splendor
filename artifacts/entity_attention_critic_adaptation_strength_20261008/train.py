"""Continue the original 16600 critic while freezing all encoder/policy weights."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import fcntl
import gc
import json
from pathlib import Path
import resource

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from splendor.alphazero_checkpoint import (load_alphazero_checkpoint, model_state_sha256,
                                          move_optimizer_state, save_alphazero_checkpoint)
from splendor.alphazero_recording import environment_identity, read_json, sha256, source_identity, write_json
from splendor.alphazero_train import AlphazeroLearner, LearnerConfig
from splendor.features import GLOBAL_FEATURE_NAMES, SCALES
from splendor.rng import DeterministicRng

ROOT = Path(__file__).resolve().parents[2]
ART = Path(__file__).resolve().parent
OUT = ROOT / 'runs/entity_attention_critic_adaptation_strength_20261008'
TEMP = ROOT / 'runtime/entity_attention_critic_adaptation_strength_20261008'
REFERENCE = ROOT / 'artifacts/entity_attention_encoder_critic_refit_20261008'
BASE = ROOT / 'runs/entity_attention_iteration_trend_seed1_20261008/seed1_steps50/rounds/round-001/checkpoints/final.pt'
NODES = (0, 50, 200, 500)
COMMAND = ('conda run --no-capture-output -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 '
           'MKL_NUM_THREADS=1 python -u artifacts/entity_attention_critic_adaptation_strength_20261008/train.py --resume')


def identity(path):
    return {'path': str(path.resolve()), 'sha256': sha256(path), 'bytes': path.stat().st_size}


def emit(**row):
    print(json.dumps(row, ensure_ascii=False), flush=True)


def frozen_hash(model):
    return model_state_sha256({k: v for k, v in model.state_dict().items() if not k.startswith('critic.')})


def prepare():
    path = ART / 'training_specification.json'
    if path.exists():
        return read_json(path)
    prior = read_json(REFERENCE / 'specification.json')
    assert prior['checkpoints']['16600'] == identity(BASE)
    feature_manifest = read_json(REFERENCE / 'features_16600.json')
    payload = load_alphazero_checkpoint(BASE, device='cpu')
    spec = {'protocol': '16600-original-critic-adaptation-v1', 'base': identity(BASE),
            'inputs': [identity(REFERENCE / name) for name in
                       ('specification.json', 'dataset.json', 'batches.json', 'features_16600.json')],
            'features': feature_manifest['files'], 'initialization': 'Original 16600 critic and complete AdamW state retained; no reset.',
            'learner_config': payload.learner_config, 'nodes': NODES, 'steps': 500,
            'sampling_seed': prior['sampling_seed'], 'batch_size': 256, 'sampling': '128 recent + 128 history train positions',
            'learning_rate': payload.optimizer_config['learning_rate'], 'weight_decay': payload.optimizer_config['weight_decay'],
            'critic_clip': 1.0, 'device': 'cpu', 'dtype': 'float32',
            'late_score_threshold': 10, 'phase_rule': 'max(self_public.score, opponent_public.score)>=10',
            'source': source_identity(ROOT, TEMP / 'training_source'), 'script': identity(Path(__file__)),
            'environment': environment_identity(), 'command': COMMAND,
            'scope': 'One fixed-batch continuation trajectory. Frozen complete encoder/policy and their optimizer states. No search or matches in this entry point.'}
    write_json(path, spec)
    return spec


def phase_labels(dataset):
    path = TEMP / 'position_phases.npy'
    if path.exists():
        phases = np.load(path, allow_pickle=False)
        assert phases.shape == (dataset['positions'],)
        return phases
    phases = np.empty(dataset['positions'], dtype=np.bool_)
    own = GLOBAL_FEATURE_NAMES.index('self_score_scaled')
    other = GLOBAL_FEATURE_NAMES.index('opponent_score_scaled')
    for g in dataset['games']:
        stat = Path(g['path']).stat()
        assert (stat.st_size, stat.st_mtime_ns) == (g['archive_size'], g['archive_mtime_ns'])
        with np.load(g['path'], allow_pickle=False) as archive:
            global_features = archive['global_features']
        scores = np.rint(global_features[:, (own, other)]*SCALES.player_score).astype(np.int64)
        phases[g['offset']:g['offset']+g['decision_count']] = scores.max(axis=1)>=10
    temporary = path.with_suffix('.npy.tmp')
    with temporary.open('wb') as stream:
        np.save(stream, phases, allow_pickle=False)
    temporary.replace(path)
    write_json(ART / 'phase_labels.json', {'array': identity(path), 'early_positions': int((~phases).sum()),
                                         'late_positions': int(phases.sum()), 'threshold': 10,
                                         'rule': 'max public score>=10, before the actual root decision; all original positions retained'})
    return phases


def load_features(spec, dataset):
    parts = []
    for item in spec['features']:
        p = torch.load(item['path'], weights_only=True)
        assert p['encoder_sha256'] == spec['base']['sha256']
        assert p['dataset_sha256'] == spec['inputs'][1]['sha256']
        parts.append(p)
    result = tuple(torch.cat([p[k] for p in parts]) for k in ('x', 'y', 'w'))
    assert result[0].shape == (dataset['positions'], 128)
    return result


def save(learner, step, metrics, spec):
    path = OUT / f'checkpoints/step-{step:06d}.pt'
    saved = save_alphazero_checkpoint(path, model=learner.model, optimizer=learner.optimizer,
        learner_config=asdict(learner.config) | {'policy_loss_weight': 0.0},
        round_index=learner.round_index, round_train_step=step, total_train_step=16600+step,
        round_steps_target=500, parent_checkpoint=spec['base'], replay_identity=learner.replay_identity,
        sampling_rng_state=learner.sampling_rng.snapshot(),
        metadata={'protocol': spec['protocol'], 'extra_critic_steps': step, 'encoder_policy_frozen': True,
                  'diagnostic_only_not_formal_resume': True, 'sampling': 'External frozen batches.json ledger'})
    write_json(OUT / 'state.json', {'step': step, 'checkpoint': identity(path), 'checkpoint_identity': saved,
                                  'training_metrics': metrics})


def evaluate(learner, step, features, phases, dataset):
    path = ART / f'mse/step-{step:06d}.json'
    checkpoint = OUT / f'checkpoints/step-{step:06d}.pt'
    if path.exists():
        result = read_json(path)
        assert result['checkpoint']['sha256'] == sha256(checkpoint)
        return result
    x, y, w = features
    learner.model.eval()
    errors = []
    with torch.no_grad():
        for start in range(0, len(x), 4096):
            prediction = learner.model.critic.network(x[start:start+4096]).squeeze(-1)
            errors.append(((prediction-y[start:start+4096]).square()*w[start:start+4096]).double())
    error, weight = torch.cat(errors).numpy(), w.double().numpy()
    rows = []
    for g in dataset['games']:
        left, right = g['offset'], g['offset']+g['decision_count']
        e, v, late = error[left:right], weight[left:right], phases[left:right]
        rows.append([g['round_index'], g['game_id'], int(g['split']=='validation'), g['decision_count'],
                     float(e.sum()), float(v.sum()), int((~late).sum()), int(late.sum()),
                     float(e[~late].sum()), float(v[~late].sum()), float(e[late].sum()), float(v[late].sum())])
    result = {'step': step, 'checkpoint': identity(checkpoint),
              'columns': ['round', 'game', 'validation', 'positions', 'value_sq_sum', 'weight_sum',
                          'early_positions', 'late_positions', 'early_value_sq_sum', 'early_weight_sum',
                          'late_value_sq_sum', 'late_weight_sum'], 'per_game': rows}
    write_json(path, result)
    a = np.array(rows)
    emit(event='mse', step=step, train_mse=float(a[a[:, 2]==0, 4].sum()/a[a[:, 2]==0, 5].sum()),
         validation_mse=float(a[a[:, 2]==1, 4].sum()/a[a[:, 2]==1, 5].sum()))
    return result


def summarize(spec, dataset, checks):
    rows0 = np.array(read_json(ART / 'mse/step-000000.json')['per_game'])
    nodes = []
    for step in NODES:
        raw = read_json(ART / f'mse/step-{step:06d}.json')
        rows = np.array(raw['per_game'])
        assert np.array_equal(rows[:, :4], rows0[:, :4]) and np.array_equal(rows[:, 5:8], rows0[:, 5:8])
        assert np.allclose(rows[:, 4], rows[:, 8]+rows[:, 10], rtol=0, atol=1e-10)
        aggregates = {}
        for label, mask in (('all', np.ones(len(rows), dtype=bool)), ('recent_77_80', rows[:, 0]>=77),
                            ('history_66_76', rows[:, 0]<77), ('new_B2_80', rows[:, 0]==80)):
            aggregates[label] = {}
            for split, val in (('train', 0), ('validation', 1)):
                selected = rows[mask & (rows[:, 2]==val)]
                aggregates[label][split] = {
                    phase: {'MSE': float(selected[:, numerator].sum()/selected[:, denominator].sum()),
                            'positions': int(selected[:, positions].sum()), 'weight_sum': float(selected[:, denominator].sum())}
                    for phase, numerator, denominator, positions in (('all', 4, 5, 3), ('early', 8, 9, 6), ('late', 10, 11, 7))}
        cohorts = [{'round_index': c['round_index'], 'label': c['label'],
                    'MSE': {split: float(rows[(rows[:, 0]==c['round_index']) & (rows[:, 2]==val), 4].sum()/
                                         rows[(rows[:, 0]==c['round_index']) & (rows[:, 2]==val), 5].sum())
                            for split, val in (('train', 0), ('validation', 1))}}
                   for c in dataset['cohorts']]
        nodes.append({'step': step, 'checkpoint': raw['checkpoint'], 'aggregates': aggregates, 'cohorts': cohorts})
    baseline = read_json(ROOT / 'artifacts/entity_attention_critic_only_replay_20261008/results.json')
    for split in ('train', 'validation'):
        assert abs(nodes[0]['aggregates']['all'][split]['all']['MSE']-
                   baseline['aggregates']['all'][split]['MSE']['16600_control']) < 1e-6
    checks.update(zero_step_mse_matches_original_16600=True, evaluated_games_per_node=3000,
                  phase_partition_sums_match=True, cpu_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    result = {'protocol': spec['protocol'], 'specification': identity(ART / 'training_specification.json'),
              'nodes': nodes, 'verification': checks,
              'scope': 'Completed original-critic continuation and full frozen train/holdout MSE; search has not run.'}
    write_json(ART / 'mse.json', result)
    lines = ['# 16600原critic适配与后续搜索棋力检验', '',
             '已完成原critic额外0/50/200/500步；固定encoder和policy，保留16600模型与完整AdamW状态，'
             '沿用上次同一500批次（128近期＋128历史、lr/weight_decay1e-4、critic clip1），CPU FP32。'
             '这是同一轨迹的四个前缀，没有重新初始化critic。搜索对战尚未启动。', '',
             '| 额外步数 | 完整训练MSE | 完整留出MSE | 前中期留出MSE | 后期留出MSE |',
             '|---:|---:|---:|---:|---:|']
    for n in nodes:
        tr, val = n['aggregates']['all']['train'], n['aggregates']['all']['validation']
        lines.append(f"| {n['step']} | {tr['all']['MSE']:.6f} | {val['all']['MSE']:.6f} | {val['early']['MSE']:.6f} | {val['late']['MSE']:.6f} |")
    lines += ['', '完整15代3000局，train2701局/221955位置，留出299局/24568位置；保留原整局划分。'
              '阶段按真实根决策前双方最高公开分数是否≥10划分，阶段绝对MSE不能直接当成棋力。'
              '完整逐局分子分母在mse/，逐代与近期／历史统计在mse.json。', '',
              '核验：0步完整模型参数等于16600；0步MSE复现已有完整统计；500步encoder/policy参数及其AdamW状态不变；'
              '冻结特征缓存来自同一16600检查点，128维global token缓存已在此前实验核验与完整value前向一致。', '',
              '训练与恢复命令（只训练和评估MSE，不执行搜索）：', '', '```bash', COMMAND, '```', '',
              '诊断checkpoint：runs/entity_attention_critic_adaptation_strength_20261008/checkpoints/step-{000000,000050,000200,000500}.pt。'
              '每50步保存可恢复状态，MSE按节点原子保存；完成后resume直接返回。模型仅用于独立诊断，不是正式hybrid恢复入口。', '']
    (ART / 'README.md').write_text('\n'.join(lines), encoding='utf-8')
    emit(event='training_and_mse_complete', checkpoints=[n['checkpoint']['path'] for n in nodes], verification=checks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    TEMP.mkdir(parents=True, exist_ok=True)
    with (OUT / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (ART / 'training_specification.json').exists():
            assert args.resume
        spec = prepare()
        for item in [spec['base'], spec['script'], *spec['inputs'], *spec['features']]:
            assert sha256(Path(item['path'])) == item['sha256'], f'Input changed: {item["path"]}'
        assert source_identity(ROOT)['source_sha256'] == spec['source']['source_sha256']
        if (ART / 'mse.json').exists():
            emit(event='already_complete', mse=str(ART / 'mse.json'))
            return
        dataset = read_json(REFERENCE / 'dataset.json')
        batches = read_json(REFERENCE / 'batches.json')['batches']
        features, phases = load_features(spec, dataset), phase_labels(dataset)
        original = load_alphazero_checkpoint(BASE, device='cpu')
        state = read_json(OUT / 'state.json') if (OUT / 'state.json').exists() else None
        payload = load_alphazero_checkpoint(Path(state['checkpoint']['path']) if state else BASE, device='cpu')
        if state:
            assert sha256(Path(state['checkpoint']['path'])) == state['checkpoint']['sha256']
        config = replace(LearnerConfig(**spec['learner_config']), device='cpu')
        learner = AlphazeroLearner(payload.build_model(device='cpu'), config,
                                   sampling_rng=DeterministicRng.restore(original.sampling_rng_state))
        learner.optimizer.load_state_dict(payload.optimizer_state_dict)
        move_optimizer_state(learner.optimizer, learner.device)
        learner.round_index = original.round_index+1
        learner.replay_identity = {'digest': spec['inputs'][1]['sha256'], 'dataset_path': spec['inputs'][1]['path']}
        for name, p in learner.model.named_parameters():
            p.requires_grad_(name.startswith('critic.'))
        before = model_state_sha256({k: v for k, v in original.model_state_dict.items() if not k.startswith('critic.')})
        step, metrics = (state['step'], state['training_metrics']) if state else (0, [])
        if not state:
            assert model_state_sha256(learner.model.state_dict()) == original.model_sha256
            save(learner, step, metrics, spec)
        for node in NODES:
            if node < step:
                assert (ART / f'mse/step-{node:06d}.json').exists()
                continue
            for update in range(step+1, node+1):
                ids = torch.tensor(batches[update-1], dtype=torch.long)
                x, y, w = [array[ids] for array in features]
                prediction = learner.model.critic.network(x).squeeze(-1)
                loss = ((prediction-y).square()*w).sum()/w.sum()
                assert torch.isfinite(loss)
                learner.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = float(clip_grad_norm_(learner.model.critic.parameters(), spec['critic_clip']))
                assert np.isfinite(norm)
                assert all(p.grad is None for name, p in learner.model.named_parameters() if not name.startswith('critic.'))
                learner.optimizer.step()
                step = update
                metrics.append({'step': step, 'value_mse_on_batch_before_update': float(loss.detach()),
                                'gradient_norm': norm, 'gradient_clipped': norm>1.0})
                if step % 50 == 0:
                    assert frozen_hash(learner.model)==before
                    save(learner, step, metrics, spec)
                    emit(event='training', **metrics[-1])
            evaluate(learner, node, features, phases, dataset)
        assert frozen_hash(learner.model)==before
        actual = learner.optimizer.state_dict()['state']
        for i, (name, _) in enumerate(learner.model.named_parameters()):
            if name.startswith('critic.'):
                continue
            for key, value in original.optimizer_state_dict['state'][i].items():
                assert torch.equal(value, actual[i][key]) if isinstance(value, torch.Tensor) else value==actual[i][key]
        write_json(ART / 'training.json', {'metrics': metrics, 'steps': step, 'critic_state_retained': True,
                                         'frozen_optimizer_states_unchanged': True})
        summarize(spec, dataset, {'zero_step_model_equal_original': True, 'encoder_policy_parameters_unchanged': True,
                                  'noncritic_adamw_states_unchanged': True, 'critic_optimizer_state_retained': True})


if __name__ == '__main__':
    main()
