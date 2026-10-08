"""Matched fresh-critic fitting on frozen 16500/16600 global features."""
from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from splendor.alphazero_checkpoint import load_alphazero_checkpoint, model_state_sha256
from splendor.alphazero_recording import environment_identity, read_json, sha256, source_identity, write_json
from splendor.entity_attention import GlobalValueHead
from splendor.offline_train import make_index
from splendor.selfplay_data import collate_selfplay_samples

ROOT = Path(__file__).resolve().parents[2]
ART = Path(__file__).resolve().parent
OUT = ROOT / 'runs/entity_attention_encoder_critic_refit_20261008'
TEMP = ROOT / 'runtime/entity_attention_encoder_critic_refit_20261008'
FORMAL = ROOT / 'runs/entity_attention_iteration_trend_seed1_20261008/seed1_steps50'
INPUTS = {'16500': FORMAL / 'checkpoints/fork_initial.pt',
          '16600': FORMAL / 'rounds/round-001/checkpoints/final.pt'}
SNAPSHOT = FORMAL / 'rounds/round-001/replay_snapshot.json'
NODES = (0, 50, 200, 500)
COMMAND = ('conda run --no-capture-output -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 '
           'MKL_NUM_THREADS=1 python -u artifacts/entity_attention_encoder_critic_refit_20261008/run.py --resume')


def emit(**record):
    print(json.dumps(record, ensure_ascii=False), flush=True)


def identity(path):
    return {'path': str(path.resolve()), 'bytes': path.stat().st_size, 'sha256': sha256(path)}


def save_tensor(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def frozen_hash(model):
    return model_state_sha256({k: v for k, v in model.state_dict().items() if not k.startswith('critic.')})


def prepare():
    spec_path = ART / 'specification.json'
    if spec_path.exists():
        return read_json(spec_path)
    payloads = {name: load_alphazero_checkpoint(path, device='cpu') for name, path in INPUTS.items()}
    assert [p.total_train_step for p in payloads.values()] == [16500, 16600]
    assert payloads['16500'].model_config == payloads['16600'].model_config
    groups = [p.optimizer_state_dict['param_groups'][0] for p in payloads.values()]
    options = ('lr', 'betas', 'eps', 'weight_decay', 'amsgrad', 'maximize', 'foreach', 'capturable', 'differentiable', 'fused')
    optimizers = [{key: group[key] for key in options} for group in groups]
    assert optimizers[0] == optimizers[1]
    snapshot = read_json(SNAPSHOT)
    recent = snapshot['split_training']['recent_round_indices']
    assert recent == [77, 78, 79, 80]
    games, offset = [], 0
    for g in sorted(snapshot['games'], key=lambda g: (g['round_index'], g['game_id'])):
        stat = Path(g['path']).stat()
        games.append({**g, 'offset': offset, 'archive_size': stat.st_size, 'archive_mtime_ns': stat.st_mtime_ns})
        offset += g['decision_count']
    assert len(games) == 3000
    teachers = {t['round_index']: t for t in snapshot['split_training']['teachers']}
    cohorts = [{'round_index': r, 'label': f'Split{r-58}' if r < 79 else f'B{r-78}',
                'teacher': teachers[r]['checkpoint_identity']} for r in sorted(teachers)]
    dataset = {'snapshot': identity(SNAPSHOT), 'games': games, 'positions': offset, 'cohorts': cohorts,
               'recent_rounds': recent, 'scope': 'Original whole-game train/validation; all 15 cohorts and all positions.'}
    write_json(ART / 'dataset.json', dataset)
    pools = {bucket: np.concatenate([np.arange(g['offset'], g['offset'] + g['decision_count'], dtype=np.int64)
                                    for g in games if g['split'] == 'train' and
                                    ((g['round_index'] in recent) == (bucket == 'recent'))])
             for bucket in ('recent', 'history')}
    rng = np.random.default_rng(2026100802)
    batches = [np.concatenate([pool[rng.integers(len(pool), size=128)] for pool in pools.values()]).tolist()
               for _ in range(500)]
    write_json(ART / 'batches.json', {'seed': 2026100802, 'columns': 'Global position indices in dataset.json game-offset order',
                                   'sampling': '128 recent + 128 history train positions; uniform with replacement within each bucket',
                                   'pool_positions': {k: len(v) for k, v in pools.items()}, 'batches': batches})
    torch.manual_seed(2026100801)
    head = GlobalValueHead(payloads['16500'].model_config['hidden'])
    save_tensor(OUT / 'initial_critic.pt', head.state_dict())
    spec = {'protocol': 'two-frozen-encoders-matched-fresh-critic-v1',
            'checkpoints': {k: identity(p) for k, p in INPUTS.items()}, 'snapshot': identity(SNAPSHOT),
            'dataset': identity(ART / 'dataset.json'), 'batches': identity(ART / 'batches.json'),
            'initial_critic': identity(OUT / 'initial_critic.pt'),
            'initial_critic_parameter_sha256': model_state_sha256(head.state_dict()),
            'initialization_seed': 2026100801, 'sampling_seed': 2026100802,
            'initialization': 'Identical fresh random head in both arms; optimizer moments empty, original critic weights discarded.',
            'optimizer': {'name': 'AdamW', **optimizers[0]}, 'gradient_clip': 1.0, 'nodes': NODES, 'steps': 500,
            'batch_size': 256, 'value_loss_weight': 1.0, 'dtype': 'float32', 'device': 'cpu', 'threads': 1,
            'feature_cache': 'Only the 128-dimensional global token consumed by GlobalValueHead; no stochastic encoder layers.',
            'source': source_identity(ROOT, TEMP / 'source'), 'script': identity(Path(__file__)),
            'environment': environment_identity(), 'command': COMMAND,
            'scope': 'Same-data, same-budget fitting capacity comparison; 500 steps is not a convergence guarantee. No collection or matches.'}
    write_json(spec_path, spec)
    return spec


def build_head(spec):
    head = GlobalValueHead(128)
    head.load_state_dict(torch.load(OUT / 'initial_critic.pt', weights_only=True))
    assert model_state_sha256(head.state_dict()) == spec['initial_critic_parameter_sha256']
    optimizer = torch.optim.AdamW(head.parameters(), **{k: v for k, v in spec['optimizer'].items() if k != 'name'})
    assert len(optimizer.state) == 0
    return head, optimizer


def encode(arm, spec, dataset):
    model = load_alphazero_checkpoint(INPUTS[arm], device='cpu').build_model(device='cpu')
    model.critic.load_state_dict(torch.load(OUT / 'initial_critic.pt', weights_only=True))
    model.requires_grad_(False)
    model.eval()
    before = frozen_hash(model)
    index = make_index(read_json(SNAPSHOT), 64 * 1024**2)
    lookup = {g.key: g for g in index.games}
    records = []
    for cohort in dataset['cohorts']:
        r = cohort['round_index']
        path = TEMP / f'features/{arm}/round-{r:03d}.pt'
        if path.exists():
            saved = torch.load(path, weights_only=True)
            assert saved['encoder_sha256'] == spec['checkpoints'][arm]['sha256']
            assert saved['dataset_sha256'] == spec['dataset']['sha256']
        else:
            xs, ys, ws = [], [], []
            parity = None
            for descriptor in dataset['games']:
                if descriptor['round_index'] != r:
                    continue
                game = lookup[(r, descriptor['game_id'])]
                stat = game.path.stat()
                assert (stat.st_size, stat.st_mtime_ns) == (descriptor['archive_size'], descriptor['archive_mtime_ns'])
                for start in range(0, game.decision_count, 256):
                    batch = collate_selfplay_samples([index.cache.sample(game, i) for i in
                                                     range(start, min(start+256, game.decision_count))])
                    with torch.no_grad():
                        state = model.encode_state(model._prepare(batch.decisions))
                        x = state[0][:, 0].contiguous().clone()
                        if parity is None:
                            parity = float((model.critic.network(x).squeeze(-1)-model.value_only(batch.decisions)).abs().max())
                            assert parity == 0.0
                    xs.append(x)
                    ys.append(torch.from_numpy(batch.value_targets.copy()))
                    ws.append(torch.from_numpy(batch.value_weights.copy()))
            saved = {'encoder_sha256': spec['checkpoints'][arm]['sha256'], 'dataset_sha256': spec['dataset']['sha256'],
                     'x': torch.cat(xs), 'y': torch.cat(ys), 'w': torch.cat(ws),
                     'cached_head_matches_full_value_exactly': True, 'parity_max_abs_difference': parity}
            assert frozen_hash(model) == before
            save_tensor(path, saved)
        expected = sum(g['decision_count'] for g in dataset['games'] if g['round_index'] == r)
        assert saved['x'].shape == (expected, 128) and saved['y'].shape == saved['w'].shape == (expected,)
        assert torch.isfinite(saved['x']).all() and torch.isfinite(saved['y']).all()
        assert torch.all(saved['w'] >= 0)
        records.append(identity(path))
        emit(event='feature_cache', encoder=arm, cohort=cohort['label'], positions=expected)
        del saved
        gc.collect()
    assert frozen_hash(model) == before
    write_json(ART / f'features_{arm}.json', {'encoder_noncritic_parameter_sha256': before,
              'encoder_and_policy_unchanged': True, 'cached_head_matches_full_value_exactly': True,
              'files': records, 'decode_cache': index.cache.stats()})
    del model, index, lookup
    gc.collect()


def load_features(arm, dataset):
    parts = [torch.load(TEMP / f'features/{arm}/round-{c["round_index"]:03d}.pt', weights_only=True)
             for c in dataset['cohorts']]
    arrays = tuple(torch.cat([p[k] for p in parts]) for k in ('x', 'y', 'w'))
    assert len(arrays[0]) == dataset['positions']
    return arrays


def evaluate(arm, step, head, features, dataset):
    path = ART / f'evaluations/encoder-{arm}-step-{step:06d}.json'
    checkpoint = OUT / arm / f'checkpoints/step-{step:06d}.pt'
    if path.exists():
        saved = read_json(path)
        assert saved['checkpoint']['sha256'] == sha256(checkpoint)
        return saved
    x, y, w = features
    head.eval()
    sums = []
    with torch.no_grad():
        for start in range(0, len(x), 4096):
            prediction = head.network(x[start:start+4096]).squeeze(-1)
            sums.append(((prediction-y[start:start+4096]).square()*w[start:start+4096]).double())
    errors = torch.cat(sums).numpy()
    weights = w.double().numpy()
    rows = []
    for g in dataset['games']:
        left, right = g['offset'], g['offset']+g['decision_count']
        rows.append([g['round_index'], g['game_id'], int(g['split']=='validation'), g['decision_count'],
                     float(errors[left:right].sum()), float(weights[left:right].sum())])
    saved = {'encoder': arm, 'step': step, 'checkpoint': identity(checkpoint),
             'columns': ['round_index', 'game_id', 'validation', 'positions', 'value_sq_sum', 'value_weight_sum'],
             'per_game': rows}
    write_json(path, saved)
    emit(event='evaluation', encoder=arm, step=step,
         train_mse=ratio(np.array(rows)[np.array(rows)[:, 2]==0]),
         validation_mse=ratio(np.array(rows)[np.array(rows)[:, 2]==1]))
    return saved


def ratio(rows):
    return float(rows[:, 4].sum()/rows[:, 5].sum())


def train(arm, spec, dataset, batches):
    features = load_features(arm, dataset)
    head, optimizer = build_head(spec)
    state_path = OUT / arm / 'state.json'
    if state_path.exists():
        state = read_json(state_path)
        checkpoint = torch.load(state['checkpoint']['path'], weights_only=True)
        assert sha256(Path(state['checkpoint']['path'])) == state['checkpoint']['sha256']
        assert checkpoint['specification_sha256'] == sha256(ART / 'specification.json')
        head.load_state_dict(checkpoint['critic_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        step, metrics = checkpoint['step'], checkpoint['metrics']
    else:
        step, metrics = 0, []
    def save():
        path = OUT / arm / f'checkpoints/step-{step:06d}.pt'
        save_tensor(path, {'protocol': spec['protocol'], 'encoder': spec['checkpoints'][arm],
                          'specification_sha256': sha256(ART / 'specification.json'), 'step': step,
                          'critic_state_dict': head.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                          'metrics': metrics, 'diagnostic_only': True})
        write_json(state_path, {'step': step, 'checkpoint': identity(path)})
    if not state_path.exists():
        save()
    for node in NODES:
        if node < step:
            assert (ART / f'evaluations/encoder-{arm}-step-{node:06d}.json').exists()
            continue
        for update in range(step+1, node+1):
            ids = torch.tensor(batches[update-1], dtype=torch.long)
            x, y, w = [array[ids] for array in features]
            head.train()
            prediction = head.network(x).squeeze(-1)
            loss = ((prediction-y).square()*w).sum()/w.sum()
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = float(clip_grad_norm_(head.parameters(), spec['gradient_clip']))
            assert np.isfinite(norm)
            optimizer.step()
            step = update
            metrics.append({'step': step, 'value_mse_on_batch_before_update': float(loss.detach()),
                            'gradient_norm': norm, 'gradient_clipped': norm > spec['gradient_clip'],
                            'batch_sha256': hashlib.sha256(ids.numpy().astype('<i8').tobytes()).hexdigest()})
            if step % 50 == 0:
                save()
                emit(event='training', encoder=arm, **metrics[-1])
        evaluate(arm, node, head, features, dataset)
    write_json(ART / f'training_{arm}.json', {'encoder': spec['checkpoints'][arm], 'steps': step, 'metrics': metrics,
              'initial_critic_parameter_sha256': spec['initial_critic_parameter_sha256'], 'optimizer_initial_state_empty': True,
              'final_critic_parameter_sha256': model_state_sha256(head.state_dict()),
              'features': identity(ART / f'features_{arm}.json')})
    del features, head, optimizer
    gc.collect()


def paired_interval(a, b, seed):
    rng = np.random.default_rng(seed)
    strata = [np.flatnonzero(a[:, 0]==r) for r in np.unique(a[:, 0])]
    difference, weight = b[:, 4]-a[:, 4], a[:, 5]
    values = []
    for start in range(0, 10000, 256):
        size = min(256, 10000-start)
        numerators, denominators = np.zeros(size), np.zeros(size)
        for group in strata:
            ids = group[rng.integers(len(group), size=(size, len(group)))]
            numerators += difference[ids].sum(axis=1)
            denominators += weight[ids].sum(axis=1)
        values.extend((numerators/denominators).tolist())
    return {'difference_16600_minus_16500': ratio(b)-ratio(a), 'lower': float(np.quantile(values, .025)),
            'upper': float(np.quantile(values, .975)), 'paired_whole_games': True,
            'stratified_by_teacher': True, 'replicates': 10000, 'seed': seed}


def summarize(spec, dataset):
    nodes = []
    evaluations = []
    for node in NODES:
        paths = [ART / f'evaluations/encoder-{arm}-step-{node:06d}.json' for arm in INPUTS]
        a, b = [np.array(read_json(path)['per_game'], dtype=np.float64) for path in paths]
        assert np.array_equal(a[:, :4], b[:, :4]) and np.array_equal(a[:, 5], b[:, 5])
        groups = {}
        conditions = {'all': np.ones(len(a), dtype=bool), 'recent_77_80': a[:, 0]>=77,
                      'history_66_76': a[:, 0]<77, 'new_B2_80': a[:, 0]==80}
        for label, condition in conditions.items():
            groups[label] = {}
            for split, val in (('train', 0), ('validation', 1)):
                mask = condition & (a[:, 2]==val)
                x, y = a[mask], b[mask]
                groups[label][split] = {'games': len(x), 'positions': int(x[:, 3].sum()),
                                       'MSE': {'16500': ratio(x), '16600': ratio(y)}}
                if label == 'all':
                    groups[label][split]['paired_difference_95ci'] = paired_interval(x, y, 2026100803+node+1000*val)
        cohorts = [{'round_index': c['round_index'], 'label': c['label'],
                    'MSE': {split: {arm: ratio(rows[(rows[:, 0]==c['round_index']) & (rows[:, 2]==val)])
                                    for arm, rows in (('16500', a), ('16600', b))}
                            for split, val in (('train', 0), ('validation', 1))}}
                   for c in dataset['cohorts']]
        nodes.append({'step': node, 'aggregates': groups, 'cohorts': cohorts})
        evaluations.extend(identity(path) for path in paths)
    training = [read_json(ART / f'training_{arm}.json') for arm in INPUTS]
    assert [m['batch_sha256'] for m in training[0]['metrics']] == [m['batch_sha256'] for m in training[1]['metrics']]
    result = {'protocol': spec['protocol'], 'specification': identity(ART / 'specification.json'),
              'nodes': nodes, 'evaluations': evaluations,
              'verification': {'identical_initial_critic': True, 'fresh_optimizers': True,
                               'same_500_batches_in_order': True, 'encoders_and_policies_frozen': True,
                               'cache_equals_full_value_forward': True, 'full_original_splits': True,
                               'evaluated_games_per_node': len(dataset['games']),
                               'cpu_peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}}
    write_json(ART / 'results.json', result)
    lines = ['# 冻结16500与16600 encoder，重新拟合相同critic', '',
             '两边丢弃原critic权重及优化器动量，使用完全相同的新随机critic（seed2026100801）。'
             '各自冻结全部encoder/policy，在同一B2活动窗口15代3000局、原整局划分上训练。'
             'AdamW lr/weight_decay均1e-4、betas=(0.9,0.999)、eps=1e-8、clip1；'
             '每步128近期＋128历史train位置、同一500步账本，CPU FP32。', '',
             '| 步数 | 16500训练MSE | 16600训练MSE | 16500留出MSE | 16600留出MSE | 留出Δ(16600−16500)及95%区间 |',
             '|---:|---:|---:|---:|---:|---:|']
    for item in nodes:
        train = item['aggregates']['all']['train']['MSE']
        val = item['aggregates']['all']['validation']['MSE']
        ci = item['aggregates']['all']['validation']['paired_difference_95ci']
        lines.append(f"| {item['step']} | {train['16500']:.6f} | {train['16600']:.6f} | {val['16500']:.6f} | {val['16600']:.6f} | {ci['difference_16600_minus_16500']:+.6f} [{ci['lower']:+.6f},{ci['upper']:+.6f}] |")
    lines += ['', '训练2701局／221955位置，留出299局／24568位置；近期4代留出79局，最新B2留出12局。'
              '逐代及近期／历史的各节点MSE见results.json，逐局加权平方误差与分母保存在evaluations/。'
              '区间为10000次整局配对bootstrap并按教师分层，只衡量固定数据上的两个配对拟合轨迹；'
              '未做多初始化重复或棋力评测。500步不是收敛保证，不能直接宣称绝对拟合上限。', '',
              '冻结global token只编码一次，GlobalValueHead仅使用该128维token；每代首批缓存路径与完整value前向完全一致，'
              'encoder/policy参数哈希保持不变。两边500个训练batch哈希逐项一致；AdamW状态从空开始。', '',
              '## 启动与恢复', '', '从仓库根目录执行同一命令，首次准备并运行，中断后接续：', '', '```bash', COMMAND, '```', '',
              '每50步独立保存critic与AdamW状态，报告节点为0/50/200/500；encoder特征按代原子保存并复用。'
              'runs/entity_attention_encoder_critic_refit_20261008/保存初始化和诊断checkpoint；'
              'runtime/entity_attention_encoder_critic_refit_20261008/保存特征缓存与源码快照。正式训练模型没有替换。', '']
    (ART / 'README.md').write_text('\n'.join(lines), encoding='utf-8')
    emit(event='complete', results=str(ART / 'results.json'), verification=result['verification'])


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
        if (ART / 'specification.json').exists():
            assert args.resume, 'Existing experiment requires --resume'
        spec = prepare()
        for record in [*spec['checkpoints'].values(), spec['snapshot'], spec['dataset'], spec['batches'], spec['script'], spec['initial_critic']]:
            assert sha256(Path(record['path'])) == record['sha256'], f'Input changed: {record["path"]}'
        assert source_identity(ROOT)['source_sha256'] == spec['source']['source_sha256']
        if (ART / 'results.json').exists():
            emit(event='already_complete', results=str(ART / 'results.json'))
            return
        dataset = read_json(ART / 'dataset.json')
        batches = read_json(ART / 'batches.json')['batches']
        for arm in INPUTS:
            encode(arm, spec, dataset)
            train(arm, spec, dataset, batches)
        summarize(spec, dataset)


if __name__ == '__main__':
    main()
