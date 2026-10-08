"""Finite, frozen-teacher collection and split-clipped student diagnostic."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from splendor.alphazero_checkpoint import load_alphazero_checkpoint, move_optimizer_state, save_alphazero_checkpoint
from splendor.alphazero_recording import append_json, environment_identity, read_json, sha256, source_identity, write_json
from splendor.alphazero_train import AlphazeroLearner, LearnerConfig
from splendor.compact_trace import decode_request
from splendor.evaluation import FixedEvaluationConfig, run_fixed_evaluation
from splendor.offline_train import make_index
from splendor.rng import DeterministicRng
from splendor.search_inference import FrozenSearchInference, InferenceConfig
from splendor.selfplay_data import collate_selfplay_samples, game_path, load_game
from splendor.selfplay_fast import FastSelfPlayConfig, generate, validate_run

ROOT = Path(__file__).resolve().parents[2]
ART = Path(__file__).resolve().parent
OUT = ROOT / 'runs/entity_attention_chance_training_probe_20261007'
RUNTIME = ROOT / 'runtime/entity_attention_chance_training_probe_20261007'
INITIAL = ROOT / 'runs/entity_attention_hybrid_split_p4_v15_from_round50_seed1/rounds/round-019/checkpoints/final.pt'
INITIAL_SHA = 'dbfbe066144a485d15ec633aa87c85ac27864f142183725b86562e4ba44ed2c4'
ARMS = ('native', 'chance', 'cpw')
NODES = (0, 100, 300, 500)
NOISE_DIRS = {'native': 'entity_attention_target_noise_20261007',
              'chance': 'entity_attention_chance_search_20261007', 'cpw': 'entity_attention_chance_pw_noise_20261007'}

def emit(**kwargs):
    print(json.dumps(kwargs, ensure_ascii=False), flush=True)

def record(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha256(path), 'bytes': path.stat().st_size}

def settings():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

def check_source():
    spec = read_json(ART / 'specification.json')
    assert sha256(INITIAL) == INITIAL_SHA
    for item in spec['inputs']:
        assert sha256(Path(item['path'])) == item['sha256'], item['path']
    source = spec['source']
    for path, expected in source['source_files'].items():
        assert sha256(ROOT / path) == expected, path
    for path, expected in source['native_extensions'].items():
        assert sha256(Path(path)) == expected, path
    return spec

def checkpoint(arm, step):
    return INITIAL if step == 0 else OUT / arm / f'checkpoints/step-{step:06d}.pt'

def eval_config(arm, step, mode, *, pairs=50, confirm=False, fixed_value=False):
    inference = {'mode': 'eager', 'actor_projection': False, 'candidate_buckets': [16, 32, 48, 64]}
    search = {'simulations': 1024, 'determinizations': 1, 'seed': 16850000,
              'root_noise_fraction': 0.0, 'dirichlet_alpha': .3, 'c_puct': 1.5,
              'fold_forced_phases': True, 'max_simulation_transitions': 512}
    a = {'kind': 'greedy_checkpoint', 'name': f'{arm}_{step}_{mode}', 'checkpoint': str(checkpoint(arm, step))}
    b = {'kind': 'greedy_checkpoint', 'name': 'frozen_16500_greedy', 'checkpoint': str(INITIAL)}
    if mode != 'greedy':
        a.update(kind='mcts_checkpoint', search_backend='native_chance' if mode == 'cpw' else 'native',
                 search_config=search, inference=inference)
        if mode == 'cpw':
            a['chance_widening'] = {'k': .25, 'alpha': .5}
        b.update(kind='mcts_checkpoint', name='frozen_16500_native', search_backend='native',
                 search_config=search.copy(), inference=inference.copy())
    raw = {'policy_a': a, 'policy_b': b, 'pairs': pairs,
           'seed_start': 16870000 if confirm else 16830000,
           'policy_seed_base': 16880000 if confirm else 16840000,
           'bootstrap_seed': 16890000 if confirm else 16860000, 'bootstrap_resamples': 10000,
           'device': 'cuda', 'workers': 1, 'max_completed_turns': 400,
           'adjudicate_deterministic_policy_cycles': True,
           'execution': {'backend': 'serial' if mode == 'greedy' or fixed_value else 'native_batched',
                         'active_games': pairs * 2, 'max_batch_size': 200,
                         'candidate_buckets': [16, 32, 48, 64], 'actor_projection': False,
                         'inference_mode': 'eager', 'max_batch_wait_ms': 0},
           'progress_every': 5, 'summary_every': 5,
           'metadata': {'protocol': 'chance-training-probe-v1', 'arm': arm, 'student_step': step,
                        'evaluation_mode': mode, 'confirm': confirm, 'fixed_value': fixed_value}}
    if mode == 'greedy' or fixed_value:
        raw.pop('execution')  # Default execution is the supported serial evaluator.
    return raw

def prepare():
    assert not (ART / 'specification.json').exists()
    assert sha256(INITIAL) == INITIAL_SHA
    base = read_json(ROOT / 'configs/entity_attention_v1/hybrid_split_policy_value.json')['selfplay']
    inputs = [record(INITIAL), record(Path(__file__)), record(ART / 'run_suite.py'),
              record(ROOT / 'configs/entity_attention_v1/hybrid_split_policy_value.json')]
    for arm in ARMS:
        config = json.loads(json.dumps(base))
        config.update(run_id=f'chance-training-probe-{arm}', round_index=0,
                      checkpoint=str(INITIAL), num_games=200, seed=16800000, split_seed=16810000,
                      record_search_diagnostics=True)
        config['search_config']['seed'] = 16820000
        config['execution']['total_active_games'] = 200
        if arm != 'native':
            config['search_backend'] = 'native_chance'
        if arm == 'cpw':
            config['chance_widening'] = {'k': .25, 'alpha': .5}
        parsed = FastSelfPlayConfig.from_dict(config)
        write_json(ART / f'configs/collect_{arm}.json', parsed.to_dict())
        inputs.append(record(ART / f'configs/collect_{arm}.json'))
        smoke = parsed.to_dict()
        smoke.update(num_games=2, run_id=f'chance-training-smoke-{arm}', seed=16900000, validation_fraction=0)
        smoke['search_config']['simulations'] = 16
        smoke['execution'].update(search_workers=2, total_active_games=2, max_batch_size=4,
                                   local_microbatch=2, batch_buckets=[2, 4])
        write_json(ART / f'configs/smoke_{arm}.json', smoke)
        inputs.append(record(ART / f'configs/smoke_{arm}.json'))
    for name in ('root_bank.json', 'per_root.json', 'raw_targets.npz'):
        inputs.append(record(ROOT / 'artifacts/entity_attention_target_noise_20261007' / name))
    for arm in ('chance', 'cpw'):
        inputs.append(record(ROOT / 'artifacts' / NOISE_DIRS[arm] / 'per_root.json'))
    bank = read_json(ROOT / 'artifacts/entity_attention_target_noise_20261007/root_bank.json')
    inputs.extend(bank['inputs'])
    panels = []
    for arm in ARMS:
        for step in (100, 500):
            for mode in ('greedy', 'native'):
                name = f'{arm}_{step}_{mode}'
                write_json(ART / f'configs/{name}.json', eval_config(arm, step, mode))
                inputs.append(record(ART / f'configs/{name}.json'))
                panels.append({'id': name, 'arm': arm, 'step': step, 'mode': mode, 'pairs': 50})
        name = f'{arm}_500_cpw'
        write_json(ART / f'configs/{name}.json', eval_config(arm, 500, 'cpw'))
        inputs.append(record(ART / f'configs/{name}.json'))
        panels.append({'id': name, 'arm': arm, 'step': 500, 'mode': 'cpw', 'pairs': 50})
    write_json(ART / 'configs/initial_native.json', eval_config('native', 0, 'native'))
    inputs.append(record(ART / 'configs/initial_native.json'))
    panels.insert(0, {'id': 'initial_native', 'arm': 'native', 'step': 0, 'mode': 'native', 'pairs': 50})
    source = source_identity(ROOT, RUNTIME / 'source')
    extras = [ROOT / 'src/splendor/native_search.cpp', ROOT / 'setup.py',
              ROOT / 'tests/test_multiworker_gpu.py', ROOT / 'tests/test_native_chance_search.py',
              ROOT / 'tests/test_hybrid_training.py', *[p for p in (ROOT / 'data').glob('**/*') if p.is_file()]]
    source['additional_files'] = {}
    for path in extras:
        target = RUNTIME / 'source' / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        source['additional_files'][str(path.relative_to(ROOT))] = sha256(path)
    for path in ROOT.glob('src/splendor/_native_search*.so'):
        shutil.copyfile(path, RUNTIME / 'source' / path.relative_to(ROOT))
    spec = {'protocol': 'chance-training-probe-v1', 'status': 'prepared', 'client_date': '2026-10-07',
            'initial': record(INITIAL), 'arms': list(ARMS), 'observations': list(NODES),
            'source': source, 'inputs': inputs, 'panels': panels,
            'training': {'steps': 500, 'policy_batch': 256, 'value_batch': 256,
                         'policy_max_grad_norm': 1., 'value_max_grad_norm': 1.,
                         'value_trunk_detached': True, 'data': '100% each arm new train split, no history',
                         'optimizer': 'Restore full original AdamW state and both RNG states at each arm start'},
            'scope': 'One frozen16500-step teacher, one200-game collection per algorithm, fixed labels and finite split training; no self-iteration or causal isolation of label noise from trajectories.',
            'commands': {}, 'outputs': []}
    prefix = 'conda run --no-capture-output -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u '
    spec['commands'] = {'prepare': prefix + str(Path(__file__).relative_to(ROOT)) + ' prepare',
                        'run': prefix + str((ART / 'run_suite.py').relative_to(ROOT)),
                        'analyze': prefix + str(Path(__file__).relative_to(ROOT)) + ' analyze'}
    write_json(ART / 'specification.json', spec)
    emit(event='prepared', arms=list(ARMS), collection_games=600, training_updates=1500, evaluation_panels=len(panels))

def collect(arm, smoke=False):
    check_source()
    name = f'smoke_{arm}' if smoke else f'collect_{arm}'
    config = FastSelfPlayConfig.from_dict(read_json(ART / f'configs/{name}.json'))
    root = OUT / (f'smoke-{arm}' if smoke else f'{arm}/selfplay')
    started = time.perf_counter()
    summary = generate(config, root, resume=(root / 'manifest.json').exists(),
                       progress_callback=lambda done,total: emit(event='collection_progress', arm=arm, completed_games=done, total_games=total), progress_interval_seconds=5)
    checked = validate_run(root, replay=True)
    assert summary['completed_games'] == config.num_games and not checked['missing_games']
    result = {'arm': arm, 'smoke': smoke, 'wall_seconds_including_validation': time.perf_counter()-started,
              'summary': summary, 'checked_validation': checked, 'environment': environment_identity(),
              'config': record(ART / f'configs/{name}.json'), 'manifest': record(root / 'manifest.json')}
    write_json(ART / f'{name}_result.json', result)
    emit(event='collection_verified', arm=arm, smoke=smoke, games=config.num_games,
         seconds=result['wall_seconds_including_validation'])

def dataset(arm):
    root = OUT / arm / 'selfplay'
    games = []
    for path in sorted((root / 'games').glob('game-*.npz')):
        _, meta = load_game(path)
        games.append({k: meta[k] for k in ('round_index','run_id','game_id','split','decision_count','policy_samples')}
                     | record(path))
    assert len(games) == 200
    return {'games': games, 'collection_manifest': record(root / 'manifest.json')}

def restore():
    payload = load_alphazero_checkpoint(INITIAL, device='cuda')
    learner = AlphazeroLearner(payload.build_model(device='cuda'), LearnerConfig(**payload.learner_config),
                               sampling_rng=DeterministicRng.restore(payload.sampling_rng_state))
    learner.optimizer.load_state_dict(payload.optimizer_state_dict)
    move_optimizer_state(learner.optimizer, learner.device)
    learner.round_index = payload.round_index
    learner.total_train_step = payload.total_train_step
    learner.round_train_step = 0
    learner.round_steps_target = 500
    torch.set_rng_state(payload.torch_cpu_rng_state.cpu())
    torch.cuda.set_rng_state_all([s.cpu() for s in payload.torch_cuda_rng_states])
    return learner, payload

def common_inputs():
    bank = read_json(ROOT / 'artifacts/entity_attention_target_noise_20261007/root_bank.json')
    labels = {}
    wanted = {r['game_key']: [] for r in bank['positions']}
    for row in bank['positions']:
        wanted[row['game_key']].append(row)
    for item in bank['inputs']:
        arrays, meta = load_game(Path(item['path']))
        for row in wanted.get(f"{meta['run_id']}/{meta['game_id']}", []):
            labels[row['id']] = int(arrays['value_target'][row['decision_index']])
    targets = {}
    for arm, folder in NOISE_DIRS.items():
        rows = read_json(ROOT / 'artifacts' / folder / 'per_root.json')
        targets[arm] = {r['root_id']: np.asarray(r['mean_target'], dtype=np.float64) for r in rows if r['arm']=='D'}
    return bank['positions'], labels, targets

def common_metrics(model, data):
    rows, labels, targets = data
    evaluator = FrozenSearchInference(model.eval(), device='cuda', config=InferenceConfig(mode='eager', actor_projection=False))
    results = evaluator.policy_value([decode_request(r['request']) for r in rows])
    metrics = []
    for row, (logits, value) in zip(rows, results):
        logit = np.asarray(logits, dtype=np.float64)
        logp = logit - np.logaddexp.reduce(logit)
        kls = {}
        for arm in ARMS:
            pi = targets[arm][row['id']]
            assert len(pi) == len(logit)
            nz = pi > 0
            kls[arm] = float(np.sum(pi[nz]*(np.log(pi[nz])-logp[nz])))
        metrics.append({'root_id': row['id'], 'game_key': row['game_key'], 'stage': row['stage'],
                        'value': value, 'value_target': labels[row['id']],
                        'value_sq_error': (value-labels[row['id']])**2,
                        'network_chosen_index': int(logit.argmax()), 'policy_kl_to_frozen_mean_targets': kls})
    summary = {}
    for stage in ('all','early','middle','late'):
        subset = metrics if stage == 'all' else [r for r in metrics if r['stage']==stage]
        summary[stage] = {'roots': len(subset), 'value_mse': float(np.mean([r['value_sq_error'] for r in subset])),
                          'value_sign_accuracy': float(np.mean([r['value']*r['value_target'] > 0 for r in subset])),
                          'policy_kl_to_frozen_mean_targets': {a: float(np.mean([r['policy_kl_to_frozen_mean_targets'][a] for r in subset])) for a in ARMS}}
    return {'summary': summary, 'per_root': metrics}

def train(arm):
    spec = check_source()
    assert not (ART / f'train_{arm}.json').exists()
    data = dataset(arm)
    write_json(ART / f'dataset_{arm}.json', data)
    index = make_index(data, 1024**3)
    pool = index.position_pool(split='train', round_index=0)
    samples = {split: [index.cache.sample(g,i) for g in index.split_games(split) for i in range(g.decision_count)]
               for split in ('train','validation')}
    assert samples['validation']
    learner, payload = restore()
    initial_sampling = asdict(learner.sampling_rng.snapshot())
    torch_rng_sha = hashlib.sha256(torch.get_rng_state().cpu().numpy().tobytes()).hexdigest()
    common = common_inputs()
    curve, ledger = [], np.empty((500,2,256,2), dtype=np.int32)
    started = time.perf_counter()
    optimizer_seconds = diagnostics_seconds = 0.
    output = OUT / arm
    for step in range(501):
        if step in NODES:
            tick = time.perf_counter()
            observation = {'arm': arm, 'step': step, 'total_train_step': learner.total_train_step,
                           'elapsed_seconds': time.perf_counter()-started,
                           **{split: learner.evaluate_samples(rows) for split, rows in samples.items()},
                           'common': common_metrics(learner.model, common)}
            for split in ('train','validation'):
                games = index.split_games(split)
                observation[split]['games'] = len(games)
            if step:
                path = checkpoint(arm, step)
                save_alphazero_checkpoint(path, model=learner.model, optimizer=learner.optimizer,
                    learner_config=asdict(learner.config), round_index=learner.round_index,
                    round_train_step=step, total_train_step=learner.total_train_step, round_steps_target=500,
                    parent_checkpoint=record(INITIAL), replay_identity={'dataset_sha256':sha256(ART / f'dataset_{arm}.json')},
                    sampling_rng_state=learner.sampling_rng.snapshot(),
                    metadata={'diagnostic': 'chance-training-probe-v1', 'teacher_algorithm':arm,
                              'value_trunk_detached':True, 'separate_gradient_clipping':True})
                observation['checkpoint'] = record(path)
            curve.append(observation)
            append_json(ART / f'curve_{arm}.jsonl', observation)
            diagnostics_seconds += time.perf_counter()-tick
            emit(event='fit_observation', arm=arm, step=step,
                 train_kl=observation['train']['policy_kl'], validation_kl=observation['validation']['policy_kl'],
                 validation_value_mse=observation['validation']['value_loss'])
        if step == 500:
            break
        tick = time.perf_counter()
        draws = [[pool.sample(learner.sampling_rng) for _ in range(256)] for _ in range(2)]
        policy_batch, value_batch = [collate_selfplay_samples([x[0] for x in batch]) for batch in draws]
        references = [x[1] for batch in draws for x in batch]
        for head in range(2):
            ledger[step,head] = [(x[1].game_id,x[1].decision_index) for x in draws[head]]
        row = learner.train_split_step(policy_batch,value_batch,references=references,
                                       policy_max_grad_norm=1., value_max_grad_norm=1.)
        append_json(ART / f'metrics_{arm}.jsonl', row)
        optimizer_seconds += time.perf_counter()-tick
    np.save(output / 'batch_references.npy', ledger)
    result = {'arm':arm, 'steps':500, 'curve':curve, 'initial_sampling_rng':initial_sampling,
              'initial_torch_cpu_rng_sha256':torch_rng_sha, 'final_sampling_rng':asdict(learner.sampling_rng.snapshot()),
              'batch_ledger':record(output / 'batch_references.npy'), 'dataset':record(ART / f'dataset_{arm}.json'),
              'teacher':record(INITIAL), 'optimizer_start_identity':'restored full checkpoint',
              'optimizer_seconds':optimizer_seconds, 'diagnostics_seconds':diagnostics_seconds,
              'wall_seconds':time.perf_counter()-started, 'environment':environment_identity(),
              'policy_draws_per_train_position':128000/len(pool), 'value_draws_per_train_position':128000/len(pool),
              'input_hashes_verified':True, 'checkpoints':[x['checkpoint'] for x in curve if x['step']]}
    write_json(ART / f'train_{arm}.json', result)
    emit(event='training_done', arm=arm, seconds=result['wall_seconds'])

def evaluate(case):
    check_source()
    config_path = ART / f'configs/{case}.json'
    raw = read_json(config_path)
    config = FixedEvaluationConfig.from_dict(raw)
    output = ART / f'evaluations/{case}.json'
    emit(event='evaluation_start',case=case,pairs=len(config.game_seeds))
    result = run_fixed_evaluation(config,output, resume=output.exists(), require_scored_pairs=True,
        progress_callback=lambda done,total: emit(event='evaluation_progress', case=case, completed_pairs=done, pairs=total))
    assert result.summary['passed']
    write_json(ART / f'evaluations/{case}_recording.json', {
        'config':record(config_path), 'output':record(output),
        'checkpoint_a':record(Path(config.policy_a.checkpoint)), 'checkpoint_b':record(Path(config.policy_b.checkpoint)),
        'environment':environment_identity()})
    emit(event='evaluation_done',case=case,score=result.summary['win_rate_score'])

def bootstrap_difference(a,b):
    def scores(result):
        payoff = {'a_win':1., 'a_loss':0., 'draw':.5}
        return np.array([np.mean([payoff[g['outcome_for_a']] for g in p['games']]) for p in result['pairs']])
    assert [p['game_seed'] for p in a['pairs']]==[p['game_seed'] for p in b['pairs']]
    delta = scores(a)-scores(b)
    rng = np.random.default_rng(16860000)
    values = delta[rng.integers(0,len(delta),size=(10000,len(delta)))].mean(axis=1)
    return {'difference':float(delta.mean()), 'lower':float(np.quantile(values,.025)),
            'upper':float(np.quantile(values,.975)), 'pairs':len(delta)}

def analyze():
    spec = check_source()
    panel_results = {}
    for panel in spec['panels']:
        name = panel['id']
        path = ART / f'evaluations/{name}.json'
        if path.exists():
            result = read_json(path)
            assert result['passed'] and result['valid_games']==panel['pairs']*2
            panel_results[name] = result
    differences = {}
    for mode in ('greedy','native','cpw'):
        for step in ((100,500) if mode != 'cpw' else (500,)):
            baseline=f'native_{step}_{mode}'
            for arm in ('chance','cpw'):
                name=f'{arm}_{step}_{mode}'
                if name in panel_results and baseline in panel_results:
                    differences[f'{name}_minus_{baseline}']=bootstrap_difference(panel_results[name],panel_results[baseline])
    collection = {}
    for arm in ARMS:
        result = read_json(ART / f'collect_{arm}_result.json')
        roots = []
        phase_counts=Counter()
        for item in read_json(ART / f'dataset_{arm}.json')['games']:
            assert sha256(Path(item['path']))==item['sha256']
            arrays, meta = load_game(Path(item['path']))
            assert meta['checkpoint_sha256']==INITIAL_SHA
            assert len(meta['search_diagnostics'])==meta['decision_count']
            for row in meta['search_diagnostics']:
                i=row['decision_index'];assert row['stats']['simulations']==int(arrays['simulations'][i])
                phase_counts[str(row['phase'])]+=1
                if row['stats']['simulations']:
                    assert row['stats']['simulations']==1024 and row['stats']['transition_cap_hits']==0
                    roots.append(row)
        strata={}
        for label,lo,hi in (('all',0,100),('score_0_4',0,5),('score_5_9',5,10),('score_10_plus',10,100)):
            group=[r for r in roots if lo <= r['max_visible_score'] < hi]
            sims=sum(r['stats']['simulations'] for r in group)
            strata[label]={'roots':len(group),'simulations':sims,
                'mean_depth':sum(r['stats']['tree_depth_sum'] for r in group)/sims if sims else None,
                'terminal_fraction':sum(r['stats']['terminal_hits'] for r in group)/sims if sims else None}
            if arm!='native':
                strata[label].update({k:sum(r['chance'][k] for r in group) for k in
                    (('chance_draws','root_hidden_samples','information_set_hits') + (('chance_pw_fresh_samples','chance_pw_reuses') if arm=='cpw' else ()))})
        collection[arm]={'wall_seconds_including_validation':result['wall_seconds_including_validation'],
                         'summary':result['summary'],'search_strata':strata,'phase_decisions':dict(phase_counts)}
    training={a:read_json(ART / f'train_{a}.json') for a in ARMS}
    assert len({json.dumps(x['initial_sampling_rng'],sort_keys=True) for x in training.values()})==1
    assert len({x['initial_torch_cpu_rng_sha256'] for x in training.values()})==1
    for arm in ARMS:
        ledger=np.load(training[arm]['batch_ledger']['path']); assert ledger.shape==(500,2,256,2)
        games={g['game_id']:g for g in read_json(ART / f'dataset_{arm}.json')['games']}
        assert all(games[int(gid)]['split']=='train' for gid in np.unique(ledger[...,0]))
        for cp in training[arm]['checkpoints']:assert sha256(Path(cp['path']))==cp['sha256']
        metrics=[json.loads(s) for s in (ART / f'metrics_{arm}.jsonl').read_text().splitlines()]
        assert len(metrics)==500 and all(m['training_mode']=='split_policy_detached_value' and m['batch_size']==512 for m in metrics)
        training[arm]['gradient_summary']={
            'policy_clipped_steps':sum(m['policy_grad_clipped'] for m in metrics),
            'value_clipped_steps':sum(m['value_grad_clipped'] for m in metrics),
            'mean_policy_grad_norm':float(np.mean([m['policy_grad_norm'] for m in metrics])),
            'mean_value_grad_norm':float(np.mean([m['value_grad_norm'] for m in metrics]))}
    output={'protocol':'chance-training-probe-analysis-v1','complete_main_panels':len(panel_results)==len(spec['panels']),
            'collection':collection,'training':training,
            'panels':{k:{field:v[field] for field in ('win_rate_score','paired_bootstrap','a_wins','draws','a_losses','valid_games','policy_a_stats','policy_b_stats','execution_stats','elapsed_seconds_this_invocation') if field != 'execution_stats' or 'execution_stats' in v} for k,v in panel_results.items()},
            'paired_differences':differences,
            'limitations':['One teacher/start, one collection/training seed, 200 games per algorithm, latest-only fixed labels, no self-iteration.',
                           'Trajectory and outcome distributions differ between arms; within-arm loss improvements are comparable, raw losses across teachers are not.',
                           'DEV50-pair curves are screening; independent confirmation and training seeds are needed for small gains.',
                           'Common-bank value labels belong to archived teacher continuation; empirical mean search targets are diagnostics, not optimal answers.']}
    write_json(ART / 'analysis.json',output)
    emit(event='analysis_done',panels=len(panel_results),complete=output['complete_main_panels'],differences=differences)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','collect','train','evaluate','analyze'))
    parser.add_argument('--arm',choices=ARMS)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--case')
    args=parser.parse_args()
    settings()
    if args.action=='prepare':prepare()
    elif args.action=='collect':collect(args.arm,args.smoke)
    elif args.action=='train':train(args.arm)
    elif args.action=='evaluate':evaluate(args.case)
    elif args.action=='analyze':analyze()
