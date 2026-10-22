"""Post-hoc checkpoint EMA, original per-round holdouts, then one direct arena.

No optimizer updates or self-play. EMA uses endpoint samples every50 updates;
it approximates, rather than reconstructs, EMA after every optimizer update.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import gc
import json
import math
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import torch

from splendor.actor_critic import MODEL_WEIGHTS_FORMAT_VERSION
from splendor.alphazero_checkpoint import load_alphazero_checkpoint, model_state_sha256
from splendor.alphazero_recording import environment_identity, read_json, sha256, source_identity, write_json
from splendor.alphazero_train import AlphazeroLearner, LearnerConfig
from splendor.compact_game_cache import CompactGameCache
from splendor.features import FEATURE_SCHEMA_VERSION
from splendor.offline_train import _digest, make_index
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.pool_fitting import ratios
from splendor.pool_iteration import panel
from splendor.replay import RULESET_ID
from splendor.selfplay_data import collate_selfplay_samples

ROOT = Path(__file__).resolve().parents[2]
ART = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / 'runs/entity_attention_ema_17_19_20261020'
CONFIG = ART / 'config.json'
COLUMNS = ('positions', 'policy_weight', 'ce', 'kl', 'target_entropy', 'entropy', 'top1',
           'sq_error', 'pred', 'pred_sq', 'target', 'target_sq', 'product')


def emit(event, **values):
    print(json.dumps({'event': event, **values}, ensure_ascii=False), flush=True)


def identity(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha256(Path(path))}


def checked(record):
    path = Path(record['path'])
    if sha256(path) != record['sha256']:
        raise ValueError(f'frozen input changed: {path}')
    return path


def trajectory(config):
    """One initial value and160 distinct50-step updates; no final/initial aliases."""
    source = ROOT / config['source_run']
    for iteration in range(config['warmup_first_iteration'], max(config['target_iterations']) + 1):
        stage = source / 'rounds' / f'round-{iteration-1:03d}'
        for step in range(config['checkpoint_interval'], config['updates_per_iteration'] + 1,
                          config['checkpoint_interval']):
            yield iteration, step, stage / 'checkpoints' / f'step-{step:09d}.pt'


def validation_records(snapshot, references):
    games = {(g['round_index'], g['game_id']): g for g in snapshot['games']}
    selected = {}
    for ref in references:
        g = games[(ref['round_index'], ref['game_id'])]
        if (g['split'] != 'validation' or g['run_id'] != ref['run_id']
                or not 0 <= ref['decision_index'] < g['decision_count']):
            raise ValueError('reference is not an original validation position')
        selected[(g['round_index'], g['game_id'])] = g
    return list(selected.values())


def prepare(output, report):
    config = read_json(CONFIG)
    path = output / 'plan.json'
    if path.exists():
        plan = read_json(path)
        if plan['config'] != config or plan['script_sha256'] != sha256(Path(__file__)):
            raise ValueError('resume requires the frozen experiment configuration and script')
        return plan
    source = ROOT / config['source_run']
    source_config = read_json(ROOT / config['source_config'])
    first = source / 'rounds' / f'round-{config["warmup_first_iteration"]-1:03d}' / 'checkpoints/initial.pt'
    initial = load_alphazero_checkpoint(first)
    checkpoints = []
    boundaries = []
    for iteration, step, checkpoint in trajectory(config):
        checkpoints.append({'iteration': iteration, 'step': step,
            'total_train_step': initial.total_train_step + (iteration-config['warmup_first_iteration'])*1000 + step,
            **identity(checkpoint)})
        if step == config['checkpoint_interval']:
            boundaries.append({'iteration': iteration, **identity(checkpoint.parent / 'initial.pt')})
    panels = []
    archives = {}
    for iteration in config['target_iterations']:
        stage = source / 'rounds' / f'round-{iteration-1:03d}'
        refs_path = stage / 'validation_references.json'
        snapshot_path = stage / 'replay_snapshot.json'
        refs, snapshot = read_json(refs_path), read_json(snapshot_path)
        if _digest(refs['references']) != refs['digest']:
            raise ValueError('source validation reference digest mismatch')
        games = validation_records(snapshot, refs['references'])
        for game in games:
            key = game['path']
            if key not in archives:
                stat = Path(key).stat()
                if (stat.st_size, stat.st_mtime_ns) != (game['size'], game['mtime_ns']):
                    raise ValueError('validation archive differs from source replay snapshot')
                archives[key] = identity(key)
        baseline_path = stage / 'validation/step-000001000.json'
        baseline = read_json(baseline_path)
        raw_model = identity(stage / 'checkpoints/step-000001000.pt')
        if (baseline['checkpoint_sha256'] != raw_model['sha256']
                or baseline['validation_references_digest'] != refs['digest']
                or baseline['snapshot_digest'] != snapshot['digest']):
            raise ValueError('source baseline, checkpoint and holdout identities differ')
        p = {'iteration': iteration, 'raw_checkpoint': raw_model,
            'final_alias':identity(stage / 'checkpoints/final.pt'),
            'references_identity': identity(refs_path), 'snapshot_identity': identity(snapshot_path),
            'baseline_identity': identity(baseline_path), 'references': refs['references'],
            'reference_digest': refs['digest'], 'games': games, 'baseline': baseline['metrics']}
        p['digest'] = _digest(p)
        panels.append(p)
    plan = {'protocol': config['protocol'], 'config': config, 'source_config': source_config,
        'source_config_identity_at_freeze': identity(ROOT / config['source_config']),
        'initial_checkpoint': identity(first), 'initial_total_step': initial.total_train_step,
        'model_config': initial.model_config, 'checkpoints': checkpoints, 'boundaries': boundaries,
        'panels': panels, 'validation_archives': list(archives.values()),
        'script_sha256': sha256(Path(__file__)),
        'scope': 'DEV original2048 holdouts per round; one common half-life; no optimizer, collection or teacher updates; approximate sparse checkpoint EMA'}
    plan['digest'] = _digest(plan)
    write_json(path, plan)
    write_json(report / 'plan.json', plan)
    write_json(report / 'source_config.json', source_config)
    emit('prepared', checkpoints=len(checkpoints), holdouts=[len(p['references']) for p in panels])
    return plan


def ema_update(average, state, elapsed, half_life):
    if elapsed <= 0 or half_life <= 0 or set(average) != set(state):
        raise ValueError('invalid EMA interval, half-life or state keys')
    decay = 2. ** (-elapsed / half_life)
    for name, value in state.items():
        if average[name].shape != value.shape or not value.is_floating_point():
            raise ValueError('EMA requires aligned floating-point weights and buffers')
        average[name].mul_(decay).add_(value.to(torch.float64), alpha=1-decay)


def save_frozen(path, model_config, state, metadata):
    """Existing read-only model schema, deliberately no optimizer/RNG payload."""
    values = {k: v.to(torch.float32).clone() for k, v in state.items()}
    payload = {'checkpoint_schema': 'splendor-frozen-actor-critic-v1',
        'format_version': MODEL_WEIGHTS_FORMAT_VERSION,
        'feature_schema_version': FEATURE_SCHEMA_VERSION, 'config': model_config,
        'state_dict': values, 'metadata': {'rules_version': RULESET_ID, 'evaluation_only': True,
            'model_sha256': model_state_sha256(values), **metadata}}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.pt.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)
    return identity(path) | {'model_sha256': payload['metadata']['model_sha256']}


def build_emas(plan, output, report):
    manifest_path = output / 'ema_models.json'
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest['plan_digest'] != plan['digest']:
            raise ValueError('EMA manifest differs from plan')
        for item in manifest['models']:
            checked(item)
        return manifest
    initial = load_alphazero_checkpoint(checked(plan['initial_checkpoint']))
    averages = {h: {k: v.to(torch.float64).clone() for k,v in initial.model_state_dict.items()}
                for h in plan['config']['half_life_steps']}
    previous_step = initial.total_train_step
    previous_hash = initial.model_sha256
    boundary = {r['iteration']: r for r in plan['boundaries']}
    models = []
    for item in plan['checkpoints']:
        if item['step'] == plan['config']['checkpoint_interval']:
            start = load_alphazero_checkpoint(checked(boundary[item['iteration']]))
            if start.model_sha256 != previous_hash or start.total_train_step != previous_step:
                raise ValueError('training trajectory is discontinuous at round boundary')
        payload = load_alphazero_checkpoint(checked(item))
        if (payload.model_config != plan['model_config'] or payload.total_train_step != item['total_train_step']
                or model_state_sha256(payload.model_state_dict) != payload.model_sha256):
            raise ValueError('source checkpoint model, step or parameter hash mismatch')
        for half_life, average in averages.items():
            ema_update(average, payload.model_state_dict, payload.total_train_step-previous_step, half_life)
        previous_step, previous_hash = payload.total_train_step, payload.model_sha256
        if item['step'] == 1000:
            emit('ema_round', iteration=item['iteration'])
        if item['iteration'] in plan['config']['target_iterations'] and item['step'] == 1000:
            final = next(p['final_alias'] for p in plan['panels'] if p['iteration'] == item['iteration'])
            if load_alphazero_checkpoint(checked(final)).model_sha256 != payload.model_sha256:
                raise ValueError('final.pt is not the source1000-step model')
            for half_life, average in averages.items():
                metadata = {'plan_digest': plan['digest'], 'iteration': item['iteration'],
                    'total_train_step': previous_step, 'half_life_steps': half_life,
                    'initial_residual_weight': 2.**(-(previous_step-initial.total_train_step)/half_life),
                    'sampling_interval': 50, 'accumulator_dtype': 'float64', 'export_dtype': 'float32'}
                target = output / 'checkpoints' / f'round-{item["iteration"]:03d}-half-{half_life:04d}.pt'
                models.append({**save_frozen(target, plan['model_config'], average, metadata), **metadata})
    manifest = {'plan_digest': plan['digest'], 'models': models}
    write_json(manifest_path, manifest)
    write_json(report / 'ema_models.json', manifest)
    return manifest


def summarize(per_game):
    metric = ratios(dict(zip(COLUMNS, map(float, per_game.sum(0)))))
    n = metric['positions']
    metric.update(value_prediction_mean=metric['pred']/n,
        value_prediction_std=math.sqrt(max(0., metric['pred_sq']/n-(metric['pred']/n)**2)),
        value_target_mean=metric['target']/n, kl_plus_mse=metric['policy_kl']+metric['value_mse'])
    return metric


def evaluate_samples(model, samples, game_ids, game_count, batch_size):
    model.eval()
    rows = []
    with torch.inference_mode():
        for start in range(0, len(samples), batch_size):
            part = samples[start:start+batch_size]
            batch = collate_selfplay_samples(part)
            logits, values = model(batch.decisions)
            mask = torch.as_tensor(batch.decisions.candidate_mask)
            lp = torch.log_softmax(logits.masked_fill(~mask, -torch.inf), -1)
            lp = torch.where(mask, lp, 0.).numpy().astype(np.float64)
            pred = values.numpy().astype(np.float64)
            target = batch.policy_targets.astype(np.float64)
            logs = np.zeros_like(target)
            np.log(target, out=logs, where=target>0)
            pw, y = batch.policy_weights.astype(np.float64), batch.value_targets.astype(np.float64)
            if not np.all(batch.value_weights == 1):
                raise ValueError('source experiment requires original unit value weights')
            ce, th = -(target*lp).sum(1), -(target*logs).sum(1)
            probs = np.where(batch.decisions.candidate_mask, np.exp(lp), 0.)
            entropy = -(probs*lp).sum(1)
            best = np.where(batch.decisions.candidate_mask, lp, -np.inf).argmax(1)
            top1 = target[np.arange(len(part)),best] >= target.max(1)-1e-7
            rows.append(np.column_stack((np.ones(len(part)),pw,ce*pw,(ce-th)*pw,th*pw,
                entropy*pw,top1*pw,(pred-y)**2,pred,pred**2,y,y**2,pred*y)))
    positions = np.concatenate(rows)
    if not np.isfinite(positions).all():
        raise ValueError('nonfinite holdout predictions')
    per_game = np.column_stack([np.bincount(game_ids, positions[:,i], minlength=game_count)
                              for i in range(len(COLUMNS))])
    return summarize(per_game), per_game


def evaluate_holdouts(plan, models, output, report):
    result_path = report / 'validation.json'
    if result_path.exists():
        result = read_json(result_path)
        if result['plan_digest'] != plan['digest']:
            raise ValueError('validation result differs from plan')
        return result
    for record in plan['validation_archives']:
        checked(record)
    results = []
    for p in plan['panels']:
        for name in ('references_identity','snapshot_identity','baseline_identity'):
            checked(p[name])
        index = make_index({'games':p['games']}, 1024**3)
        index.cache = CompactGameCache(1024**3)
        by_key = {g.key:g for g in index.games}
        samples = [index.cache.sample(by_key[(r['round_index'],r['game_id'])],r['decision_index'])
                   for r in p['references']]
        keys = sorted({(r['run_id'],r['round_index'],r['game_id']) for r in p['references']})
        key_index = {k:i for i,k in enumerate(keys)}
        ids = np.array([key_index[(r['run_id'],r['round_index'],r['game_id'])] for r in p['references']])
        specs = [{'half_life_steps':None, **p['raw_checkpoint']}] + [m for m in models['models'] if m['iteration']==p['iteration']]
        points = []
        for spec in specs:
            model = load_frozen_actor_critic_checkpoint(checked(spec),device='cpu')
            metrics, per_game = evaluate_samples(model,samples,ids,len(keys),plan['config']['validation_batch_size'])
            verification = None
            if spec['half_life_steps'] is None:
                differences = {key:metrics[key]-p['baseline'][old] for key,old in
                    (('policy_kl','policy_kl'),('policy_ce','policy_loss'),('value_mse','value_loss'))}
                if max(map(abs,differences.values()))>5e-5:
                    raise AssertionError(f'raw model does not reproduce original holdout: {differences}')
                reference = AlphazeroLearner.evaluate_samples(SimpleNamespace(model=model,device=torch.device('cpu'),
                    config=LearnerConfig(device='cpu',validation_batch_size=256)),samples[:64])
                small,_ = evaluate_samples(model,samples[:64],ids[:64],len(keys),256)
                joint_errors = {key:small[key]-reference[old] for key,old in
                    (('policy_kl','policy_kl'),('policy_ce','policy_loss'),('value_mse','value_loss'))}
                if max(map(abs,joint_errors.values()))>5e-5:
                    raise AssertionError('metric implementation differs from learner')
                verification = {'original_gpu_baseline_differences':differences,'joint_learner_differences':joint_errors}
            tag = 'raw' if spec['half_life_steps'] is None else f'half-{spec["half_life_steps"]:04d}'
            directory = report / 'validation_raw'
            directory.mkdir(exist_ok=True)
            path = directory / f'round-{p["iteration"]:03d}-{tag}.npz'
            np.savez_compressed(path,per_game=per_game)
            point = {'iteration':p['iteration'],'half_life_steps':spec['half_life_steps'],
                'checkpoint':{k:spec[k] for k in ('path','sha256')},'panel_digest':p['digest'],
                'metrics':metrics,'per_game':identity(path),'reference_verification':verification}
            points.append(point)
            write_json(directory / f'round-{p["iteration"]:03d}-{tag}.json',point)
            emit('validation',iteration=p['iteration'],half_life=spec['half_life_steps'],
                kl=metrics['policy_kl'],mse=metrics['value_mse'],bias=metrics['value_bias'])
            del model
        results.append({'iteration':p['iteration'],'panel_digest':p['digest'],
            'reference_digest':p['reference_digest'],'game_keys':keys,'columns':COLUMNS,'points':points})
        del samples,index
        gc.collect()
    result = {'plan_digest':plan['digest'],'groups':results}
    write_json(result_path,result)
    return result


def select_half_life(validation):
    candidates = {}
    for group in validation['groups']:
        raw = next(p['metrics'] for p in group['points'] if p['half_life_steps'] is None)
        for p in group['points']:
            h = p['half_life_steps']
            if h is None:
                continue
            delta = {k:p['metrics'][k]-raw[k] for k in ('policy_kl','value_mse','kl_plus_mse')}
            candidates.setdefault(h,[]).append({'iteration':group['iteration'],**delta})
    scores = [{'half_life_steps':h,'per_round':rows,
               'mean_delta':{k:float(np.mean([r[k] for r in rows]))
                   for k in ('policy_kl','value_mse','kl_plus_mse')}} for h,rows in sorted(candidates.items())]
    selected = min(scores,key=lambda r:(r['mean_delta']['kl_plus_mse'],r['half_life_steps']))
    return {'half_life_steps':selected['half_life_steps'],'candidates':scores,
        'criterion':'equal-round mean delta(KL+MSE); lower is better; one common half-life; exact ties choose shorter',
        'scope':'DEV preselection, not strength evidence; no per-round parameter selection'}


def evaluation_config(plan, ema_path, raw_path, kind):
    raw = json.loads(json.dumps(plan['source_config'][f'{kind}_evaluation']))
    spec = plan['config'][kind]
    raw.update(spec)
    raw['bootstrap_resamples'] = 10000
    raw['policy_a'].update(checkpoint=str(Path(ema_path).resolve()),name='EMA_R19')
    raw['policy_b'].update(checkpoint=str(Path(raw_path).resolve()),name='RAW_R19')
    if kind=='search':
        raw['execution'] = json.loads(json.dumps(plan['source_config']['selfplay']['execution']))
        raw['execution']['pipeline_slots'] = 1
        for side in ('policy_a','policy_b'):
            raw[side]['search_config']['root_noise_fraction'] = 0.
    return raw


def bootstrap_deltas(plan, validation):
    """Whole-game paired intervals per round; never treat three panels as iid."""
    rows = []
    for group in validation['groups']:
        raw = next(p for p in group['points'] if p['half_life_steps'] is None)
        with np.load(checked(raw['per_game'])) as saved:
            base = saved['per_game']
        rng = np.random.default_rng(plan['config']['bootstrap_seed']+group['iteration'])
        weights = rng.multinomial(len(base),np.full(len(base),1/len(base)),size=plan['config']['bootstrap_resamples'])
        for point in group['points']:
            if point['half_life_steps'] is None:
                continue
            with np.load(checked(point['per_game'])) as saved:
                values = saved['per_game']
            changes = {}
            draws = []
            for metric,numerator,denominator in (('policy_kl','kl','policy_weight'),('value_mse','sq_error','positions')):
                col,den = COLUMNS.index(numerator),COLUMNS.index(denominator)
                if not np.array_equal(base[:,den],values[:,den]):
                    raise ValueError('paired holdout population changed')
                draw = (weights@(values[:,col]-base[:,col]))/(weights@base[:,den])
                changes[metric] = {'delta':point['metrics'][metric]-raw['metrics'][metric],
                    'ci95':np.quantile(draw,[.025,.975]).tolist()}
                draws.append(draw)
            changes['kl_plus_mse'] = {'delta':point['metrics']['kl_plus_mse']-raw['metrics']['kl_plus_mse'],
                'ci95':np.quantile(draws[0]+draws[1],[.025,.975]).tolist()}
            rows.append({'iteration':group['iteration'],'half_life_steps':point['half_life_steps'],'changes':changes})
    return rows


def render(report, validation, selection, arenas, *, complete):
    lines = ['# 第17–19轮：稀疏checkpoint EMA回溯测试','',
        '状态：'+('complete' if complete else 'incomplete；已完成留出预选，正式对战尚未全部完成')+'。', '',
        '每轮使用原有2048个留出references，顺序、重复抽样、标签和权重不变；三轮各自的pool不同。CPU FP32推理，float64统计；各轮raw与原GPU指标核对，容差5e-5。', '',
        '从第12轮训练前权重初始化，按每50步端点累计；半衰期以训练更新次数定义，衰减2^(-50/H)，float64累积、float32导出。initial/final别名不重复计入。它近似逐步EMA，不含缺失的中间49步。只平均完整网络，不包含optimizer／RNG，不改在线学生或教师。', '',
        '| 轮 | 模型 | Policy KL | Value MSE | KL+MSE | Value偏差 | 预测标准差 | 相关性 |',
        '| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for group in validation['groups']:
        for p in group['points']:
            title = 'raw' if p['half_life_steps'] is None else f'EMA H={p["half_life_steps"]}'
            m=p['metrics']
            lines.append(f"| {group['iteration']} | {title} | "+' | '.join(f'{m[k]:.6f}' for k in
                ('policy_kl','value_mse','kl_plus_mse','value_bias','value_prediction_std','value_correlation'))+' |')
    lines+=['','统一预选半衰期：**'+str(selection['half_life_steps'])+'次更新**。按三轮相对raw的KL+MSE变化等权均值最低选择；每轮不另选参数。该预选使用DEV，不能当作独立棋力验证。','',
        '| 半衰期 | 平均ΔKL | 平均ΔMSE | 平均Δ(KL+MSE) |','| ---: | ---: | ---: | ---: |']
    for item in selection['candidates']:
        lines.append(f"| {item['half_life_steps']} | "+' | '.join(f"{item['mean_delta'][k]:+.6f}" for k in ('policy_kl','value_mse','kl_plus_mse'))+' |')
    lines+=['','逐轮按整局配对bootstrap区间见`paired_changes.json`；不同轮次重叠，不据此计算三轮独立重复的显著性。','',
        '正式对战仅第19轮EMA对同轮raw；新种子与原R43面板分离。greedy1000pair，双方S1024/N1搜索500pair、无根噪声/T0；执行参数复用冻结采集配置，owner单槽。每pair交换先后手。','',
        '| 对战 | pair | EMA得分 | 95%区间 |','| --- | ---: | ---: | --- |']
    for kind, item in arenas.items():
        ci=item['paired_bootstrap']
        lines.append(f"| {kind} | {item['completed_pairs']} | {item['win_rate_score']*100:.3f}% | [{ci['lower']*100:.3f}%,{ci['upper']*100:.3f}%] |")
    (report/'results.md').write_text('\n'.join(lines)+'\n')


def run(output, report):
    output.mkdir(parents=True,exist_ok=True)
    report.mkdir(parents=True,exist_ok=True)
    with (output/'.ema.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=prepare(output,report)
        def live(stage,**values):
            write_json(report/'live.json',{'stage':stage,'updated_utc':datetime.now(timezone.utc).isoformat(),
                'plan_digest':plan['digest'],**values})
        live('build_ema')
        models=build_emas(plan,output,report)
        live('validation')
        validation=evaluate_holdouts(plan,models,output,report)
        selection=select_half_life(validation)
        selection.update(plan_digest=plan['digest'],validation_sha256=sha256(report/'validation.json'))
        write_json(report/'selection.json',selection)
        write_json(report/'paired_changes.json',{'resamples':plan['config']['bootstrap_resamples'],
            'seed_rule':'bootstrap_seed + iteration','unit':'paired whole original game',
            'comparisons':bootstrap_deltas(plan,validation)})
        chosen=next(m for m in models['models'] if m['iteration']==19 and m['half_life_steps']==selection['half_life_steps'])
        raw=next(p['raw_checkpoint'] for p in plan['panels'] if p['iteration']==19)
        arenas={}
        render(report,validation,selection,arenas,complete=False)
        for kind in ('greedy','search'):
            live(kind,half_life_steps=selection['half_life_steps'])
            config=evaluation_config(plan,checked(chosen),checked(raw),kind)
            write_json(report/f'{kind}_config.json',config)
            result=panel(config,Path(chosen['path']),Path(raw['path']),output/f'{kind}.json',opponent_name='RAW_R19').summary
            if result['completed_pairs']!=plan['config'][kind]['pairs']:
                raise ValueError('formal arena has incomplete paired population')
            arenas[kind]={k:v for k,v in result.items() if k!='pairs'}
            write_json(report/f'{kind}_summary.json',arenas[kind])
            write_json(report/f'{kind}_paired_outcomes.json',{'pairs':[{
                'pair_id':p['pair_id'],'game_seed':p['game_seed'],
                'games':[{k:g[k] for k in ('game_in_pair','first_player','status','outcome_for_a','scores','policy_seeds')}
                         for g in p['games']]} for p in result['pairs']]})
            render(report,validation,selection,arenas,complete=kind=='search')
            emit('arena_complete',kind=kind,score=result['win_rate_score'],pairs=result['completed_pairs'])
        live('complete',half_life_steps=selection['half_life_steps'])
        write_json(report/'complete.json',{'plan_digest':plan['digest'],'selection':selection,
            'arenas':arenas,'scope':'post-hoc checkpoint averaging only; no claim about EMA teachers in future self-iteration'})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','run'))
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--report',type=Path,default=ART)
    args=parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    args.output.mkdir(parents=True,exist_ok=True)
    args.report.mkdir(parents=True,exist_ok=True)
    if args.command=='prepare':
        plan=prepare(args.output,args.report)
        emit('prepare_complete',plan_digest=plan['digest'])
        return
    source=args.output/'source'
    if not (args.report/'execution.json').exists():
        code=source_identity(ROOT,source)
        shutil.copyfile(__file__,source/'experiment.py')
        write_json(args.report/'execution.json',{'command':sys.argv,'script_sha256':sha256(Path(__file__)),
            'source':code,'environment':environment_identity(),'config_sha256':sha256(CONFIG)})
    try:
        run(args.output,args.report)
    except BaseException as error:
        write_json(args.report/'failure.json',{'type':type(error).__name__,'message':str(error),
            'utc':datetime.now(timezone.utc).isoformat()})
        raise


if __name__=='__main__':
    main()
