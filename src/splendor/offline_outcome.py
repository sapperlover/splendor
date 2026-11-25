"""Controlled scratch scalar/WDL, MC/fixed-lambda offline comparisons.

Uses the established immutable 4-worker pool and streaming training driver.
Only the parent reads the small lambda-target mmap; no shared pool is mutated.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import math
import os
import shutil
import sys
import time
from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .actor_critic import build_actor_critic
from .alphazero_checkpoint import load_alphazero_checkpoint
from .alphazero_recording import append_json, read_json, sha256, source_identity, write_json
from .alphazero_train import (_cpu_scalar_sums, _torch_batch, alphazero_loss_sums,
                              label_statistics, metric_ratios, run_training_round)
from .compact_game_cache import CompactGameCache
from .features import EncodedStateBatch, FEATURE_SCHEMA_VERSION, GLOBAL_FEATURE_NAMES
from .hybrid_train import HybridReferenceWriter
from .offline_mixed_budget import MixedBudgetOfflineConfig, make_batch_drawer, prepare_inputs as prepare_base
from .offline_objectives import digest
from .rng import DeterministicRng
from .selfplay_data import STATE_FIELDS, collate_selfplay_samples
from .training_prefetch import SharedReadonlyPrefetch
from .value_outcome import (OutcomeLearner, archived_search_lambda_targets, scalar_lambda_targets,
                            terminal_one_hot, vector_lambda_targets)


@dataclass(frozen=True)
class OutcomeOfflineConfig(MixedBudgetOfflineConfig):
    outcome_objective: str = 'scalar_mc'
    outcome_lambda: float = .8
    outcome_lambda_source: str = 'network'
    outcome_teacher_checkpoint: str | None = None
    outcome_teacher_selection: str | None = None
    outcome_teacher_sha256: str | None = None
    target_batch_size: int = 1024

    def __post_init__(self):
        super().__post_init__()
        if self.model_config.get('architecture') != 'entity_attention_v1':
            raise ValueError('Outcome comparison requires base entity attention')
        if self.outcome_objective not in ('scalar_mc', 'scalar_lambda', 'wdl_mc', 'wdl_lambda'):
            raise ValueError('Unknown outcome objective')
        if not math.isfinite(self.outcome_lambda) or not 0 <= self.outcome_lambda <= 1:
            raise ValueError('outcome_lambda must be in [0,1]')
        if self.outcome_lambda_source not in ('network', 'archive_search'):
            raise ValueError('Unknown lambda target source')
        sources = (self.outcome_teacher_checkpoint, self.outcome_teacher_selection)
        if self.outcome_lambda_source == 'archive_search':
            if self.outcome_objective != 'scalar_lambda' or any(x is not None for x in sources) or self.outcome_teacher_sha256 is not None:
                raise ValueError('Archived search targets require scalar_lambda and no network teacher')
        elif self.outcome_objective.endswith('_lambda') and sum(x is not None for x in sources) != 1:
            raise ValueError('Fixed lambda targets require exactly one teacher checkpoint or selection')
        if type(self.target_batch_size) is not int or self.target_batch_size <= 0:
            raise ValueError('target_batch_size must be positive')


def prepare_inputs(config, output):
    index, pool, identity, fixed, plan = prepare_base(config, output)
    if config.outcome_objective.startswith('wdl_'):
        plan['initialization']['model_config'] = dict(config.model_config) | {
            'architecture': 'entity_attention_wdl_v1'}
    plan.update(outcome_objective=config.outcome_objective,
        wdl_order=['win','draw','loss'] if config.outcome_objective.startswith('wdl_') else None,
        lambda_target_policy=('generate once with one fixed teacher, then immutable reuse from student step zero'
                              if config.outcome_objective.endswith('_lambda') else None),
        outcome_teacher_checkpoint=config.outcome_teacher_checkpoint,
        outcome_teacher_selection=config.outcome_teacher_selection,
        outcome_teacher_sha256=config.outcome_teacher_sha256,
        outcome_lambda_source=config.outcome_lambda_source,
        selection_criterion='full_validation.policy_kl + full_validation.value_loss (true-terminal scalar MSE)',
        training_budget='fixed exposure budget; no group-specific early stopping',
        lambda_clock='strictly later MAIN_ACTION of the same player; exact terminal tail')
    if config.outcome_lambda_source == 'archive_search':
        plan.update(lambda_target_policy='generate once from archived search roots, then immutable reuse from student step zero',
            root_value_reduction='equal world mean of sum(W)/sum(N); original unpruned search statistics',
            missing_root_policy='mask targets dependent on missing roots; retain policy and true-terminal metrics')
    write_json(output/'plan.json', plan)
    return index, pool, identity, fixed, plan


def fixed_teacher(config, output):
    """Bind the external teacher before allocating the full training cache."""
    selection = None
    if config.outcome_teacher_selection is not None:
        selection = read_json(Path(config.outcome_teacher_selection))
        checkpoint = Path(selection['checkpoint'])
    else:
        checkpoint = Path(config.outcome_teacher_checkpoint)
    actual_sha = sha256(checkpoint)
    if config.outcome_teacher_sha256 is not None and actual_sha != config.outcome_teacher_sha256:
        raise ValueError('Configured fixed teacher checkpoint SHA changed')
    if selection is not None and actual_sha != selection['checkpoint_sha256']:
        raise ValueError('Selected fixed teacher checkpoint SHA changed')
    payload = load_alphazero_checkpoint(checkpoint)
    if config.outcome_objective == 'wdl_lambda' and payload.model_config['architecture'] != 'entity_attention_wdl_v1':
        raise ValueError('Vector lambda targets require a WDL teacher; scalar Q cannot determine WDL')
    record = dict(protocol='fixed-outcome-teacher-v1', checkpoint=str(checkpoint.resolve()),
        checkpoint_sha256=actual_sha, train_step=payload.round_train_step,
        source_selection=config.outcome_teacher_selection)
    destination = output/'fixed_teacher.json'
    if destination.exists():
        if read_json(destination) != record:
            raise ValueError('Fixed teacher identity changed')
    else:
        write_json(destination, record)
    return record


def initialize(config, identity, steps):
    torch.manual_seed(config.initialization_seed)
    model = dict(config.model_config)
    if config.outcome_objective.startswith('wdl_'):
        model['architecture'] = 'entity_attention_wdl_v1'
    learner = OutcomeLearner(build_actor_critic(model), config.learner,
                             sampling_rng=DeterministicRng(config.sampling_seed))
    learner.replay_identity, learner.round_steps_target = identity, steps
    learner.parent_checkpoint = dict(kind='random_initialization', seed=config.initialization_seed,
                                     model_config=asdict(learner.model.config), optimizer='fresh AdamW')
    return learner


def evaluate(learner, index):
    """All groups use the same true-terminal MSE/KL; WDL adds NLL/Brier/ECE."""
    totals = Counter(); bins = np.zeros((10, 3), dtype=np.float64)
    was_training = learner.model.training
    learner.model.eval()

    def flush(samples):
        batch = collate_selfplay_samples(samples); t = _torch_batch(batch, learner.device)
        with torch.inference_mode():
            extra = {}
            if learner.model.config.architecture == 'entity_attention_wdl_v1':
                logits, values, wdl = learner.model.forward_with_wdl(batch.decisions)
                target = torch.as_tensor(terminal_one_hot(batch.value_targets), device=learner.device)
                prob = wdl.softmax(-1); weight = t['value_weights']
                extra = dict(wdl_nll_sum=(-(target*wdl.log_softmax(-1)).sum(-1)*weight).sum(),
                             wdl_brier_sum=((prob-target).square().sum(-1)*weight).sum())
                confidence, chosen = prob.max(-1)
                ids = (confidence*10).long().clamp_max(9)
                counts = torch.zeros((10,3), dtype=torch.float64, device=learner.device)
                counts[:,0].scatter_add_(0, ids, weight.double())
                counts[:,1].scatter_add_(0, ids, (confidence*weight).double())
                counts[:,2].scatter_add_(0, ids, ((chosen==target.argmax(-1))*weight).double())
                bins[:] += counts.cpu().numpy()
            else:
                logits, values = learner.model(batch.decisions)
            terms = alphazero_loss_sums(logits, values, **t)
            sums = {k:getattr(terms,k) for k in ('policy_ce_sum','policy_weight_sum','value_sq_sum',
                                                'value_weight_sum','entropy_sum')}
            totals.update(_cpu_scalar_sums(sums | extra))
        totals.update(label_statistics(batch)); totals['positions'] += len(samples)

    pending = []
    try:
        for game in index.split_games('validation'):
            for i in range(game.decision_count):
                pending.append(index.cache.sample(game, i))
                if len(pending) == learner.config.validation_batch_size:
                    flush(pending); pending=[]
        if pending:
            flush(pending)
    finally:
        learner.model.train(was_training)
    metrics = dict(totals) | metric_ratios(totals)
    if 'wdl_nll_sum' in totals:
        n = totals['value_weight_sum']
        metrics.update(wdl_nll=totals['wdl_nll_sum']/n, wdl_brier=totals['wdl_brier_sum']/n,
                       wdl_ece=float(np.abs(bins[:,1]-bins[:,2]).sum()/n),
                       wdl_calibration_bins=bins.tolist())
    return metrics


def offsets_for(games):
    offsets = {}; size=0
    for g in games:
        offsets[g.key]=size; size+=g.decision_count
    return offsets, size


def archive_lambda_targets(config, output, training_index):
    """Generate one CPU-only target/mask pair directly from immutable train NPZs."""
    games = training_index.split_games('train'); offsets, size = offsets_for(games)
    folder = output/'lambda_targets'/'archived_search'
    identity = dict(protocol='same-player-next-main-archived-search-lambda-v1',
        dataset_digest=read_json(output/'dataset.json')['digest'], outcome_lambda=config.outcome_lambda,
        game_order_digest=digest([(g.key,g.decision_count) for g in games]), positions=size, train_only=True,
        root_value_reduction='equal world mean of sum(W)/sum(N); original unpruned search statistics',
        missing_root_policy='mask dependent targets; finite terminal placeholders have zero training weight',
        clock='strictly later MAIN_ACTION of the same player; no own future main action uses true terminal')
    if folder.exists():
        record = read_json(folder/'manifest.json')
        if (record['identity'] != identity or sha256(folder/'targets.npy') != record['target_sha256']
                or sha256(folder/'mask.npy') != record['mask_sha256']):
            raise ValueError('Frozen archived lambda target/mask identity changed')
    else:
        started=time.perf_counter();partial=folder.with_name(folder.name+'.partial')
        if partial.exists():
            if read_json(partial/'identity.json') != identity:
                raise ValueError('Unfinished archived labels belong to another data identity')
            shutil.rmtree(partial)
        partial.mkdir(parents=True,exist_ok=False);write_json(partial/'identity.json',identity)
        targets=np.lib.format.open_memmap(partial/'targets.npy',mode='w+',dtype=np.float32,shape=(size,))
        mask=np.lib.format.open_memmap(partial/'mask.npy',mode='w+',dtype=np.bool_,shape=(size,))
        counts=Counter();budgets=Counter()
        fields=('global_features','player_id','value_target','candidate_offsets','valid_world_count',
                'per_world_root_visit_counts','per_world_root_value_sums','simulations')
        print(f'归档搜索λ：开始一次性生成Train标签，共{len(games)}局；不调用网络或搜索',flush=True)
        for number,game in enumerate(games,1):
            with np.load(game.path,allow_pickle=False) as archive:
                arrays={name:archive[name] for name in fields}
                metadata=json.loads(archive['metadata_json'].tobytes().decode('utf-8'))
            labels,available,root_valid=archived_search_lambda_targets(arrays,metadata,config.outcome_lambda)
            begin=offsets[game.key];end=begin+game.decision_count
            targets[begin:end]=labels;mask[begin:end]=available
            main=arrays['global_features'][:,GLOBAL_FEATURE_NAMES.index('phase_main_action')]>.5
            counts.update(positions=len(available),main_positions=int(main.sum()),
                root_positions=int(root_valid.sum()),eligible_positions=int(available.sum()),
                eligible_main_positions=int(available[main].sum()))
            budgets.update(int(x) for x in arrays['simulations'][root_valid])
            if number%1000==0 or number==len(games):
                write_json(partial/'progress.json',dict(status='building_targets',games=number,
                    total_games=len(games),counts=dict(counts),seconds=time.perf_counter()-started))
                print(f'归档搜索λ {number}/{len(games)}局，有效标签{counts["eligible_positions"]}/{counts["positions"]}',flush=True)
        targets.flush();mask.flush();del targets,mask
        for name in ('targets.npy','mask.npy'):
            with (partial/name).open('rb') as handle:os.fsync(handle.fileno())
        write_json(partial/'manifest.json',dict(identity=identity,target_sha256=sha256(partial/'targets.npy'),
            mask_sha256=sha256(partial/'mask.npy'),generation_seconds=time.perf_counter()-started,
            counts=dict(counts),root_budget_counts=dict(budgets),
            target_coverage=counts['eligible_positions']/counts['positions'],
            root_coverage=counts['root_positions']/counts['main_positions']))
        partial.rename(folder)
        print(f'归档搜索λ标签和掩码已冻结，耗时{time.perf_counter()-started:.1f}s',flush=True)
    return (np.load(folder/'targets.npy',mmap_mode='r'), offsets,
            np.load(folder/'mask.npy',mmap_mode='r'))


def lambda_targets(config, output, teacher_identity, training_index):
    """Publish labels + teacher/data identity atomically, preserving student RNG."""
    games = training_index.split_games('train'); offsets, size = offsets_for(games)
    checkpoint = Path(teacher_identity['checkpoint'])
    if sha256(checkpoint) != teacher_identity['checkpoint_sha256']:
        raise ValueError('Fixed teacher checkpoint SHA changed before target generation')
    wdl = config.outcome_objective == 'wdl_lambda'
    folder = output/'lambda_targets'/'fixed'
    identity = dict(protocol='same-player-next-main-fixed-lambda-v2', teacher=teacher_identity,
        outcome_objective=config.outcome_objective,
        dataset_digest=read_json(output/'dataset.json')['digest'], outcome_lambda=config.outcome_lambda,
        game_order_digest=digest([(g.key,g.decision_count) for g in games]), positions=size,
        scalar_bootstrap_transform=None if wdl else 'clip[-1,1]',
        wdl_order=['win','draw','loss'] if wdl else None, train_only=True)
    if folder.exists():
        record = read_json(folder/'manifest.json')
        if record['identity'] != identity or sha256(folder/'targets.npy') != record['sha256']:
            raise ValueError('Frozen lambda target identity changed')
    else:
        generation_started = time.perf_counter()
        partial = folder.with_name(folder.name+'.partial')
        if partial.exists():
            if read_json(partial/'identity.json') != identity:
                raise ValueError('Unfinished lambda generation belongs to another teacher/data identity')
            # This unpublished generation is owned by this exact run and can
            # be regenerated deterministically after an interrupted inference.
            shutil.rmtree(partial)
        partial.mkdir(parents=True, exist_ok=False)
        write_json(partial/'identity.json', identity)
        devices = list(range(torch.cuda.device_count())) if config.learner.device.startswith('cuda') else []
        with torch.random.fork_rng(devices=devices):
            teacher = load_alphazero_checkpoint(checkpoint).build_model(device=config.learner.device)
            teacher.eval(); teacher.requires_grad_(False)
            shape = (size,3) if wdl else (size,)
            predictions = np.zeros(shape,dtype=np.float32)
            chunks=[]; indices=[]; queued=0
            diagnostics=dict(predicted_positions=0,clipped_positions=0)

            def flush():
                nonlocal chunks, indices, queued
                if not chunks:return
                state=EncodedStateBatch(feature_schema_version=FEATURE_SCHEMA_VERSION,
                    **{k:np.concatenate([c[k] for c in chunks]) for k in STATE_FIELDS})
                with torch.inference_mode():
                    prediction = teacher.wdl_logits_from_state(state).softmax(-1) if wdl else teacher.value_from_state(state)
                    values=prediction.cpu().numpy()
                    diagnostics['predicted_positions']+=len(values)
                    if not wdl:diagnostics['clipped_positions']+=int((np.abs(values)>1).sum())
                    predictions[np.concatenate(indices)] = values
                chunks=[];indices=[];queued=0

            print(f'固定λ教师：开始一次性预测训练局面，共{len(games)}局',flush=True)
            for number, game in enumerate(games,1):
                sample=training_index.cache.sample(game,0); arrays=sample.arrays
                ids=np.flatnonzero(arrays['global_features'][:,GLOBAL_FEATURE_NAMES.index('phase_main_action')]>.5)
                for start in range(0,len(ids),config.target_batch_size):
                    part=ids[start:start+config.target_batch_size]
                    chunks.append({k:arrays[k][part] for k in STATE_FIELDS})
                    indices.append(offsets[game.key]+part);queued+=len(part)
                    if queued>=config.target_batch_size:flush()
                if number%1000==0 or number==len(games):
                    write_json(partial/'progress.json',dict(status='teacher_inference',games=number,
                        total_games=len(games),seconds=time.perf_counter()-generation_started))
                    print(f'λ教师预测 {number}/{len(games)}局',flush=True)
            flush(); del teacher
        write_json(partial/'progress.json',dict(status='building_targets',total_games=len(games),
            seconds=time.perf_counter()-generation_started))
        targets=np.lib.format.open_memmap(partial/'targets.npy',mode='w+',dtype=np.float32,shape=shape)
        target_function = vector_lambda_targets if wdl else scalar_lambda_targets
        for game in games:
            arrays=training_index.cache.sample(game,0).arrays; begin=offsets[game.key];end=begin+game.decision_count
            targets[begin:end]=target_function(predictions[begin:end],arrays['player_id'],
                arrays['global_features'][:,GLOBAL_FEATURE_NAMES.index('phase_main_action')]>.5,
                arrays['value_target'],config.outcome_lambda)
        targets.flush();del targets,predictions
        with (partial/'targets.npy').open('rb') as f:os.fsync(f.fileno())
        write_json(partial/'manifest.json',dict(identity=identity,sha256=sha256(partial/'targets.npy'),
            generation_diagnostics=diagnostics,
            generation_seconds=time.perf_counter()-generation_started))
        write_json(partial/'progress.json',dict(status='ready',total_games=len(games),
            seconds=time.perf_counter()-generation_started))
        partial.rename(folder)
        print(f'固定λ训练目标发布，耗时{time.perf_counter()-generation_started:.1f}s',flush=True)
    return np.load(folder/'targets.npy',mmap_mode='r'), offsets


def _complete_training(output, plan, checkpoint, wall_started, training_sessions, *, recovered):
    """Publish the terminal checkpoint only after its full validation is committed."""
    step=plan['steps_target']
    row=read_json(output/'full_validation'/f'step-{step:09d}.json')
    if (row['step']!=step or row['checkpoint_sha256']!=sha256(checkpoint)
            or row['dataset_digest']!=read_json(output/'dataset.json')['digest']
            or row['metrics']['positions']!=plan['splits']['validation']['positions']):
        raise ValueError('Terminal validation checkpoint/data identity changed')
    final=output/'checkpoints/final.pt'
    temporary=final.with_suffix('.pt.tmp')
    shutil.copyfile(checkpoint,temporary)
    with temporary.open('rb') as handle:os.fsync(handle.fileno())
    temporary.replace(final)
    write_json(output/'progress.json',dict(status='complete',step=step,steps_target=step,
        training_sessions=training_sessions,finalization_recovered=recovered,
        elapsed_seconds_this_session_including_cache_targets_validation=time.perf_counter()-wall_started))
    return dict(status='complete',step=step)


def run(config, output, *, prepare_only=False, generate_targets_only=False, max_steps_this_session=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    with (output/'.lock').open('a') as lock, ExitStack() as cleanup:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        index,pool,identity,_,plan=prepare_inputs(config,output)
        if prepare_only:return plan
        if max_steps_this_session is not None and max_steps_this_session<=0:
            raise ValueError('max_steps_this_session must be positive')
        wall_started=time.perf_counter()
        archived = config.outcome_lambda_source == 'archive_search'
        checkpoints=output/'checkpoints'
        terminal_checkpoint=checkpoints/f'step-{plan["steps_target"]:09d}.pt'
        ending=terminal_checkpoint.exists()
        teacher = (fixed_teacher(config, output) if config.outcome_objective.endswith('_lambda')
                   and not archived and (generate_targets_only or not ending) else None)
        if generate_targets_only:
            if archived:
                archive_lambda_targets(config,output,index)
                return dict(status='archived_targets_ready')
            if teacher is None:
                raise ValueError('Only the lambda arm has generated targets')
            torch.set_num_threads(1)
            lambda_targets(config, output, teacher, index)
            return dict(status='fixed_targets_ready', teacher=teacher)
        progress_path = output/'progress.json'
        if progress_path.exists() and read_json(progress_path)['status']=='complete':
            final = output/'checkpoints/final.pt'
            recorded = read_json(output/'full_validation'/f'step-{plan["steps_target"]:09d}.json')
            if (sha256(final)!=recorded['checkpoint_sha256'] or
                    recorded['dataset_digest']!=read_json(output/'dataset.json')['digest']):
                raise ValueError('Completed output checkpoint/data identity changed')
            return dict(status='complete',step=plan['steps_target'])
        if ending and (output/'full_validation'/f'step-{plan["steps_target"]:09d}.json').exists():
            # No optimizer steps remain. Verify on CPU without rebuilding the
            # training pool, labels, model or optimizer just to finish publishing.
            payload=load_alphazero_checkpoint(terminal_checkpoint)
            saved_config=dict(payload.learner_config);requested_config=asdict(config.learner)
            saved_config.pop('checkpoint_every_steps');requested_config.pop('checkpoint_every_steps')
            if (payload.round_train_step!=plan['steps_target'] or payload.round_steps_target!=plan['steps_target']
                    or payload.round_index!=0 or payload.replay_identity.get('digest')!=identity.get('digest')
                    or saved_config!=requested_config):
                raise ValueError('Terminal checkpoint progress/configuration identity changed')
            return _complete_training(output,plan,terminal_checkpoint,wall_started,[],recovered=True)
        torch.set_num_threads(1)
        archive_targets=archive_lambda_targets(config,output,index) if archived and not ending else None
        drawer=make_batch_drawer(index,pool,config.learner.batch_size) if not ending else None
        prefetch=None
        if config.training_workers and not ending:
            def warmup_progress(row):
                write_json(output/'prefetch_progress.json',row|{'status':'warming'})
                print(f'共享缓存 {row["games"]}/{row["total_games"]}局，{row["arena_bytes"]/2**30:.2f}GiB',flush=True)
            prefetch=SharedReadonlyPrefetch(index,drawer,config.training_workers,
                progress=warmup_progress)
            cleanup.callback(prefetch.close)
            write_json(output/'prefetch_progress.json',dict(status='ready',workers=config.training_workers,
                arena_bytes=prefetch.used_bytes,payload_bytes=index.cache.bytes,
                warmup_seconds=prefetch.warmup_seconds))
        validation_index=copy.copy(index);validation_index.cache=CompactGameCache(256*1024**2)
        targets = lambda_targets(config, output, teacher, index) if teacher is not None else None
        checkpoints.mkdir(exist_ok=True)
        saved=sorted(checkpoints.glob('step-*.pt'));latest=saved[-1] if saved else checkpoints/'initial.pt'
        if latest.exists():
            learner=OutcomeLearner.resume(latest,config.learner,replay_identity=identity,
                round_index=0,round_steps_target=plan['steps_target'])
        else:
            learner=initialize(config,identity,plan['steps_target']);learner.save(latest)
        if targets is not None:
            learner.outcome_targets, learner.outcome_offsets = targets
            learner.outcome_teacher_step = teacher['train_step']
        if archive_targets is not None:
            learner.outcome_targets,learner.outcome_offsets,learner.outcome_target_mask=archive_targets
            learner.outcome_target_source='archive_search'
        session=str(time.time_ns())
        write_json(output/'source'/session/'identity.json',source_identity(Path(__file__).resolve().parents[2],output/'source'/session)|
                   dict(command=sys.argv,config=asdict(config)))
        eval_steps=set(plan['validation_steps']);curve=output/'validation_curve.jsonl'

        def validate(current,path):
            step=current.round_train_step
            write_json(output/'progress.json',dict(status='incomplete',step=step,
                steps_target=plan['steps_target'],checkpoint=str(path)))
            destination=output/'full_validation'/f'step-{step:09d}.json'
            if (step==0 or step in eval_steps) and not destination.exists():
                metrics=evaluate(current,validation_index)
                row=dict(step=step,checkpoint=str(path),checkpoint_sha256=sha256(path),
                    dataset_digest=read_json(output/'dataset.json')['digest'],metrics=metrics)
                write_json(destination,row);append_json(curve,row)
                print(f'完整留出 step={step} KL={metrics["policy_kl"]:.6f} MSE={metrics["value_loss"]:.6f}',flush=True)
            elif destination.exists() and (step==0 or step in eval_steps):
                if read_json(destination)['checkpoint_sha256']!=sha256(path):
                    raise ValueError('Committed validation checkpoint changed')

        validate(learner,latest)
        if learner.round_train_step==plan['steps_target']:
            return _complete_training(output,plan,latest,wall_started,[],recovered=True)
        writer=HybridReferenceWriter(output/'references',session,{'policy':identity,'value':identity},
            (config.learner.batch_size,config.learner.batch_size),('policy','value'))
        cleanup.callback(writer.close)
        limit=plan['steps_target'] if max_steps_this_session is None else min(
            plan['steps_target'],learner.round_train_step+max_steps_this_session)
        summaries=[run_training_round(learner,steps_target=plan['steps_target'],current_pool=pool,
                history_pool=None,validation_samples=[],validation_games=len(index.split_games('validation')),
                checkpoint_dir=checkpoints,metrics_path=output/'training_metrics.jsonl',
                reference_writer=writer,checkpoint_callback=validate,
                checkpoint_steps=sorted(eval_steps),max_steps_this_session=limit-learner.round_train_step,
                batch_drawer=drawer,joint_separate_batches=True,batch_prefetch=prefetch,
                progress_callback=lambda r:print(f'训练 step={r["round_train_step"]} loss={r["total_loss"]:.6f}',flush=True)
                    if r['round_train_step']%100==0 else None)]
        complete=learner.round_train_step==plan['steps_target']
        if complete:
            return _complete_training(output,plan,terminal_checkpoint,wall_started,summaries,recovered=False)
        write_json(output/'progress.json',dict(status='incomplete',
            step=learner.round_train_step,steps_target=plan['steps_target'],training_sessions=summaries,
            elapsed_seconds_this_session_including_cache_targets_validation=time.perf_counter()-wall_started))
        return dict(status='incomplete',step=learner.round_train_step)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    phases=parser.add_mutually_exclusive_group()
    phases.add_argument('--prepare-only',action='store_true')
    phases.add_argument('--select-only',action='store_true')
    phases.add_argument('--generate-targets-only',action='store_true')
    parser.add_argument('--max-steps-this-session',type=int)
    args=parser.parse_args();config=OutcomeOfflineConfig.from_dict(read_json(args.config))
    if args.select_only:
        from .offline_scratch import select_checkpoint
        print(select_checkpoint(config,args.output,input_preparer=prepare_inputs))
    else:
        print(run(config,args.output,prepare_only=args.prepare_only,
            generate_targets_only=args.generate_targets_only,max_steps_this_session=args.max_steps_this_session))


if __name__=='__main__':main()
