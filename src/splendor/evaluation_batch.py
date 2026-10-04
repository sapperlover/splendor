"""Cooperative checked games and model-isolated native inference batches.

One owner advances all games. Each ready search job already has its next leaf,
so waiting for another producer cannot improve supply; tail batches dispatch
immediately. The checked runner coroutine owns adjudication and error scenes.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, replace
from pathlib import Path

import torch

from .native_policy import NativeBatchRunner, NativePUCTSearchPolicy, _default_catalog
from .opponent_pool import load_frozen_actor_critic_checkpoint
from .mcts_policy import MCTSConfig
from .runner import _game_coroutine, _match_from_run
from .search_inference import FrozenSearchInference, InferenceConfig


@dataclass(frozen=True)
class BatchEvaluationConfig:
    backend: str = "native_batched"
    active_games: int = 100
    max_batch_size: int = 64
    candidate_buckets: tuple[int, ...] = (16, 32, 48, 64)
    inference_mode: str = "compiled"
    actor_projection: bool = True
    max_batch_wait_ms: float = 0.0

    def __post_init__(self):
        if self.backend != "native_batched":
            raise ValueError("unknown evaluation execution backend")
        if type(self.active_games) is not int or self.active_games <= 0:
            raise ValueError("active_games must be a positive integer")
        if type(self.max_batch_size) is not int or self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be a positive integer")
        if self.max_batch_wait_ms != 0:
            raise ValueError("cooperative evaluation dispatches all available supply with wait=0")
        InferenceConfig(mode=self.inference_mode, actor_projection=self.actor_projection,
                        candidate_buckets=tuple(self.candidate_buckets), compile_failure_policy="fail_fast")


def evaluate_batched(config, pair_ids, log_root: Path, failure_root: Path, on_pair):
    from .evaluation import (_ParallelPairTask, ParallelPairResult, _append_game_log,
                             _game_log_record, _numeric_policy_stats, _sum_numeric_stats)
    execution = BatchEvaluationConfig(**config.execution)
    torch.set_num_threads(1)
    catalog = _default_catalog()
    evaluators, runners, model_keys = {}, {}, []
    creation_started = time.perf_counter()
    for spec in (config.policy_a, config.policy_b):
        checkpoint_sha = hashlib.sha256(Path(spec.checkpoint).read_bytes()).hexdigest()
        inference = (InferenceConfig(**spec.inference) if spec.inference is not None else
                     InferenceConfig(mode=execution.inference_mode, actor_projection=execution.actor_projection,
                                     candidate_buckets=tuple(execution.candidate_buckets),
                                     compile_failure_policy="fail_fast", native_transport="reused"))
        key = (checkpoint_sha, json.dumps(inference.to_dict(), sort_keys=True))
        model_keys.append(key)
        if key not in evaluators:
            model = load_frozen_actor_critic_checkpoint(Path(spec.checkpoint), device=config.device)
            evaluators[key] = FrozenSearchInference(model, device=config.device, config=inference)
            runners[key] = NativeBatchRunner(evaluators[key], catalog, capacity=execution.max_batch_size,
                                             buckets=tuple(execution.candidate_buckets))
    creation_seconds = time.perf_counter() - creation_started
    todo = deque((pair_id, seat) for pair_id in pair_ids for seat in (0, 1))
    active, pairs = [], defaultdict(list)
    model_costs = {key: {"batch_wall_seconds": 0., "root_start_seconds": 0.,
                         "decision_latency_seconds": [], "width_rows": Counter(), "width_batches": Counter(),
                         "physical_candidate_positions": 0, "actual_candidate_positions": 0}
                   for key in evaluators}
    game_records = []
    rule_seconds = sidecar_seconds = 0.

    def finish(session, run):
        nonlocal sidecar_seconds
        task = session['task']
        match = _match_from_run(run, game_seed=task.game_seed, policy_seeds=session['seeds'],
            policy_names=(config.policy_a.name, config.policy_b.name),
            pair_id=task.pair_id, game_in_pair=session['seat'])
        stats = [_numeric_policy_stats(policy) for policy in session['policies']]
        started = time.perf_counter()
        _append_game_log(task, _game_log_record(match, *stats))
        sidecar_seconds += time.perf_counter() - started
        game_records.append({'pair_id': task.pair_id, 'seat': session['seat'],
                             'wall_seconds': time.perf_counter() - session['started']})
        pairs[task.pair_id].append((match, stats))
        if len(pairs[task.pair_id]) == 2:
            games = sorted(pairs.pop(task.pair_id), key=lambda item: item[0].game_in_pair)
            totals = [{}, {}]
            for _, item_stats in games:
                for i in range(2):
                    _sum_numeric_stats(totals[i], item_stats[i])
            on_pair(ParallelPairResult(tuple(item[0] for item in games), *totals))

    def advance(session, action=None, error=None, initial=False):
        nonlocal rule_seconds
        started = time.perf_counter()
        try:
            if initial:
                actor, request = next(session['game'])
            elif error is not None:
                actor, request = session['game'].throw(error)
            else:
                actor, request = session['game'].send(action)
        except StopIteration as completed:
            rule_seconds += time.perf_counter() - started
            finish(session, completed.value)
            return False
        rule_seconds += time.perf_counter() - started
        policy = session['policies'][actor]
        key = model_keys[actor]
        before = time.perf_counter()
        try:
            session['job'] = policy.start_job(request, add_root_noise=False, temperature=0.)
        except Exception as error:
            model_costs[key]['root_start_seconds'] += time.perf_counter() - before
            return advance(session, error=error)
        model_costs[key]['root_start_seconds'] += time.perf_counter() - before
        session.update(actor=actor, request=request, root_started=before)
        return True

    def create(pair_id, seat):
        task = _ParallelPairTask(config.policy_a, config.policy_b, pair_id, config.game_seeds[pair_id],
            config.policy_seed_base, config.max_completed_turns, str(failure_root),
            config.adjudicate_deterministic_policy_cycles, str(log_root))
        seeds = (config.policy_seed_base + pair_id * 2, config.policy_seed_base + pair_id * 2 + 1)
        policies = []
        for i, spec in enumerate((config.policy_a, config.policy_b)):
            values = dict(spec.search_config)
            values['seed'] = int(values.get('seed', 0)) ^ seeds[i]
            evaluator = evaluators[model_keys[i]]
            policies.append(NativePUCTSearchPolicy(evaluator.model, config=MCTSConfig(**values),
                device=config.device, catalog=catalog, evaluator=evaluator, model_version=model_keys[i][0]))
        game = _game_coroutine(task.game_seed, first_player=seat, policies=policies, policy_seeds=seeds,
            max_completed_turns=config.max_completed_turns, catalog=catalog,
            failure_dir=failure_root / f'pair-{pair_id:06d}-game-{seat}',
            adjudicate_deterministic_policy_cycles=config.adjudicate_deterministic_policy_cycles,
            raise_on_error=False, lean=True)
        session = {'task': task, 'seat': seat, 'seeds': seeds, 'policies': policies,
                   'game': game, 'started': time.perf_counter()}
        return session if advance(session, initial=True) else None

    while todo or active:
        while todo and len(active) < execution.active_games:
            session = create(*todo.popleft())
            if session is not None:
                active.append(session)
        for session in tuple(active):
            if session['job'].done:
                key = model_keys[session['actor']]
                model_costs[key]['decision_latency_seconds'].append(time.perf_counter() - session['root_started'])
                action = session['request'].legal_actions[session['job'].result.chosen_candidate_index]
                if not advance(session, action=action):
                    active.remove(session)
        for key, runner in runners.items():
            sessions = [s for s in active if model_keys[s['actor']] == key and not s['job'].done]
            if not sessions:
                continue
            started = time.perf_counter()
            try:
                runner.run([s['job'] for s in sessions], context={s['job']: s for s in sessions})
            except Exception as error:
                # Preserve attribution; abandon all jobs submitted in this
                # failed batch and classify affected games through the runner.
                report = runner.failure_report(error)
                path = failure_root / f'batch-{runner.batches:09d}-{key[0][:8]}.json'
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({'model_sha256': key[0], 'error': str(error),
                    'scope': report['scope'], 'games': [s['task'].pair_id for s in sessions]}, indent=2))
                for session in sessions:
                    session['job'].cancel()
                    advance(session, error=error)
                    active.remove(session)
            model_costs[key]['batch_wall_seconds'] += time.perf_counter() - started
            for record in runner.batch_records:
                width = 0 if record['state_only'] else record['width']
                model_costs[key]['width_rows'][str(width)] += record['rows']
                model_costs[key]['width_batches'][str(width)] += 1
                if width:
                    model_costs[key]['physical_candidate_positions'] += record['rows'] * width
                    model_costs[key]['actual_candidate_positions'] += int(record['identity'][:, 3].sum())
    result = {'scope': 'this invocation; mutually exclusive owner host intervals',
              'model_creation_seconds': creation_seconds, 'checked_rule_seconds': rule_seconds,
              'sidecar_seconds': sidecar_seconds, 'games': game_records, 'models': {}}
    for key, evaluator in evaluators.items():
        costs = model_costs[key]
        runner = runners[key]
        stats = evaluator.stats.to_dict()
        latencies = costs.pop('decision_latency_seconds')
        import numpy as np
        result['models'][key[0] + ':' + str(model_keys.index(key))] = {
            **costs, 'inference': stats, 'native_fill_seconds': runner.fill_seconds,
            'native_consume_seconds': runner.consume_seconds, 'root_rows': runner.root_rows,
            'policy_rows': runner.policy_rows, 'value_rows': runner.value_rows,
            'decision_latency': {str(p): float(np.percentile(latencies, p)) if latencies else None
                                 for p in (50, 90, 99)},
            'batch_non_inference_seconds': costs['batch_wall_seconds'] - sum(stats[k] for k in
                ('forward_seconds', 'h2d_seconds', 'd2h_seconds', 'wait_seconds'))}
    return result
