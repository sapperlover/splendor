"""Checked paired games using the collector's model-routed CUDA pool."""
from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch

from .mcts_policy import MCTSConfig
from .native_policy import (
    ChanceWideningConfig, NativeChancePUCTSearchPolicy, NativePUCTSearchPolicy, NativeGreedyPolicy,
    _default_catalog,
)
from .opponent_pool import load_frozen_actor_critic_checkpoint
from .replay import to_primitive
from .runner import _game_coroutine, _match_from_run
from .search_inference import FrozenSearchInference, InferenceConfig
from .selfplay_fast import ExecutionConfig, FastSelfPlayConfig


def pool_inference_config(execution):
    """The execution section is authoritative, just as in collection."""
    return InferenceConfig(
        mode=execution.inference_mode, actor_projection=execution.actor_projection,
        candidate_buckets=tuple(execution.candidate_buckets),
        compile_mode=execution.compile_mode,
        compile_failure_policy=execution.compile_failure_policy,
        native_transport=execution.native_transport,
        batch_buckets=tuple(execution.batch_buckets))


def validate_pool_evaluation(config):
    execution = ExecutionConfig(**config.execution)
    if execution.backend != "native_cuda_pool":
        raise ValueError("pooled evaluation requires native_cuda_pool")
    if execution.compile_mode == 'native-cudagraph' and config.device != 'cuda':
        raise ValueError("native-cudagraph evaluation requires device=cuda")
    inference = pool_inference_config(execution)
    for spec in (config.policy_a, config.policy_b):
        if spec.kind != "greedy_checkpoint" and (
                spec.kind != "mcts_checkpoint" or spec.search_backend not in ("native", "native_chance")):
            raise ValueError("native_cuda_pool evaluation requires native MCTS or greedy checkpoint policies")
        # Avoid silently ignoring a policy-specific inference configuration.
        for name, value in (spec.inference or {}).items():
            if json.dumps(value) != json.dumps(inference.to_dict().get(name)):
                raise ValueError(f"policy inference {name} conflicts with pooled execution")
    return execution


@dataclass(frozen=True)
class _EvaluationSource:
    config: object
    model_keys: tuple[str, str]
    log_root: Path
    failure_root: Path

    def __call__(self, _pool_config, game_id, evaluator, _identity):
        return _EvaluationSession(self, game_id, evaluator)


class _EvaluationSession:
    """Collector session protocol; checked runner owns every real transition."""

    def __init__(self, source, game_id, evaluator):
        from .evaluation import _ParallelPairTask
        self.source, self.game_id = source, game_id
        config = source.config
        self.pair_id, self.seat = divmod(game_id, 2)
        self.task = _ParallelPairTask(config.policy_a, config.policy_b, self.pair_id,
            config.game_seeds[self.pair_id], config.policy_seed_base, config.max_completed_turns,
            str(source.failure_root), config.adjudicate_deterministic_policy_cycles,
            str(source.log_root))
        self.seeds = (config.policy_seed_base + self.pair_id * 2,
                      config.policy_seed_base + self.pair_id * 2 + 1)
        catalog = _default_catalog()
        policies = []
        for actor, spec in enumerate((config.policy_a, config.policy_b)):
            if spec.kind == 'greedy_checkpoint':
                policies.append(NativeGreedyPolicy(None, catalog=catalog,
                    device=evaluator.device, evaluator=evaluator, model_version=source.model_keys[actor]))
                continue
            search = dict(spec.search_config)
            search['seed'] = int(search.get('seed', 0)) ^ self.seeds[actor]
            cls = NativeChancePUCTSearchPolicy if spec.search_backend == 'native_chance' else NativePUCTSearchPolicy
            kwargs = ({} if spec.chance_widening is None else
                      {'chance_widening': ChanceWideningConfig(**spec.chance_widening)})
            policies.append(cls(None, config=MCTSConfig(**search), catalog=catalog,
                device=evaluator.device, evaluator=evaluator,
                model_version=source.model_keys[actor], **kwargs))
        self.policies = tuple(policies)
        self.game = _game_coroutine(self.task.game_seed, first_player=self.seat,
            policies=self.policies, policy_seeds=self.seeds,
            max_completed_turns=config.max_completed_turns, catalog=catalog,
            failure_dir=source.failure_root / f'pair-{self.pair_id:06d}-game-{self.seat}',
            adjudicate_deterministic_policy_cycles=config.adjudicate_deterministic_policy_cycles,
            raise_on_error=False, lean=True)
        self.records, self.decision_latencies_ms = [], []
        self.job = self.request = self.run = None
        self.started = time.perf_counter()
        self.rule_seconds = self.root_start_seconds = 0.
        self._advance(initial=True)

    @property
    def finished(self):
        return self.run is not None

    def _advance(self, *, initial=False, action=None, error=None):
        started = time.perf_counter()
        try:
            self.actor, self.request = (next(self.game) if initial else
                self.game.throw(error) if error is not None else self.game.send(action))
        except StopIteration as completed:
            self.run = completed.value
        self.rule_seconds += time.perf_counter() - started

    def ensure_job(self):
        while not self.finished and self.job is None:
            self.job_started = time.perf_counter()
            try:
                self.job = self.policies[self.actor].start_job(
                    self.request, add_root_noise=False, temperature=0.)
            except Exception as error:
                self._advance(error=error)
            self.root_start_seconds += time.perf_counter() - self.job_started
            if self.job is not None and self.job.done:
                self.take_result()

    def take_result(self):
        if self.job is None or not self.job.done:
            raise RuntimeError("evaluation search result is not ready")
        action = self.request.legal_actions[self.job.result.chosen_candidate_index]
        self.decision_latencies_ms.append((time.perf_counter() - self.job_started) * 1000)
        self.records.append(None)
        self.job = None
        self._advance(action=action)

    def commit(self, _root, _fingerprint, _identity):
        from .evaluation import _append_game_log, _game_log_record, _numeric_policy_stats
        config = self.source.config
        match = _match_from_run(self.run, game_seed=self.task.game_seed, policy_seeds=self.seeds,
            policy_names=(config.policy_a.name, config.policy_b.name),
            pair_id=self.pair_id, game_in_pair=self.seat)
        stats = tuple(_numeric_policy_stats(policy) for policy in self.policies)
        started = time.perf_counter()
        _append_game_log(self.task, _game_log_record(match, *stats))
        return {'game_id': self.game_id, 'decisions': match.decision_count,
            'simulations': sum(item.get('simulations', 0) for item in stats),
            'match': match, 'policy_stats': stats,
            'wall_seconds': time.perf_counter() - self.started,
            'checked_rule_seconds': self.rule_seconds,
            'root_start_seconds': self.root_start_seconds,
            'sidecar_seconds': time.perf_counter() - started}

    def failure_scene(self):
        return {'game_id': self.game_id, 'pair_id': self.pair_id, 'seat': self.seat,
            'game_seed': self.task.game_seed, 'policy_seeds': self.seeds,
            'root_decision_index': len(self.records),
            'root_request': to_primitive(self.request),
            'search_context': None if self.job is None else
                to_primitive(self.job.failure_context_for_world(None))}


def evaluate_pooled(config, pair_ids, log_root, failure_root, on_pair):
    from .evaluation import ParallelPairResult, _sum_numeric_stats
    from .multiworker_gpu import collect
    execution = validate_pool_evaluation(config)
    torch.set_num_threads(execution.torch_threads)
    inference = pool_inference_config(execution)
    evaluators, model_keys = {}, []
    started = time.perf_counter()
    for spec in (config.policy_a, config.policy_b):
        sha = hashlib.sha256(Path(spec.checkpoint).read_bytes()).hexdigest()
        model_keys.append(sha)
        if sha not in evaluators:
            model = load_frozen_actor_critic_checkpoint(Path(spec.checkpoint), device=config.device)
            evaluator = FrozenSearchInference(model, device=config.device, config=inference)
            if config.device == 'cpu':
                # Explicit CPU reference execution uses the same packed shared
                # layout; it is not a fallback for failed CUDA inference.
                from .policy_averaging_pool import _CPUOwner
                evaluator = _CPUOwner(evaluator)
            evaluators[sha] = evaluator
    creation_seconds = time.perf_counter() - started
    if config.device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    identity = hashlib.sha256(json.dumps(config.to_dict(), sort_keys=True).encode()).hexdigest()
    pool = FastSelfPlayConfig(protocol='mcts-selfplay-v2', run_id='fixed-evaluation',
        round_index=0, checkpoint=config.policy_a.checkpoint, num_games=len(config.game_seeds) * 2,
        seed=0, search_config=MCTSConfig(**(config.policy_a.search_config or config.policy_b.search_config or {})),
        execution=execution, add_root_noise=False, validation_fraction=0.,
        max_completed_turns=config.max_completed_turns)
    source = _EvaluationSource(config, tuple(model_keys), Path(log_root), Path(failure_root))
    pairs = defaultdict(list)

    def completed(summary):
        match = summary['match']
        pairs[match.pair_id].append(summary)
        if len(pairs[match.pair_id]) == 2:
            games = sorted(pairs.pop(match.pair_id), key=lambda game: game['match'].game_in_pair)
            totals = [{}, {}]
            for game in games:
                for actor in (0, 1):
                    _sum_numeric_stats(totals[actor], game['policy_stats'][actor])
            on_pair(ParallelPairResult(tuple(game['match'] for game in games), *totals))

    results, owner = collect(pool, Path(failure_root),
        tuple(pair_id * 2 + seat for pair_id in pair_ids for seat in (0, 1)), identity, identity,
        model_evaluators=evaluators, session_factory=source, completion_callback=completed)
    games = [game for worker in results for game in worker['completed']]
    return {**owner, 'backend': 'native_cuda_pool', 'model_creation_seconds': creation_seconds,
        'scope': 'this invocation; rule/sidecar times summed across search workers',
        'models': {sha: {'inference': evaluator.stats.to_dict()} for sha, evaluator in evaluators.items()},
        **{key: sum(game[key] for game in games) for key in
            ('checked_rule_seconds', 'root_start_seconds', 'sidecar_seconds')},
        'games': [{key: game[key] for key in ('game_id', 'wall_seconds', 'simulations', 'decisions')}
                  for game in sorted(games, key=lambda item: item['game_id'])]}
