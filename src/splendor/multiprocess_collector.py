"""Spawn-based CPU environment workers with selectable opponent inference.

The workers own authoritative game engines and perform all rule, observation,
legal-action, and feature-encoding work.  The legacy backend proxies neural
opponent requests to the parent.  The worker-local backend loads historical
CPU models once and receives one frozen latest snapshot per rollout, avoiding
all action-level neural-opponent IPC.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import resource
import time
import traceback
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np
import torch

from .actor_critic import ActorCritic
from .actor_critic import ActorCriticConfig
from .actions import Action, ClaimNoble, Pay, ReturnTokens
from .contracts import DecisionRequest, Policy
from .features import EncodedDecision, collate_encoded_decisions, encode_decision
from .heuristic_policy import DevelopmentPolicy, ScorePolicy
from .neural_policy import NeuralInferenceTiming, SamplingActorCriticPolicy
from .opponent_pool import load_frozen_actor_critic_checkpoint
from .ppo import PolicySampler, RolloutBatch, RolloutSample, group_rollout_by_episode
from .random_policy import RandomPolicy
from .replay import ReplayRecord, to_primitive
from .result import TerminalResult
from .rng import DeterministicRng, RngState
from .training_env import LearningDecision, LearningEnvironment, LearningTransition


MULTIPROCESS_COLLECTOR_FORMAT_VERSION = 1
OpponentInferenceBackend = Literal["central", "worker_cpu"]
_WORKER_RNG_OFFSET = 6_700_000_000_000_000
# Exact two-player upper bounds from the frozen rules: 45 combined main
# choices; at most C(10, 5)=252 gold allocations; C(8, 5)=56 ways to return
# three among six token colors; and three nobles in play.
_MAIN_ACTION_CANDIDATE_CAPACITY = 45
_PAYMENT_CANDIDATE_CAPACITY = 252
_RETURN_CANDIDATE_CAPACITY = 56
_NOBLE_CANDIDATE_CAPACITY = 3


@dataclass(frozen=True, slots=True)
class MultiprocessEpisodeSpec:
    """Serializable instructions for constructing one worker-owned episode."""

    episode_index: int
    episode_id: str
    game_seed: int
    first_player: int
    learning_player: int
    learning_policy_seed: int
    opponent_seed: int
    opponent_kind: str
    opponent_id: str
    max_completed_turns: int
    failure_dir: str

    def __post_init__(self) -> None:
        if self.episode_index < 0:
            raise ValueError("episode_index must be non-negative")
        if self.learning_player not in (0, 1) or self.first_player not in (0, 1):
            raise ValueError("player seats must be 0 or 1")
        if self.opponent_kind not in (
            "random",
            "score",
            "development",
            "latest",
            "historical",
        ):
            raise ValueError(f"unsupported opponent kind: {self.opponent_kind}")
        if not self.episode_id or not self.opponent_id:
            raise ValueError("episode and opponent ids must not be empty")


@dataclass(frozen=True, slots=True)
class MultiprocessEpisodeLaunch:
    """Parent launch record for either central or worker-local inference."""

    spec: MultiprocessEpisodeSpec
    neural_opponent_model: ActorCritic | None = None

    def __post_init__(self) -> None:
        neural = self.spec.opponent_kind in ("latest", "historical")
        if not neural and self.neural_opponent_model is not None:
            raise ValueError("fixed opponent specs cannot carry a neural model")


@dataclass(frozen=True, slots=True)
class WorkerHistoricalModel:
    opponent_id: str
    checkpoint_path: str

    def __post_init__(self) -> None:
        if not self.opponent_id:
            raise ValueError("worker historical opponent id must not be empty")
        if not self.checkpoint_path:
            raise ValueError("worker historical checkpoint path must not be empty")


@dataclass(frozen=True, slots=True)
class MultiprocessTransition:
    """IPC-minimal learner transition; omits the parent-owned old decision."""

    episode_id: str
    learning_player: int
    chosen_candidate_index: int
    chosen_action: Action
    reward: float
    terminated: bool
    truncated: bool
    next_decision: LearningDecision | None
    terminal_result: TerminalResult | None
    state_digest_after: str
    completed_turns_after: tuple[int, int]


@dataclass(frozen=True, slots=True)
class CompletedLearningEpisode:
    """Picklable completed-episode view used by the existing metric code."""

    episode_id: str
    learning_player: int
    transitions: tuple[MultiprocessTransition, ...]
    terminal_result: TerminalResult | None
    terminated: bool
    truncated: bool
    opponent_decision_count: int
    replay_record: ReplayRecord
    final_state: dict[str, Any]


@dataclass(frozen=True, slots=True)
class WorkerStatistics:
    worker_id: int
    pid: int
    episodes_started: int
    learner_steps: int
    environment_encoding_seconds: float
    neural_opponent_wait_seconds: float
    neural_opponent_inference_seconds: float
    neural_opponent_decisions: int
    neural_opponent_inference_calls: int
    opponent_model_sync_seconds: float
    opponent_model_sync_count: int
    opponent_model_count: int
    torch_num_threads: int
    cpu_seconds: float
    max_rss_mib: float
    rng: RngState


@dataclass(frozen=True, slots=True)
class MultiprocessCollectorSnapshot:
    format_version: int
    num_workers: int
    envs_per_worker: int
    next_episode_index: int
    active_episode_count: int
    worker_statistics: tuple[WorkerStatistics, ...]


@dataclass(frozen=True, slots=True)
class _StartEpisode:
    slot_id: int
    spec: MultiprocessEpisodeSpec


@dataclass(frozen=True, slots=True)
class _StepLearner:
    slot_id: int
    decision_id: int
    candidate_index: int


@dataclass(frozen=True, slots=True)
class _OpponentAction:
    request_id: int
    candidate_index: int


@dataclass(frozen=True, slots=True)
class _SnapshotWorker:
    request_id: int


@dataclass(frozen=True, slots=True)
class _CloseWorker:
    pass


@dataclass(frozen=True, slots=True)
class _SyncWorkerOpponents:
    request_id: int
    latest_opponent_id: str
    latest_model_config: dict[str, object]
    latest_state_dict: dict[str, torch.Tensor]
    historical_models: tuple[WorkerHistoricalModel, ...]


@dataclass(frozen=True, slots=True)
class _EpisodeReady:
    worker_id: int
    slot_id: int
    decision: LearningDecision


@dataclass(frozen=True, slots=True)
class _TransitionReady:
    worker_id: int
    slot_id: int
    transition: MultiprocessTransition
    completed: CompletedLearningEpisode | None


@dataclass(frozen=True, slots=True)
class _NeuralOpponentRequest:
    worker_id: int
    slot_id: int
    request_id: int
    opponent_id: str
    encoded: EncodedDecision


@dataclass(frozen=True, slots=True)
class _WorkerSnapshotReady:
    worker_id: int
    request_id: int
    statistics: WorkerStatistics


@dataclass(frozen=True, slots=True)
class _WorkerInitialized:
    worker_id: int
    historical_model_count: int


@dataclass(frozen=True, slots=True)
class _WorkerOpponentsReady:
    worker_id: int
    request_id: int
    model_count: int


@dataclass(frozen=True, slots=True)
class _WorkerFailure:
    worker_id: int
    slot_id: int | None
    error_type: str
    detail: str
    traceback_text: str


@dataclass(slots=True)
class _WorkerRuntime:
    worker_id: int
    output_queue: Any
    opponent_response_queue: Any
    request_counter: int = 0
    episodes_started: int = 0
    learner_steps: int = 0
    environment_encoding_seconds: float = 0.0
    neural_opponent_wait_seconds: float = 0.0
    opponent_timing: NeuralInferenceTiming = field(default_factory=NeuralInferenceTiming)
    opponent_model_sync_seconds: float = 0.0
    opponent_model_sync_count: int = 0
    cpu_started: float = 0.0

    def statistics(
        self,
        rng: DeterministicRng,
        *,
        opponent_model_count: int,
    ) -> WorkerStatistics:
        return WorkerStatistics(
            worker_id=self.worker_id,
            pid=os.getpid(),
            episodes_started=self.episodes_started,
            learner_steps=self.learner_steps,
            environment_encoding_seconds=self.environment_encoding_seconds,
            neural_opponent_wait_seconds=self.neural_opponent_wait_seconds,
            neural_opponent_inference_seconds=self.opponent_timing.seconds,
            neural_opponent_decisions=self.opponent_timing.decisions,
            neural_opponent_inference_calls=self.opponent_timing.calls,
            opponent_model_sync_seconds=self.opponent_model_sync_seconds,
            opponent_model_sync_count=self.opponent_model_sync_count,
            opponent_model_count=opponent_model_count,
            torch_num_threads=torch.get_num_threads(),
            cpu_seconds=time.process_time() - self.cpu_started,
            max_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
            rng=rng.snapshot(),
        )


class _RemoteNeuralOpponentPolicy:
    """Worker-side Policy boundary; no model or CUDA object exists here."""

    def __init__(
        self,
        runtime: _WorkerRuntime,
        *,
        slot_id: int,
        opponent_id: str,
    ) -> None:
        self._runtime = runtime
        self._slot_id = slot_id
        self._opponent_id = opponent_id

    def choose(self, request: DecisionRequest):
        encoded = encode_decision(request)
        request_id = self._runtime.request_counter
        self._runtime.request_counter += 1
        self._runtime.output_queue.put(
            _NeuralOpponentRequest(
                worker_id=self._runtime.worker_id,
                slot_id=self._slot_id,
                request_id=request_id,
                opponent_id=self._opponent_id,
                encoded=encoded,
            )
        )
        wait_started = time.perf_counter()
        response = self._runtime.opponent_response_queue.get()
        self._runtime.neural_opponent_wait_seconds += time.perf_counter() - wait_started
        if not isinstance(response, _OpponentAction) or response.request_id != request_id:
            raise RuntimeError("neural-opponent response does not match its request")
        if not 0 <= response.candidate_index < len(request.legal_actions):
            raise RuntimeError("central neural opponent returned an invalid candidate index")
        if encoded.candidate_actions != request.legal_actions:
            raise RuntimeError("worker candidate handles changed during neural inference")
        return request.legal_actions[response.candidate_index]


def _fixed_opponent(kind: str, seed: int) -> Policy:
    if kind == "random":
        return RandomPolicy(seed)
    if kind == "score":
        return ScorePolicy()
    if kind == "development":
        return DevelopmentPolicy()
    raise ValueError(f"{kind!r} is not a fixed opponent")


def _compact_transition(transition: LearningTransition) -> MultiprocessTransition:
    return MultiprocessTransition(
        episode_id=transition.episode_id,
        learning_player=transition.learning_player,
        chosen_candidate_index=transition.chosen_candidate_index,
        chosen_action=transition.chosen_action,
        reward=transition.reward,
        terminated=transition.terminated,
        truncated=transition.truncated,
        next_decision=transition.next_decision,
        terminal_result=transition.terminal_result,
        state_digest_after=transition.state_digest_after,
        completed_turns_after=transition.completed_turns_after,
    )


def _completed_episode(environment: LearningEnvironment) -> CompletedLearningEpisode:
    state = environment._engine._debug_state()
    return CompletedLearningEpisode(
        episode_id=environment.episode_id,
        learning_player=environment.learning_player,
        transitions=tuple(
            _compact_transition(transition) for transition in environment.transitions
        ),
        terminal_result=environment.terminal_result,
        terminated=environment.terminated,
        truncated=environment.truncated,
        opponent_decision_count=environment.opponent_decision_count,
        replay_record=environment.replay_record,
        final_state=to_primitive(state),
    )


def _worker_main(
    worker_id: int,
    command_queue: Any,
    opponent_response_queue: Any,
    output_queue: Any,
    worker_seed: int,
    restored_rng: RngState | None,
    opponent_inference_backend: OpponentInferenceBackend,
    initial_historical_models: tuple[WorkerHistoricalModel, ...],
) -> None:
    # Importing torch is unavoidable because encoded/transition dataclasses use
    # tensors elsewhere, but workers must never initialize or address CUDA.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # A spawned interpreter normally permits this exactly once.
        pass
    runtime = _WorkerRuntime(
        worker_id=worker_id,
        output_queue=output_queue,
        opponent_response_queue=opponent_response_queue,
        cpu_started=time.process_time(),
    )
    worker_rng = (
        DeterministicRng.restore(restored_rng)
        if restored_rng is not None
        else DeterministicRng(worker_seed)
    )
    environments: dict[int, LearningEnvironment] = {}
    opponent_ids_by_slot: dict[int, str] = {}
    opponent_models: dict[str, ActorCritic] = {}
    historical_model_ids: set[str] = set()
    current_latest_id: str | None = None
    current_slot: int | None = None
    try:
        if opponent_inference_backend == "worker_cpu":
            for item in initial_historical_models:
                opponent_models[item.opponent_id] = (
                    load_frozen_actor_critic_checkpoint(
                        Path(item.checkpoint_path), device="cpu"
                    )
                )
                historical_model_ids.add(item.opponent_id)
        output_queue.put(
            _WorkerInitialized(worker_id, len(historical_model_ids))
        )
        while True:
            command = command_queue.get()
            if isinstance(command, _CloseWorker):
                return
            if isinstance(command, _SyncWorkerOpponents):
                if opponent_inference_backend != "worker_cpu":
                    raise RuntimeError(
                        "central-inference worker received local model synchronization"
                    )
                sync_started = time.perf_counter()
                desired_historical = {
                    item.opponent_id: item for item in command.historical_models
                }
                for opponent_id, item in desired_historical.items():
                    if opponent_id not in opponent_models:
                        opponent_models[opponent_id] = (
                            load_frozen_actor_critic_checkpoint(
                                Path(item.checkpoint_path), device="cpu"
                            )
                        )
                historical_model_ids = set(desired_historical)
                latest = ActorCritic(
                    ActorCriticConfig(**command.latest_model_config)
                )
                latest.load_state_dict(command.latest_state_dict, strict=True)
                latest.eval().requires_grad_(False)
                opponent_models[command.latest_opponent_id] = latest
                current_latest_id = command.latest_opponent_id

                active_ids = set(opponent_ids_by_slot.values())
                retained = historical_model_ids | active_ids | {current_latest_id}
                for opponent_id in tuple(opponent_models):
                    if opponent_id not in retained:
                        opponent_models.pop(opponent_id)
                runtime.opponent_model_sync_seconds += (
                    time.perf_counter() - sync_started
                )
                runtime.opponent_model_sync_count += 1
                output_queue.put(
                    _WorkerOpponentsReady(
                        worker_id,
                        command.request_id,
                        len(opponent_models),
                    )
                )
                continue
            if isinstance(command, _SnapshotWorker):
                output_queue.put(
                    _WorkerSnapshotReady(
                        worker_id,
                        command.request_id,
                        runtime.statistics(
                            worker_rng,
                            opponent_model_count=len(opponent_models),
                        ),
                    )
                )
                continue
            if isinstance(command, _StartEpisode):
                current_slot = command.slot_id
                if current_slot in environments:
                    raise RuntimeError("worker received a duplicate active slot")
                spec = command.spec
                # A worker-local deterministic stream makes scheduling state
                # explicit in checkpoints without influencing game randomness.
                worker_rng.randbelow(2**31)
                policy: Policy
                if spec.opponent_kind in ("latest", "historical"):
                    if opponent_inference_backend == "worker_cpu":
                        model = opponent_models.get(spec.opponent_id)
                        if model is None:
                            raise RuntimeError(
                                "worker has no synchronized model for neural opponent "
                                f"{spec.opponent_id!r}"
                            )
                        policy = SamplingActorCriticPolicy(
                            model,
                            seed=spec.opponent_seed,
                            device="cpu",
                            timing=runtime.opponent_timing,
                        )
                        opponent_ids_by_slot[current_slot] = spec.opponent_id
                    else:
                        policy = _RemoteNeuralOpponentPolicy(
                            runtime,
                            slot_id=current_slot,
                            opponent_id=spec.opponent_id,
                        )
                else:
                    policy = _fixed_opponent(spec.opponent_kind, spec.opponent_seed)
                started = time.perf_counter()
                wait_before = runtime.neural_opponent_wait_seconds
                inference_before = runtime.opponent_timing.seconds
                environment = LearningEnvironment.new(
                    spec.game_seed,
                    first_player=spec.first_player,
                    learning_player=spec.learning_player,
                    opponent_policy=policy,
                    episode_id=spec.episode_id,
                    max_completed_turns=spec.max_completed_turns,
                    failure_dir=Path(spec.failure_dir),
                )
                runtime.environment_encoding_seconds += max(
                    0.0,
                    time.perf_counter() - started
                    - (runtime.neural_opponent_wait_seconds - wait_before)
                    - (runtime.opponent_timing.seconds - inference_before),
                )
                runtime.episodes_started += 1
                environments[current_slot] = environment
                if environment.done:
                    completed = _completed_episode(environment)
                    environments.pop(current_slot)
                    opponent_ids_by_slot.pop(current_slot, None)
                    raise RuntimeError(
                        f"episode {completed.episode_id} ended without a learner decision"
                    )
                output_queue.put(
                    _EpisodeReady(worker_id, current_slot, environment.current_decision())
                )
                current_slot = None
                continue
            if isinstance(command, _StepLearner):
                current_slot = command.slot_id
                environment = environments.get(current_slot)
                if environment is None:
                    raise RuntimeError("worker received a step for an inactive slot")
                started = time.perf_counter()
                wait_before = runtime.neural_opponent_wait_seconds
                inference_before = runtime.opponent_timing.seconds
                transition = environment.step_learning_action(
                    command.candidate_index,
                    decision_id=command.decision_id,
                )
                runtime.environment_encoding_seconds += max(
                    0.0,
                    time.perf_counter() - started
                    - (runtime.neural_opponent_wait_seconds - wait_before)
                    - (runtime.opponent_timing.seconds - inference_before),
                )
                runtime.learner_steps += 1
                completed = None
                if transition.terminated or transition.truncated:
                    completed = _completed_episode(environment)
                    environments.pop(current_slot)
                    opponent_ids_by_slot.pop(current_slot, None)
                output_queue.put(
                    _TransitionReady(
                        worker_id,
                        current_slot,
                        _compact_transition(transition),
                        completed,
                    )
                )
                current_slot = None
                continue
            raise RuntimeError(f"unknown worker command: {type(command).__name__}")
    except BaseException as error:
        output_queue.put(
            _WorkerFailure(
                worker_id=worker_id,
                slot_id=current_slot,
                error_type=type(error).__name__,
                detail=str(error),
                traceback_text=traceback.format_exc(),
            )
        )


MultiprocessEnvironmentFactory = Callable[[int], MultiprocessEpisodeLaunch]


@dataclass(slots=True)
class _CentralSlot:
    launch: MultiprocessEpisodeLaunch
    worker_id: int
    learner_rng: torch.Generator
    opponent_rng: torch.Generator | None
    current_decision: LearningDecision | None = None


class MultiprocessRolloutCollector:
    """Parallel CPU environments with central or worker-local opponents."""

    def __init__(
        self,
        sampler: PolicySampler,
        environment_factory: MultiprocessEnvironmentFactory,
        *,
        num_workers: int,
        envs_per_worker: int,
        master_seed: int,
        start_episode_index: int = 0,
        restored_worker_rng_states: Sequence[RngState] | None = None,
        opponent_inference_backend: OpponentInferenceBackend = "central",
        worker_historical_models: Mapping[str, str | Path] | None = None,
    ) -> None:
        if num_workers <= 0 or envs_per_worker <= 0:
            raise ValueError("num_workers and envs_per_worker must be positive")
        if start_episode_index < 0:
            raise ValueError("start_episode_index must be non-negative")
        if restored_worker_rng_states is not None and len(restored_worker_rng_states) != num_workers:
            raise ValueError("one restored RNG state is required per worker")
        if opponent_inference_backend not in ("central", "worker_cpu"):
            raise ValueError("unsupported opponent inference backend")
        if opponent_inference_backend == "central" and worker_historical_models:
            raise ValueError(
                "central opponent inference cannot preload worker models"
            )
        self.sampler = sampler
        self.environment_factory = environment_factory
        self.num_workers = num_workers
        self.envs_per_worker = envs_per_worker
        self.active_envs = num_workers * envs_per_worker
        self.master_seed = master_seed
        self.opponent_inference_backend = opponent_inference_backend
        self._next_episode_index = start_episode_index
        self._slots: dict[int, _CentralSlot] = {}
        self._completed_environments: list[CompletedLearningEpisode] = []
        self._pending_events: deque[Any] = deque()
        self._closed = False
        self._snapshot_request_id = 0
        self._opponent_sync_request_id = 0
        self.ipc_wait_seconds = 0.0
        self.opponent_inference_seconds = 0.0
        self.opponent_inference_calls = 0
        self.opponent_inference_items = 0
        self.opponent_model_sync_seconds = 0.0
        self._latest_worker_statistics: tuple[WorkerStatistics, ...] = ()
        self._worker_timing_baseline: tuple[WorkerStatistics, ...] = ()
        historical = tuple(
            WorkerHistoricalModel(opponent_id, str(Path(path).resolve()))
            for opponent_id, path in sorted(
                (worker_historical_models or {}).items()
            )
        )
        self._worker_historical_models = {
            item.opponent_id: item for item in historical
        }
        self._worker_model_ids = set(self._worker_historical_models)

        self._context = mp.get_context("spawn")
        self._output_queue = self._context.Queue()
        self._command_queues = [self._context.Queue() for _ in range(num_workers)]
        self._opponent_response_queues = [
            self._context.Queue() for _ in range(num_workers)
        ]
        restored = (
            tuple(restored_worker_rng_states)
            if restored_worker_rng_states is not None
            else (None,) * num_workers
        )
        self._workers = []
        for worker_id in range(num_workers):
            process = self._context.Process(
                target=_worker_main,
                args=(
                    worker_id,
                    self._command_queues[worker_id],
                    self._opponent_response_queues[worker_id],
                    self._output_queue,
                    _WORKER_RNG_OFFSET + master_seed * 1_000 + worker_id,
                    restored[worker_id],
                    opponent_inference_backend,
                    historical,
                ),
                name=f"splendor-env-{worker_id}",
            )
            process.start()
            self._workers.append(process)
        initialized: set[int] = set()
        while len(initialized) < self.num_workers:
            event = self._get_event()
            if not isinstance(event, _WorkerInitialized):
                raise RuntimeError(
                    "unexpected event during worker initialization: "
                    f"{type(event).__name__}"
                )
            initialized.add(event.worker_id)
            if event.historical_model_count != len(historical):
                raise RuntimeError("worker did not load every historical model")

    @property
    def start_method(self) -> str:
        return self._context.get_start_method()

    @property
    def next_episode_index(self) -> int:
        return self._next_episode_index

    @property
    def active_environment_count(self) -> int:
        return len(self._slots)

    @property
    def active_policy_rng_states(self) -> tuple[tuple[str, torch.Tensor], ...]:
        return tuple(
            (
                slot.launch.spec.episode_id,
                slot.learner_rng.get_state().clone(),
            )
            for _, slot in sorted(self._slots.items())
        )

    @property
    def mean_inference_batch_size(self) -> float:
        return self.sampler.mean_inference_batch_size

    @property
    def mean_opponent_inference_batch_size(self) -> float:
        return (
            self.opponent_inference_items / self.opponent_inference_calls
            if self.opponent_inference_calls
            else 0.0
        )

    @property
    def worker_local_opponents(self) -> bool:
        return self.opponent_inference_backend == "worker_cpu"

    @property
    def worker_pids(self) -> tuple[int, ...]:
        return tuple(process.pid or -1 for process in self._workers)

    def reset_timing(self) -> None:
        self.ipc_wait_seconds = 0.0
        self.opponent_inference_seconds = 0.0
        self.opponent_inference_calls = 0
        self.opponent_inference_items = 0
        self.opponent_model_sync_seconds = 0.0
        if self.worker_local_opponents:
            if not self._latest_worker_statistics:
                self._latest_worker_statistics = self.worker_statistics()
            self._worker_timing_baseline = self._latest_worker_statistics

    def sync_worker_opponents(
        self,
        *,
        latest_opponent_id: str,
        latest_model: ActorCritic,
        historical_models: Mapping[str, str | Path],
    ) -> None:
        """Synchronize one rollout's worker-local frozen opponent registry."""

        self._ensure_open()
        if not self.worker_local_opponents:
            raise ValueError(
                "worker opponent synchronization requires worker_cpu inference"
            )
        if not latest_opponent_id:
            raise ValueError("latest opponent id must not be empty")
        if any(slot.current_decision is None for slot in self._slots.values()):
            raise ValueError(
                "worker opponent synchronization requires learner boundaries"
            )
        historical = tuple(
            WorkerHistoricalModel(opponent_id, str(Path(path).resolve()))
            for opponent_id, path in sorted(historical_models.items())
        )
        state_dict = {
            name: tensor.detach().cpu().clone()
            for name, tensor in latest_model.state_dict().items()
        }
        request_id = self._opponent_sync_request_id
        self._opponent_sync_request_id += 1
        command = _SyncWorkerOpponents(
            request_id=request_id,
            latest_opponent_id=latest_opponent_id,
            latest_model_config=asdict(latest_model.config),
            latest_state_dict=state_dict,
            historical_models=historical,
        )
        started = time.perf_counter()
        for command_queue in self._command_queues:
            command_queue.put(command)
        ready: set[int] = set()
        while len(ready) < self.num_workers:
            event = self._get_event()
            if (
                isinstance(event, _WorkerOpponentsReady)
                and event.request_id == request_id
            ):
                ready.add(event.worker_id)
            else:
                raise RuntimeError(
                    "unexpected event during worker model synchronization: "
                    f"{type(event).__name__}"
                )
        self.opponent_model_sync_seconds += time.perf_counter() - started
        self._worker_historical_models = {
            item.opponent_id: item for item in historical
        }
        self._worker_model_ids = {
            latest_opponent_id,
            *(item.opponent_id for item in historical),
        }

    def take_completed_environments(self) -> tuple[CompletedLearningEpisode, ...]:
        completed = tuple(self._completed_environments)
        self._completed_environments.clear()
        return completed

    def collect(self, steps: int) -> RolloutBatch:
        if steps <= 0:
            raise ValueError("rollout steps must be positive")
        self._ensure_open()
        samples: list[RolloutSample] = []
        while len(samples) < steps:
            self._fill_slots()
            remaining = steps - len(samples)
            ready = tuple(
                slot
                for _, slot in sorted(self._slots.items())
                if slot.current_decision is not None
            )
            selected = ready[:remaining]
            if not selected:
                raise RuntimeError("multiprocess collector has no ready learner decision")
            samples.extend(self._step(selected))
        return group_rollout_by_episode(samples)

    def drain_active_environments(self) -> RolloutBatch:
        self._ensure_open()
        samples: list[RolloutSample] = []
        while self._slots:
            ready = tuple(
                slot
                for _, slot in sorted(self._slots.items())
                if slot.current_decision is not None
            )
            if not ready:
                raise RuntimeError("multiprocess drain has no ready learner decision")
            samples.extend(self._step(ready))
        return group_rollout_by_episode(samples)

    def snapshot(self) -> MultiprocessCollectorSnapshot:
        """Snapshot worker RNG/statistics at the required episode boundary."""

        self._ensure_open()
        if self._slots:
            raise ValueError("multiprocess collector snapshot requires an episode boundary")
        request_id = self._snapshot_request_id
        self._snapshot_request_id += 1
        for command_queue in self._command_queues:
            command_queue.put(_SnapshotWorker(request_id))
        replies: dict[int, WorkerStatistics] = {}
        while len(replies) < self.num_workers:
            event = self._get_event()
            if isinstance(event, _NeuralOpponentRequest):
                self._serve_opponent_requests((event,))
            elif isinstance(event, _WorkerSnapshotReady) and event.request_id == request_id:
                replies[event.worker_id] = event.statistics
            else:
                raise RuntimeError(
                    f"unexpected event while snapshotting: {type(event).__name__}"
                )
        return MultiprocessCollectorSnapshot(
            format_version=MULTIPROCESS_COLLECTOR_FORMAT_VERSION,
            num_workers=self.num_workers,
            envs_per_worker=self.envs_per_worker,
            next_episode_index=self.next_episode_index,
            active_episode_count=0,
            worker_statistics=tuple(replies[index] for index in range(self.num_workers)),
        )

    def worker_statistics(self) -> tuple[WorkerStatistics, ...]:
        if self._slots:
            # Statistics commands are safe with active but currently idle slots.
            if any(slot.current_decision is None for slot in self._slots.values()):
                raise ValueError("cannot snapshot workers during an in-flight step")
        request_id = self._snapshot_request_id
        self._snapshot_request_id += 1
        for command_queue in self._command_queues:
            command_queue.put(_SnapshotWorker(request_id))
        replies: dict[int, WorkerStatistics] = {}
        while len(replies) < self.num_workers:
            event = self._get_event()
            if isinstance(event, _WorkerSnapshotReady) and event.request_id == request_id:
                replies[event.worker_id] = event.statistics
            elif isinstance(event, _NeuralOpponentRequest):
                self._serve_opponent_requests((event,))
            else:
                self._pending_events.append(event)
        result = tuple(replies[index] for index in range(self.num_workers))
        self._latest_worker_statistics = result
        if self.worker_local_opponents and self._worker_timing_baseline:
            self.opponent_inference_seconds = sum(
                after.neural_opponent_inference_seconds
                - before.neural_opponent_inference_seconds
                for before, after in zip(
                    self._worker_timing_baseline, result, strict=True
                )
            )
            self.opponent_inference_items = sum(
                after.neural_opponent_decisions
                - before.neural_opponent_decisions
                for before, after in zip(
                    self._worker_timing_baseline, result, strict=True
                )
            )
            self.opponent_inference_calls = sum(
                after.neural_opponent_inference_calls
                - before.neural_opponent_inference_calls
                for before, after in zip(
                    self._worker_timing_baseline, result, strict=True
                )
            )
        return result

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for command_queue in self._command_queues:
            command_queue.put(_CloseWorker())
        for process in self._workers:
            process.join(timeout=5.0)
        for process in self._workers:
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)
        for channel in (
            *self._command_queues,
            *self._opponent_response_queues,
            self._output_queue,
        ):
            channel.close()
            channel.join_thread()

    def __enter__(self) -> MultiprocessRolloutCollector:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("multiprocess collector is closed")
        failed = [process for process in self._workers if process.exitcode not in (None, 0)]
        if failed:
            raise RuntimeError(
                "environment worker exited unexpectedly: "
                + ", ".join(f"pid={p.pid} exit={p.exitcode}" for p in failed)
            )

    def _worker_loads(self) -> list[int]:
        loads = [0] * self.num_workers
        for slot in self._slots.values():
            loads[slot.worker_id] += 1
        return loads

    def _fill_slots(self) -> None:
        started_slots: list[int] = []
        loads = self._worker_loads()
        while len(self._slots) < self.active_envs:
            preferred = self._next_episode_index % self.num_workers
            worker_id = next(
                (
                    (preferred + offset) % self.num_workers
                    for offset in range(self.num_workers)
                    if loads[(preferred + offset) % self.num_workers]
                    < self.envs_per_worker
                ),
                None,
            )
            if worker_id is None:
                raise AssertionError("worker capacities disagree with active slot count")
            launch = self.environment_factory(self._next_episode_index)
            if launch.spec.episode_index != self._next_episode_index:
                raise ValueError("environment factory returned the wrong episode index")
            neural = launch.spec.opponent_kind in ("latest", "historical")
            if self.worker_local_opponents:
                if launch.neural_opponent_model is not None:
                    raise ValueError(
                        "worker-local opponent launch must not carry a parent model"
                    )
                if neural and launch.spec.opponent_id not in self._worker_model_ids:
                    raise ValueError(
                        "worker-local opponent was not synchronized: "
                        f"{launch.spec.opponent_id}"
                    )
            elif neural != (launch.neural_opponent_model is not None):
                raise ValueError(
                    "central neural opponent launch requires exactly one model"
                )
            slot_id = self._next_episode_index
            self._next_episode_index += 1
            learner_rng = torch.Generator(device="cpu")
            learner_rng.manual_seed(launch.spec.learning_policy_seed)
            opponent_rng = None
            if (
                not self.worker_local_opponents
                and launch.neural_opponent_model is not None
            ):
                opponent_rng = torch.Generator(device="cpu")
                opponent_rng.manual_seed(launch.spec.opponent_seed)
            self._slots[slot_id] = _CentralSlot(
                launch=launch,
                worker_id=worker_id,
                learner_rng=learner_rng,
                opponent_rng=opponent_rng,
            )
            loads[worker_id] += 1
            started_slots.append(slot_id)
            self._command_queues[worker_id].put(_StartEpisode(slot_id, launch.spec))
        ready: dict[int, LearningDecision] = {}
        while len(ready) < len(started_slots):
            event = self._get_event()
            if isinstance(event, _NeuralOpponentRequest):
                requests = self._gather_immediate_opponent_requests(event)
                self._serve_opponent_requests(requests)
            elif isinstance(event, _EpisodeReady) and event.slot_id in started_slots:
                ready[event.slot_id] = event.decision
            else:
                raise RuntimeError(
                    f"unexpected event while filling slots: {type(event).__name__}"
                )
        for slot_id, decision in ready.items():
            self._slots[slot_id].current_decision = decision

    def _step(self, selected: Sequence[_CentralSlot]) -> list[RolloutSample]:
        ordered = tuple(sorted(selected, key=lambda slot: slot.launch.spec.episode_index))
        decisions = tuple(slot.current_decision for slot in ordered)
        if any(decision is None for decision in decisions):
            raise RuntimeError("selected slot has no current learner decision")
        concrete = tuple(decision for decision in decisions if decision is not None)
        policy_samples = self.sampler.sample_batch(
            tuple(decision.encoded for decision in concrete),
            generators=tuple(slot.learner_rng for slot in ordered),
        )
        expected_slots: set[int] = set()
        for slot, decision, policy_sample in zip(
            ordered, concrete, policy_samples, strict=True
        ):
            slot_id = slot.launch.spec.episode_index
            expected_slots.add(slot_id)
            slot.current_decision = None
            self._command_queues[slot.worker_id].put(
                _StepLearner(
                    slot_id,
                    decision.decision_id,
                    policy_sample.chosen_candidate_index,
                )
            )

        transitions: dict[int, _TransitionReady] = {}
        while transitions.keys() != expected_slots:
            event = self._get_event()
            if isinstance(event, _NeuralOpponentRequest):
                requests = self._gather_immediate_opponent_requests(event)
                self._serve_opponent_requests(requests)
            elif isinstance(event, _TransitionReady) and event.slot_id in expected_slots:
                if event.slot_id in transitions:
                    raise RuntimeError("worker returned a duplicate transition")
                transitions[event.slot_id] = event
            else:
                raise RuntimeError(
                    f"unexpected event while stepping: {type(event).__name__}"
                )

        successor_slots = tuple(
            slot
            for slot in ordered
            if not transitions[slot.launch.spec.episode_index].transition.terminated
        )
        successor_values = self.sampler.value_batch(
            tuple(
                transitions[slot.launch.spec.episode_index].transition.next_decision.encoded
                for slot in successor_slots
                if transitions[slot.launch.spec.episode_index].transition.next_decision
                is not None
            )
        )
        if len(successor_values) != len(successor_slots):
            raise AssertionError("non-terminal worker transition has no successor")
        bootstrap = {
            slot.launch.spec.episode_index: value
            for slot, value in zip(successor_slots, successor_values, strict=True)
        }

        samples: list[RolloutSample] = []
        for slot, decision, policy_sample in zip(
            ordered, concrete, policy_samples, strict=True
        ):
            slot_id = slot.launch.spec.episode_index
            report = transitions[slot_id]
            transition = report.transition
            if transition.chosen_candidate_index != policy_sample.chosen_candidate_index:
                raise RuntimeError("worker executed a different learner candidate")
            if transition.chosen_action != decision.legal_actions[
                policy_sample.chosen_candidate_index
            ]:
                raise RuntimeError("candidate handle mapping changed across IPC")
            samples.append(
                RolloutSample(
                    encoded=decision.encoded,
                    chosen_candidate_index=policy_sample.chosen_candidate_index,
                    old_logprob=policy_sample.old_logprob,
                    old_value=policy_sample.old_value,
                    reward=transition.reward,
                    terminated=transition.terminated,
                    truncated=transition.truncated,
                    episode_id=transition.episode_id,
                    learning_player=transition.learning_player,
                    bootstrap_value=(
                        0.0 if transition.terminated else bootstrap[slot_id]
                    ),
                )
            )
            if report.completed is not None:
                self._completed_environments.append(report.completed)
                self._slots.pop(slot_id)
            else:
                if transition.next_decision is None:
                    raise AssertionError("active worker transition lost its successor")
                slot.current_decision = transition.next_decision
        return samples

    def _get_event(self, *, timeout: float = 60.0) -> Any:
        if self._pending_events:
            event = self._pending_events.popleft()
        else:
            started = time.perf_counter()
            try:
                event = self._output_queue.get(timeout=timeout)
            except queue.Empty as error:
                self._ensure_open()
                raise TimeoutError("timed out waiting for an environment worker") from error
            finally:
                self.ipc_wait_seconds += time.perf_counter() - started
        if isinstance(event, _WorkerFailure):
            raise RuntimeError(
                f"worker {event.worker_id} failed in slot {event.slot_id}: "
                f"{event.error_type}: {event.detail}\n{event.traceback_text}"
            )
        return event

    def _gather_immediate_opponent_requests(
        self, first: _NeuralOpponentRequest
    ) -> tuple[_NeuralOpponentRequest, ...]:
        requests = [first]
        # A tiny bounded window coalesces requests that are already arriving.
        # Determinism does not depend on which requests make this window: every
        # model forward below uses the same canonical shape and worker row.
        deadline = time.perf_counter() + 0.001
        while len(requests) < self.num_workers:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                event = self._output_queue.get(timeout=remaining)
            except queue.Empty:
                break
            if isinstance(event, _WorkerFailure):
                raise RuntimeError(
                    f"worker {event.worker_id} failed: {event.error_type}: {event.detail}\n"
                    f"{event.traceback_text}"
                )
            if isinstance(event, _NeuralOpponentRequest):
                requests.append(event)
            else:
                self._pending_events.append(event)
        return tuple(requests)

    def _serve_opponent_requests(
        self, requests: Sequence[_NeuralOpponentRequest]
    ) -> None:
        grouped: dict[tuple[str, int, int], list[_NeuralOpponentRequest]] = defaultdict(list)
        models: dict[tuple[str, int, int], ActorCritic] = {}
        for request in requests:
            slot = self._slots.get(request.slot_id)
            if slot is None or slot.worker_id != request.worker_id:
                raise RuntimeError("neural-opponent request refers to an unknown slot")
            model = slot.launch.neural_opponent_model
            if model is None or slot.opponent_rng is None:
                raise RuntimeError("fixed opponent unexpectedly requested central inference")
            if request.opponent_id != slot.launch.spec.opponent_id:
                raise RuntimeError("neural-opponent identity changed within an episode")
            first_action = request.encoded.candidate_actions[0]
            if isinstance(first_action, Pay):
                candidate_capacity = _PAYMENT_CANDIDATE_CAPACITY
            elif isinstance(first_action, ReturnTokens):
                candidate_capacity = _RETURN_CANDIDATE_CAPACITY
            elif isinstance(first_action, ClaimNoble):
                candidate_capacity = _NOBLE_CANDIDATE_CAPACITY
            else:
                candidate_capacity = _MAIN_ACTION_CANDIDATE_CAPACITY
            if len(request.encoded.candidate_actions) > candidate_capacity:
                raise RuntimeError(
                    "neural-opponent legal set exceeds its canonical capacity"
                )
            key = (request.opponent_id, id(model), candidate_capacity)
            grouped[key].append(request)
            models[key] = model

        for key, group in grouped.items():
            group.sort(key=lambda item: item.slot_id)
            model = models[key]
            started = time.perf_counter()
            # Every inference for a given worker topology and phase has the
            # same batch/candidate shape.  A real request always occupies its
            # worker-id row; unused rows duplicate a legal observation and are
            # discarded.  Opportunistic coalescing can therefore change only
            # wasted work, never a real request's numerical execution shape.
            canonical = [group[0].encoded] * self.num_workers
            for item in group:
                canonical[item.worker_id] = item.encoded
            batch = collate_encoded_decisions(tuple(canonical))
            candidate_capacity = key[2]
            padding = candidate_capacity - batch.candidate_features.shape[1]
            if padding < 0:
                raise RuntimeError("collated candidates exceed canonical capacity")
            if padding:
                batch = replace(
                    batch,
                    candidate_features=np.pad(
                        batch.candidate_features,
                        ((0, 0), (0, padding), (0, 0)),
                    ),
                    candidate_mask=np.pad(
                        batch.candidate_mask,
                        ((0, 0), (0, padding)),
                        constant_values=False,
                    ),
                )
            was_training = model.training
            model.eval()
            with torch.inference_mode():
                distribution, _ = model.distribution(batch)
                probabilities = distribution.probs.detach().cpu()
                chosen = []
                for request in group:
                    rng = self._slots[request.slot_id].opponent_rng
                    if rng is None:
                        raise AssertionError("neural opponent lost its RNG")
                    chosen.append(
                        int(
                            torch.multinomial(
                                probabilities[request.worker_id],
                                num_samples=1,
                                generator=rng,
                            ).item()
                        )
                    )
            model.train(was_training)
            self.opponent_inference_seconds += time.perf_counter() - started
            self.opponent_inference_calls += 1
            self.opponent_inference_items += len(group)
            for request, candidate_index in zip(group, chosen, strict=True):
                self._opponent_response_queues[request.worker_id].put(
                    _OpponentAction(request.request_id, candidate_index)
                )


def restored_worker_rng_states(
    snapshot: dict[str, Any] | None,
    *,
    num_workers: int,
    envs_per_worker: int,
) -> tuple[RngState, ...] | None:
    """Validate serialized collector metadata and recover worker streams."""

    if snapshot is None:
        return None
    if snapshot.get("format_version") != MULTIPROCESS_COLLECTOR_FORMAT_VERSION:
        raise ValueError("unsupported multiprocess collector checkpoint format")
    if int(snapshot["num_workers"]) != num_workers:
        raise ValueError("resume num_workers differs from its checkpoint")
    if int(snapshot["envs_per_worker"]) != envs_per_worker:
        raise ValueError("resume envs_per_worker differs from its checkpoint")
    if int(snapshot["active_episode_count"]) != 0:
        raise ValueError("5C resume only supports episode-boundary checkpoints")
    statistics = snapshot["worker_statistics"]
    if len(statistics) != num_workers:
        raise ValueError("checkpoint has the wrong worker RNG count")
    return tuple(RngState(**item["rng"]) for item in statistics)
