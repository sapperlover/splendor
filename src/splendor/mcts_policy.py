"""Independent-determinization PUCT over policy-safe engine reconstructions."""

from __future__ import annotations

import math
import random
import time
from itertools import count
from dataclasses import asdict, dataclass, field

import torch

from .actions import Action
from .actor_critic import ActorCritic
from .catalog import Catalog, load_default_catalog
from .contracts import DecisionRequest
from .engine import GameEngine
from .features import (
    EncodedDecision,
    collate_encoded_decisions,
    collate_encoded_states,
    encode_decision,
    encode_decision_state,
)
from .result import TerminalResult
from .search_inference import FrozenSearchInference
from .search_policy import _request_seed, _request_public_word, _seed_from_public_word, determinize_request


@dataclass(frozen=True, slots=True)
class MCTSConfig:
    simulations: int = 256
    determinizations: int = 2
    c_puct: float = 1.5
    seed: int = 0
    fold_forced_phases: bool = True
    max_simulation_transitions: int = 512
    root_noise_fraction: float = 0.25
    dirichlet_alpha: float = 0.3

    def __post_init__(self) -> None:
        for name in ("simulations", "determinizations", "max_simulation_transitions"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.determinizations > self.simulations:
            raise ValueError("determinizations cannot exceed simulations")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        if type(self.fold_forced_phases) is not bool:
            raise ValueError("fold_forced_phases must be bool")
        for name in ("c_puct", "root_noise_fraction", "dirichlet_alpha"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.c_puct <= 0 or self.dirichlet_alpha <= 0:
            raise ValueError("c_puct and dirichlet_alpha must be positive")
        if not 0 <= self.root_noise_fraction <= 1:
            raise ValueError("root_noise_fraction must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class MCTSSearchResult:
    chosen_candidate_index: int
    searchable: bool
    root_network_prior: tuple[float, ...]
    root_used_prior: tuple[float, ...]
    simulation_budgets: tuple[int, ...]
    per_world_root_visit_counts: tuple[tuple[int, ...], ...]
    root_visit_counts: tuple[int, ...]
    policy_target: tuple[float, ...]
    behavior_policy: tuple[float, ...]
    simulations: int
    max_tree_depth: int
    transition_cap_hits: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class MCTSStats:
    decisions: int = 0
    searchable_decisions: int = 0
    single_candidate_decisions: int = 0
    determinizations: int = 0
    simulations: int = 0
    expanded_nodes: int = 0
    state_transitions: int = 0
    rule_seconds: float = 0.0
    folded_actions: int = 0
    terminal_hits: int = 0
    transition_cap_hits: int = 0
    model_batches: int = 0
    model_positions: int = 0
    feature_seconds: float = 0.0
    inference_seconds: float = 0.0
    total_seconds: float = 0.0
    changed_action_count: int = 0
    tree_depth_sum: int = 0

    def to_dict(self) -> dict[str, int | float]:
        data = asdict(self)
        data.pop("rule_seconds")  # Preserve the stable v1 archive layout.
        return data


@dataclass(slots=True)
class _Edge:
    visits: int = 0
    value_sum: float = 0.0
    child: _Node | None = None
    terminal: TerminalResult | None = None
    actions: tuple[Action, ...] = ()


@dataclass(slots=True)
class _Node:
    engine: GameEngine
    request: DecisionRequest
    prior: tuple[float, ...]
    edges: list[_Edge] = field(default_factory=list)
    state_version: int = field(init=False)

    def __post_init__(self) -> None:
        self.state_version = self.engine.state_version
        if not self.edges:
            self.edges = [_Edge() for _ in self.prior]


def _softmax(logits: tuple[float, ...]) -> tuple[float, ...]:
    if not logits or any(not math.isfinite(x) for x in logits):
        raise ValueError("network logits must be finite and nonempty")
    peak = max(logits)
    weights = [math.exp(x - peak) for x in logits]
    total = sum(weights)
    return tuple(x / total for x in weights)


def _mean_world_visits(
    counts: tuple[tuple[int, ...], ...], budgets: tuple[int, ...]
) -> tuple[float, ...]:
    if not counts or len(counts) != len(budgets):
        raise ValueError("one positive budget is required per world")
    width = len(counts[0])
    if any(b <= 0 or len(row) != width or sum(row) != b for row, b in zip(counts, budgets)):
        raise ValueError("root visit counts do not match simulation budgets")
    return tuple(sum(row[i] / budget for row, budget in zip(counts, budgets)) / len(counts) for i in range(width))


def _temperature_policy(target: tuple[float, ...], temperature: float) -> tuple[float, ...]:
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and nonnegative")
    if temperature == 0:
        return target
    power = 1.0 / temperature
    peak = max(target)
    if not math.isfinite(power):
        tied = tuple(float(p == peak) for p in target)
        return tuple(x / sum(tied) for x in tied)
    log_peak = math.log(peak)
    weights = [math.exp((math.log(p) - log_peak) * power) if p > 0 else 0.0 for p in target]
    total = sum(weights)
    return tuple(x / total for x in weights)


class PUCTSearchPolicy:
    def __init__(
        self,
        model: ActorCritic,
        *,
        config: MCTSConfig | None = None,
        device: torch.device | str = "cpu",
        catalog: Catalog | None = None,
        engine_mode: str = "fast",
        evaluator: FrozenSearchInference | None = None,
    ) -> None:
        self.evaluator = evaluator or FrozenSearchInference(model, device=device)
        self.device = self.evaluator.device
        self.model = self.evaluator.model
        self.engine_mode = engine_mode
        self.config = config or MCTSConfig()
        self.catalog = catalog or load_default_catalog()
        self.stats = MCTSStats()
        self.last_search_result: MCTSSearchResult | None = None

    def _infer(self, requests: tuple[DecisionRequest, ...]) -> tuple[tuple[tuple[float, ...], float], ...]:
        before_features = self.evaluator.stats.feature_seconds
        before_inference = self.evaluator.stats.forward_seconds
        rows = self.evaluator.policy_value(requests)
        self.stats.feature_seconds += self.evaluator.stats.feature_seconds - before_features
        self.stats.inference_seconds += self.evaluator.stats.forward_seconds - before_inference
        self.stats.model_batches += int(bool(requests))
        return tuple((_softmax(logits), max(-1.0, min(1.0, value))) for logits, value in rows)

    def _critic(self, request: DecisionRequest) -> float:
        before_features = self.evaluator.stats.feature_seconds
        before_inference = self.evaluator.stats.forward_seconds
        value = self.evaluator.value_only((request,))[0]
        self.stats.feature_seconds += self.evaluator.stats.feature_seconds - before_features
        self.stats.inference_seconds += self.evaluator.stats.forward_seconds - before_inference
        self.stats.model_batches += 1
        return max(-1.0, min(1.0, value))

    def _select(self, node: _Node, root_player: int) -> int:
        total = sum(edge.visits for edge in node.edges)
        sign = 1 if node.engine.current_player == root_player else -1
        def score(index: int) -> tuple[float, float, int]:
            edge = node.edges[index]
            q = edge.value_sum / edge.visits if edge.visits else 0.0
            u = self.config.c_puct * node.prior[index] * math.sqrt(1 + total) / (1 + edge.visits)
            return (sign * q + u, node.prior[index], -index)
        return max(range(len(node.edges)), key=score)

    def start_job(
        self, request: DecisionRequest, *, add_root_noise: bool = False,
        temperature: float = 0.0, sampling_seed: int | None = None,
        root_encoded: EncodedDecision | None = None,
    ) -> MCTSSearchJob:
        return MCTSSearchJob(self, request, add_root_noise=add_root_noise,
                             temperature=temperature, sampling_seed=sampling_seed,
                             root_encoded=root_encoded)

    def search(
        self, request: DecisionRequest, *, add_root_noise: bool = False,
        temperature: float = 0.0, sampling_seed: int | None = None,
    ) -> MCTSSearchResult:
        job = self.start_job(request, add_root_noise=add_root_noise,
                             temperature=temperature, sampling_seed=sampling_seed)
        while not job.done:
            pending = job.pending_requests()
            policy_value = tuple(item for item in pending if item.kind == "policy_value")
            critic_only = tuple(item for item in pending if item.kind == "value_only")
            if policy_value:
                for item, (prior, value) in zip(policy_value, self._infer(tuple(x.request for x in policy_value))):
                    job.submit(item.request_id, prior, value, prior_is_normalized=True)
            if critic_only:
                for item in critic_only:
                    job.submit(item.request_id, None, self._critic(item.request))
        return job.result

    def choose(self, request: DecisionRequest) -> Action:
        result = self.search(request)
        return request.legal_actions[result.chosen_candidate_index]


@dataclass(frozen=True, slots=True)
class PendingSearchEvaluation:
    request_id: str
    kind: str
    request: DecisionRequest
    encoded: EncodedDecision | None = None


@dataclass(slots=True)
class _WorldJob:
    root: _Node
    budget: int
    completed: int = 0


@dataclass(slots=True)
class _PendingLeaf:
    world_index: int
    path: list[_Edge]
    engine: GameEngine
    request: DecisionRequest
    depth: int
    cap: bool
    branch_actions: tuple[Action, ...]
    simulation_index: int


_JOB_IDS = count()


class MCTSSearchJob:
    """One root with at most one unresolved leaf per independent world."""

    def __init__(
        self, policy: PUCTSearchPolicy, request: DecisionRequest, *,
        add_root_noise: bool, temperature: float, sampling_seed: int | None,
        root_encoded: EncodedDecision | None = None,
    ) -> None:
        _temperature_policy((1.0,), temperature)
        if type(add_root_noise) is not bool:
            raise ValueError("add_root_noise must be bool")
        if sampling_seed is not None and type(sampling_seed) is not int:
            raise ValueError("sampling_seed must be an integer or None")
        self.policy = policy
        self.request = request
        self.add_root_noise = add_root_noise
        self.temperature = temperature
        self.random_seed = policy.config.seed if sampling_seed is None else sampling_seed
        self.public_word = _request_public_word(request) if len(request.legal_actions) > 1 else 0
        self.started = time.perf_counter()
        self.job_id = next(_JOB_IDS)
        self.sequence = 0
        self._pending: dict[str, tuple[PendingSearchEvaluation, _PendingLeaf | None]] = {}
        self._worlds: list[_WorldJob] = []
        self._result: MCTSSearchResult | None = None
        self._cancelled = False
        self._max_depth = 0
        self._cap_before = policy.stats.transition_cap_hits
        self._current_world: int | None = None
        self._current_simulation: int | None = None
        self._branch_actions: list[Action] = []
        self._attempted_action: Action | None = None
        self.network_prior: tuple[float, ...] = ()
        self.used_prior: tuple[float, ...] = ()
        self.budgets: tuple[int, ...] = ()
        policy.stats.decisions += 1
        if len(request.legal_actions) == 1:
            policy.stats.single_candidate_decisions += 1
            self._complete(MCTSSearchResult(0, False, (1.0,), (1.0,), (), (), (0,), (1.0,), (1.0,), 0, 0, 0))
        else:
            policy.stats.searchable_decisions += 1
            self._emit("policy_value", request, None, root_encoded)

    @property
    def done(self) -> bool:
        return self._result is not None or self._cancelled

    @property
    def result(self) -> MCTSSearchResult:
        if self._cancelled:
            raise RuntimeError("search job was cancelled")
        if self._result is None:
            raise RuntimeError("search job has not finished")
        return self._result

    def pending_requests(self) -> tuple[PendingSearchEvaluation, ...]:
        return tuple(entry[0] for entry in self._pending.values())

    def cancel(self) -> None:
        self._cancelled = True
        self._pending.clear()
        self._worlds.clear()

    def failure_context(self, request_id: str | None = None) -> dict[str, object]:
        """Describe the pending or currently advancing branch on the cold path."""
        pending_entry = self._pending.get(request_id) if request_id is not None else None
        leaf = pending_entry[1] if pending_entry is not None else None
        if pending_entry is None:
            request_id = None
        world_index = leaf.world_index if leaf is not None else self._current_world
        branch = leaf.branch_actions if leaf is not None else self._branch_actions
        return {
            "algorithm": "determinized-puct", "search_protocol": "determinized-puct-v2-batched",
            "search_config": asdict(self.policy.config),
            "root_request": self.request, "sampling_seed": self.random_seed,
            "world_index": world_index,
            "determinization_seed": (
                None if world_index is None else _seed_from_public_word(
                    self.public_word, self.policy.config.seed, world_index)
            ),
            "simulation_index": leaf.simulation_index if leaf is not None else self._current_simulation,
            "branch_actions": list(branch),
            "attempted_action": None if leaf is not None else self._attempted_action,
            "request_id": request_id,
        }

    def _emit(self, kind: str, request: DecisionRequest, leaf: _PendingLeaf | None,
              encoded: EncodedDecision | None = None) -> None:
        request_id = f"{self.job_id}:{self.sequence}"
        self.sequence += 1
        self._pending[request_id] = (PendingSearchEvaluation(request_id, kind, request, encoded), leaf)
        self.policy.stats.model_positions += 1

    def _complete(self, result: MCTSSearchResult) -> None:
        self._result = result
        self.policy.stats.total_seconds += time.perf_counter() - self.started
        self.policy.last_search_result = result
        self._worlds.clear()

    def _start_worlds(self, network_prior: tuple[float, ...]) -> None:
        self.network_prior = network_prior
        used = network_prior
        size = len(network_prior)
        if self.add_root_noise:
            rng = random.Random(_seed_from_public_word(self.public_word, self.random_seed, 0x4E4F4953))
            alpha = self.policy.config.dirichlet_alpha
            eta = [1.0] * size if alpha > 1e306 else [rng.gammavariate(alpha, 1.0) for _ in range(size)]
            peak = max(eta)
            if peak == 0.0:
                selected = rng.randrange(size)
                eta = [float(i == selected) for i in range(size)]
            else:
                eta = [e / peak for e in eta]
            eta_sum = sum(eta)
            used = tuple((1 - self.policy.config.root_noise_fraction) * p + self.policy.config.root_noise_fraction * e / eta_sum for p, e in zip(network_prior, eta))
        self.used_prior = used
        n = self.policy.config.determinizations
        s = self.policy.config.simulations
        self.budgets = tuple(s // n + (j < s % n) for j in range(n))
        for j, budget in enumerate(self.budgets):
            seed = _seed_from_public_word(self.public_word, self.policy.config.seed, j)
            engine = determinize_request(self.request, seed=seed, catalog=self.policy.catalog,
                                         engine_mode=self.policy.engine_mode)
            self._worlds.append(_WorldJob(_Node(engine, self.request, used), budget))
            self.policy.stats.determinizations += 1
        for j in range(n):
            self._advance_world(j)

    def _backup(self, world_index: int, path: list[_Edge], depth: int, value: float) -> None:
        for edge in path:
            edge.visits += 1
            edge.value_sum += value
        self.policy.stats.tree_depth_sum += depth
        self.policy.stats.simulations += 1
        self._max_depth = max(self._max_depth, depth)
        self._worlds[world_index].completed += 1

    def _advance_world(self, world_index: int) -> None:
        world = self._worlds[world_index]
        root_player = world.root.engine.current_player
        while world.completed < world.budget:
            self._current_world = world_index
            self._current_simulation = world.completed
            self._branch_actions = []
            self._attempted_action = None
            node = world.root
            path: list[_Edge] = []
            transitions = 0
            while True:
                index = self.policy._select(node, root_player)
                edge = node.edges[index]
                path.append(edge)
                depth = len(path)
                action = node.request.legal_actions[index]
                if edge.child is not None or edge.terminal is not None:
                    self._branch_actions.extend(edge.actions)
                if edge.terminal is not None:
                    self.policy.stats.terminal_hits += 1
                    self._backup(world_index, path, depth, float(edge.terminal.payoff(root_player)))
                    break
                if edge.child is not None:
                    node = edge.child
                    continue
                rule_started = time.perf_counter()
                child = node.engine.fork()
                self._attempted_action = action
                branch_start = len(self._branch_actions)
                if self.policy.config.fold_forced_phases:
                    try:
                        response, transitions, folded = child.apply_candidate_folded(
                            index, max_transitions=self.policy.config.max_simulation_transitions,
                            action_trace=self._branch_actions,
                            state_version=node.state_version)
                    except Exception:
                        self._attempted_action = child.last_attempted_action
                        raise
                    self.policy.stats.folded_actions += folded
                else:
                    response = child.apply_candidate(index, state_version=node.state_version)
                    self._branch_actions.append(action)
                    transitions = 1
                edge.actions = tuple(self._branch_actions[branch_start:])
                self._attempted_action = None
                self.policy.stats.state_transitions += transitions
                self.policy.stats.rule_seconds += time.perf_counter() - rule_started
                if isinstance(response, TerminalResult):
                    edge.terminal = response
                    self.policy.stats.terminal_hits += 1
                    self._backup(world_index, path, depth, float(response.payoff(root_player)))
                    break
                cap = transitions >= self.policy.config.max_simulation_transitions
                if cap:
                    self.policy.stats.transition_cap_hits += 1
                self._emit("value_only" if cap else "policy_value", response,
                           _PendingLeaf(world_index, path, child, response, depth, cap,
                                        tuple(self._branch_actions), world.completed))
                return
        if all(w.completed == w.budget for w in self._worlds) and not self._pending:
            self._finish()

    def submit(
        self, request_id: str, logits: tuple[float, ...] | None, value: float,
        *, prior_is_normalized: bool = False,
    ) -> None:
        if self._cancelled or self.done:
            raise RuntimeError("search job is cancelled or complete")
        try:
            pending, leaf = self._pending[request_id]
        except KeyError as error:
            raise ValueError(f"unknown or already resolved search request {request_id}") from error
        if not math.isfinite(value):
            raise ValueError("network value must be finite")
        if pending.kind == "policy_value":
            if logits is None:
                raise ValueError("policy/value request requires logits")
            if len(logits) != len(pending.request.legal_actions):
                raise ValueError("network logits must match legal actions")
            prior = tuple(logits) if prior_is_normalized else _softmax(tuple(logits))
            if prior_is_normalized and (
                any(not math.isfinite(p) or p < 0 for p in prior) or not any(p > 0 for p in prior)
            ):
                raise ValueError("network prior must be finite, nonnegative and nonempty")
        else:
            if logits is not None:
                raise ValueError("value-only request cannot receive logits")
            prior = ()
        critic = max(-1.0, min(1.0, value))
        # Preserve the pending branch on validation failure; after consumption,
        # _advance_world failures must instead describe the newly advancing branch.
        del self._pending[request_id]
        if leaf is None:
            self._start_worlds(prior)
            return
        root_player = self._worlds[leaf.world_index].root.engine.current_player
        edge = leaf.path[-1]
        if not leaf.cap:
            edge.child = _Node(leaf.engine, leaf.request, prior)
            self.policy.stats.expanded_nodes += 1
        value_for_root = critic if leaf.engine.current_player == root_player else -critic
        self._backup(leaf.world_index, leaf.path, leaf.depth, value_for_root)
        self._advance_world(leaf.world_index)

    def _finish(self) -> None:
        size = len(self.request.legal_actions)
        rows = tuple(tuple(edge.visits for edge in world.root.edges) for world in self._worlds)
        target = _mean_world_visits(rows, self.budgets)
        counts = tuple(sum(row[i] for row in rows) for i in range(size))
        behavior = _temperature_policy(target, self.temperature)
        if self.temperature == 0:
            chosen = max(range(size), key=lambda i: (target[i], self.network_prior[i], -i))
            behavior = tuple(float(i == chosen) for i in range(size))
        else:
            rng = random.Random(_seed_from_public_word(self.public_word, self.random_seed, 0x53414D50))
            draw = rng.random()
            chosen = size - 1
            cumulative = 0.0
            for i, probability in enumerate(behavior):
                cumulative += probability
                if draw < cumulative:
                    chosen = i
                    break
        raw_choice = max(range(size), key=lambda i: (self.network_prior[i], -i))
        self.policy.stats.changed_action_count += chosen != raw_choice
        result = MCTSSearchResult(chosen, True, self.network_prior, self.used_prior,
                                  self.budgets, rows, counts, target, behavior,
                                  self.policy.config.simulations, self._max_depth,
                                  self.policy.stats.transition_cap_hits - self._cap_before)
        self._complete(result)
