"""Independent-determinization PUCT over policy-safe engine reconstructions."""

from __future__ import annotations

import math
import random
import time
from dataclasses import asdict, dataclass, field

import torch

from .actions import Action
from .actor_critic import ActorCritic
from .catalog import Catalog, load_default_catalog
from .contracts import DecisionRequest
from .engine import GameEngine
from .features import (
    collate_encoded_decisions,
    collate_encoded_states,
    encode_decision,
    encode_decision_state,
)
from .result import TerminalResult
from .search_policy import _request_seed, determinize_request


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
        return asdict(self)


@dataclass(slots=True)
class _Edge:
    visits: int = 0
    value_sum: float = 0.0
    child: _Node | None = None
    terminal: TerminalResult | None = None


@dataclass(slots=True)
class _Node:
    engine: GameEngine
    request: DecisionRequest
    prior: tuple[float, ...]
    edges: list[_Edge] = field(default_factory=list)

    def __post_init__(self) -> None:
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
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.config = config or MCTSConfig()
        self.catalog = catalog or load_default_catalog()
        self.stats = MCTSStats()
        self.last_search_result: MCTSSearchResult | None = None

    def _infer(self, requests: tuple[DecisionRequest, ...]) -> tuple[tuple[tuple[float, ...], float], ...]:
        """Single batch boundary for future cross-game inference scheduling."""
        started = time.perf_counter()
        batch = collate_encoded_decisions(tuple(encode_decision(r) for r in requests))
        self.stats.feature_seconds += time.perf_counter() - started
        started = time.perf_counter()
        with torch.inference_mode():
            logits, values = self.model(batch)
            mask = torch.as_tensor(batch.candidate_mask, device=logits.device)
            logits = self.model.masked_logits(logits, mask)
            raw_logits = logits.detach().cpu().tolist()
            raw_values = values.detach().cpu().tolist()
        self.stats.inference_seconds += time.perf_counter() - started
        self.stats.model_batches += 1
        self.stats.model_positions += len(requests)
        result = []
        for request, row, raw in zip(requests, raw_logits, raw_values):
            if not math.isfinite(raw):
                raise ValueError("network value must be finite")
            result.append((_softmax(tuple(row[: len(request.legal_actions)])), max(-1.0, min(1.0, raw))))
        return tuple(result)

    def _critic(self, request: DecisionRequest) -> float:
        started = time.perf_counter()
        batch = collate_encoded_states((encode_decision_state(request),))
        self.stats.feature_seconds += time.perf_counter() - started
        started = time.perf_counter()
        with torch.inference_mode():
            raw = float(self.model.value_from_state(batch).detach().cpu()[0])
        self.stats.inference_seconds += time.perf_counter() - started
        self.stats.model_batches += 1
        self.stats.model_positions += 1
        if not math.isfinite(raw):
            raise ValueError("network value must be finite")
        return max(-1.0, min(1.0, raw))

    def _select(self, node: _Node, root_player: int) -> int:
        total = sum(edge.visits for edge in node.edges)
        sign = 1 if node.engine.current_player == root_player else -1
        def score(index: int) -> tuple[float, float, int]:
            edge = node.edges[index]
            q = edge.value_sum / edge.visits if edge.visits else 0.0
            u = self.config.c_puct * node.prior[index] * math.sqrt(1 + total) / (1 + edge.visits)
            return (sign * q + u, node.prior[index], -index)
        return max(range(len(node.edges)), key=score)

    def _simulate(self, root: _Node) -> int:
        node = root
        root_player = root.engine.current_player
        path: list[_Edge] = []
        transitions = 0
        depth = 0
        while True:
            index = self._select(node, root_player)
            edge = node.edges[index]
            path.append(edge)
            depth += 1
            if edge.terminal is not None:
                self.stats.terminal_hits += 1
                value = float(edge.terminal.payoff(root_player))
                break
            if edge.child is not None:
                node = edge.child
                continue
            child_engine = node.engine.fork()
            response = child_engine.apply(node.request.legal_actions[index])
            transitions += 1
            self.stats.state_transitions += 1
            while (
                self.config.fold_forced_phases
                and isinstance(response, DecisionRequest)
                and len(response.legal_actions) == 1
                and transitions < self.config.max_simulation_transitions
            ):
                response = child_engine.apply(response.legal_actions[0])
                transitions += 1
                self.stats.state_transitions += 1
                self.stats.folded_actions += 1
            if isinstance(response, TerminalResult):
                edge.terminal = response
                self.stats.terminal_hits += 1
                value = float(response.payoff(root_player))
            elif transitions >= self.config.max_simulation_transitions:
                self.stats.transition_cap_hits += 1
                critic = self._critic(response)
                value = critic if child_engine.current_player == root_player else -critic
            else:
                prior, critic = self._infer((response,))[0]
                edge.child = _Node(child_engine, response, prior)
                self.stats.expanded_nodes += 1
                value = critic if child_engine.current_player == root_player else -critic
            break
        for edge in path:
            edge.visits += 1
            edge.value_sum += value
        self.stats.tree_depth_sum += depth
        return depth

    def search(
        self,
        request: DecisionRequest,
        *,
        add_root_noise: bool = False,
        temperature: float = 0.0,
        sampling_seed: int | None = None,
    ) -> MCTSSearchResult:
        _temperature_policy((1.0,), temperature)
        if type(add_root_noise) is not bool:
            raise ValueError("add_root_noise must be bool")
        if sampling_seed is not None and type(sampling_seed) is not int:
            raise ValueError("sampling_seed must be an integer or None")
        started = time.perf_counter()
        self.stats.decisions += 1
        size = len(request.legal_actions)
        if size == 1:
            self.stats.single_candidate_decisions += 1
            result = MCTSSearchResult(0, False, (1.0,), (1.0,), (), (), (0,), (1.0,), (1.0,), 0, 0, 0)
        else:
            self.stats.searchable_decisions += 1
            network_prior, _ = self._infer((request,))[0]
            used_prior = network_prior
            random_seed = self.config.seed if sampling_seed is None else sampling_seed
            if add_root_noise:
                rng = random.Random(_request_seed(request, random_seed, 0x4E4F4953))
                alpha = self.config.dirichlet_alpha
                # At this scale the Dirichlet is uniform at float precision;
                # Python's gamma sampler can loop forever after 9*alpha overflows.
                eta = [1.0] * size if alpha > 1e306 else [rng.gammavariate(alpha, 1.0) for _ in range(size)]
                peak = max(eta)
                if peak == 0.0:
                    selected = rng.randrange(size)
                    eta = [float(i == selected) for i in range(size)]
                else:
                    eta = [e / peak for e in eta]
                eta_sum = sum(eta)
                used_prior = tuple((1 - self.config.root_noise_fraction) * p + self.config.root_noise_fraction * e / eta_sum for p, e in zip(network_prior, eta))
            n = self.config.determinizations
            budgets = tuple(self.config.simulations // n + (j < self.config.simulations % n) for j in range(n))
            rows = []
            max_depth = 0
            cap_before = self.stats.transition_cap_hits
            for j, budget in enumerate(budgets):
                engine = determinize_request(request, seed=_request_seed(request, self.config.seed, j), catalog=self.catalog)
                root = _Node(engine, request, used_prior)
                self.stats.determinizations += 1
                for _ in range(budget):
                    max_depth = max(max_depth, self._simulate(root))
                    self.stats.simulations += 1
                rows.append(tuple(edge.visits for edge in root.edges))
            row_tuple = tuple(rows)
            target = _mean_world_visits(row_tuple, budgets)
            counts = tuple(sum(row[i] for row in row_tuple) for i in range(size))
            behavior = _temperature_policy(target, temperature)
            if temperature == 0:
                chosen = max(range(size), key=lambda i: (target[i], network_prior[i], -i))
                behavior = tuple(float(i == chosen) for i in range(size))
            else:
                rng = random.Random(_request_seed(request, random_seed, 0x53414D50))
                draw = rng.random()
                chosen = size - 1
                cumulative = 0.0
                for i, probability in enumerate(behavior):
                    cumulative += probability
                    if draw < cumulative:
                        chosen = i
                        break
            raw_choice = max(range(size), key=lambda i: (network_prior[i], -i))
            self.stats.changed_action_count += chosen != raw_choice
            result = MCTSSearchResult(chosen, True, network_prior, used_prior, budgets, row_tuple, counts, target, behavior, self.config.simulations, max_depth, self.stats.transition_cap_hits - cap_before)
        self.stats.total_seconds += time.perf_counter() - started
        self.last_search_result = result
        return result

    def choose(self, request: DecisionRequest) -> Action:
        result = self.search(request)
        return request.legal_actions[result.chosen_candidate_index]
