"""Information-set-safe policy/critic lookahead over the existing engine.

The live GameState is never passed to this policy. Each search tree starts
from a determinization reconstructed solely from the policy's DecisionRequest:
public cards and the acting player's private reservations are fixed, while
future deck order and the opponent's blind reservations are sampled with a
deterministic policy-local seed.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from ._state import (
    DeckState,
    GameState,
    MarketState,
    PendingTurn,
    PlayerState,
    ReservedCard,
)
from .actions import Action
from .actor_critic import ActorCritic
from .catalog import Catalog, load_default_catalog
from .contracts import DecisionRequest, Observation, PlayerPublicState
from .engine import GameEngine
from .features import collate_encoded_decisions, encode_decision
from .model import ReservationSource, Tier
from .replay import new_replay, to_primitive
from .result import TerminalResult
from .rng import DeterministicRng


@dataclass(frozen=True, slots=True)
class SearchConfig:
    """Small, deliberately direct policy-guided minimax configuration."""

    top_k: int = 5
    depth: int = 4
    determinizations: int = 1
    seed: int = 0
    leaf_value_clip: float | None = 1.0

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.depth <= 0:
            raise ValueError("depth must be positive")
        if self.determinizations <= 0:
            raise ValueError("determinizations must be positive")
        if self.leaf_value_clip is not None and self.leaf_value_clip <= 0:
            raise ValueError("leaf_value_clip must be positive or None")


@dataclass(slots=True)
class SearchStats:
    decisions: int = 0
    determinizations: int = 0
    expanded_nodes: int = 0
    leaf_nodes: int = 0
    model_batches: int = 0
    model_positions: int = 0
    inference_seconds: float = 0.0
    total_seconds: float = 0.0

    def to_dict(self) -> dict[str, int | float]:
        return {
            "decisions": self.decisions,
            "determinizations": self.determinizations,
            "expanded_nodes": self.expanded_nodes,
            "leaf_nodes": self.leaf_nodes,
            "model_batches": self.model_batches,
            "model_positions": self.model_positions,
            "inference_seconds": self.inference_seconds,
            "total_seconds": self.total_seconds,
        }


@dataclass(slots=True)
class _TreeNode:
    engine: GameEngine
    response: DecisionRequest | TerminalResult
    actor: int | None
    depth: int
    children: list[tuple[Action, int]]
    value: float | None = None


def _request_seed(request: DecisionRequest, base_seed: int, sample: int) -> int:
    """Derive a platform-stable stream from public policy input only."""

    payload = json.dumps(
        to_primitive(request),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    public_word = int.from_bytes(digest[:8], "big")
    return (
        public_word
        ^ (base_seed & ((1 << 64) - 1))
        ^ ((sample + 1) * 0x9E3779B97F4A7C15)
    ) & ((1 << 64) - 1)


def _market_ids(observation: Observation) -> tuple[str, ...]:
    return tuple(
        card.card_id
        for cards in (
            observation.market.tier_one,
            observation.market.tier_two,
            observation.market.tier_three,
        )
        for card in cards
    )


def _own_reservations(observation: Observation) -> tuple[ReservedCard, ...]:
    blind: dict[Tier, list[str]] = {tier: [] for tier in Tier}
    for card in observation.self_private.blind_reserved_cards:
        blind[card.tier].append(card.card_id)
    for values in blind.values():
        values.sort()

    result: list[ReservedCard] = []
    for reserved in observation.self_public.reserved_cards:
        if reserved.source is ReservationSource.MARKET:
            if reserved.card is None:
                raise ValueError("a public market reservation is missing its card")
            result.append(ReservedCard(reserved.card.card_id, reserved.source))
            continue
        if reserved.card is not None:
            raise ValueError("a blind reservation unexpectedly exposes a card")
        candidates = blind[reserved.tier]
        if not candidates:
            raise ValueError("private blind cards do not match public reservations")
        result.append(ReservedCard(candidates.pop(0), reserved.source))
    if any(blind.values()):
        raise ValueError("private blind cards do not match public reservations")
    return tuple(result)


def _known_opponent_reservations(
    observation: Observation,
) -> tuple[ReservedCard, ...]:
    result = []
    for reserved in observation.opponent_public.reserved_cards:
        if reserved.source is ReservationSource.DECK:
            continue
        if reserved.card is None:
            raise ValueError("a public market reservation is missing its card")
        result.append(ReservedCard(reserved.card.card_id, reserved.source))
    return tuple(result)


def _player_state(
    public: PlayerPublicState,
    reservations: tuple[ReservedCard, ...],
) -> PlayerState:
    return PlayerState(
        tokens=public.tokens,
        purchased_card_ids=public.purchased_card_ids,
        reserved_cards=reservations,
        noble_ids=public.noble_ids,
    )


def _determinized_state(
    request: DecisionRequest,
    catalog: Catalog,
    *,
    seed: int,
) -> GameState:
    """Reconstruct one complete root-relative state from safe information."""

    observation = request.observation
    own_reserved = _own_reservations(observation)
    opponent_known = _known_opponent_reservations(observation)

    visible_ids = {
        *_market_ids(observation),
        *observation.self_public.purchased_card_ids,
        *observation.opponent_public.purchased_card_ids,
        *(reserved.card_id for reserved in own_reserved),
        *(reserved.card_id for reserved in opponent_known),
    }
    expected_visible = (
        len(_market_ids(observation))
        + len(observation.self_public.purchased_card_ids)
        + len(observation.opponent_public.purchased_card_ids)
        + len(own_reserved)
        + len(opponent_known)
    )
    if len(visible_ids) != expected_visible:
        raise ValueError("observation contains a development card in multiple locations")

    rng = DeterministicRng(seed)
    deck_by_tier: dict[Tier, tuple[str, ...]] = {}
    blind_by_tier: dict[Tier, list[str]] = {tier: [] for tier in Tier}
    for tier in Tier:
        unknown = sorted(
            card.card_id
            for card in catalog.cards
            if card.tier is tier and card.card_id not in visible_ids
        )
        rng.shuffle(unknown)
        blind_count = sum(
            reserved.source is ReservationSource.DECK and reserved.tier is tier
            for reserved in observation.opponent_public.reserved_cards
        )
        deck_count = observation.deck_counts[tier.value - 1]
        if len(unknown) != blind_count + deck_count:
            raise ValueError(
                f"tier {tier.value} hidden-card count does not match observation"
            )
        blind_by_tier[tier] = unknown[:blind_count]
        deck_by_tier[tier] = tuple(unknown[blind_count:])

    opponent_reserved: list[ReservedCard] = []
    blind_cursor = {tier: 0 for tier in Tier}
    for reserved in observation.opponent_public.reserved_cards:
        if reserved.source is ReservationSource.MARKET:
            if reserved.card is None:
                raise ValueError("a public market reservation is missing its card")
            opponent_reserved.append(
                ReservedCard(reserved.card.card_id, ReservationSource.MARKET)
            )
            continue
        index = blind_cursor[reserved.tier]
        opponent_reserved.append(
            ReservedCard(
                blind_by_tier[reserved.tier][index],
                ReservationSource.DECK,
            )
        )
        blind_cursor[reserved.tier] += 1

    pending = None
    if observation.pending is not None:
        pending = PendingTurn(
            main_action=observation.pending.main_action,
            tokens_to_return=observation.pending.tokens_to_return,
            eligible_noble_ids=tuple(
                noble.noble_id for noble in observation.pending.eligible_nobles
            ),
        )

    claimed_nobles = {
        *observation.self_public.noble_ids,
        *observation.opponent_public.noble_ids,
    }
    in_play_nobles = tuple(
        noble.noble_id for noble in observation.nobles_in_play
    )
    out_of_game = tuple(
        noble.noble_id
        for noble in catalog.nobles
        if noble.noble_id not in claimed_nobles
        and noble.noble_id not in in_play_nobles
    )
    first_player = 0 if observation.self_is_first_player else 1
    trigger = (
        None
        if not observation.final_round_triggered
        else (0 if observation.final_round_triggered_by_self else 1)
    )
    return GameState(
        supply=observation.supply,
        decks=DeckState(
            deck_by_tier[Tier.ONE],
            deck_by_tier[Tier.TWO],
            deck_by_tier[Tier.THREE],
        ),
        market=MarketState(
            tuple(card.card_id for card in observation.market.tier_one),
            tuple(card.card_id for card in observation.market.tier_two),
            tuple(card.card_id for card in observation.market.tier_three),
        ),
        players=(
            _player_state(observation.self_public, own_reserved),
            _player_state(observation.opponent_public, tuple(opponent_reserved)),
        ),
        nobles_in_play=in_play_nobles,
        nobles_out_of_game=out_of_game,
        first_player=first_player,
        current_player=0,
        completed_turns=(
            observation.own_completed_turns,
            observation.opponent_completed_turns,
        ),
        consecutive_forced_skips=observation.consecutive_forced_skips,
        phase=observation.phase,
        pending=pending,
        final_round_triggered_by=trigger,
        result=None,
        rng=rng.snapshot(),
    )


def determinize_request(
    request: DecisionRequest,
    *,
    seed: int,
    catalog: Catalog | None = None,
    failure_dir: Path | None = None,
) -> GameEngine:
    """Build a legal simulator whose root view exactly equals the input."""

    catalog = catalog or load_default_catalog()
    state = _determinized_state(request, catalog, seed=seed)
    engine = GameEngine(
        state,
        catalog,
        new_replay(catalog, seed, state.first_player),
        failure_dir or Path.cwd() / "runtime" / "search_failures",
    )
    rebuilt = engine.decision()
    if not isinstance(rebuilt, DecisionRequest):
        raise ValueError("a non-terminal request determinized to terminal")
    if rebuilt != request:
        raise ValueError("determinized root does not reproduce DecisionRequest")
    return engine


def _clone_engine(engine: GameEngine) -> GameEngine:
    """Fork a search-owned immutable engine state, never a live match state."""

    return engine.fork(
        failure_dir=Path.cwd() / "runtime" / "search_failures"
    )


def _terminal_value(result: TerminalResult) -> float:
    if len(result.winner_ids) == 2:
        return 0.0
    return 1.0 if result.winner_ids[0] == 0 else -1.0


class PolicyValueSearchPolicy:
    """Top-k policy-guided minimax with critic evaluation at leaf requests.

    A ply is one model decision boundary, not necessarily one completed player
    turn. PAYMENT, RETURN_TOKENS, and CHOOSE_NOBLE therefore count exactly as
    the model sees them during PPO training.
    """

    def __init__(
        self,
        model: ActorCritic,
        *,
        config: SearchConfig | None = None,
        device: torch.device | str = "cpu",
        catalog: Catalog | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.catalog = catalog or load_default_catalog()
        self.config = config or SearchConfig()
        self.stats = SearchStats()

    def _infer(
        self,
        requests: list[DecisionRequest],
    ) -> tuple[list[list[float]], list[float]]:
        if not requests:
            return [], []
        started = time.perf_counter()
        encoded = tuple(encode_decision(request) for request in requests)
        batch = collate_encoded_decisions(encoded)
        was_training = self.model.training
        self.model.eval()
        with torch.inference_mode():
            logits, values = self.model(batch)
            mask = torch.as_tensor(batch.candidate_mask, device=logits.device)
            masked = self.model.masked_logits(logits, mask)
        self.model.train(was_training)
        cpu_logits = masked.detach().cpu()
        cpu_values = values.detach().cpu()
        self.stats.model_batches += 1
        self.stats.model_positions += len(requests)
        self.stats.inference_seconds += time.perf_counter() - started
        return (
            [
                row[: len(request.legal_actions)].tolist()
                for row, request in zip(cpu_logits, requests, strict=True)
            ],
            cpu_values.tolist(),
        )

    def _top_indices(self, logits: list[float]) -> tuple[int, ...]:
        count = min(self.config.top_k, len(logits))
        return tuple(
            sorted(range(len(logits)), key=lambda index: (-logits[index], index))[
                :count
            ]
        )

    def _infer_values(self, requests: list[DecisionRequest]) -> list[float]:
        if not requests:
            return []
        started = time.perf_counter()
        encoded = tuple(encode_decision(request) for request in requests)
        batch = collate_encoded_decisions(encoded)
        was_training = self.model.training
        self.model.eval()
        with torch.inference_mode():
            values = self.model.value_only(batch)
        self.model.train(was_training)
        self.stats.model_batches += 1
        self.stats.model_positions += len(requests)
        self.stats.inference_seconds += time.perf_counter() - started
        return values.detach().cpu().tolist()

    def choose(self, request: DecisionRequest) -> Action:
        started = time.perf_counter()
        self.stats.decisions += 1
        if len(request.legal_actions) == 1:
            self.stats.total_seconds += time.perf_counter() - started
            return request.legal_actions[0]

        roots: list[int] = []
        nodes: list[_TreeNode] = []
        for sample in range(self.config.determinizations):
            seed = _request_seed(request, self.config.seed, sample)
            engine = determinize_request(
                request,
                seed=seed,
                catalog=self.catalog,
            )
            roots.append(len(nodes))
            nodes.append(
                _TreeNode(
                    engine=engine,
                    response=request,
                    actor=0,
                    depth=0,
                    children=[],
                )
            )
        self.stats.determinizations += len(roots)

        frontier = list(roots)
        root_logits: list[float] | None = None
        for depth in range(self.config.depth):
            active = [
                index
                for index in frontier
                if isinstance(nodes[index].response, DecisionRequest)
            ]
            if not active:
                frontier = []
                break
            active_requests = []
            for index in active:
                response = nodes[index].response
                if not isinstance(response, DecisionRequest):
                    raise AssertionError("active tree node is terminal")
                active_requests.append(response)
            logits_rows, _ = self._infer(active_requests)
            if depth == 0:
                root_logits = logits_rows[0]
            next_frontier: list[int] = []
            for node_index, logits in zip(active, logits_rows, strict=True):
                node = nodes[node_index]
                node_request = node.response
                if not isinstance(node_request, DecisionRequest):
                    raise AssertionError("active tree node is terminal")
                for action_index in self._top_indices(logits):
                    action = node_request.legal_actions[action_index]
                    child_engine = _clone_engine(node.engine)
                    child_response = child_engine.apply(action)
                    child_index = len(nodes)
                    child = _TreeNode(
                        engine=child_engine,
                        response=child_response,
                        actor=(
                            None
                            if isinstance(child_response, TerminalResult)
                            else child_engine.current_player
                        ),
                        depth=depth + 1,
                        children=[],
                        value=(
                            _terminal_value(child_response)
                            if isinstance(child_response, TerminalResult)
                            else None
                        ),
                    )
                    nodes.append(child)
                    node.children.append((action, child_index))
                    self.stats.expanded_nodes += 1
                    if child.value is None:
                        next_frontier.append(child_index)
            frontier = next_frontier

        leaf_indices = [
            index
            for index in frontier
            if isinstance(nodes[index].response, DecisionRequest)
        ]
        leaf_requests = []
        for index in leaf_indices:
            response = nodes[index].response
            if not isinstance(response, DecisionRequest):
                raise AssertionError("leaf tree node is terminal")
            leaf_requests.append(response)
        leaf_values = self._infer_values(leaf_requests)
        for node_index, raw_value in zip(leaf_indices, leaf_values, strict=True):
            node = nodes[node_index]
            if self.config.leaf_value_clip is not None:
                limit = self.config.leaf_value_clip
                raw_value = max(-limit, min(limit, raw_value))
            if node.actor == 1:
                raw_value = -raw_value
            node.value = raw_value

        self.stats.leaf_nodes += sum(not node.children for node in nodes)
        for node in reversed(nodes):
            if node.value is not None:
                continue
            if not node.children or node.actor is None:
                raise AssertionError("search tree contains an unevaluated node")
            child_values = []
            for _, child_index in node.children:
                child_value = nodes[child_index].value
                if child_value is None:
                    raise AssertionError("child value was not backed up")
                child_values.append(child_value)
            node.value = (
                max(child_values) if node.actor == 0 else min(child_values)
            )

        if root_logits is None:
            raise AssertionError("search root was not evaluated")
        values_by_action: dict[Action, list[float]] = {}
        for root_index in roots:
            for action, child_index in nodes[root_index].children:
                value = nodes[child_index].value
                if value is None:
                    raise AssertionError("root child was not evaluated")
                values_by_action.setdefault(action, []).append(value)
        candidates = []
        for index, action in enumerate(request.legal_actions):
            values = values_by_action.get(action)
            if values is None:
                continue
            if len(values) != self.config.determinizations:
                raise AssertionError("root action is missing a determinization")
            candidates.append(
                (
                    sum(values) / len(values),
                    root_logits[index],
                    -index,
                    action,
                )
            )
        if not candidates:
            raise AssertionError("search produced no root action")
        chosen = max(candidates, key=lambda item: item[:3])[3]
        self.stats.total_seconds += time.perf_counter() - started
        return chosen
