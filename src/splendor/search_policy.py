"""Information-set-safe policy/critic lookahead over the existing engine.

The live GameState is never passed to this policy. Each search tree starts
from a determinization reconstructed solely from the policy's DecisionRequest:
public cards and the acting player's private reservations are fixed, while
future deck order and the opponent's blind reservations are sampled with a
deterministic policy-local seed.

Search shape (rewritten for depth sweeps; equivalence with the previous
implementation is pinned by ``tests/test_search_policy_equivalence.py``):

* the root expands its policy top-k candidates with a full window per child,
  so every root action value stays *exact* and the final choice matches the
  pre-alpha-beta reference;
* interior nodes use fail-soft alpha-beta ordered by the acting player's own
  policy logits, which prunes most of the tree while returning the same
  minimax value;
* leaves are evaluated by the critic from a state-only encoding that still
  carries the complete pending-turn context (payment target, tokens to return,
  eligible nobles);
* ``fold_forced_phases`` optionally collapses single-legal-action chains
  (``PAYMENT`` / ``RETURN_TOKENS`` / ``CHOOSE_NOBLE`` and any other forced
  decision) into one macro transition, so depth is spent on real choices.
  Folding changes the searched tree and is therefore off by default.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

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
from .features import (
    collate_encoded_decisions,
    collate_encoded_states,
    encode_decision,
    encode_decision_state,
)
from .model import ReservationSource, Tier
from .replay import new_replay, to_primitive
from .result import TerminalResult
from .rng import DeterministicRng


@dataclass(frozen=True, slots=True)
class SearchConfig:
    """Policy-guided alpha-beta configuration.

    ``depth`` counts model decision plies, exactly as before: the root is one
    ply and the horizon is evaluated with the critic.  ``fold_forced_phases``
    opts into macro transitions; see the module docstring.
    """

    top_k: int = 5
    depth: int = 4
    determinizations: int = 1
    seed: int = 0
    leaf_value_clip: float | None = 1.0
    fold_forced_phases: bool = False

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
    folded_decisions: int = 0
    pruned_branches: int = 0
    searchable_decisions: int = 0
    changed_action_count: int = 0
    chosen_rank_sum: int = 0
    chosen_rank_top1_count: int = 0
    chosen_rank_top2_count: int = 0
    chosen_rank_top3_count: int = 0
    chosen_rank_beyond3_count: int = 0
    chosen_rank_counts: dict[int, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, int | float]:
        result: dict[str, int | float] = {
            "decisions": self.decisions,
            "determinizations": self.determinizations,
            "expanded_nodes": self.expanded_nodes,
            "leaf_nodes": self.leaf_nodes,
            "model_batches": self.model_batches,
            "model_positions": self.model_positions,
            "inference_seconds": self.inference_seconds,
            "total_seconds": self.total_seconds,
            "folded_decisions": self.folded_decisions,
            "pruned_branches": self.pruned_branches,
            "searchable_decisions": self.searchable_decisions,
            "changed_action_count": self.changed_action_count,
            "chosen_rank_sum": self.chosen_rank_sum,
            "chosen_rank_top1_count": self.chosen_rank_top1_count,
            "chosen_rank_top2_count": self.chosen_rank_top2_count,
            "chosen_rank_top3_count": self.chosen_rank_top3_count,
            "chosen_rank_beyond3_count": self.chosen_rank_beyond3_count,
        }
        result.update(
            {
                f"chosen_rank_{rank}_count": count
                for rank, count in sorted(self.chosen_rank_counts.items())
            }
        )
        return result


def decision_quality_metrics(
    stats: Mapping[str, int | float],
) -> dict[str, float]:
    """Derive the search-vs-raw-policy metrics from summed counters.

    ``chosen_policy_rank`` is the mean 1-based rank of the action the search
    picked inside the raw policy's own ordering over all legal actions.
    ``search_changed_raw_action_rate`` is the share of decisions whose final
    action differs from the raw policy's argmax; the ``_searchable`` variant
    excludes single-legal-action decisions, which the search never explores.
    Per-search averages are normalised by executed searches (one per
    determinization), and ``cutoff_rate`` is cut-offs per expanded node.
    ``chosen_rank_N_rate`` is the exact rank-N share over searchable decisions.
    """

    decisions = float(stats.get("decisions", 0))
    searchable = float(stats.get("searchable_decisions", 0))
    searches = float(stats.get("determinizations", 0))
    expanded = float(stats.get("expanded_nodes", 0))
    leaf_nodes = float(stats.get("leaf_nodes", 0))
    pruned = float(stats.get("pruned_branches", 0))
    changed = float(stats.get("changed_action_count", 0))

    def per_search(value: float) -> float:
        return (value / searches) if searches else 0.0

    chain_rate = (changed / decisions) if decisions else 0.0
    searchable_rate = (changed / searchable) if searchable else 0.0
    result: dict[str, float] = {
        "search_calls": decisions,
        "search_expansions": searches,
        "searchable_decisions": searchable,
        "nodes_per_search": per_search(expanded),
        "leaf_nodes_per_search": per_search(leaf_nodes),
        "model_positions_per_search": per_search(stats.get("model_positions", 0)),
        "folded_plies_per_search": per_search(stats.get("folded_decisions", 0)),
        "wall_ms_per_search": 1000.0 * per_search(stats.get("total_seconds", 0.0)),
        "inference_ms_per_search": 1000.0
        * per_search(stats.get("inference_seconds", 0.0)),
        "alpha_beta_cutoffs": pruned,
        "cutoff_rate": (pruned / expanded) if expanded else 0.0,
        "chosen_policy_rank": (stats.get("chosen_rank_sum", 0) / searchable)
        if searchable
        else 0.0,
        "search_changed_raw_action_count": changed,
        "search_changed_raw_action_rate": chain_rate,
        "search_changed_raw_action_rate_searchable": searchable_rate,
        # Kept for continuity with the earlier artifacts.
        "search_changed_action_rate": chain_rate,
        "search_changed_action_rate_searchable": searchable_rate,
        "chosen_rank_top1_rate": (stats.get("chosen_rank_top1_count", 0) / searchable)
        if searchable
        else 0.0,
        "chosen_rank_top2_rate": (stats.get("chosen_rank_top2_count", 0) / searchable)
        if searchable
        else 0.0,
        "chosen_rank_top3_rate": (stats.get("chosen_rank_top3_count", 0) / searchable)
        if searchable
        else 0.0,
        "searchable_decision_share": (searchable / decisions) if decisions else 0.0,
    }
    for name, value in stats.items():
        if not name.startswith("chosen_rank_") or not name.endswith("_count"):
            continue
        rank = name[len("chosen_rank_") : -len("_count")]
        if rank.isdigit():
            result[f"chosen_rank_{rank}_rate"] = (
                float(value) / searchable if searchable else 0.0
            )
    return {name: float(value) for name, value in result.items()}


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
    """Top-k policy-guided alpha-beta with critic evaluation at leaf requests.

    A ply is one model decision boundary, not necessarily one completed player
    turn.  PAYMENT, RETURN_TOKENS, and CHOOSE_NOBLE therefore count exactly as
    the model sees them during PPO training, unless ``fold_forced_phases``
    collapses single-action chains into macro transitions.
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
        self.stats = SearchStats(
            chosen_rank_counts={
                rank: 0 for rank in range(1, self.config.top_k + 1)
            }
        )
        self.last_root_values: dict[Action, float] | None = None
        self.last_root_bound_values: dict[Action, float] | None = None

    # -- inference ---------------------------------------------------------

    def _infer_logits(self, request: DecisionRequest) -> list[float]:
        """Masked actor logits for one request (root or interior node)."""

        started = time.perf_counter()
        batch = collate_encoded_decisions((encode_decision(request),))
        was_training = self.model.training
        self.model.eval()
        with torch.inference_mode():
            logits, _ = self.model(batch)
            mask = torch.as_tensor(batch.candidate_mask, device=logits.device)
            masked = self.model.masked_logits(logits, mask)
        self.model.train(was_training)
        self.stats.model_batches += 1
        self.stats.model_positions += 1
        self.stats.inference_seconds += time.perf_counter() - started
        return masked.detach().cpu()[0, : len(request.legal_actions)].tolist()

    def _leaf_value(self, request: DecisionRequest, actor: int) -> float:
        """Critic value at the horizon, from the state-only encoding."""

        started = time.perf_counter()
        batch = collate_encoded_states((encode_decision_state(request),))
        was_training = self.model.training
        self.model.eval()
        with torch.inference_mode():
            values = self.model.value_from_state(batch)
        self.model.train(was_training)
        self.stats.model_batches += 1
        self.stats.model_positions += 1
        self.stats.inference_seconds += time.perf_counter() - started
        value = float(values.detach().cpu()[0])
        if self.config.leaf_value_clip is not None:
            limit = self.config.leaf_value_clip
            value = max(-limit, min(limit, value))
        return value if actor == 0 else -value

    # -- tree --------------------------------------------------------------

    def _top_indices(self, logits: list[float]) -> tuple[int, ...]:
        """Top-k candidate indices, most preferred by the acting player first."""

        return self._ranked_indices(logits)[: self.config.top_k]

    @staticmethod
    def _ranked_indices(logits: list[float]) -> tuple[int, ...]:
        """Full raw-policy ordering over every legal action."""

        return tuple(
            sorted(range(len(logits)), key=lambda index: (-logits[index], index))
        )

    def _record_choice_metrics(
        self,
        request: DecisionRequest,
        ranked: tuple[int, ...],
        chosen: Action,
    ) -> None:
        """Rank the chosen action inside the raw policy's own ordering."""

        chosen_index = request.legal_actions.index(chosen)
        rank = ranked.index(chosen_index) + 1
        self.stats.searchable_decisions += 1
        self.stats.chosen_rank_sum += rank
        self.stats.chosen_rank_counts[rank] = (
            self.stats.chosen_rank_counts.get(rank, 0) + 1
        )
        if rank == 1:
            self.stats.chosen_rank_top1_count += 1
        if rank <= 2:
            self.stats.chosen_rank_top2_count += 1
        if rank <= 3:
            self.stats.chosen_rank_top3_count += 1
        if rank > 3:
            self.stats.chosen_rank_beyond3_count += 1
        if chosen_index != ranked[0]:
            self.stats.changed_action_count += 1

    def _expand_child(
        self,
        engine: GameEngine,
        action: Action,
    ) -> tuple[GameEngine, DecisionRequest | TerminalResult]:
        """Apply one action on a forked search engine, folding forced plies."""

        child = _clone_engine(engine)
        response = child.apply(action)
        if not self.config.fold_forced_phases:
            return child, response
        folded = 0
        while isinstance(response, DecisionRequest) and len(response.legal_actions) == 1:
            response = child.apply(response.legal_actions[0])
            folded += 1
        self.stats.folded_decisions += folded
        return child, response

    def _child_value(
        self,
        child: GameEngine,
        response: DecisionRequest | TerminalResult,
        remaining: int,
        alpha: float,
        beta: float,
    ) -> float:
        if isinstance(response, TerminalResult):
            self.stats.leaf_nodes += 1
            return _terminal_value(response)
        if remaining == 0:
            self.stats.leaf_nodes += 1
            return self._leaf_value(response, child.current_player)
        return self._alpha_beta(child, response, remaining, alpha, beta)

    def _alpha_beta(
        self,
        engine: GameEngine,
        request: DecisionRequest,
        remaining: int,
        alpha: float,
        beta: float,
    ) -> float:
        """Fail-soft alpha-beta value of ``request`` in root perspective.

        ``remaining`` counts the plies still to expand below this node, so
        ``remaining == 1`` means its children are critic leaves.
        """

        logits = self._infer_logits(request)
        maximizing = engine.current_player == 0
        best = -math.inf if maximizing else math.inf
        # Policy logit order explores each node's most preferred child first,
        # which is the ordering that prunes most for both max and min nodes.
        for action_index in self._top_indices(logits):
            action = request.legal_actions[action_index]
            child, response = self._expand_child(engine, action)
            self.stats.expanded_nodes += 1
            value = self._child_value(child, response, remaining - 1, alpha, beta)
            if maximizing:
                if value > best:
                    best = value
                if best > alpha:
                    alpha = best
                # Strict cutoff: a pruned branch is proven strictly worse than
                # the current best, never merely equal, so the root can prune
                # without losing a value tie that the (value, logit, index)
                # rule would resolve differently.
                cutoff = best > beta
            else:
                if value < best:
                    best = value
                if best < beta:
                    beta = best
                cutoff = best < alpha
            if cutoff:
                self.stats.pruned_branches += 1
                break
        return best

    # -- entry point -------------------------------------------------------

    def choose(self, request: DecisionRequest) -> Action:
        started = time.perf_counter()
        self.stats.decisions += 1
        self.last_root_values = None
        self.last_root_bound_values = None
        if len(request.legal_actions) == 1:
            self.stats.total_seconds += time.perf_counter() - started
            return request.legal_actions[0]
        if self.config.determinizations == 1:
            chosen = self._choose_single_determinization(request)
        else:
            chosen = self._choose_combined_determinizations(request)
        self.stats.total_seconds += time.perf_counter() - started
        return chosen

    def _choose_single_determinization(self, request: DecisionRequest) -> Action:
        """Root search with strict pruning, keeping the chosen value exact.

        Every root child is searched with a lower bound equal to the best exact
        value found so far.  Interior cutoffs are strict, so a cut-off child is
        *proven* strictly worse than that bound and can never win the
        ``(value, logit, index)`` tie-break; its value is not needed.  A child
        that returns exactly at the bound may be an exact tie, so it is
        re-searched with a full window.
        """

        seed = _request_seed(request, self.config.seed, 0)
        engine = determinize_request(
            request,
            seed=seed,
            catalog=self.catalog,
        )
        self.stats.determinizations += 1
        logits = self._infer_logits(request)
        ranked = self._ranked_indices(logits)
        remaining = self.config.depth - 1
        exact: dict[Action, float] = {}
        bounds: dict[Action, float] = {}
        alpha = -math.inf
        for action_index in ranked[: self.config.top_k]:
            action = request.legal_actions[action_index]
            child, response = self._expand_child(engine, action)
            self.stats.expanded_nodes += 1
            # One ulp below the best exact value: a child that comes back at or
            # above it was searched strictly inside its window and is therefore
            # exact, while a cut-off child is proven strictly worse than alpha.
            window = (
                alpha
                if alpha == -math.inf
                else math.nextafter(alpha, -math.inf)
            )
            value = self._child_value(child, response, remaining, window, math.inf)
            if value >= alpha:
                exact[action] = value
                if value > alpha:
                    alpha = value
                continue
            bounds[action] = value
        if not exact:
            raise AssertionError("search produced no root action")
        candidates = []
        for index, action in enumerate(request.legal_actions):
            value = exact.get(action)
            if value is None:
                continue
            candidates.append((value, logits[index], -index, action))
        self.last_root_values = {item[3]: item[0] for item in candidates}
        self.last_root_bound_values = dict(bounds)
        chosen = max(candidates, key=lambda item: item[:3])[3]
        self._record_choice_metrics(request, ranked, chosen)
        return chosen

    def _choose_combined_determinizations(self, request: DecisionRequest) -> Action:
        """Average exact root action values across determinizations."""

        values_by_action: dict[Action, list[float]] = {}
        root_logits: list[float] | None = None
        for sample in range(self.config.determinizations):
            seed = _request_seed(request, self.config.seed, sample)
            engine = determinize_request(
                request,
                seed=seed,
                catalog=self.catalog,
            )
            self.stats.determinizations += 1
            logits = self._infer_logits(request)
            if root_logits is None:
                root_logits = logits
            root_remaining = self.config.depth - 1
            for action_index in self._top_indices(logits):
                action = request.legal_actions[action_index]
                child, response = self._expand_child(engine, action)
                self.stats.expanded_nodes += 1
                # Every root child is searched with a full window so its value
                # stays exact; interior pruning cannot change the root choice.
                value = self._child_value(
                    child,
                    response,
                    root_remaining,
                    -math.inf,
                    math.inf,
                )
                values_by_action.setdefault(action, []).append(value)

        if root_logits is None:
            raise AssertionError("search root was not evaluated")
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
        self.last_root_values = {item[3]: item[0] for item in candidates}
        chosen = max(candidates, key=lambda item: item[:3])[3]
        self._record_choice_metrics(
            request, self._ranked_indices(root_logits), chosen
        )
        return chosen
