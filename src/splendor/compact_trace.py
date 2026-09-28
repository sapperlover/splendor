"""Versioned JSON action traces and public search-request decoding."""

from __future__ import annotations

from typing import Any, Mapping

from .catalog import Catalog, load_default_catalog
from .contracts import (
    CardView, DecisionRequest, MarketView, NobleView, Observation,
    PendingObservation, PlayerPrivateState, PlayerPublicState, PublicReservedCard,
)
from .engine import GameEngine
from .model import Color, DecisionPhase, GemVector, ReservationSource, Tier, TokenVector
from .replay import RULESET_ID, _decode_action, _decode_result, to_primitive
from .result import TerminalResult
from .rng import RNG_ALGORITHM

TRACE_VERSION = "compact-action-trace-v2"


def _tokens(value: Mapping[str, Any]) -> TokenVector:
    return TokenVector(*(int(value[color.value]) for color in Color))


def _gems(value: Mapping[str, Any]) -> GemVector:
    return GemVector(*(int(value[color.value]) for color in Color if color is not Color.GOLD))


def _card(value: Mapping[str, Any]) -> CardView:
    return CardView(str(value["card_id"]), Tier(int(value["tier"])), Color(value["bonus"]),
                    int(value["points"]), _gems(value["cost"]))


def _noble(value: Mapping[str, Any]) -> NobleView:
    return NobleView(str(value["noble_id"]), int(value["points"]), _gems(value["requirement"]))


def _public(value: Mapping[str, Any]) -> PlayerPublicState:
    return PlayerPublicState(
        _tokens(value["tokens"]), _gems(value["bonuses"]), int(value["score"]),
        tuple(value["purchased_card_ids"]), tuple(value["noble_ids"]),
        tuple(PublicReservedCard(Tier(int(r["tier"])), ReservationSource(r["source"]),
                                 None if r["card"] is None else _card(r["card"]))
              for r in value["reserved_cards"]),
    )


def decode_request(value: Mapping[str, Any]) -> DecisionRequest:
    """Decode only public information needed to reconstruct a search root."""
    if value.get("__type__") != "DecisionRequest":
        raise ValueError("unsupported search request JSON")
    raw = value["observation"]
    pending_raw = raw["pending"]
    pending = None if pending_raw is None else PendingObservation(
        _decode_action(pending_raw["main_action"]),
        None if pending_raw["payment_target"] is None else _card(pending_raw["payment_target"]),
        int(pending_raw["tokens_to_return"]),
        tuple(_noble(n) for n in pending_raw["eligible_nobles"]),
    )
    market = raw["market"]
    observation = Observation(
        DecisionPhase(raw["phase"]), bool(raw["self_is_first_player"]),
        _tokens(raw["supply"]), tuple(raw["deck_counts"]),
        MarketView(*(
            tuple(_card(card) for card in market[name])
            for name in ("tier_one", "tier_two", "tier_three")
        )),
        tuple(_noble(n) for n in raw["nobles_in_play"]),
        _public(raw["self_public"]),
        PlayerPrivateState(tuple(_card(c) for c in raw["self_private"]["blind_reserved_cards"])),
        _public(raw["opponent_public"]),
        tuple(bool(x) for x in raw["public_purchased"]),
        int(raw["own_completed_turns"]), int(raw["opponent_completed_turns"]),
        int(raw["consecutive_forced_skips"]), bool(raw["final_round_triggered"]),
        bool(raw["final_round_triggered_by_self"]), pending,
        _decode_result(raw.get("result")),
    )
    return DecisionRequest(observation, tuple(_decode_action(a) for a in value["legal_actions"]))


def new_trace(game_id: int, game_seed: int, first_player: int, catalog: Catalog) -> dict[str, Any]:
    return {
        "version": TRACE_VERSION, "game_id": game_id,
        "game_seed": game_seed, "first_player": first_player,
        "ruleset_id": RULESET_ID, "catalog_fingerprint": catalog.data_fingerprint,
        "rng_algorithm": RNG_ALGORITHM, "actions": [], "terminal_result": None,
    }


def append_action(trace: dict[str, Any], request: DecisionRequest, player_id: int, index: int) -> None:
    trace["actions"].append({
        "decision_index": len(trace["actions"]), "player_id": player_id,
        "phase": request.observation.phase.value, "candidate_index": index,
        "action": to_primitive(request.legal_actions[index]),
    })


def replay_trace(trace: Mapping[str, Any], *, require_terminal: bool = True) -> DecisionRequest | TerminalResult:
    """Checked re-execution of actual choices, independent of neural inference."""
    catalog = load_default_catalog()
    if (trace.get("version") != TRACE_VERSION or trace.get("ruleset_id") != RULESET_ID
            or trace.get("catalog_fingerprint") != catalog.data_fingerprint
            or trace.get("rng_algorithm") != RNG_ALGORITHM):
        raise ValueError("trace version or catalog mismatch")
    engine = GameEngine.new(int(trace["game_seed"]), first_player=int(trace["first_player"]),
                            catalog=catalog, mode="checked")
    current = engine.decision()
    for position, item in enumerate(trace["actions"]):
        if not isinstance(current, DecisionRequest):
            raise ValueError(f"trace action {position}: game already terminal")
        index = int(item["candidate_index"])
        action = _decode_action(item["action"])
        if (item["decision_index"] != position or item["player_id"] != engine.current_player
                or item["phase"] != current.observation.phase.value
                or not 0 <= index < len(current.legal_actions)
                or current.legal_actions[index] != action):
            raise ValueError(f"trace action {position}: actor, phase or action mismatch")
        current = engine.apply(action)
    if require_terminal:
        if not isinstance(current, TerminalResult):
            raise ValueError(f"trace ended before formal terminal at decision {len(trace['actions'])}")
        if to_primitive(current) != trace.get("terminal_result"):
            raise ValueError("trace terminal result mismatch")
        return current
    return current
