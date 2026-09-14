from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, NoReturn

from ._state import DeckState, GameState, MarketState, PendingTurn, PlayerState, ReservedCard
from .actions import (
    Action,
    BuyMarket,
    BuyReserved,
    ClaimNoble,
    Pay,
    ReserveDeck,
    ReserveMarket,
    ReturnTokens,
    TakeDistinct,
    TakePair,
)
from .catalog import Catalog
from .invariants import InvariantViolation, assert_game_state
from .model import Color, DecisionPhase, ReservationSource, Tier, TokenVector
from .result import TerminalReason, TerminalResult
from .rng import RNG_ALGORITHM, RngState


REPLAY_SCHEMA_VERSION = 2
RULESET_ID = "splendor-base-two-player-v2-forced-skip"


@dataclass(frozen=True, slots=True)
class ReplayHeader:
    schema_version: int
    ruleset_id: str
    data_fingerprint: str
    rng_algorithm: str
    game_seed: int
    first_player: int


@dataclass(frozen=True, slots=True)
class ActionEvent:
    sequence: int
    actor: int
    phase: DecisionPhase
    action: Action
    automatic: bool
    state_digest_before: str
    state_digest_after: str


@dataclass(frozen=True, slots=True)
class ForcedSkipEvent:
    """Automatic engine event; this is deliberately not a player Action."""

    sequence: int
    actor: int
    state_digest_before: str
    state_digest_after: str
    phase: DecisionPhase = DecisionPhase.MAIN_ACTION
    automatic: bool = True
    action: None = None


ReplayEvent = ActionEvent | ForcedSkipEvent


@dataclass(frozen=True, slots=True)
class ReplayRecord:
    header: ReplayHeader
    events: tuple[ReplayEvent, ...] = ()
    result: TerminalResult | None = None


def new_replay(catalog: Catalog, game_seed: int, first_player: int) -> ReplayRecord:
    return ReplayRecord(
        ReplayHeader(
            schema_version=REPLAY_SCHEMA_VERSION,
            ruleset_id=RULESET_ID,
            data_fingerprint=catalog.data_fingerprint,
            rng_algorithm=RNG_ALGORITHM,
            game_seed=game_seed,
            first_player=first_player,
        )
    )


def to_primitive(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "__type__": type(value).__name__,
            **{field.name: to_primitive(getattr(value, field.name)) for field in fields(value)},
        }
    if isinstance(value, tuple):
        return [to_primitive(item) for item in value]
    if isinstance(value, list):
        return [to_primitive(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_primitive(item) for key, item in value.items()}
    return value


def state_digest(state: GameState) -> str:
    payload = json.dumps(to_primitive(state), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def save_game(path: Path, state: GameState, replay: ReplayRecord) -> None:
    payload = {"state": to_primitive(state), "replay": to_primitive(replay)}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_game(path: Path, catalog: Catalog) -> tuple[GameState, ReplayRecord]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    replay = _decode_replay(payload["replay"])
    header = replay.header
    if header.schema_version != REPLAY_SCHEMA_VERSION:
        raise ValueError(f"unsupported save schema: {header.schema_version}")
    if header.ruleset_id != RULESET_ID:
        raise ValueError(f"wrong ruleset: {header.ruleset_id}")
    if header.data_fingerprint != catalog.data_fingerprint:
        raise ValueError("save uses different normative data")
    if header.rng_algorithm != RNG_ALGORITHM:
        raise ValueError(f"unsupported replay RNG: {header.rng_algorithm}")
    return _decode_state(payload["state"]), replay


def _decode_tokens(data: dict[str, Any]) -> TokenVector:
    return TokenVector(*(int(data[color.value]) for color in Color))


def _decode_action(data: dict[str, Any]) -> Action:
    kind = data["__type__"]
    if kind == "TakeDistinct":
        return TakeDistinct(tuple(Color(color) for color in data["colors"]))
    if kind == "TakePair":
        return TakePair(Color(data["color"]))
    if kind == "ReserveMarket":
        return ReserveMarket(str(data["card_id"]))
    if kind == "ReserveDeck":
        return ReserveDeck(Tier(int(data["tier"])))
    if kind == "BuyMarket":
        return BuyMarket(str(data["card_id"]))
    if kind == "BuyReserved":
        return BuyReserved(str(data["card_id"]))
    if kind == "Pay":
        return Pay(_decode_tokens(data["tokens"]))
    if kind == "ReturnTokens":
        return ReturnTokens(_decode_tokens(data["tokens"]))
    if kind == "ClaimNoble":
        return ClaimNoble(str(data["noble_id"]))
    raise ValueError(f"unknown action type: {kind}")


def _decode_player(data: dict[str, Any]) -> PlayerState:
    return PlayerState(
        tokens=_decode_tokens(data["tokens"]),
        purchased_card_ids=tuple(data["purchased_card_ids"]),
        reserved_cards=tuple(
            ReservedCard(str(card["card_id"]), ReservationSource(card["source"]))
            for card in data["reserved_cards"]
        ),
        noble_ids=tuple(data["noble_ids"]),
    )


def _decode_result(data: dict[str, Any] | None) -> TerminalResult | None:
    if data is None:
        return None
    return TerminalResult(
        scores=tuple(data["scores"]),
        purchased_card_counts=tuple(data["purchased_card_counts"]),
        winner_ids=tuple(data["winner_ids"]),
        completed_turns=tuple(data["completed_turns"]),
        reason=TerminalReason(data.get("reason", TerminalReason.FINAL_ROUND.value)),
    )


def _decode_state(data: dict[str, Any]) -> GameState:
    deck = data["decks"]
    market = data["market"]
    pending_data = data["pending"]
    pending = None
    if pending_data is not None:
        pending = PendingTurn(
            main_action=_decode_action(pending_data["main_action"]),
            tokens_to_return=int(pending_data["tokens_to_return"]),
            eligible_noble_ids=tuple(pending_data["eligible_noble_ids"]),
        )
    rng = data["rng"]
    return GameState(
        supply=_decode_tokens(data["supply"]),
        decks=DeckState(tuple(deck["tier_one"]), tuple(deck["tier_two"]), tuple(deck["tier_three"])),
        market=MarketState(
            tuple(market["tier_one"]),
            tuple(market["tier_two"]),
            tuple(market["tier_three"]),
        ),
        players=tuple(_decode_player(player) for player in data["players"]),
        nobles_in_play=tuple(data["nobles_in_play"]),
        nobles_out_of_game=tuple(data["nobles_out_of_game"]),
        first_player=int(data["first_player"]),
        current_player=int(data["current_player"]),
        completed_turns=tuple(data["completed_turns"]),
        consecutive_forced_skips=int(data.get("consecutive_forced_skips", 0)),
        phase=DecisionPhase(data["phase"]),
        pending=pending,
        final_round_triggered_by=data["final_round_triggered_by"],
        result=_decode_result(data["result"]),
        rng=RngState(str(rng["algorithm"]), int(rng["state"]), int(rng["draws"])),
    )


def _decode_replay(data: dict[str, Any]) -> ReplayRecord:
    header = data["header"]
    events: list[ReplayEvent] = []
    for event in data["events"]:
        if event.get("__type__") == "ForcedSkipEvent":
            if (
                DecisionPhase(event["phase"]) is not DecisionPhase.MAIN_ACTION
                or not bool(event["automatic"])
                or event["action"] is not None
            ):
                raise ValueError("invalid forced-skip replay event")
            events.append(
                ForcedSkipEvent(
                    sequence=int(event["sequence"]),
                    actor=int(event["actor"]),
                    state_digest_before=str(event["state_digest_before"]),
                    state_digest_after=str(event["state_digest_after"]),
                )
            )
            continue
        events.append(
            ActionEvent(
                sequence=int(event["sequence"]),
                actor=int(event["actor"]),
                phase=DecisionPhase(event["phase"]),
                action=_decode_action(event["action"]),
                automatic=bool(event["automatic"]),
                state_digest_before=str(event["state_digest_before"]),
                state_digest_after=str(event["state_digest_after"]),
            )
        )
    return ReplayRecord(
        header=ReplayHeader(
            schema_version=int(header["schema_version"]),
            ruleset_id=str(header["ruleset_id"]),
            data_fingerprint=str(header["data_fingerprint"]),
            rng_algorithm=str(header["rng_algorithm"]),
            game_seed=int(header["game_seed"]),
            first_player=int(header["first_player"]),
        ),
        events=tuple(events),
        result=_decode_result(data["result"]),
    )


class InvariantGuard:
    """Validate or atomically save a replayable failure scene and re-raise."""

    def __init__(self, catalog: Catalog, replay: ReplayRecord, failure_dir: Path) -> None:
        self.catalog = catalog
        self.replay = replay
        self.failure_dir = failure_dir

    def check(self, state: GameState, attempted_action: Action | None = None) -> None:
        try:
            assert_game_state(state, self.catalog)
        except InvariantViolation as error:
            self.fail(state, error, attempted_action)

    def fail(self, state: GameState, error: Exception, attempted_action: Action | None = None) -> NoReturn:
        code = str(getattr(error, "code", type(error).__name__))
        detail = str(getattr(error, "detail", str(error)))
        report = {
            "schema_version": REPLAY_SCHEMA_VERSION,
            "error": {"code": code, "detail": detail},
            "attempted_action": to_primitive(attempted_action),
            "replay": to_primitive(self.replay),
            "state": to_primitive(state),
            "state_digest": state_digest(state),
        }
        path = self._unused_path(code, len(self.replay.events))
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
        error.add_note(f"failure snapshot: {path}")
        raise error

    def _unused_path(self, code: str, sequence: int) -> Path:
        base = self.failure_dir / f"failure-{sequence:06d}-{code}.json"
        if not base.exists():
            return base
        suffix = 1
        while True:
            candidate = self.failure_dir / f"failure-{sequence:06d}-{code}-{suffix}.json"
            if not candidate.exists():
                return candidate
            suffix += 1
