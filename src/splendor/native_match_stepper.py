"""Authoritative native real-game stepper with the existing public trace contract."""

from __future__ import annotations

from ._state import DeckState, GameState, MarketState, PendingTurn, PlayerState, ReservedCard
from .actions import (BuyMarket, BuyReserved, ClaimNoble, Pay, ReserveDeck,
                      ReserveMarket, ReturnTokens, TakeDistinct, TakePair)
from .catalog import Catalog
from .compact_trace import append_action, new_trace
from .contracts import DecisionRequest
from .engine import GameEngine
from .model import Color, DecisionPhase, ReservationSource, Tier, TokenVector
from .native_policy import _default_catalog, _extension, _native_catalog
from .observation import build_observation
from .result import TerminalReason, TerminalResult
from .rules import player_score


_PHASES = tuple(DecisionPhase(name) for name in (
    "main_action", "payment", "return_tokens", "choose_noble", "terminal"
))
_COLORS = tuple(Color(name) for name in ("white", "blue", "green", "red", "black"))


class NativeMatchStepper:
    def __init__(self, game_id: int, game_seed: int, first_player: int, *, mode: str) -> None:
        if mode != "fast":
            raise ValueError("native real-game stepper requires fast engine mode")
        self.catalog = _default_catalog()
        setup = GameEngine.new(game_seed, first_player=first_player,
                               catalog=self.catalog, mode="fast")
        initial = setup._debug_state()
        self._rng = initial.rng
        self._nobles_out = initial.nobles_out_of_game
        self.native = _extension().NativeGame(_native_catalog(self.catalog), initial)
        self.engine = self
        self.trace = new_trace(game_id, game_seed, first_player, self.catalog)
        self.decision_count = 0
        self.current = self._response()

    def _action(self, raw):
        kind, identity, tier, values = raw
        if kind == 0:
            return TakeDistinct(tuple(_COLORS[i] for i in range(5) if values[i]))
        if kind == 1:
            return TakePair(_COLORS[next(i for i in range(5) if values[i])])
        if kind == 2:
            return ReserveMarket(self.catalog.cards[identity].card_id)
        if kind == 3:
            return ReserveDeck(Tier(tier + 1))
        if kind == 4:
            return BuyMarket(self.catalog.cards[identity].card_id)
        if kind == 5:
            return BuyReserved(self.catalog.cards[identity].card_id)
        if kind == 6:
            return Pay(TokenVector(*values))
        if kind == 7:
            return ReturnTokens(TokenVector(*values))
        return ClaimNoble(self.catalog.nobles[identity].noble_id)

    def _state(self, snapshot) -> GameState:
        cards = self.catalog.cards
        nobles = self.catalog.nobles
        player_rows = []
        for row in snapshot["players"]:
            player_rows.append(PlayerState(
                tokens=TokenVector(*row["tokens"]),
                purchased_card_ids=tuple(cards[i].card_id for i in row["purchased"]),
                reserved_cards=tuple(ReservedCard(
                    cards[i].card_id, ReservationSource.MARKET if source == 0 else ReservationSource.DECK
                ) for i, source in row["reserved"]),
                noble_ids=tuple(nobles[i].noble_id for i in row["nobles"]),
            ))
        players = tuple(player_rows)
        pending = None
        if snapshot["pending"] is not None:
            pending = PendingTurn(
                self._action(snapshot["pending"]),
                tokens_to_return=snapshot["return_count"],
                eligible_noble_ids=tuple(nobles[i].noble_id for i in snapshot["eligible"]),
            )
        result = None
        if snapshot["phase"] == 4:
            winner = snapshot["winner"]
            result = TerminalResult(
                scores=tuple(player_score(player, self.catalog) for player in players),
                purchased_card_counts=tuple(len(player.purchased_card_ids) for player in players),
                winner_ids=(0, 1) if winner == -1 else (winner,),
                completed_turns=tuple(snapshot["turns"]),
                reason=(TerminalReason.FINAL_ROUND if snapshot["trigger"] >= 0 else
                        TerminalReason.CONSECUTIVE_FORCED_SKIPS),
            )
        return GameState(
            supply=TokenVector(*snapshot["supply"]),
            decks=DeckState(*(tuple(cards[i].card_id for i in tier) for tier in snapshot["decks"])),
            market=MarketState(*(tuple(cards[i].card_id for i in tier) for tier in snapshot["market"])),
            players=players,
            nobles_in_play=tuple(nobles[i].noble_id for i in snapshot["nobles"]),
            nobles_out_of_game=self._nobles_out,
            first_player=snapshot["first"], current_player=snapshot["actor"],
            completed_turns=tuple(snapshot["turns"]),
            consecutive_forced_skips=snapshot["skips"],
            phase=_PHASES[snapshot["phase"]], pending=pending,
            final_round_triggered_by=None if snapshot["trigger"] < 0 else snapshot["trigger"],
            result=result, rng=self._rng,
        )

    def _response(self) -> DecisionRequest | TerminalResult:
        state = self._state(self.native.snapshot())
        if state.result is not None:
            return state.result
        return DecisionRequest(
            build_observation(state, state.current_player, self.catalog),
            tuple(self._action(action) for action in self.native.actions()),
        )

    @property
    def request(self) -> DecisionRequest:
        if not isinstance(self.current, DecisionRequest):
            raise RuntimeError("match is already terminal")
        return self.current

    @property
    def terminal(self) -> TerminalResult | None:
        return self.current if isinstance(self.current, TerminalResult) else None

    @property
    def current_player(self) -> int:
        return self.native.snapshot()["actor"]

    @property
    def forced_skip_count(self) -> int:
        return self.native.snapshot()["forced_skip_total"]

    @property
    def completed_player_turns(self) -> int:
        return sum(self.native.snapshot()["turns"])

    def apply(self, candidate_index: int) -> DecisionRequest | TerminalResult:
        request = self.request
        if not 0 <= candidate_index < len(request.legal_actions):
            raise ValueError("candidate index out of range")
        actor = self.current_player
        self.native.apply_index(candidate_index)
        append_action(self.trace, request, actor, candidate_index)
        self.decision_count += 1
        self.current = self._response()
        if isinstance(self.current, TerminalResult):
            from .replay import to_primitive
            self.trace["terminal_result"] = to_primitive(self.current)
        return self.current
