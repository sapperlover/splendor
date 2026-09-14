from __future__ import annotations

from collections import Counter

from ._state import GameState
from .actions import BuyMarket, BuyReserved, MAIN_ACTION_TYPES
from .catalog import Catalog
from .model import DecisionPhase, TIERS, TOKEN_COLORS
from .rng import DeterministicRng
from .result import TerminalReason
from .rules import determine_winners, eligible_noble_ids, legal_payments_for_card, player_score


class InvariantViolation(AssertionError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"[{code}] {detail}")


def _require(condition: bool, code: str, detail: str) -> None:
    if not condition:
        raise InvariantViolation(code, detail)


def _assert_nonnegative_vector(name: str, values: object) -> None:
    for value in values:  # type: ignore[union-attr]
        _require(type(value) is int and value >= 0, "nonnegative_integer", f"{name} contains {value!r}")


def assert_game_state(state: GameState, catalog: Catalog) -> None:
    """Fail on the first corrupt fact. This function never repairs state."""

    _require(len(state.players) == 2, "player_count", "the engine supports exactly two players")
    _require(state.first_player in (0, 1), "first_player", repr(state.first_player))
    _require(state.current_player in (0, 1), "current_player", repr(state.current_player))
    _require(isinstance(state.phase, DecisionPhase), "phase", repr(state.phase))
    _require(
        state.final_round_triggered_by is None or state.final_round_triggered_by in (0, 1),
        "terminal_trigger",
        repr(state.final_round_triggered_by),
    )
    try:
        DeterministicRng.restore(state.rng)
    except ValueError as error:
        raise InvariantViolation("rng_state", str(error)) from error
    _require(
        all(type(turns) is int and turns >= 0 for turns in state.completed_turns),
        "completed_turns",
        repr(state.completed_turns),
    )
    _require(
        type(state.consecutive_forced_skips) is int
        and 0 <= state.consecutive_forced_skips <= 2,
        "forced_skip_streak",
        repr(state.consecutive_forced_skips),
    )
    if state.phase is not DecisionPhase.TERMINAL and state.consecutive_forced_skips:
        _require(
            state.consecutive_forced_skips == 1
            and state.phase is DecisionPhase.MAIN_ACTION
            and state.pending is None,
            "forced_skip_streak",
            "a live forced-skip streak must await the next MAIN_ACTION",
        )
    first, other = state.first_player, 1 - state.first_player
    _require(
        state.completed_turns[first] in (state.completed_turns[other], state.completed_turns[other] + 1),
        "turn_balance",
        repr(state.completed_turns),
    )
    expected_actor = first if state.completed_turns[first] == state.completed_turns[other] else other
    if state.phase is not DecisionPhase.TERMINAL:
        _require(state.current_player == expected_actor, "turn_order", f"expected player {expected_actor}")
        if state.final_round_triggered_by is not None:
            _require(
                state.final_round_triggered_by == first
                and state.completed_turns[first] == state.completed_turns[other] + 1,
                "final_round_progress",
                "a non-terminal final round must await the second player's reply",
            )

    _assert_nonnegative_vector("supply", state.supply)
    for player_id, player in enumerate(state.players):
        _assert_nonnegative_vector(f"players[{player_id}].tokens", player.tokens)
        _require(
            len(player.reserved_cards) <= catalog.constants.max_reserved_cards,
            "reserve_limit",
            f"player {player_id} has {len(player.reserved_cards)} reserved cards",
        )

    for color in TOKEN_COLORS:
        actual = state.supply.for_color(color) + sum(
            player.tokens.for_color(color) for player in state.players
        )
        expected = catalog.constants.initial_tokens.for_color(color)
        _require(actual == expected, "token_conservation", f"{color.value}: expected {expected}, got {actual}")

    card_locations = list(state.decks.all_cards()) + list(state.market.all_cards())
    for player in state.players:
        card_locations.extend(player.purchased_card_ids)
        card_locations.extend(reserved.card_id for reserved in player.reserved_cards)
    expected_cards = {card.card_id for card in catalog.cards}
    counts = Counter(card_locations)
    duplicates = sorted(card_id for card_id, count in counts.items() if count != 1)
    _require(not duplicates, "card_uniqueness", f"non-unique locations: {duplicates}")
    _require(set(card_locations) == expected_cards, "card_coverage", "card locations do not match catalog")

    for tier in TIERS:
        deck = state.decks.for_tier(tier)
        market = state.market.for_tier(tier)
        _require(
            len(market) <= catalog.constants.face_up_per_tier,
            "market_capacity",
            f"tier {tier.value} has {len(market)} cards",
        )
        _require(
            not deck or len(market) == catalog.constants.face_up_per_tier,
            "market_refill",
            f"tier {tier.value} has a non-empty deck but only {len(market)} market cards",
        )
        for card_id in deck + market:
            _require(catalog.card(card_id).tier is tier, "card_tier", f"{card_id} is in tier {tier.value}")

    noble_locations = list(state.nobles_in_play) + list(state.nobles_out_of_game)
    for player in state.players:
        noble_locations.extend(player.noble_ids)
    expected_nobles = {noble.noble_id for noble in catalog.nobles}
    noble_counts = Counter(noble_locations)
    duplicate_nobles = sorted(noble_id for noble_id, count in noble_counts.items() if count != 1)
    _require(not duplicate_nobles, "noble_uniqueness", f"non-unique locations: {duplicate_nobles}")
    _require(set(noble_locations) == expected_nobles, "noble_coverage", "noble locations do not match catalog")
    selected_nobles = len(state.nobles_in_play) + sum(len(player.noble_ids) for player in state.players)
    _require(
        selected_nobles == catalog.constants.nobles_in_play,
        "nobles_selected",
        f"expected {catalog.constants.nobles_in_play}, got {selected_nobles}",
    )
    if state.final_round_triggered_by is not None:
        trigger_score = player_score(state.players[state.final_round_triggered_by], catalog)
        _require(
            trigger_score >= catalog.constants.end_game_prestige,
            "terminal_trigger_score",
            f"player {state.final_round_triggered_by} triggered at {trigger_score} points",
        )

    _assert_phase_context(state, catalog)
    _assert_terminal_result(state, catalog)


def _assert_phase_context(state: GameState, catalog: Catalog) -> None:
    player = state.players[state.current_player]
    limit = catalog.constants.max_tokens_at_turn_end
    pending = state.pending

    for player_id, candidate in enumerate(state.players):
        may_exceed = state.phase is DecisionPhase.RETURN_TOKENS and player_id == state.current_player
        if not may_exceed:
            _require(
                candidate.tokens.total <= limit,
                "turn_end_token_limit",
                f"player {player_id} has {candidate.tokens.total}",
            )

    if state.phase in (DecisionPhase.MAIN_ACTION, DecisionPhase.TERMINAL):
        _require(pending is None, "phase_context", f"{state.phase.value} must not have pending context")
    else:
        _require(pending is not None, "phase_context", f"{state.phase.value} requires pending context")
    if pending is None:
        return

    _require(isinstance(pending.main_action, MAIN_ACTION_TYPES), "phase_context", "pending action is not a main action")

    if state.phase is DecisionPhase.PAYMENT:
        _require(
            isinstance(pending.main_action, (BuyMarket, BuyReserved)),
            "payment_context",
            "payment must follow a buy target",
        )
        _require(pending.tokens_to_return == 0 and not pending.eligible_noble_ids, "payment_context", "extraneous fields")
        if isinstance(pending.main_action, BuyMarket):
            _require(
                pending.main_action.card_id in state.market.all_cards(),
                "payment_target",
                "market purchase target is no longer in the market",
            )
        elif isinstance(pending.main_action, BuyReserved):
            reserved_ids = {reserved.card_id for reserved in player.reserved_cards}
            _require(
                pending.main_action.card_id in reserved_ids,
                "payment_target",
                "reserved purchase target is not held by the current player",
            )
        card = catalog.card(pending.main_action.card_id)
        _require(
            bool(legal_payments_for_card(player, card, catalog)),
            "payment_options",
            "payment target has no legal payment",
        )
    elif state.phase is DecisionPhase.RETURN_TOKENS:
        excess = player.tokens.total - limit
        _require(excess > 0, "return_context", "return phase requires excess tokens")
        _require(pending.tokens_to_return == excess, "return_context", f"expected {excess} tokens to return")
        _require(not pending.eligible_noble_ids, "return_context", "nobles are checked only after returning")
    elif state.phase is DecisionPhase.CHOOSE_NOBLE:
        expected = set(eligible_noble_ids(player, state.nobles_in_play, catalog))
        _require(pending.tokens_to_return == 0, "noble_context", "return must finish first")
        _require(expected and set(pending.eligible_noble_ids) == expected, "noble_context", "eligible nobles mismatch")
        _require(
            len(pending.eligible_noble_ids) == len(set(pending.eligible_noble_ids)),
            "noble_context",
            "duplicate eligible nobles",
        )


def _assert_terminal_result(state: GameState, catalog: Catalog) -> None:
    if state.phase is not DecisionPhase.TERMINAL:
        _require(state.result is None, "terminal_result", "non-terminal state has a result")
        return

    _require(state.result is not None, "terminal_result", "terminal state has no result")
    result = state.result
    if result is None:  # narrowed by the assertion above
        return
    scores = tuple(player_score(player, catalog) for player in state.players)
    counts = tuple(len(player.purchased_card_ids) for player in state.players)
    _require(result.scores == scores, "result_score", f"expected {scores}, got {result.scores}")
    _require(result.purchased_card_counts == counts, "result_card_count", f"expected {counts}")
    _require(result.completed_turns == state.completed_turns, "result_turns", "completed turns mismatch")
    if result.reason is TerminalReason.FINAL_ROUND:
        _require(
            state.final_round_triggered_by in (0, 1),
            "terminal_trigger",
            "final-round terminal game was never triggered",
        )
        _require(
            state.completed_turns[0] == state.completed_turns[1],
            "terminal_turn_balance",
            repr(state.completed_turns),
        )
        _require(max(scores) >= catalog.constants.end_game_prestige, "terminal_score", repr(scores))
        _require(
            result.winner_ids == determine_winners(scores, counts),
            "result_winners",
            "winner/tiebreak mismatch",
        )
    elif result.reason is TerminalReason.CONSECUTIVE_FORCED_SKIPS:
        _require(state.final_round_triggered_by is None, "terminal_trigger", "forced-skip draw has final trigger")
        _require(state.consecutive_forced_skips == 2, "forced_skip_streak", "forced-skip draw requires two skips")
        _require(result.winner_ids == (0, 1), "result_winners", "forced-skip terminal must be a draw")
    else:
        _require(False, "terminal_reason", repr(result.reason))
