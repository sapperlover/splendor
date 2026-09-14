"""Shared pure rule derivations.

The state machine and invariant checker both call these functions.  Future
legal-action generation must extend this module instead of duplicating them in
policies, observation encoders, or tests.
"""

from __future__ import annotations

from dataclasses import replace
from itertools import combinations

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
    MAIN_ACTION_TYPES,
)
from .catalog import CardDefinition, Catalog
from .model import DecisionPhase, GEM_COLORS, GemVector, ReservationSource, TIERS, TOKEN_COLORS, Tier, TokenVector
from .result import TerminalReason, TerminalResult


def player_bonuses(player: PlayerState, catalog: Catalog) -> GemVector:
    counts = {name: 0 for name in ("white", "blue", "green", "red", "black")}
    for card_id in player.purchased_card_ids:
        counts[catalog.card(card_id).bonus.value] += 1
    return GemVector(**counts)


def player_score(player: PlayerState, catalog: Catalog) -> int:
    return sum(catalog.card(card_id).points for card_id in player.purchased_card_ids) + sum(
        catalog.noble(noble_id).points for noble_id in player.noble_ids
    )


def eligible_noble_ids(player: PlayerState, noble_ids: tuple[str, ...], catalog: Catalog) -> tuple[str, ...]:
    bonuses = player_bonuses(player, catalog)
    return tuple(
        noble_id
        for noble_id in noble_ids
        if all(
            have >= need
            for have, need in zip(bonuses, catalog.noble(noble_id).requirement, strict=True)
        )
    )


def determine_winners(scores: tuple[int, int], card_counts: tuple[int, int]) -> tuple[int, ...]:
    highest = max(scores)
    finalists = tuple(player for player, score in enumerate(scores) if score == highest)
    if len(finalists) == 1:
        return finalists
    fewest_cards = min(card_counts[player] for player in finalists)
    return tuple(player for player in finalists if card_counts[player] == fewest_cards)


def legal_actions(state: GameState, catalog: Catalog) -> tuple[Action, ...]:
    """The only source of legal action candidates for the implemented slice."""

    if state.phase is DecisionPhase.MAIN_ACTION:
        return _legal_main_actions(state, catalog)
    if state.phase is DecisionPhase.PAYMENT:
        return _legal_payment_actions(state, catalog)
    if state.phase is DecisionPhase.RETURN_TOKENS:
        return _legal_returns(state, catalog)
    if state.phase is DecisionPhase.CHOOSE_NOBLE:
        return _legal_nobles(state, catalog)
    return ()


def _legal_main_actions(state: GameState, catalog: Catalog) -> tuple[Action, ...]:
    actions: list[Action] = []
    available = tuple(color for color in GEM_COLORS if state.supply.for_color(color) > 0)
    if len(available) >= 3:
        actions.extend(TakeDistinct(colors) for colors in combinations(available, 3))
    elif available:
        actions.append(TakeDistinct(available))

    actions.extend(
        TakePair(color)
        for color in GEM_COLORS
        if state.supply.for_color(color) >= 4
    )

    player = state.players[state.current_player]
    if len(player.reserved_cards) < catalog.constants.max_reserved_cards:
        actions.extend(ReserveMarket(card_id) for card_id in sorted(state.market.all_cards()))
        actions.extend(
            ReserveDeck(tier)
            for tier in TIERS
            if state.decks.for_tier(tier)
        )

    actions.extend(
        BuyMarket(card_id)
        for card_id in sorted(state.market.all_cards())
        if legal_payments_for_card(player, catalog.card(card_id), catalog)
    )
    actions.extend(
        BuyReserved(card_id)
        for card_id in sorted(reserved.card_id for reserved in player.reserved_cards)
        if legal_payments_for_card(player, catalog.card(card_id), catalog)
    )
    return tuple(actions)


def discounted_need(player: PlayerState, card: CardDefinition, catalog: Catalog) -> GemVector:
    """Return the five-color cost remaining after permanent card bonuses."""

    bonuses = player_bonuses(player, catalog)
    return GemVector(
        *(max(0, cost - bonus) for cost, bonus in zip(card.cost, bonuses, strict=True))
    )


def legal_payments_for_card(
    player: PlayerState,
    card: CardDefinition,
    catalog: Catalog,
) -> tuple[Pay, ...]:
    """Enumerate every distinct actual token payment for one purchase target.

    Gold use is a choice per gem color.  Distinct gold allocations that yield
    the same actual six-color payment are represented by one ``Pay`` action.
    """

    need = tuple(discounted_need(player, card, catalog))
    held = tuple(player.tokens)
    choices: set[Pay] = set()

    def enumerate_gold(index: int, gold_used: int, ordinary: list[int]) -> None:
        if index == len(GEM_COLORS):
            choices.add(Pay(TokenVector(*ordinary, gold_used)))
            return

        for gold_for_color in range(need[index] + 1):
            ordinary_paid = need[index] - gold_for_color
            if ordinary_paid > held[index] or gold_used + gold_for_color > player.tokens.gold:
                continue
            ordinary.append(ordinary_paid)
            enumerate_gold(index + 1, gold_used + gold_for_color, ordinary)
            ordinary.pop()

    enumerate_gold(0, 0, [])
    return tuple(sorted(choices, key=lambda action: tuple(action.tokens)))


def _legal_payment_actions(state: GameState, catalog: Catalog) -> tuple[Action, ...]:
    pending = state.pending
    if pending is None or not isinstance(pending.main_action, (BuyMarket, BuyReserved)):
        return ()
    card = catalog.card(pending.main_action.card_id)
    return legal_payments_for_card(state.players[state.current_player], card, catalog)


def _legal_returns(state: GameState, catalog: Catalog) -> tuple[Action, ...]:
    player = state.players[state.current_player]
    excess = player.tokens.total - catalog.constants.max_tokens_at_turn_end
    if excess <= 0:
        return ()

    held = tuple(player.tokens)
    choices: list[ReturnTokens] = []

    def enumerate_vectors(index: int, remaining: int, values: list[int]) -> None:
        if index == len(TOKEN_COLORS):
            if remaining == 0:
                choices.append(ReturnTokens(TokenVector(*values)))
            return
        for amount in range(min(held[index], remaining) + 1):
            values.append(amount)
            enumerate_vectors(index + 1, remaining - amount, values)
            values.pop()

    enumerate_vectors(0, excess, [])
    return tuple(choices)


def _legal_nobles(state: GameState, catalog: Catalog) -> tuple[Action, ...]:
    player = state.players[state.current_player]
    eligible = eligible_noble_ids(player, state.nobles_in_play, catalog)
    return tuple(ClaimNoble(noble_id) for noble_id in sorted(eligible))


def apply_legal_action(state: GameState, action: Action, catalog: Catalog) -> GameState:
    """Apply an action already proven to be in ``legal_actions``."""

    if isinstance(action, MAIN_ACTION_TYPES) and state.consecutive_forced_skips:
        state = replace(state, consecutive_forced_skips=0)

    if isinstance(action, ReturnTokens):
        return _apply_return(state, action, catalog)
    if isinstance(action, Pay):
        return _apply_payment(state, action, catalog)
    if isinstance(action, ClaimNoble):
        return _apply_claim_noble(state, action, catalog)
    if isinstance(action, (BuyMarket, BuyReserved)):
        return replace(
            state,
            phase=DecisionPhase.PAYMENT,
            pending=PendingTurn(action),
        )
    if isinstance(action, (TakeDistinct, TakePair, ReserveMarket, ReserveDeck)):
        changed = _apply_main(state, action, catalog)
        player = changed.players[changed.current_player]
        excess = player.tokens.total - catalog.constants.max_tokens_at_turn_end
        if excess > 0:
            return replace(
                changed,
                phase=DecisionPhase.RETURN_TOKENS,
                pending=PendingTurn(action, tokens_to_return=excess),
            )
        return _begin_turn_settlement(changed, action, catalog)
    raise TypeError(f"action is outside the implemented rule slice: {type(action).__name__}")


def apply_forced_skip(state: GameState, catalog: Catalog) -> GameState:
    """Complete a choice-free turn only when MAIN_ACTION has no candidate."""

    if state.phase is not DecisionPhase.MAIN_ACTION or state.pending is not None:
        raise ValueError("forced skip requires MAIN_ACTION without pending context")
    if _legal_main_actions(state, catalog):
        raise ValueError("forced skip is forbidden while a legal main action exists")
    if state.consecutive_forced_skips not in (0, 1):
        raise ValueError("invalid forced-skip streak")

    actor = state.current_player
    completed = list(state.completed_turns)
    completed[actor] += 1
    completed_turns = tuple(completed)
    streak = state.consecutive_forced_skips + 1

    # A forced skip still completes the outstanding final-round reply.
    if (
        state.final_round_triggered_by is not None
        and completed_turns[0] == completed_turns[1]
    ):
        scores = tuple(player_score(player, catalog) for player in state.players)
        card_counts = tuple(len(player.purchased_card_ids) for player in state.players)
        return replace(
            state,
            completed_turns=completed_turns,
            consecutive_forced_skips=streak,
            phase=DecisionPhase.TERMINAL,
            result=TerminalResult(
                scores=scores,
                purchased_card_counts=card_counts,
                winner_ids=determine_winners(scores, card_counts),
                completed_turns=completed_turns,
                reason=TerminalReason.FINAL_ROUND,
            ),
        )

    if streak == 2:
        scores = tuple(player_score(player, catalog) for player in state.players)
        card_counts = tuple(len(player.purchased_card_ids) for player in state.players)
        return replace(
            state,
            completed_turns=completed_turns,
            consecutive_forced_skips=streak,
            phase=DecisionPhase.TERMINAL,
            result=TerminalResult(
                scores=scores,
                purchased_card_counts=card_counts,
                winner_ids=(0, 1),
                completed_turns=completed_turns,
                reason=TerminalReason.CONSECUTIVE_FORCED_SKIPS,
            ),
        )

    return replace(
        state,
        current_player=1 - actor,
        completed_turns=completed_turns,
        consecutive_forced_skips=streak,
    )


def _apply_main(
    state: GameState,
    action: TakeDistinct | TakePair | ReserveMarket | ReserveDeck,
    catalog: Catalog,
) -> GameState:
    actor = state.current_player
    player = state.players[actor]
    supply = state.supply
    decks = state.decks
    market = state.market

    if isinstance(action, TakeDistinct):
        taken = TokenVector(**{color.value: 1 for color in action.colors})
        player = replace(player, tokens=player.tokens.plus(taken))
        supply = supply.minus(taken)
    elif isinstance(action, TakePair):
        taken = TokenVector(**{action.color.value: 2})
        player = replace(player, tokens=player.tokens.plus(taken))
        supply = supply.minus(taken)
    elif isinstance(action, ReserveMarket):
        card = catalog.card(action.card_id)
        cards = market.for_tier(card.tier)
        market = market.replacing_tier(card.tier, tuple(card_id for card_id in cards if card_id != action.card_id))
        player = replace(
            player,
            reserved_cards=player.reserved_cards + (ReservedCard(action.card_id, ReservationSource.MARKET),),
        )
        decks, market = _refill_market(decks, market, card.tier)
        supply, player = _take_gold_if_available(supply, player)
    elif isinstance(action, ReserveDeck):
        cards = decks.for_tier(action.tier)
        card_id = cards[-1]
        decks = decks.replacing_tier(action.tier, cards[:-1])
        player = replace(
            player,
            reserved_cards=player.reserved_cards + (ReservedCard(card_id, ReservationSource.DECK),),
        )
        supply, player = _take_gold_if_available(supply, player)

    players = list(state.players)
    players[actor] = player
    return replace(state, supply=supply, decks=decks, market=market, players=tuple(players))


def _take_gold_if_available(supply: TokenVector, player: PlayerState) -> tuple[TokenVector, PlayerState]:
    if supply.gold == 0:
        return supply, player
    gold = TokenVector(gold=1)
    return supply.minus(gold), replace(player, tokens=player.tokens.plus(gold))


def _refill_market(decks: DeckState, market: MarketState, tier: Tier) -> tuple[DeckState, MarketState]:
    cards = decks.for_tier(tier)
    if not cards:
        return decks, market
    drawn = cards[-1]
    decks = decks.replacing_tier(tier, cards[:-1])
    market = market.replacing_tier(tier, market.for_tier(tier) + (drawn,))
    return decks, market


def _apply_return(state: GameState, action: ReturnTokens, catalog: Catalog) -> GameState:
    actor = state.current_player
    player = state.players[actor]
    players = list(state.players)
    players[actor] = replace(player, tokens=player.tokens.minus(action.tokens))
    returned = replace(state, supply=state.supply.plus(action.tokens), players=tuple(players))
    if state.pending is None:
        raise ValueError("return requires pending turn context")
    return _begin_turn_settlement(returned, state.pending.main_action, catalog)


def _apply_payment(state: GameState, action: Pay, catalog: Catalog) -> GameState:
    pending = state.pending
    if pending is None or not isinstance(pending.main_action, (BuyMarket, BuyReserved)):
        raise ValueError("payment requires a pending purchase target")

    actor = state.current_player
    player = state.players[actor]
    target = pending.main_action
    card_id = target.card_id
    decks = state.decks
    market = state.market

    if isinstance(target, BuyMarket):
        card = catalog.card(card_id)
        market = market.replacing_tier(
            card.tier,
            tuple(candidate for candidate in market.for_tier(card.tier) if candidate != card_id),
        )
        decks, market = _refill_market(decks, market, card.tier)
    else:
        player = replace(
            player,
            reserved_cards=tuple(
                reserved for reserved in player.reserved_cards if reserved.card_id != card_id
            ),
        )

    player = replace(
        player,
        tokens=player.tokens.minus(action.tokens),
        purchased_card_ids=player.purchased_card_ids + (card_id,),
    )
    players = list(state.players)
    players[actor] = player
    purchased = replace(
        state,
        supply=state.supply.plus(action.tokens),
        decks=decks,
        market=market,
        players=tuple(players),
    )
    return _begin_turn_settlement(purchased, target, catalog)


def _begin_turn_settlement(
    state: GameState,
    main_action: TakeDistinct | TakePair | ReserveMarket | ReserveDeck | BuyMarket | BuyReserved,
    catalog: Catalog,
) -> GameState:
    actor = state.current_player
    eligible = tuple(
        sorted(eligible_noble_ids(state.players[actor], state.nobles_in_play, catalog))
    )
    if eligible:
        return replace(
            state,
            phase=DecisionPhase.CHOOSE_NOBLE,
            pending=PendingTurn(main_action, eligible_noble_ids=eligible),
        )
    return _finish_turn(state, catalog)


def _apply_claim_noble(state: GameState, action: ClaimNoble, catalog: Catalog) -> GameState:
    actor = state.current_player
    player = state.players[actor]
    players = list(state.players)
    players[actor] = replace(player, noble_ids=player.noble_ids + (action.noble_id,))
    claimed = replace(
        state,
        players=tuple(players),
        nobles_in_play=tuple(
            noble_id for noble_id in state.nobles_in_play if noble_id != action.noble_id
        ),
    )
    return _finish_turn(claimed, catalog)


def _finish_turn(state: GameState, catalog: Catalog) -> GameState:
    actor = state.current_player
    completed = list(state.completed_turns)
    completed[actor] += 1
    completed_turns = tuple(completed)
    trigger = state.final_round_triggered_by
    if (
        trigger is None
        and player_score(state.players[actor], catalog) >= catalog.constants.end_game_prestige
    ):
        trigger = actor

    if trigger is not None and completed_turns[0] == completed_turns[1]:
        scores = tuple(player_score(player, catalog) for player in state.players)
        card_counts = tuple(len(player.purchased_card_ids) for player in state.players)
        result = TerminalResult(
            scores=scores,
            purchased_card_counts=card_counts,
            winner_ids=determine_winners(scores, card_counts),
            completed_turns=completed_turns,
            reason=TerminalReason.FINAL_ROUND,
        )
        return replace(
            state,
            completed_turns=completed_turns,
            phase=DecisionPhase.TERMINAL,
            pending=None,
            final_round_triggered_by=trigger,
            result=result,
        )

    return replace(
        state,
        current_player=1 - actor,
        completed_turns=completed_turns,
        phase=DecisionPhase.MAIN_ACTION,
        pending=None,
        final_round_triggered_by=trigger,
    )
