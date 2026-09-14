from __future__ import annotations

import inspect
from dataclasses import replace
from pathlib import Path

import splendor.heuristic_policy as heuristic_module
from splendor._state import GameState, PendingTurn, ReservedCard
from splendor.actions import BuyMarket, ClaimNoble, TakeDistinct
from splendor.catalog import Catalog
from splendor.contracts import DecisionRequest
from splendor.engine import GameEngine
from splendor.heuristic_policy import DevelopmentPolicy, ScorePolicy
from splendor.invariants import assert_game_state
from splendor.model import Color, DecisionPhase, GEM_COLORS, ReservationSource, Tier, TokenVector
from splendor.observation import build_observation
from splendor.random_policy import RandomPolicy
from splendor.replay import new_replay
from splendor.rules import eligible_noble_ids, legal_actions


def _engine(state: GameState, catalog: Catalog, path: Path) -> GameEngine:
    return GameEngine(state, catalog, new_replay(catalog, 123, state.first_player), path)


def _swap_market_card(
    state: GameState,
    catalog: Catalog,
    *,
    incoming: str,
    outgoing: str,
) -> GameState:
    tier = catalog.card(incoming).tier
    assert catalog.card(outgoing).tier is tier
    deck = tuple(outgoing if card_id == incoming else card_id for card_id in state.decks.for_tier(tier))
    market = tuple(incoming if card_id == outgoing else card_id for card_id in state.market.for_tier(tier))
    return replace(
        state,
        decks=state.decks.replacing_tier(tier, deck),
        market=state.market.replacing_tier(tier, market),
    )


def _purchase_state(
    initial_state: GameState,
    catalog: Catalog,
    *,
    low_card: str,
    low_outgoing: str,
    high_card: str,
    high_outgoing: str,
    tokens: TokenVector,
) -> GameState:
    state = _swap_market_card(
        initial_state,
        catalog,
        incoming=low_card,
        outgoing=low_outgoing,
    )
    state = _swap_market_card(
        state,
        catalog,
        incoming=high_card,
        outgoing=high_outgoing,
    )
    state = replace(
        state,
        supply=state.supply.minus(tokens),
        players=(replace(state.players[0], tokens=tokens), state.players[1]),
    )
    assert_game_state(state, catalog)
    return state


def _multi_noble_state(initial_state: GameState, catalog: Catalog) -> GameState:
    noble_ids = initial_state.nobles_in_play[:2]
    required = {
        color: max(catalog.noble(noble_id).requirement.for_color(color) for noble_id in noble_ids)
        for color in GEM_COLORS
    }
    purchased: list[str] = []
    remaining = {tier: list(initial_state.decks.for_tier(tier)) for tier in Tier}
    for color, count in required.items():
        choices = [
            card.card_id
            for card in catalog.cards
            if card.bonus is color and card.card_id in remaining[card.tier]
        ][:count]
        purchased.extend(choices)
        for card_id in choices:
            remaining[catalog.card(card_id).tier].remove(card_id)
    player = replace(initial_state.players[0], purchased_card_ids=tuple(purchased))
    eligible = tuple(sorted(eligible_noble_ids(player, initial_state.nobles_in_play, catalog)))
    state = replace(
        initial_state,
        decks=replace(
            initial_state.decks,
            tier_one=tuple(remaining[Tier.ONE]),
            tier_two=tuple(remaining[Tier.TWO]),
            tier_three=tuple(remaining[Tier.THREE]),
        ),
        players=(player, initial_state.players[1]),
        phase=DecisionPhase.CHOOSE_NOBLE,
        pending=PendingTurn(
            TakeDistinct((Color.WHITE, Color.BLUE, Color.GREEN)),
            eligible_noble_ids=eligible,
        ),
    )
    assert len(eligible) >= 2
    assert_game_state(state, catalog)
    return state


def test_heuristics_have_request_only_policy_boundary() -> None:
    source = inspect.getsource(heuristic_module)
    for forbidden in ("GameState", "._state", ".engine", ".replay", ".rules", ".rng"):
        assert forbidden not in source
    for policy_type in (ScorePolicy, DevelopmentPolicy):
        assert tuple(inspect.signature(policy_type.choose).parameters) == ("self", "request")


def test_all_baselines_choose_only_candidates_in_every_player_phase(
    initial_state: GameState,
    catalog: Catalog,
    tmp_path: Path,
) -> None:
    main_request = _engine(initial_state, catalog, tmp_path / "main").decision()
    assert isinstance(main_request, DecisionRequest)

    purchase_state = _purchase_state(
        initial_state,
        catalog,
        low_card="007",
        low_outgoing="001",
        high_card="046",
        high_outgoing="041",
        tokens=TokenVector(green=3, black=4, gold=2),
    )
    payment_engine = _engine(purchase_state, catalog, tmp_path / "payment")
    payment_request = payment_engine.apply(BuyMarket("046"))
    assert isinstance(payment_request, DecisionRequest)
    assert payment_request.observation.phase is DecisionPhase.PAYMENT

    held = TokenVector(white=2, blue=2, green=2, red=2, black=1)
    return_state = replace(
        initial_state,
        supply=initial_state.supply.minus(held),
        players=(replace(initial_state.players[0], tokens=held), initial_state.players[1]),
    )
    return_engine = _engine(return_state, catalog, tmp_path / "return")
    return_request = return_engine.apply(
        TakeDistinct((Color.WHITE, Color.BLUE, Color.GREEN))
    )
    assert isinstance(return_request, DecisionRequest)
    assert return_request.observation.phase is DecisionPhase.RETURN_TOKENS

    noble_state = _multi_noble_state(initial_state, catalog)
    noble_request = _engine(noble_state, catalog, tmp_path / "noble").decision()
    assert isinstance(noble_request, DecisionRequest)
    assert noble_request.observation.phase is DecisionPhase.CHOOSE_NOBLE

    requests = (main_request, payment_request, return_request, noble_request)
    for policy in (RandomPolicy(7), ScorePolicy(), DevelopmentPolicy()):
        for request in requests:
            assert policy.choose(request) in request.legal_actions
            assert all("skip" not in type(action).__name__.lower() for action in request.legal_actions)


def test_unknown_opponent_blind_card_cannot_change_heuristic_choice(
    initial_state: GameState,
    catalog: Catalog,
) -> None:
    first_hidden, second_hidden = initial_state.decks.tier_one[:2]
    first_state = replace(
        initial_state,
        decks=initial_state.decks.replacing_tier(
            Tier.ONE,
            tuple(card for card in initial_state.decks.tier_one if card != first_hidden),
        ),
        players=(
            initial_state.players[0],
            replace(
                initial_state.players[1],
                reserved_cards=(ReservedCard(first_hidden, ReservationSource.DECK),),
            ),
        ),
    )
    second_state = replace(
        initial_state,
        decks=initial_state.decks.replacing_tier(
            Tier.ONE,
            tuple(card for card in initial_state.decks.tier_one if card != second_hidden),
        ),
        players=(
            initial_state.players[0],
            replace(
                initial_state.players[1],
                reserved_cards=(ReservedCard(second_hidden, ReservationSource.DECK),),
            ),
        ),
    )
    assert_game_state(first_state, catalog)
    assert_game_state(second_state, catalog)
    first = DecisionRequest(build_observation(first_state, 0, catalog), legal_actions(first_state, catalog))
    second = DecisionRequest(build_observation(second_state, 0, catalog), legal_actions(second_state, catalog))
    assert first == second
    for policy in (ScorePolicy(), DevelopmentPolicy()):
        assert policy.choose(first) == policy.choose(second)


def test_score_prefers_high_points_while_development_prefers_cheap_early_bonus(
    initial_state: GameState,
    catalog: Catalog,
) -> None:
    state = _purchase_state(
        initial_state,
        catalog,
        low_card="007",
        low_outgoing="001",
        high_card="046",
        high_outgoing="041",
        tokens=TokenVector(green=3, black=4, gold=2),
    )
    observation = build_observation(state, 0, catalog)
    choices = (BuyMarket("007"), BuyMarket("046"))
    assert all(action in legal_actions(state, catalog) for action in choices)
    request = DecisionRequest(observation, choices)

    assert ScorePolicy().choose(request) == BuyMarket("046")
    assert DevelopmentPolicy().choose(request) == BuyMarket("007")


def test_development_noble_bonus_preference_differs_from_score_tie_break(
    initial_state: GameState,
    catalog: Catalog,
) -> None:
    state = _purchase_state(
        initial_state,
        catalog,
        low_card="007",
        low_outgoing="001",
        high_card="039",
        high_outgoing="002",
        tokens=TokenVector(white=3, green=3),
    )
    selected = ("N02", "N04", "N08")
    state = replace(
        state,
        nobles_in_play=selected,
        nobles_out_of_game=tuple(
            noble.noble_id for noble in catalog.nobles if noble.noble_id not in selected
        ),
    )
    assert_game_state(state, catalog)
    choices = (BuyMarket("007"), BuyMarket("039"))
    assert all(action in legal_actions(state, catalog) for action in choices)
    request = DecisionRequest(build_observation(state, 0, catalog), choices)

    assert ScorePolicy().choose(request) == BuyMarket("039")
    assert DevelopmentPolicy().choose(request) == BuyMarket("007")


def test_heuristic_tie_break_does_not_depend_on_candidate_order(tmp_path: Path) -> None:
    request = GameEngine.new(81, first_player=0, failure_dir=tmp_path).decision()
    assert isinstance(request, DecisionRequest)
    reversed_request = DecisionRequest(request.observation, tuple(reversed(request.legal_actions)))
    for policy in (ScorePolicy(), DevelopmentPolicy()):
        assert policy.choose(request) == policy.choose(reversed_request)
