from __future__ import annotations

from pathlib import Path

import pytest

from splendor._state import DeckState, GameState, MarketState, PlayerState
from splendor.catalog import Catalog, load_catalog
from splendor.model import DecisionPhase, Tier, TokenVector
from splendor.rng import DeterministicRng


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def catalog() -> Catalog:
    return load_catalog(ROOT / "data")


@pytest.fixture
def initial_state(catalog: Catalog) -> GameState:
    by_tier = {
        tier: tuple(card.card_id for card in catalog.cards if card.tier is tier)
        for tier in Tier
    }
    market_by_tier = {
        tier: cards[: catalog.constants.face_up_per_tier]
        for tier, cards in by_tier.items()
    }
    deck_by_tier = {
        tier: cards[catalog.constants.face_up_per_tier :]
        for tier, cards in by_tier.items()
    }
    selected = tuple(noble.noble_id for noble in catalog.nobles[: catalog.constants.nobles_in_play])
    unused = tuple(noble.noble_id for noble in catalog.nobles[catalog.constants.nobles_in_play :])
    return GameState(
        supply=catalog.constants.initial_tokens,
        decks=DeckState(
            tier_one=deck_by_tier[Tier.ONE],
            tier_two=deck_by_tier[Tier.TWO],
            tier_three=deck_by_tier[Tier.THREE],
        ),
        market=MarketState(
            tier_one=market_by_tier[Tier.ONE],
            tier_two=market_by_tier[Tier.TWO],
            tier_three=market_by_tier[Tier.THREE],
        ),
        players=(PlayerState(TokenVector()), PlayerState(TokenVector())),
        nobles_in_play=selected,
        nobles_out_of_game=unused,
        first_player=0,
        current_player=0,
        completed_turns=(0, 0),
        consecutive_forced_skips=0,
        phase=DecisionPhase.MAIN_ACTION,
        pending=None,
        final_round_triggered_by=None,
        result=None,
        rng=DeterministicRng(123).snapshot(),
    )
