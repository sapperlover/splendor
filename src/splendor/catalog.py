from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from .model import Color, GemVector, Tier, TokenVector


@dataclass(frozen=True, slots=True)
class CardDefinition:
    card_id: str
    tier: Tier
    bonus: Color
    points: int
    cost: GemVector


@dataclass(frozen=True, slots=True)
class NobleDefinition:
    noble_id: str
    name: str
    points: int
    requirement: GemVector


@dataclass(frozen=True, slots=True)
class TwoPlayerConstants:
    initial_tokens: TokenVector
    face_up_per_tier: int
    nobles_in_play: int
    max_reserved_cards: int
    max_tokens_at_turn_end: int
    end_game_prestige: int


@dataclass(frozen=True, slots=True)
class Catalog:
    cards: tuple[CardDefinition, ...]
    nobles: tuple[NobleDefinition, ...]
    constants: TwoPlayerConstants
    data_fingerprint: str
    _cards_by_id: Mapping[str, CardDefinition]
    _nobles_by_id: Mapping[str, NobleDefinition]

    def card(self, card_id: str) -> CardDefinition:
        return self._cards_by_id[card_id]

    def noble(self, noble_id: str) -> NobleDefinition:
        return self._nobles_by_id[noble_id]


def _gem_vector(row: Mapping[str, str], prefix: str) -> GemVector:
    return GemVector(**{color.value: int(row[f"{prefix}{color.value}"]) for color in Color if color is not Color.GOLD})


def load_catalog(data_dir: Path) -> Catalog:
    """Load the normative CSV/JSON data without copying rule values into code."""

    cards_path = data_dir / "splendor_cards.csv"
    nobles_path = data_dir / "splendor_nobles.csv"
    config_path = data_dir / "splendor_config.json"

    with cards_path.open(encoding="utf-8", newline="") as stream:
        cards = tuple(
            CardDefinition(
                card_id=row["id"],
                tier=Tier(int(row["tier"])),
                bonus=Color(row["bonus"]),
                points=int(row["points"]),
                cost=_gem_vector(row, "cost_"),
            )
            for row in csv.DictReader(stream)
        )
    with nobles_path.open(encoding="utf-8", newline="") as stream:
        nobles = tuple(
            NobleDefinition(
                noble_id=row["id"],
                name=row["name_en"],
                points=int(row["points"]),
                requirement=_gem_vector(row, "req_"),
            )
            for row in csv.DictReader(stream)
        )
    with config_path.open(encoding="utf-8") as stream:
        config = json.load(stream)

    setup = config["setup"]["2"]
    colored = int(setup["coloredTokensPerColor"])
    constants = TwoPlayerConstants(
        initial_tokens=TokenVector(
            white=colored,
            blue=colored,
            green=colored,
            red=colored,
            black=colored,
            gold=int(setup["goldTokens"]),
        ),
        face_up_per_tier=int(config["faceUpCardsPerTier"]),
        nobles_in_play=int(setup["nobles"]),
        max_reserved_cards=int(config["maxReservedCards"]),
        max_tokens_at_turn_end=int(config["maxHeldTokensAtEndOfTurn"]),
        end_game_prestige=int(config["endGamePrestige"]),
    )

    digest = hashlib.sha256()
    for path in (cards_path, nobles_path, config_path):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())

    card_map = {card.card_id: card for card in cards}
    noble_map = {noble.noble_id: noble for noble in nobles}
    if len(card_map) != len(cards) or len(noble_map) != len(nobles):
        raise ValueError("normative data contains duplicate stable IDs")
    return Catalog(
        cards=cards,
        nobles=nobles,
        constants=constants,
        data_fingerprint=digest.hexdigest(),
        _cards_by_id=MappingProxyType(card_map),
        _nobles_by_id=MappingProxyType(noble_map),
    )


def load_default_catalog() -> Catalog:
    return load_catalog(Path(__file__).resolve().parents[2] / "data")

