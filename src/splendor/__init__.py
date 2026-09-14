"""Public, policy-safe contracts for the two-player Splendor project.

Engine-only state deliberately lives in :mod:`splendor._state` and is not
re-exported here.  Policies receive only ``DecisionRequest`` values.
"""

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
from .contracts import DecisionRequest, Observation, Policy
from .engine import GameEngine, IllegalActionError, UnsupportedRuleError
from .model import Color, DecisionPhase, GemVector, ReservationSource, Tier, TokenVector
from .result import TerminalReason, TerminalResult
from .random_policy import RandomPolicy
from .runner import GameRun, RunStatus, run_random_game

__all__ = [
    "Action",
    "BuyMarket",
    "BuyReserved",
    "ClaimNoble",
    "Color",
    "DecisionPhase",
    "DecisionRequest",
    "GemVector",
    "GameEngine",
    "IllegalActionError",
    "Observation",
    "Pay",
    "Policy",
    "RandomPolicy",
    "ReserveDeck",
    "ReserveMarket",
    "ReservationSource",
    "ReturnTokens",
    "TakeDistinct",
    "TakePair",
    "TerminalResult",
    "TerminalReason",
    "Tier",
    "TokenVector",
    "UnsupportedRuleError",
    "GameRun",
    "RunStatus",
    "run_random_game",
]
