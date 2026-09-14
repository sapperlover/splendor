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
from .evaluation import EvaluationResult, PolicySpec, baseline_policy_spec, evaluate_policies
from .heuristic_policy import DevelopmentPolicy, ScorePolicy
from .model import Color, DecisionPhase, GemVector, ReservationSource, Tier, TokenVector
from .result import TerminalReason, TerminalResult
from .random_policy import RandomPolicy
from .runner import GameRun, MatchOutcome, MatchResult, RunStatus, run_match, run_random_game

__all__ = [
    "Action",
    "BuyMarket",
    "BuyReserved",
    "ClaimNoble",
    "Color",
    "DecisionPhase",
    "DecisionRequest",
    "DevelopmentPolicy",
    "EvaluationResult",
    "GemVector",
    "GameEngine",
    "IllegalActionError",
    "Observation",
    "Pay",
    "Policy",
    "PolicySpec",
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
    "MatchOutcome",
    "MatchResult",
    "RunStatus",
    "ScorePolicy",
    "baseline_policy_spec",
    "evaluate_policies",
    "run_match",
    "run_random_game",
]
