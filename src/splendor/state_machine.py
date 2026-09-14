from __future__ import annotations

from .model import DecisionPhase


# These are next *observable decision points*. Automatic refill, settlement,
# forced single choices, end checks and player advancement happen between them.
_ALLOWED_NEXT: dict[DecisionPhase, frozenset[DecisionPhase]] = {
    DecisionPhase.MAIN_ACTION: frozenset(
        {
            DecisionPhase.PAYMENT,
            DecisionPhase.RETURN_TOKENS,
            DecisionPhase.CHOOSE_NOBLE,
            DecisionPhase.MAIN_ACTION,
            DecisionPhase.TERMINAL,
        }
    ),
    DecisionPhase.PAYMENT: frozenset(
        {
            DecisionPhase.CHOOSE_NOBLE,
            DecisionPhase.MAIN_ACTION,
            DecisionPhase.TERMINAL,
        }
    ),
    DecisionPhase.RETURN_TOKENS: frozenset(
        {
            DecisionPhase.CHOOSE_NOBLE,
            DecisionPhase.MAIN_ACTION,
            DecisionPhase.TERMINAL,
        }
    ),
    DecisionPhase.CHOOSE_NOBLE: frozenset(
        {DecisionPhase.MAIN_ACTION, DecisionPhase.TERMINAL}
    ),
    DecisionPhase.TERMINAL: frozenset(),
}


def allowed_next_phases(phase: DecisionPhase) -> frozenset[DecisionPhase]:
    return _ALLOWED_NEXT[phase]


def assert_phase_transition(before: DecisionPhase, after: DecisionPhase) -> None:
    if after not in _ALLOWED_NEXT[before]:
        raise ValueError(f"illegal phase transition: {before.value} -> {after.value}")

