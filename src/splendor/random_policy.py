from __future__ import annotations

from .actions import Action
from .contracts import DecisionRequest
from .rng import DeterministicRng, RngState


class RandomPolicy:
    """Uniform baseline over engine-provided legal actions.

    The policy owns a separate deterministic RNG.  It never receives an
    engine or authoritative state and contains no legality rules.
    """

    __slots__ = ("__rng",)

    def __init__(self, seed: int) -> None:
        self.__rng = DeterministicRng(seed)

    def choose(self, request: DecisionRequest) -> Action:
        return request.legal_actions[self.__rng.randbelow(len(request.legal_actions))]

    def rng_snapshot(self) -> RngState:
        """Expose policy randomness only for validation failure reports."""

        return self.__rng.snapshot()
