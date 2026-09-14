from __future__ import annotations

from dataclasses import dataclass
from typing import MutableSequence, TypeVar


_MASK_64 = (1 << 64) - 1
_GOLDEN_GAMMA = 0x9E3779B97F4A7C15
RNG_ALGORITHM = "splitmix64-v1"
T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class RngState:
    algorithm: str
    state: int
    draws: int


class DeterministicRng:
    """Small versioned RNG whose exact state is JSON-safe and replayable."""

    def __init__(self, seed: int) -> None:
        self._state = seed & _MASK_64
        self._draws = 0

    def snapshot(self) -> RngState:
        return RngState(RNG_ALGORITHM, self._state, self._draws)

    @classmethod
    def restore(cls, snapshot: RngState) -> DeterministicRng:
        if snapshot.algorithm != RNG_ALGORITHM:
            raise ValueError(f"unsupported RNG algorithm: {snapshot.algorithm}")
        if not 0 <= snapshot.state <= _MASK_64 or snapshot.draws < 0:
            raise ValueError("invalid RNG snapshot")
        rng = cls(snapshot.state)
        rng._draws = snapshot.draws
        return rng

    def next_u64(self) -> int:
        self._state = (self._state + _GOLDEN_GAMMA) & _MASK_64
        z = self._state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK_64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK_64
        self._draws += 1
        return (z ^ (z >> 31)) & _MASK_64

    def randbelow(self, bound: int) -> int:
        if bound <= 0:
            raise ValueError("bound must be positive")
        if bound > 1 << 64:
            raise ValueError("bound exceeds RNG word size")
        threshold = (1 << 64) % bound
        while True:
            value = self.next_u64()
            if value >= threshold:
                return value % bound

    def shuffle(self, values: MutableSequence[T]) -> None:
        for index in range(len(values) - 1, 0, -1):
            other = self.randbelow(index + 1)
            values[index], values[other] = values[other], values[index]
