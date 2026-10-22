"""Self-play budget sampling and the 2019 KataGo root-target rule.

This module has no engine/model dependencies so archived targets can be
independently reconstructed from visits, value sums and the used root prior.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Sequence


@dataclass(frozen=True, slots=True)
class SearchBudgetRandomization:
    low_simulations: int = 128
    low_probability: float = 0.75
    low_determinizations: int | None = None

    def __post_init__(self) -> None:
        if type(self.low_simulations) is not int or self.low_simulations <= 0:
            raise ValueError("low_simulations must be a positive integer")
        if self.low_determinizations is not None and (
                type(self.low_determinizations) is not int or self.low_determinizations <= 0):
            raise ValueError("low_determinizations must be a positive integer or None")
        if (isinstance(self.low_probability, bool)
                or not isinstance(self.low_probability, (int, float))
                or not math.isfinite(self.low_probability)
                or not 0 <= self.low_probability <= 1):
            raise ValueError("low_probability must be finite and in [0,1]")

    def validate_search(self, simulations: int, worlds: int) -> None:
        low_worlds = worlds if self.low_determinizations is None else self.low_determinizations
        if not low_worlds <= worlds or not low_worlds <= self.low_simulations < simulations:
            raise ValueError("budget randomization requires low worlds <= high worlds and low worlds <= low_simulations < simulations")

    def to_dict(self) -> dict:
        raw = asdict(self)
        if self.low_determinizations is None:
            raw.pop("low_determinizations")
        return raw

    def is_low(self, sampling_seed: int) -> bool:
        payload = json.dumps(["splendor-playout-cap-v1", int(sampling_seed)], separators=(",", ":")).encode()
        word = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        return word / (1 << 64) < self.low_probability


def prune_policy_counts(
    visits: Sequence[int], value_sums: Sequence[float], prior: Sequence[float],
    *, c_puct: float, forced_playouts: float,
) -> tuple[int, ...]:
    """Prune each non-winner with final Q and total visits held fixed.

    Integer variant of KataGo 2019 section 3.2: remove at most floor(sqrt(kPN))
    visits while the resulting ordinary PUCT is strictly below the most-visited
    child's final PUCT. Afterwards discard non-winners with <=1 remaining visit.
    Ties for most-visited child use the used prior, then candidate order.
    """
    total = sum(visits)
    best = max(range(len(visits)), key=lambda i: (visits[i], prior[i], -i))
    factor = c_puct * math.sqrt(1 + total)
    q = [value / n if n else 0.0 for value, n in zip(value_sums, visits)]
    threshold = q[best] + factor * prior[best] / (1 + visits[best])
    result = list(visits)
    for i, n in enumerate(visits):
        if i == best or n == 0:
            continue
        lower = max(0, n - math.floor(math.sqrt(forced_playouts * prior[i] * total)))
        # Find the smallest admissible retained count, without a per-visit loop.
        lo, hi = lower, n
        while lo < hi:
            mid = (lo + hi) // 2
            if q[i] + factor * prior[i] / (1 + mid) < threshold:
                hi = mid
            else:
                lo = mid + 1
        result[i] = 0 if lo <= 1 else lo
    return tuple(result)
