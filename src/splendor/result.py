from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TerminalReason(str, Enum):
    FINAL_ROUND = "final_round"
    CONSECUTIVE_FORCED_SKIPS = "consecutive_forced_skips"


@dataclass(frozen=True, slots=True)
class TerminalResult:
    """Formal game result, including the explicit forced-skip stalemate."""

    scores: tuple[int, int]
    purchased_card_counts: tuple[int, int]
    winner_ids: tuple[int, ...]
    completed_turns: tuple[int, int]
    reason: TerminalReason = TerminalReason.FINAL_ROUND

    def payoff(self, player_id: int) -> int:
        if len(self.winner_ids) == 2:
            return 0
        return 1 if player_id in self.winner_ids else -1
