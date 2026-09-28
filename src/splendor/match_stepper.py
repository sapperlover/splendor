"""Pauseable real-game runner shared by serial and batched self-play."""

from __future__ import annotations

from .compact_trace import append_action, new_trace
from .contracts import DecisionRequest
from .engine import GameEngine
from .result import TerminalResult


class MatchStepper:
    def __init__(self, game_id: int, game_seed: int, first_player: int, *, mode: str) -> None:
        self.engine = GameEngine.new(game_seed, first_player=first_player, mode=mode)
        self.trace = new_trace(game_id, game_seed, first_player, self.engine.catalog)
        self.current = self.engine.decision()
        self.state_version = self.engine.state_version
        self.decision_count = 0

    @property
    def request(self) -> DecisionRequest:
        if not isinstance(self.current, DecisionRequest):
            raise RuntimeError("match is already terminal")
        return self.current

    @property
    def terminal(self) -> TerminalResult | None:
        return self.current if isinstance(self.current, TerminalResult) else None

    @property
    def completed_player_turns(self) -> int:
        return sum(self.engine._debug_state().completed_turns)

    def apply(self, candidate_index: int) -> DecisionRequest | TerminalResult:
        request = self.request
        actor = self.engine.current_player
        if not 0 <= candidate_index < len(request.legal_actions):
            raise ValueError("candidate index out of range")
        if self.engine.mode == "fast":
            response = self.engine.apply_candidate(candidate_index, state_version=self.state_version)
        else:
            response = self.engine.apply(request.legal_actions[candidate_index])
        append_action(self.trace, request, actor, candidate_index)
        self.decision_count += 1
        self.current = response
        self.state_version = self.engine.state_version
        if isinstance(response, TerminalResult):
            from .replay import to_primitive
            self.trace["terminal_result"] = to_primitive(response)
        return response
