"""Single-learning-player environment adapter over the authoritative engine."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .actions import Action
from .catalog import Catalog
from .contracts import DecisionRequest, Observation, Policy
from .engine import GameEngine
from .features import EncodedDecision, encode_decision
from .replay import ReplayRecord, state_digest
from .result import TerminalResult


class LearningEnvironmentError(RuntimeError):
    pass


class StaleLearningDecisionError(LearningEnvironmentError):
    code = "stale_learning_decision"


class InvalidCandidateIndexError(LearningEnvironmentError):
    code = "invalid_candidate_index"


@dataclass(frozen=True, slots=True)
class LearningDecision:
    """One policy decision and its order-preserving numeric representation."""

    decision_id: int
    request: DecisionRequest
    encoded: EncodedDecision

    @property
    def observation(self) -> Observation:
        return self.request.observation

    @property
    def legal_actions(self) -> tuple[Action, ...]:
        return self.request.legal_actions


@dataclass(frozen=True, slots=True)
class LearningTransition:
    """One learning-player decision through its next same-player boundary."""

    episode_id: str
    learning_player: int
    decision: LearningDecision
    chosen_candidate_index: int
    reward: float
    terminated: bool
    truncated: bool
    next_decision: LearningDecision | None
    terminal_result: TerminalResult | None
    replay_event_start: int
    replay_event_end: int
    state_digest_after: str
    completed_turns_after: tuple[int, int]

    @property
    def chosen_action(self) -> Action:
        return self.decision.legal_actions[self.chosen_candidate_index]


@dataclass(frozen=True, slots=True)
class LearningEpisodeRun:
    episode_id: str
    game_seed: int
    first_player: int
    learning_player: int
    learning_policy_seed: int
    opponent_seed: int
    transitions: tuple[LearningTransition, ...]
    terminal_result: TerminalResult | None
    terminated: bool
    truncated: bool
    winner_ids: tuple[int, ...] | None
    learning_decision_count: int
    opponent_decision_count: int
    completed_turns: tuple[int, int]
    completed_player_turns: int
    replay: ReplayRecord
    final_state_digest: str

    @property
    def bootstrap_decision(self) -> LearningDecision | None:
        if not self.truncated or not self.transitions:
            return None
        return self.transitions[-1].next_decision


class LearningEnvironment:
    """Advance one fixed learning seat while a frozen policy controls its opponent."""

    __slots__ = (
        "_bootstrap_decision",
        "_current",
        "_engine",
        "_episode_id",
        "_learning_player",
        "_max_completed_turns",
        "_next_decision_id",
        "_opponent_decision_count",
        "_opponent_policy",
        "_terminal_result",
        "_transitions",
        "_truncated",
    )

    def __init__(
        self,
        engine: GameEngine,
        *,
        learning_player: int,
        opponent_policy: Policy,
        episode_id: str,
        max_completed_turns: int = 400,
    ) -> None:
        if learning_player not in (0, 1):
            raise ValueError("learning_player must be 0 or 1")
        if max_completed_turns <= 0:
            raise ValueError("max_completed_turns must be positive")
        if not episode_id:
            raise ValueError("episode_id must not be empty")

        self._engine = engine
        self._learning_player = learning_player
        self._opponent_policy = opponent_policy
        self._episode_id = episode_id
        self._max_completed_turns = max_completed_turns
        self._next_decision_id = 0
        self._current: LearningDecision | None = None
        self._bootstrap_decision: LearningDecision | None = None
        self._terminal_result: TerminalResult | None = None
        self._truncated = False
        self._opponent_decision_count = 0
        self._transitions: list[LearningTransition] = []

        initial = self._advance_opponent_until_learning_boundary(engine.decision())
        if isinstance(initial, TerminalResult):
            self._terminal_result = initial
        else:
            self._current = self._make_decision(initial)

    @classmethod
    def new(
        cls,
        game_seed: int,
        *,
        learning_player: int,
        opponent_policy: Policy,
        episode_id: str | None = None,
        first_player: int | None = None,
        max_completed_turns: int = 400,
        catalog: Catalog | None = None,
        failure_dir: Path | None = None,
    ) -> LearningEnvironment:
        return cls(
            GameEngine.new(
                game_seed,
                first_player=first_player,
                catalog=catalog,
                failure_dir=failure_dir,
            ),
            learning_player=learning_player,
            opponent_policy=opponent_policy,
            episode_id=(
                episode_id
                if episode_id is not None
                else f"game-{game_seed}-learner-{learning_player}"
            ),
            max_completed_turns=max_completed_turns,
        )

    @property
    def done(self) -> bool:
        return self._terminal_result is not None or self._truncated

    @property
    def learning_player(self) -> int:
        return self._learning_player

    @property
    def opponent_policy(self) -> Policy:
        return self._opponent_policy

    @property
    def episode_id(self) -> str:
        return self._episode_id

    @property
    def max_completed_turns(self) -> int:
        return self._max_completed_turns

    @property
    def terminated(self) -> bool:
        return self._terminal_result is not None

    @property
    def truncated(self) -> bool:
        return self._truncated

    @property
    def terminal_result(self) -> TerminalResult | None:
        return self._terminal_result

    @property
    def replay_record(self) -> ReplayRecord:
        return self._engine.replay_record

    @property
    def transitions(self) -> tuple[LearningTransition, ...]:
        return tuple(self._transitions)

    @property
    def opponent_decision_count(self) -> int:
        return self._opponent_decision_count

    @property
    def bootstrap_decision(self) -> LearningDecision | None:
        return self._bootstrap_decision

    def current_decision(self) -> LearningDecision:
        if self._current is None:
            if self._terminal_result is not None:
                raise LearningEnvironmentError("episode has formally terminated")
            if self._truncated:
                raise LearningEnvironmentError("episode has been engineering-truncated")
            raise LearningEnvironmentError("learning player has no current decision")
        return self._current

    def step_learning_action(
        self,
        candidate_index: int,
        *,
        decision_id: int,
    ) -> LearningTransition:
        current = self.current_decision()
        if decision_id != current.decision_id:
            self._engine._fail(
                StaleLearningDecisionError(
                    f"stale decision {decision_id}; current decision is {current.decision_id}"
                )
            )
        if type(candidate_index) is not int or not 0 <= candidate_index < len(
            current.legal_actions
        ):
            self._engine._fail(
                InvalidCandidateIndexError(
                    f"candidate index {candidate_index!r} is outside the current legal set"
                )
            )

        action = current.legal_actions[candidate_index]
        replay_event_start = len(self._engine.replay_record.events)
        outcome = self._engine.apply(action)
        outcome = self._advance_opponent_until_learning_boundary(outcome)

        reward = 0.0
        terminated = isinstance(outcome, TerminalResult)
        truncated = False
        next_decision: LearningDecision | None = None
        terminal_result: TerminalResult | None = None
        if terminated:
            terminal_result = outcome
            reward = float(outcome.payoff(self.learning_player))
            completed_turns = outcome.completed_turns
            self._terminal_result = outcome
            self._current = None
        else:
            next_decision = self._make_decision(outcome)
            completed_turns = self._absolute_completed_turns(outcome.observation)
            truncated = sum(completed_turns) >= self.max_completed_turns
            if truncated:
                self._truncated = True
                self._bootstrap_decision = next_decision
                self._current = None
            else:
                self._current = next_decision

        transition = LearningTransition(
            episode_id=self.episode_id,
            learning_player=self.learning_player,
            decision=current,
            chosen_candidate_index=candidate_index,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            next_decision=next_decision,
            terminal_result=terminal_result,
            replay_event_start=replay_event_start,
            replay_event_end=len(self._engine.replay_record.events),
            state_digest_after=state_digest(self._engine._debug_state()),
            completed_turns_after=completed_turns,
        )
        self._transitions.append(transition)
        return transition

    def episode_run(
        self,
        *,
        learning_policy_seed: int,
        opponent_seed: int,
    ) -> LearningEpisodeRun:
        if not self.done:
            raise LearningEnvironmentError("episode is still active")
        state = self._engine._debug_state()
        result = self._terminal_result
        return LearningEpisodeRun(
            episode_id=self.episode_id,
            game_seed=self._engine.replay_record.header.game_seed,
            first_player=self._engine.replay_record.header.first_player,
            learning_player=self.learning_player,
            learning_policy_seed=learning_policy_seed,
            opponent_seed=opponent_seed,
            transitions=tuple(self._transitions),
            terminal_result=result,
            terminated=result is not None,
            truncated=self._truncated,
            winner_ids=None if result is None else result.winner_ids,
            learning_decision_count=len(self._transitions),
            opponent_decision_count=self._opponent_decision_count,
            completed_turns=state.completed_turns,
            completed_player_turns=sum(state.completed_turns),
            replay=self._engine.replay_record,
            final_state_digest=state_digest(state),
        )

    def _make_decision(self, request: DecisionRequest) -> LearningDecision:
        if self._engine.current_player != self.learning_player:
            raise LearningEnvironmentError("decision does not belong to learning_player")
        decision = LearningDecision(
            decision_id=self._next_decision_id,
            request=request,
            encoded=encode_decision(request),
        )
        self._next_decision_id += 1
        return decision

    def _advance_opponent_until_learning_boundary(
        self,
        outcome: DecisionRequest | TerminalResult,
    ) -> DecisionRequest | TerminalResult:
        while (
            isinstance(outcome, DecisionRequest)
            and self._engine.current_player != self.learning_player
        ):
            action = self.opponent_policy.choose(outcome)
            self._opponent_decision_count += 1
            outcome = self._engine.apply(action)
        return outcome

    def _absolute_completed_turns(self, observation: Observation) -> tuple[int, int]:
        if self.learning_player == 0:
            return observation.own_completed_turns, observation.opponent_completed_turns
        return observation.opponent_completed_turns, observation.own_completed_turns


def run_learning_episode(
    learning_policy: Policy,
    opponent_policy: Policy,
    *,
    game_seed: int,
    learning_player: int,
    learning_policy_seed: int,
    opponent_seed: int,
    episode_id: str | None = None,
    first_player: int | None = None,
    max_completed_turns: int = 400,
    catalog: Catalog | None = None,
    failure_dir: Path | None = None,
) -> LearningEpisodeRun:
    """Run one episode while retaining only one fixed player's decisions."""

    environment = LearningEnvironment.new(
        game_seed,
        learning_player=learning_player,
        opponent_policy=opponent_policy,
        episode_id=episode_id,
        first_player=first_player,
        max_completed_turns=max_completed_turns,
        catalog=catalog,
        failure_dir=failure_dir,
    )
    while not environment.done:
        decision = environment.current_decision()
        action = learning_policy.choose(decision.request)
        try:
            candidate_index = decision.legal_actions.index(action)
        except ValueError as error:
            environment._engine._fail(
                InvalidCandidateIndexError(
                    "learning policy returned an action outside the current legal candidates"
                )
            )
            raise AssertionError("engine failure guard returned unexpectedly") from error
        environment.step_learning_action(
            candidate_index,
            decision_id=decision.decision_id,
        )
    return environment.episode_run(
        learning_policy_seed=learning_policy_seed,
        opponent_seed=opponent_seed,
    )
