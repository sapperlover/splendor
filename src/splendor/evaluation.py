from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import median

from .contracts import Policy
from .heuristic_policy import DevelopmentPolicy, ScorePolicy
from .random_policy import RandomPolicy
from .runner import MatchOutcome, MatchResult, RunStatus, run_match


PolicyFactory = Callable[[int], Policy]


@dataclass(frozen=True, slots=True)
class PolicySpec:
    name: str
    factory: PolicyFactory

    def create(self, seed: int) -> Policy:
        return self.factory(seed)


@dataclass(frozen=True, slots=True)
class OutcomeBreakdown:
    games: int
    valid_games: int
    wins: int
    draws: int
    losses: int
    win_rate_score: float | None


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    policy_a: str
    policy_b: str
    game_seeds: tuple[int, ...]
    total_pairs: int
    total_games: int
    valid_games: int
    a_wins: int
    draws: int
    a_losses: int
    win_rate_score: float | None
    a_first: OutcomeBreakdown
    a_second: OutcomeBreakdown
    mean_completed_player_turns: float | None
    median_completed_player_turns: float | None
    mean_decisions: float | None
    forced_skips: int
    engineering_truncations: int
    exceptions: int
    illegal_actions: int
    invariant_failures: int
    matches: tuple[MatchResult, ...]

    @property
    def passed(self) -> bool:
        return (
            self.valid_games == self.total_games
            and self.engineering_truncations == 0
            and self.exceptions == 0
            and self.illegal_actions == 0
            and self.invariant_failures == 0
        )

    def to_dict(self, *, include_matches: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "policy_a": self.policy_a,
            "policy_b": self.policy_b,
            "game_seeds": list(self.game_seeds),
            "total_pairs": self.total_pairs,
            "total_games": self.total_games,
            "valid_games": self.valid_games,
            "a_wins": self.a_wins,
            "draws": self.draws,
            "a_losses": self.a_losses,
            "win_rate_score": self.win_rate_score,
            "a_first": _breakdown_dict(self.a_first),
            "a_second": _breakdown_dict(self.a_second),
            "mean_completed_player_turns": self.mean_completed_player_turns,
            "median_completed_player_turns": self.median_completed_player_turns,
            "mean_decisions": self.mean_decisions,
            "forced_skips": self.forced_skips,
            "engineering_truncations": self.engineering_truncations,
            "exceptions": self.exceptions,
            "illegal_actions": self.illegal_actions,
            "invariant_failures": self.invariant_failures,
            "passed": self.passed,
        }
        if include_matches:
            pairs: list[list[dict[str, object]]] = [[] for _ in self.game_seeds]
            for match in self.matches:
                if match.pair_id is None:
                    raise ValueError("evaluation match is missing pair_id")
                pairs[match.pair_id].append(_match_reference(match))
            result["pairs"] = [
                {
                    "pair_id": pair_id,
                    "game_seed": game_seed,
                    "games": pairs[pair_id],
                }
                for pair_id, game_seed in enumerate(self.game_seeds)
            ]
        return result


def _random_factory(seed: int) -> Policy:
    return RandomPolicy(seed)


def _score_factory(seed: int) -> Policy:
    del seed
    return ScorePolicy()


def _development_factory(seed: int) -> Policy:
    del seed
    return DevelopmentPolicy()


_BASELINES: dict[str, PolicyFactory] = {
    "random": _random_factory,
    "score": _score_factory,
    "development": _development_factory,
}


def baseline_policy_spec(name: str) -> PolicySpec:
    try:
        factory = _BASELINES[name]
    except KeyError as error:
        choices = ", ".join(sorted(_BASELINES))
        raise ValueError(f"unknown baseline policy {name!r}; choose one of: {choices}") from error
    return PolicySpec(name, factory)


def baseline_policy_names() -> tuple[str, ...]:
    return tuple(sorted(_BASELINES))


def evaluate_policies(
    policy_a: PolicySpec,
    policy_b: PolicySpec,
    game_seeds: Sequence[int],
    *,
    policy_seed_base: int = 0,
    max_completed_turns: int = 400,
    failure_dir: Path | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> EvaluationResult:
    """Evaluate paired games: A starts once and B starts once per game seed."""

    seeds = tuple(game_seeds)
    if not seeds:
        raise ValueError("at least one game seed is required")
    if len(set(seeds)) != len(seeds) or not all(type(seed) is int for seed in seeds):
        raise ValueError("game seeds must be unique integers")
    if max_completed_turns <= 0:
        raise ValueError("max_completed_turns must be positive")
    failure_dir = failure_dir or Path.cwd() / "evaluation_failures"

    matches: list[MatchResult] = []
    for pair_id, game_seed in enumerate(seeds):
        policy_seeds = (
            policy_seed_base + pair_id * 2,
            policy_seed_base + pair_id * 2 + 1,
        )
        for game_in_pair, first_player in enumerate((0, 1)):
            matches.append(
                run_match(
                    policy_a.create(policy_seeds[0]),
                    policy_b.create(policy_seeds[1]),
                    game_seed=game_seed,
                    policy_seeds=policy_seeds,
                    first_player=first_player,
                    policy_names=(policy_a.name, policy_b.name),
                    pair_id=pair_id,
                    game_in_pair=game_in_pair,
                    max_completed_turns=max_completed_turns,
                    failure_dir=failure_dir / f"pair-{pair_id:06d}-game-{game_in_pair}",
                    raise_on_error=False,
                )
            )
        if progress_callback is not None:
            progress_callback(pair_id + 1, len(seeds))

    frozen_matches = tuple(matches)
    valid = tuple(
        match
        for match in frozen_matches
        if match.status is RunStatus.NATURAL_TERMINAL and match.terminal_result is not None
    )
    a_wins = sum(match.outcome_for_a is MatchOutcome.A_WIN for match in valid)
    draws = sum(match.outcome_for_a is MatchOutcome.DRAW for match in valid)
    a_losses = sum(match.outcome_for_a is MatchOutcome.A_LOSS for match in valid)
    lengths = tuple(match.completed_player_turns for match in valid)
    decisions = tuple(match.decision_count for match in valid)
    return EvaluationResult(
        policy_a=policy_a.name,
        policy_b=policy_b.name,
        game_seeds=seeds,
        total_pairs=len(seeds),
        total_games=len(frozen_matches),
        valid_games=len(valid),
        a_wins=a_wins,
        draws=draws,
        a_losses=a_losses,
        win_rate_score=_win_rate(a_wins, draws, len(valid)),
        a_first=_outcome_breakdown(
            tuple(match for match in frozen_matches if match.first_player == 0)
        ),
        a_second=_outcome_breakdown(
            tuple(match for match in frozen_matches if match.first_player == 1)
        ),
        mean_completed_player_turns=(sum(lengths) / len(lengths) if lengths else None),
        median_completed_player_turns=(float(median(lengths)) if lengths else None),
        mean_decisions=(sum(decisions) / len(decisions) if decisions else None),
        forced_skips=sum(match.forced_skips for match in frozen_matches),
        engineering_truncations=sum(
            match.status is RunStatus.ENGINEERING_TRUNCATION for match in frozen_matches
        ),
        exceptions=sum(match.status is RunStatus.ERROR for match in frozen_matches),
        illegal_actions=sum(match.error_type == "IllegalActionError" for match in frozen_matches),
        invariant_failures=sum(match.error_type == "InvariantViolation" for match in frozen_matches),
        matches=frozen_matches,
    )


def save_evaluation(path: Path, result: EvaluationResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _win_rate(wins: int, draws: int, valid_games: int) -> float | None:
    if valid_games == 0:
        return None
    return (wins + 0.5 * draws) / valid_games


def _outcome_breakdown(matches: tuple[MatchResult, ...]) -> OutcomeBreakdown:
    valid = tuple(
        match
        for match in matches
        if match.status is RunStatus.NATURAL_TERMINAL and match.outcome_for_a is not None
    )
    wins = sum(match.outcome_for_a is MatchOutcome.A_WIN for match in valid)
    draws = sum(match.outcome_for_a is MatchOutcome.DRAW for match in valid)
    losses = sum(match.outcome_for_a is MatchOutcome.A_LOSS for match in valid)
    return OutcomeBreakdown(
        games=len(matches),
        valid_games=len(valid),
        wins=wins,
        draws=draws,
        losses=losses,
        win_rate_score=_win_rate(wins, draws, len(valid)),
    )


def _breakdown_dict(breakdown: OutcomeBreakdown) -> dict[str, object]:
    return {
        "games": breakdown.games,
        "valid_games": breakdown.valid_games,
        "wins": breakdown.wins,
        "draws": breakdown.draws,
        "losses": breakdown.losses,
        "win_rate_score": breakdown.win_rate_score,
    }


def _match_reference(match: MatchResult) -> dict[str, object]:
    terminal = match.terminal_result
    return {
        "pair_id": match.pair_id,
        "game_in_pair": match.game_in_pair,
        "game_seed": match.game_seed,
        "first_player": match.first_player,
        "policy_seeds": list(match.policy_seeds),
        "status": match.status.value,
        "outcome_for_a": None if match.outcome_for_a is None else match.outcome_for_a.value,
        "scores": None if match.scores is None else list(match.scores),
        "purchased_card_counts": (
            None if match.purchased_card_counts is None else list(match.purchased_card_counts)
        ),
        "completed_turns": list(match.completed_turns),
        "decision_count": match.decision_count,
        "forced_skips": match.forced_skips,
        "terminal_reason": None if terminal is None else terminal.reason.value,
        "replay": {
            "ruleset_id": match.replay.header.ruleset_id,
            "schema_version": match.replay.header.schema_version,
            "event_count": len(match.replay.events),
            "final_state_digest": match.final_state_digest,
        },
        "failure_scene": None if match.failure_scene is None else str(match.failure_scene),
        "error_type": match.error_type,
        "error_detail": match.error_detail,
    }
