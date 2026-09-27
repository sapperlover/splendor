"""Paired-seed score statistics for deterministic policy evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:
    from .evaluation import EvaluationResult
from .runner import MatchOutcome, RunStatus


_SCORED_STATUSES = frozenset(
    (RunStatus.NATURAL_TERMINAL, RunStatus.POLICY_CYCLE_DRAW)
)


def is_scored_status(status: RunStatus) -> bool:
    """True when a finished game contributes a score to paired statistics."""

    return status in _SCORED_STATUSES


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float
    confidence: float
    pair_count: int
    resamples: int
    seed: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def outcome_score(outcome: MatchOutcome | str) -> float:
    value = outcome.value if isinstance(outcome, MatchOutcome) else outcome
    if value == MatchOutcome.A_WIN.value:
        return 1.0
    if value == MatchOutcome.DRAW.value:
        return 0.5
    if value == MatchOutcome.A_LOSS.value:
        return 0.0
    raise ValueError(f"invalid scored outcome: {value!r}")


def pair_scores(result: EvaluationResult) -> tuple[tuple[int, ...], np.ndarray]:
    """Return one mean score per seed after validating both swapped-seat games."""

    if not result.passed:
        raise ValueError("cannot score an evaluation containing invalid games")
    grouped: dict[int, list[tuple[int, float]]] = {}
    for match in result.matches:
        if (
            match.pair_id is None
            or match.game_in_pair is None
            or match.status not in _SCORED_STATUSES
            or match.outcome_for_a is None
        ):
            raise ValueError("evaluation contains an incomplete paired match")
        grouped.setdefault(match.pair_id, []).append(
            (match.game_in_pair, outcome_score(match.outcome_for_a))
        )
    scores: list[float] = []
    for pair_id, game_seed in enumerate(result.game_seeds):
        games = sorted(grouped.get(pair_id, ()))
        if [game_index for game_index, _ in games] != [0, 1]:
            raise ValueError(
                f"pair {pair_id} for seed {game_seed} does not contain both seat swaps"
            )
        scores.append(sum(score for _, score in games) / 2.0)
    return result.game_seeds, np.asarray(scores, dtype=np.float64)


def serialized_pair_scores(
    evaluation: dict[str, object],
) -> tuple[tuple[int, ...], np.ndarray]:
    """Extract validated pair scores from ``EvaluationResult.to_dict`` output."""

    if not evaluation.get("passed"):
        raise ValueError("cannot score an evaluation containing invalid games")
    raw_pairs = evaluation.get("pairs")
    if not isinstance(raw_pairs, list):
        raise ValueError("serialized evaluation does not contain pair details")
    seeds: list[int] = []
    scores: list[float] = []
    for expected_pair_id, pair in enumerate(raw_pairs):
        if not isinstance(pair, dict) or pair.get("pair_id") != expected_pair_id:
            raise ValueError("serialized pair ids are not contiguous")
        game_seed = pair.get("game_seed")
        games = pair.get("games")
        if not isinstance(game_seed, int) or not isinstance(games, list):
            raise ValueError("serialized pair is missing its seed or games")
        ordered = sorted(games, key=lambda game: game["game_in_pair"])
        if (
            len(ordered) != 2
            or [game["game_in_pair"] for game in ordered] != [0, 1]
            or any(
                game.get("status") not in {status.value for status in _SCORED_STATUSES}
                for game in ordered
            )
            or any(game.get("game_seed") != game_seed for game in ordered)
        ):
            raise ValueError(f"seed {game_seed} does not contain both valid seat swaps")
        seeds.append(game_seed)
        scores.append(
            sum(outcome_score(game["outcome_for_a"]) for game in ordered) / 2.0
        )
    return tuple(seeds), np.asarray(scores, dtype=np.float64)


def paired_bootstrap_mean(
    values: Sequence[float] | np.ndarray,
    *,
    confidence: float = 0.95,
    resamples: int = 100_000,
    seed: int = 0,
    batch_size: int = 2_048,
) -> BootstrapInterval:
    """Percentile bootstrap of a mean, resampling whole paired-seed units."""

    samples = np.asarray(values, dtype=np.float64)
    if samples.ndim != 1 or not len(samples) or not np.isfinite(samples).all():
        raise ValueError("bootstrap values must be a non-empty finite vector")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    if resamples <= 0 or batch_size <= 0:
        raise ValueError("resamples and batch_size must be positive")
    generator = np.random.default_rng(seed)
    bootstrap_means = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, batch_size):
        stop = min(start + batch_size, resamples)
        indices = generator.integers(0, len(samples), size=(stop - start, len(samples)))
        bootstrap_means[start:stop] = samples[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(bootstrap_means, (tail, 1.0 - tail))
    return BootstrapInterval(
        estimate=float(samples.mean()),
        lower=float(lower),
        upper=float(upper),
        confidence=confidence,
        pair_count=len(samples),
        resamples=resamples,
        seed=seed,
    )


def paired_bootstrap_difference(
    left_seeds: Sequence[int],
    left_scores: Sequence[float] | np.ndarray,
    right_seeds: Sequence[int],
    right_scores: Sequence[float] | np.ndarray,
    **kwargs: float | int,
) -> BootstrapInterval:
    """Bootstrap ``left - right`` while preserving the shared seed pairing."""

    if tuple(left_seeds) != tuple(right_seeds):
        raise ValueError("paired difference requires identical ordered game seeds")
    left = np.asarray(left_scores, dtype=np.float64)
    right = np.asarray(right_scores, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError("paired difference score vectors must have equal shape")
    return paired_bootstrap_mean(left - right, **kwargs)
