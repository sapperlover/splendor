"""Policy-only surprise weighting from frozen search/teacher distributions.

Adapted from KataGo's per-game uniform/surprise mixture. We deliberately use
the archived *unnoised* root_network_prior and change only policy loss weights,
not sampling probabilities or value weights. Validation remains unweighted.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import numpy as np

from .alphazero_replay import ReplayReference
from .selfplay_data import SelfPlayBatch, SelfPlaySample


@dataclass(frozen=True)
class PolicySurpriseConfig:
    fraction: float = 0.5

    def __post_init__(self):
        if (isinstance(self.fraction, bool) or not isinstance(self.fraction, (int, float))
                or not math.isfinite(self.fraction) or not 0 <= self.fraction <= 1):
            raise ValueError("policy surprise fraction must be finite and in [0, 1]")


def game_policy_surprise(arrays: Mapping[str, np.ndarray], config: PolicySurpriseConfig):
    """Return KL(search || teacher) and effective weights for every game row.

    A fraction of each game's searchable policy mass is redistributed in
    proportion to KL. Forced moves retain zero policy weight. A game with no
    surprise retains its original weights. No validation or learner prediction
    enters the normalization, and the raw archive is never modified.
    """
    starts = arrays["candidate_offsets"][:-1]
    lengths = np.diff(arrays["candidate_offsets"])
    target = np.asarray(arrays["policy_target"], dtype=np.float64)
    prior = np.asarray(arrays["root_network_prior"], dtype=np.float64)
    # Stored float32 distributions sum to one within archive tolerance.
    target = target / np.repeat(np.add.reduceat(target, starts), lengths)
    prior = prior / np.repeat(np.add.reduceat(prior, starts), lengths)
    log_target = np.zeros_like(target)
    np.log(target, out=log_target, where=target > 0)
    terms = target * (log_target - np.log(np.maximum(prior, 1e-100)))
    surprise = np.maximum(0., np.add.reduceat(terms, starts))
    base = np.asarray(arrays["policy_weight"], dtype=np.float64)
    mass = float(base.sum())
    surprise_mass = float(np.dot(base, surprise))
    weights = base.copy()
    if surprise_mass > 0:
        weights *= (1 - config.fraction) + config.fraction * surprise * mass / surprise_mass
    return surprise, weights.astype(np.float32)


class PolicySurpriseWeighting:
    """Small per-game weight cache; deterministic across process resumes."""

    def __init__(self, config: PolicySurpriseConfig):
        self.config = config
        self.games = {}

    def apply(self, batch: SelfPlayBatch, samples: Sequence[SelfPlaySample],
              references: Sequence[ReplayReference]):
        weights, surprises = [], []
        for sample, ref in zip(samples, references, strict=True):
            key = ref.run_id, ref.round_index, ref.game_id
            if key not in self.games:
                self.games[key] = game_policy_surprise(sample.arrays, self.config)
            surprise, effective = self.games[key]
            weights.append(effective[sample.index])
            surprises.append(surprise[sample.index])
        weights = np.asarray(weights, dtype=np.float32)
        active = batch.policy_weights > 0
        selected = weights[active].astype(np.float64)
        mass = float(selected.sum())
        square_mass = float(np.dot(selected, selected))
        stats = {"fraction": self.config.fraction, "policy_weight_sum": mass,
                 "effective_policy_positions": mass * mass / square_mass if square_mass else 0.,
                 "mean_weight": float(selected.mean()) if len(selected) else 0.,
                 "max_weight": float(selected.max()) if len(selected) else 0.,
                 "mean_teacher_surprise": float(np.asarray(surprises)[active].mean()) if active.any() else 0.}
        return replace(batch, policy_weights=weights), stats
