from __future__ import annotations

import numpy as np
import pytest

from splendor.paired_statistics import (
    paired_bootstrap_difference,
    paired_bootstrap_mean,
)


def test_paired_bootstrap_constant_sample_has_exact_interval() -> None:
    interval = paired_bootstrap_mean([0.75] * 8, resamples=1_000, seed=7)

    assert interval.estimate == interval.lower == interval.upper == 0.75
    assert interval.pair_count == 8


def test_paired_difference_resamples_whole_seed_differences() -> None:
    seeds = (10, 11, 12, 13)
    left = np.asarray((1.0, 0.5, 0.5, 0.0))
    right = np.asarray((0.5, 0.5, 0.0, 0.0))
    interval = paired_bootstrap_difference(
        seeds, left, seeds, right, resamples=2_000, seed=8
    )

    assert interval.estimate == pytest.approx(0.25)
    assert interval.lower <= interval.estimate <= interval.upper


def test_paired_difference_rejects_different_seed_order() -> None:
    with pytest.raises(ValueError, match="identical ordered game seeds"):
        paired_bootstrap_difference((1, 2), (1.0, 0.0), (2, 1), (0.0, 1.0))
