from __future__ import annotations

import pytest

from splendor.analyze_5c import (
    _action_bins,
    _bin_end,
    _historical_result_summary,
    _opponent_bins,
    _quality_bins,
)


def test_bin_end_keeps_exact_boundaries_in_the_ending_bin() -> None:
    assert _bin_end(1, 100_000, 5_000_000) == 100_000
    assert _bin_end(100_000, 100_000, 5_000_000) == 100_000
    assert _bin_end(100_001, 100_000, 5_000_000) == 200_000
    assert _bin_end(5_000_000, 100_000, 5_000_000) == 5_000_000


def test_binned_training_metrics_use_counts_as_weights() -> None:
    records = [
        {
            "games_completed": 2,
            "opponent_latest_count": 2,
            "opponent_latest_train_score": 0.5,
            "opponent_historical_count": 0,
            "opponent_score_count": 0,
            "opponent_development_count": 0,
            "TakeDistinct_count": 3,
            "BuyMarket_count": 1,
            "mean_completed_player_turns": 60.0,
            "mean_final_score": 13.0,
            "train_win_rate": 0.5,
            "train_draw_rate": 0.0,
            "train_loss_rate": 0.5,
        },
        {
            "games_completed": 1,
            "opponent_latest_count": 1,
            "opponent_latest_train_score": 1.0,
            "opponent_historical_count": 0,
            "opponent_score_count": 0,
            "opponent_development_count": 0,
            "TakeDistinct_count": 1,
            "BuyMarket_count": 3,
            "mean_completed_player_turns": 30.0,
            "mean_final_score": 16.0,
            "train_win_rate": 1.0,
            "train_draw_rate": 0.0,
            "train_loss_rate": 0.0,
        },
    ]
    grouped = [(100_000, records)]

    opponent = _opponent_bins(grouped, 100_000)[0]
    assert opponent["latest_games"] == 3
    assert opponent["latest_learner_score_rate"] == pytest.approx(2.0 / 3.0)

    actions = _action_bins(grouped, 100_000)[0]
    assert actions["main_actions"] == 8
    assert actions["main_action_decisions"] == 8
    assert actions["TakeDistinct_fraction"] == pytest.approx(0.5)
    assert actions["BuyMarket_fraction"] == pytest.approx(0.5)

    quality = _quality_bins(grouped, 100_000)[0]
    assert quality["games"] == 3
    assert quality["mean_completed_player_turns"] == pytest.approx(50.0)
    assert quality["mean_final_score"] == pytest.approx(14.0)
    assert quality["learner_score_rate"] == pytest.approx(2.0 / 3.0)


def test_reversing_a_paired_historical_result_swaps_sides_and_first_player() -> None:
    result = {
        "win_rate_score": 0.7,
        "a_wins": 7,
        "draws": 0,
        "a_losses": 3,
        "a_first": {"win_rate_score": 0.8},
        "a_second": {"win_rate_score": 0.6},
        "mean_completed_player_turns": 55.0,
        "policy_cycle_draws": 0,
        "forced_skips": 1,
        "engineering_truncations": 0,
        "illegal_actions": 0,
        "invariant_failures": 0,
        "exceptions": 0,
    }

    reversed_result = _historical_result_summary(
        result,
        row_model="older",
        column_model="newer",
        reverse=True,
        source="test",
    )

    assert reversed_result["score_rate"] == pytest.approx(0.3)
    assert reversed_result["wins"] == 3
    assert reversed_result["losses"] == 7
    assert reversed_result["first_player_score"] == pytest.approx(0.4)
    assert reversed_result["second_player_score"] == pytest.approx(0.2)
