from __future__ import annotations

from pathlib import Path

from splendor.engine import replay_game
from splendor.evaluation import baseline_policy_spec, evaluate_policies
from splendor.random_policy import RandomPolicy
from splendor.replay import state_digest
from splendor.runner import MatchOutcome, RunStatus, run_match


def test_match_result_is_structured_reproducible_and_replayable(tmp_path: Path) -> None:
    arguments = {
        "game_seed": 17,
        "policy_seeds": (101, 202),
        "first_player": 1,
        "policy_names": ("random-a", "random-b"),
        "pair_id": 3,
        "game_in_pair": 1,
    }
    left = run_match(
        RandomPolicy(101),
        RandomPolicy(202),
        failure_dir=tmp_path / "left",
        **arguments,
    )
    right = run_match(
        RandomPolicy(101),
        RandomPolicy(202),
        failure_dir=tmp_path / "right",
        **arguments,
    )

    assert left.status is RunStatus.NATURAL_TERMINAL
    assert left.normal_end
    assert left.terminal_result is not None
    assert left.outcome_for_a in tuple(MatchOutcome)
    assert left.completed_player_turns == sum(left.completed_turns)
    assert left.replay == right.replay
    assert left.final_state_digest == right.final_state_digest
    replayed = replay_game(left.replay, failure_dir=tmp_path / "replay")
    assert state_digest(replayed._debug_state()) == left.final_state_digest


def test_paired_evaluation_swaps_first_player_without_losing_pair_identity(
    tmp_path: Path,
) -> None:
    left = evaluate_policies(
        baseline_policy_spec("random"),
        baseline_policy_spec("random"),
        (40, 41),
        policy_seed_base=900,
        failure_dir=tmp_path / "left",
    )
    right = evaluate_policies(
        baseline_policy_spec("random"),
        baseline_policy_spec("random"),
        (40, 41),
        policy_seed_base=900,
        failure_dir=tmp_path / "right",
    )

    assert left.passed
    assert left.total_pairs == 2
    assert left.total_games == left.valid_games == 4
    assert left.a_wins + left.draws + left.a_losses == left.valid_games
    assert left.a_first.games == left.a_second.games == 2
    assert left.to_dict() == right.to_dict()
    for pair_id, seed in enumerate((40, 41)):
        pair = tuple(match for match in left.matches if match.pair_id == pair_id)
        assert len(pair) == 2
        assert {match.game_in_pair for match in pair} == {0, 1}
        assert {match.first_player for match in pair} == {0, 1}
        assert {match.game_seed for match in pair} == {seed}
        assert len({match.policy_seeds for match in pair}) == 1


def test_all_baseline_matchups_complete_without_engine_errors(tmp_path: Path) -> None:
    matchups = (
        ("random", "random"),
        ("score", "random"),
        ("development", "random"),
        ("score", "development"),
    )
    for index, (player_a, player_b) in enumerate(matchups):
        result = evaluate_policies(
            baseline_policy_spec(player_a),
            baseline_policy_spec(player_b),
            (index,),
            policy_seed_base=1_000 + index * 10,
            failure_dir=tmp_path / f"{player_a}-{player_b}",
        )
        assert result.passed
        assert result.valid_games == 2
        assert result.engineering_truncations == 0
        assert result.exceptions == 0
        assert result.illegal_actions == 0
        assert result.invariant_failures == 0


def test_evaluation_watchdog_is_reported_not_scored(tmp_path: Path) -> None:
    result = evaluate_policies(
        baseline_policy_spec("score"),
        baseline_policy_spec("development"),
        (7,),
        max_completed_turns=1,
        failure_dir=tmp_path,
    )

    assert not result.passed
    assert result.total_games == 2
    assert result.valid_games == 0
    assert result.engineering_truncations == 2
    assert result.win_rate_score is None
    assert result.a_first.win_rate_score is None
    assert result.a_second.win_rate_score is None
