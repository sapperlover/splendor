from __future__ import annotations

import inspect
import json
from pathlib import Path

from splendor.contracts import DecisionRequest
from splendor.engine import GameEngine, replay_game
from splendor.random_policy import RandomPolicy
from splendor.replay import state_digest
from splendor.result import TerminalResult
from splendor.runner import RunStatus, run_random_game
from splendor.validation import run_stress_validation


def test_random_policy_has_the_public_policy_signature_and_is_deterministic(tmp_path: Path) -> None:
    parameters = tuple(inspect.signature(RandomPolicy.choose).parameters)
    assert parameters == ("self", "request")

    request = GameEngine.new(7, failure_dir=tmp_path).decision()
    left = RandomPolicy(99)
    right = RandomPolicy(99)
    left_choices = tuple(left.choose(request) for _ in range(20))
    right_choices = tuple(right.choose(request) for _ in range(20))
    assert left_choices == right_choices
    assert all(action in request.legal_actions for action in left_choices)


def test_complete_random_game_is_reproducible_and_replayable(tmp_path: Path) -> None:
    left = run_random_game(0, failure_dir=tmp_path / "left")
    right = run_random_game(0, failure_dir=tmp_path / "right")

    assert left.status is RunStatus.NATURAL_TERMINAL
    assert isinstance(left.result, TerminalResult)
    assert left.replay == right.replay
    assert left.final_state_digest == right.final_state_digest
    assert left.completed_turns[0] == left.completed_turns[1]
    replayed = replay_game(left.replay, failure_dir=tmp_path / "replayed")
    assert state_digest(replayed._debug_state()) == left.final_state_digest


def test_engineering_turn_watchdog_is_not_a_game_result_and_saves_scene(tmp_path: Path) -> None:
    run = run_random_game(0, max_completed_turns=1, failure_dir=tmp_path)

    assert run.status is RunStatus.ENGINEERING_TRUNCATION
    assert run.result is None
    assert sum(run.completed_turns) == 1
    assert run.replay.result is None
    assert run.failure_scene is not None and run.failure_scene.exists()
    scene = json.loads(run.failure_scene.read_text(encoding="utf-8"))
    assert scene["reason"] == "engineering-truncation"
    assert scene["seed"] == 0
    assert scene["state"]["result"] is None
    assert scene["legal_actions"]
    assert scene["engine_rng"] == scene["state"]["rng"]


def test_seed_44_forced_skip_continues_to_a_formal_terminal(tmp_path: Path) -> None:

    run = run_random_game(44, failure_dir=tmp_path, raise_on_error=False)

    assert run.status is RunStatus.NATURAL_TERMINAL
    assert isinstance(run.result, TerminalResult)
    assert run.error_type is None
    assert run.failure_scene is None
    assert run.forced_skips == 1
    assert not run.forced_skip_draw


def test_small_stress_report_counts_outcomes_and_determinism(tmp_path: Path) -> None:
    report = run_stress_validation(
        5,
        seed_start=0,
        replay_sample_count=2,
        work_dir=tmp_path,
    )

    assert report.total_games == 5
    assert report.natural_endings == 5
    assert report.engineering_truncations == 0
    assert report.exceptions == 0
    assert report.no_legal_action_failures == 0
    assert report.replay_mismatches == 0
    assert report.save_restore_mismatches == 0
    assert report.player_0_wins + report.player_1_wins + report.draws == 5
    assert report.main_actions["BuyMarket"] > 0
    assert report.phase_entries["payment"] > 0


class _RequestOnlyPolicy:
    def __init__(self) -> None:
        self.requests: list[DecisionRequest] = []

    def choose(self, request: DecisionRequest):  # type: ignore[no-untyped-def]
        self.requests.append(request)
        return request.legal_actions[0]


def test_runner_passes_only_decision_request_to_policy(tmp_path: Path) -> None:
    policies = (_RequestOnlyPolicy(), _RequestOnlyPolicy())
    run = run_random_game(
        123,
        max_completed_turns=1,
        policies=policies,
        failure_dir=tmp_path,
    )

    assert run.status is RunStatus.ENGINEERING_TRUNCATION
    assert policies[0].requests or policies[1].requests
    assert all(
        isinstance(request, DecisionRequest)
        for policy in policies
        for request in policy.requests
    )
