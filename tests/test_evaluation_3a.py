from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

from splendor._state import PlayerState
from splendor.actions import ReturnTokens, TakeDistinct
from splendor.catalog import load_default_catalog
from splendor.engine import replay_game
from splendor.evaluation import (
    PolicySpec,
    SerializablePolicySpec,
    baseline_policy_spec,
    evaluate_policies,
    evaluate_policies_parallel,
)
from splendor.model import Color, DecisionPhase, TokenVector
from splendor.paired_statistics import pair_scores, serialized_pair_scores
from splendor.random_policy import RandomPolicy
from splendor.replay import new_replay, state_digest
from splendor.runner import (
    MatchOutcome,
    RunStatus,
    canonical_state_signature,
    run_match,
)
import splendor.runner as runner_module


def test_fixed_evaluation_retries_only_unscored_pairs_and_preserves_failures(tmp_path, monkeypatch):
    import splendor.evaluation as evaluation

    config = evaluation.FixedEvaluationConfig(
        policy_a=SerializablePolicySpec.baseline("score"),
        policy_b=SerializablePolicySpec.baseline("random"), game_seeds=(17, 18))
    path = tmp_path / "evaluation.json"
    original = evaluation.run_match
    calls = []

    def injected_failure(*args, **kwargs):
        calls.append(kwargs["pair_id"])
        match = original(*args, **kwargs)
        if kwargs["pair_id"] == 0:
            return replace(match, status=RunStatus.ERROR, normal_end=False,
                           outcome_for_a=None, error_type="FailOnRecompileLimitHit",
                           error_detail="injected compiler failure")
        return match

    monkeypatch.setattr(evaluation, "run_match", injected_failure)
    failed = evaluation.run_fixed_evaluation(config, path, require_scored_pairs=True)
    assert not failed.summary["passed"] and failed.summary["exceptions"] == 2
    root = Path(str(path) + ".games")
    pair0 = (root / "pair-0000000.jsonl").read_bytes()
    pair1 = (root / "pair-0000001.jsonl").read_bytes()
    summary = path.read_bytes()
    calls.clear()

    def successful(*args, **kwargs):
        calls.append(kwargs["pair_id"])
        return original(*args, **kwargs)

    monkeypatch.setattr(evaluation, "run_match", successful)
    resumed = evaluation.run_fixed_evaluation(config, path, resume=True, require_scored_pairs=True)
    assert resumed.summary["passed"] and resumed.summary["valid_games"] == 4
    assert calls == [0, 0]
    assert (root / "pair-0000001.jsonl").read_bytes() == pair1
    archives = list((root / "retries").iterdir())
    assert len(archives) == 1
    assert (archives[0] / "pair-0000000.jsonl").read_bytes() == pair0
    assert (archives[0] / "summary.json").read_bytes() == summary
    calls.clear()
    evaluation.run_fixed_evaluation(config, path, resume=True, require_scored_pairs=True)
    assert calls == []


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


def test_pair_parallel_evaluation_matches_serial_result(tmp_path: Path) -> None:
    seeds = (140, 141, 142, 143)
    serial = evaluate_policies(
        baseline_policy_spec("random"),
        baseline_policy_spec("random"),
        seeds,
        policy_seed_base=1_900,
        failure_dir=tmp_path / "serial",
    )
    parallel = evaluate_policies_parallel(
        SerializablePolicySpec.baseline("random"),
        SerializablePolicySpec.baseline("random"),
        seeds,
        workers=2,
        policy_seed_base=1_900,
        failure_dir=tmp_path / "parallel",
    ).result

    assert parallel.passed
    assert parallel.to_dict() == serial.to_dict()


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


class _DeterministicTokenCyclePolicy:
    def choose(self, request):  # type: ignore[no-untyped-def]
        if request.observation.phase is DecisionPhase.MAIN_ACTION:
            return next(
                action
                for action in request.legal_actions
                if isinstance(action, TakeDistinct)
            )
        if request.observation.phase is DecisionPhase.RETURN_TOKENS:
            return next(
                action
                for action in request.legal_actions
                if isinstance(action, ReturnTokens)
                and action.tokens == TokenVector(red=1)
            )
        return request.legal_actions[0]


def test_canonical_signature_excludes_only_record_counters(tmp_path: Path) -> None:
    catalog = load_default_catalog()
    state = runner_module.GameEngine.new(91, catalog=catalog, failure_dir=tmp_path)._debug_state()
    signature = canonical_state_signature(state, catalog)

    assert canonical_state_signature(
        replace(state, completed_turns=(12, 12)), catalog
    ) == signature
    assert canonical_state_signature(
        replace(state, rng=replace(state.rng, draws=state.rng.draws + 99)), catalog
    ) == signature
    assert canonical_state_signature(
        replace(state, current_player=1 - state.current_player), catalog
    ) != signature
    first, second, *rest = state.decks.tier_one
    moved_deck = replace(state.decks, tier_one=(second, first, *rest))
    assert canonical_state_signature(replace(state, decks=moved_deck), catalog) != signature

    assert len(signature) == 64
    assert set(signature) <= set("0123456789abcdef")


def test_deterministic_evaluation_scores_repeated_complete_state_as_draw(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    catalog = load_default_catalog()
    initial = runner_module.GameEngine.new(
        123, first_player=0, catalog=catalog, failure_dir=tmp_path / "initial"
    )._debug_state()

    def cycle_engine_new(
        cls,
        seed,
        first_player=None,
        *,
        catalog=None,
        failure_dir=None,
    ):  # type: ignore[no-untyped-def]
        del cls
        selected_catalog = catalog or load_default_catalog()
        actor = 0 if first_player is None else first_player
        state = replace(
            initial,
            supply=TokenVector(red=1, gold=4),
            players=(
                PlayerState(TokenVector(white=4, blue=4, green=2)),
                PlayerState(TokenVector(green=2, red=3, black=4, gold=1)),
            ),
            first_player=actor,
            current_player=actor,
            completed_turns=(0, 0),
        )
        return runner_module.GameEngine(
            state,
            selected_catalog,
            new_replay(selected_catalog, seed, actor),
            failure_dir or tmp_path / "failures",
        )

    monkeypatch.setattr(
        runner_module.GameEngine,
        "new",
        classmethod(cycle_engine_new),
    )
    spec = PolicySpec("cycle", lambda seed: _DeterministicTokenCyclePolicy())
    result = evaluate_policies(
        spec,
        spec,
        (77,),
        max_completed_turns=20,
        failure_dir=tmp_path / "evaluation",
        adjudicate_deterministic_policy_cycles=True,
    )

    assert result.passed
    assert result.valid_games == result.total_games == 2
    assert result.policy_cycle_draws == result.draws == 2
    assert result.engineering_truncations == 0
    assert result.win_rate_score == 0.5
    assert all(match.status is RunStatus.POLICY_CYCLE_DRAW for match in result.matches)
    assert all(match.terminal_result is None for match in result.matches)
    assert all(match.replay.result is None for match in result.matches)
    assert all(match.failure_scene is not None for match in result.matches)
    for match in result.matches:
        scene = json.loads(match.failure_scene.read_text(encoding="utf-8"))
        assert scene["reason"] == "policy-cycle-draw"
        assert scene["context"]["first_seen_decision"] == 2
        assert scene["context"]["repeated_at_decision"] == 6
        assert scene["context"]["repeated_players"] == [0, 1]

    _, scores = pair_scores(result)
    assert scores.tolist() == [0.5]
    _, serialized_scores = serialized_pair_scores(result.to_dict())
    assert serialized_scores.tolist() == [0.5]
