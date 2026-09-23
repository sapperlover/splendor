"""Unit tests for the folded-depth ladder: plan, metrics and durability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from splendor.ladder_search import (
    GameStore,
    _aggregate,
    _decide,
    _parser,
    _rebuild,
    _run_header,
    require_scored_pair,
)


def _args(**overrides) -> argparse.Namespace:
    args = _parser().parse_args(["--checkpoint", "x.pt", "--output", "out.json"])
    args.bootstrap_resamples = 512
    for name, value in overrides.items():
        setattr(args, name, value)
    return args


def _game(
    pair_id: int,
    game_in_pair: int,
    outcome: str,
    *,
    changed: int = 0,
    rank_sum: int = 0,
    searchable: int = 0,
    status: str = "natural_terminal",
    player_turns: int = 100,
    decision_count: int = 57,
) -> dict:
    return {
        "pair_id": pair_id,
        "game_in_pair": game_in_pair,
        "game_seed": 9_100_000 + pair_id,
        "first_player": game_in_pair,
        "status": status,
        "outcome_for_a": outcome,
        "policy_seeds": [1, 2],
        "completed_player_turns": player_turns,
        "decision_count": decision_count,
        "policy_a_stats": {
            "decisions": 10,
            "searchable_decisions": searchable,
            "changed_action_count": changed,
            "chosen_rank_sum": rank_sum,
        },
        "policy_b_stats": {"decisions": 10, "searchable_decisions": 10},
    }


def _write_pair(
    store: GameStore,
    stage: int,
    block: int,
    pair_id: int,
    outcomes: tuple[str, str],
    **kwargs,
) -> None:
    directory = store.block_dir(stage, block)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"pair-{pair_id:07d}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        for game_in_pair, outcome in enumerate(outcomes):
            handle.write(
                json.dumps(
                    _game(pair_id, game_in_pair, outcome, **kwargs), sort_keys=True
                )
                + "\n"
            )
    store.clear_cache()


def _store(tmp_path: Path, **overrides) -> tuple[GameStore, argparse.Namespace]:
    args = _args(**overrides)
    store = GameStore(
        tmp_path / "out.json.games", _run_header(args, Path("x.pt").resolve())
    )
    store.start()
    return store, args


def _fill_block(
    store: GameStore, stage: int, block: int, pairs: int, outcomes: tuple[str, str]
) -> None:
    for pair_id in range(pairs):
        _write_pair(store, stage, block, pair_id, outcomes)


def test_decide_follows_threshold_floor_and_extra_blocks() -> None:
    args = _args(threshold=0.54, floor=0.50, extra_blocks=1)
    assert _decide(0.5401, args, 1) == "advance"
    assert _decide(0.54, args, 1) == "escalate"
    assert _decide(0.50, args, 1) == "escalate"
    assert _decide(0.4999, args, 1) == "stop"
    assert _decide(0.53, args, 2) == "stop"
    assert _decide(0.5401, args, 2) == "advance"
    assert _decide(0.49, args, 2) == "stop"


def test_aggregate_reports_search_and_outcome_metrics() -> None:
    args = _args(bootstrap_resamples=200)
    pairs = {
        0: [
            _game(0, 0, "a_win", changed=2, rank_sum=6, searchable=4),
            _game(0, 1, "a_win", changed=2, rank_sum=6, searchable=4),
        ],
        1: [
            _game(1, 0, "a_loss", searchable=4),
            _game(1, 1, "a_loss", searchable=4),
        ],
    }
    aggregate = _aggregate(pairs, args)
    assert aggregate["pairs"] == 2
    assert aggregate["games"] == 4
    assert aggregate["score_rate"] == pytest.approx(0.5)
    # One resampling unit per paired seed, never per game.
    assert aggregate["paired_bootstrap"]["pair_count"] == 2
    assert aggregate["paired_bootstrap_unit"] == "one_paired_seed"
    assert aggregate["game_length"]["player_turns_mean"] == pytest.approx(100.0)
    assert aggregate["game_length"]["decisions_mean"] == pytest.approx(57.0)
    change = aggregate["outcome_by_action_change"]["challenger"]
    assert change["changed_decisions"] == 4
    assert change["kept_decisions"] == 12
    # Changed decisions all belong to the pair A won; kept decisions are mostly
    # in the pair A lost.
    assert change["score_when_search_changed_action"] == pytest.approx(1.0)
    # Four kept decisions sit in the won pair, eight in the lost pair.
    assert change["score_when_search_kept_action"] == pytest.approx(4.0 / 12.0)
    quality = aggregate["decision_quality_challenger"]
    assert quality["search_changed_raw_action_count"] == 4
    assert quality["search_changed_raw_action_rate"] == pytest.approx(4 / 40)
    # rank sum 12 over all four games' searchable decisions (4 per game).
    assert quality["chosen_policy_rank"] == pytest.approx(12 / 16)


def test_up_rung_advances_above_threshold(tmp_path: Path) -> None:
    store, args = _store(
        tmp_path,
        threshold=0.54,
        floor=0.50,
        extra_blocks=1,
        pairs=4,
        start_depth=4,
        max_depth=6,
        min_depth=1,
    )
    _fill_block(store, 0, 0, 4, ("a_win", "a_win"))
    position = _rebuild(args, store)
    assert not position.finished
    assert position.stages[0]["decision"] == "advance"
    assert position.stages[0]["mode"] == "up"
    assert position.rung is not None
    assert (position.rung.mode, position.rung.incumbent_depth) == ("up", 5)
    assert position.rung.index == 1


def test_failed_up_rung_turns_into_a_downward_probe(tmp_path: Path) -> None:
    store, args = _store(
        tmp_path,
        threshold=0.54,
        floor=0.50,
        extra_blocks=1,
        pairs=4,
        start_depth=4,
        max_depth=6,
        min_depth=1,
    )
    # d5 vs d4 sits in the escalation band, then still misses the threshold.
    _fill_block(store, 0, 0, 4, ("a_win", "a_loss"))
    position = _rebuild(args, store)
    assert not position.finished
    assert position.rung is not None
    assert (position.rung.mode, position.rung.incumbent_depth) == ("up", 4)
    assert position.next_block == 1
    _fill_block(store, 0, 1, 4, ("a_win", "a_loss"))
    position = _rebuild(args, store)
    assert not position.finished
    assert position.stages[0]["decision"] == "stop"
    assert position.rung is not None
    assert (
        position.rung.mode,
        position.rung.incumbent_depth,
        position.rung.challenger_depth,
    ) == ("down", 3, 4)
    assert position.rung.index == 1


def test_downward_probe_passing_stops_the_ladder(tmp_path: Path) -> None:
    store, args = _store(
        tmp_path,
        threshold=0.54,
        floor=0.50,
        extra_blocks=1,
        pairs=4,
        start_depth=4,
        max_depth=6,
        min_depth=1,
    )
    _fill_block(store, 0, 0, 4, ("a_loss", "a_loss"))  # d5 vs d4 fails hard
    _fill_block(store, 1, 0, 4, ("a_win", "a_win"))  # d4 vs d3 passes
    position = _rebuild(args, store)
    assert position.finished
    assert position.stop_reason == "downward_probe_passed"
    assert position.stopped_at_depth == 3
    assert [stage["mode"] for stage in position.stages] == ["up", "down"]


def test_downward_probe_keeps_walking_down_and_respects_min_depth(
    tmp_path: Path,
) -> None:
    store, args = _store(
        tmp_path,
        threshold=0.54,
        floor=0.50,
        extra_blocks=1,
        pairs=4,
        start_depth=4,
        max_depth=6,
        min_depth=3,
    )
    _fill_block(store, 0, 0, 4, ("a_loss", "a_loss"))  # d5 vs d4 fails
    _fill_block(store, 1, 0, 4, ("a_loss", "a_loss"))  # d4 vs d3 fails
    position = _rebuild(args, store)
    assert position.finished
    assert position.stop_reason == "reached_min_depth"
    assert position.stopped_at_depth == 3
    assert [stage["incumbent_depth"] for stage in position.stages] == [4, 3]


def test_deepest_probe_is_d2_vs_d1(tmp_path: Path) -> None:
    store, args = _store(
        tmp_path,
        threshold=0.54,
        floor=0.50,
        extra_blocks=1,
        pairs=4,
        start_depth=3,
        max_depth=4,
        min_depth=1,
    )
    _fill_block(store, 0, 0, 4, ("a_loss", "a_loss"))  # d4 vs d3 fails
    _fill_block(store, 1, 0, 4, ("a_loss", "a_loss"))  # d3 vs d2 fails
    position = _rebuild(args, store)
    assert not position.finished
    assert position.rung is not None
    assert (
        position.rung.mode,
        position.rung.incumbent_depth,
        position.rung.challenger_depth,
    ) == ("down", 1, 2)


def test_game_log_is_durable_per_game(tmp_path: Path) -> None:
    store, _ = _store(tmp_path, pairs=2)
    _write_pair(store, 0, 0, 0, ("a_win", "a_loss"))
    directory = store.block_dir(0, 0)
    (directory / "pair-0000001.jsonl").write_text(
        json.dumps(_game(1, 0, "a_win"), sort_keys=True) + "\n", encoding="utf-8"
    )
    store.clear_cache()
    assert set(store.scored_pairs(0, 0)) == {0}
    assert store.games(0, 0)[1][0]["game_in_pair"] == 0


def test_unscored_games_do_not_count_as_a_finished_pair(tmp_path: Path) -> None:
    store, _ = _store(tmp_path, pairs=2)
    _write_pair(store, 0, 0, 0, ("a_win", "a_loss"), status="engineering_truncation")
    _write_pair(store, 0, 0, 1, ("a_win", "a_loss"))
    assert set(store.scored_pairs(0, 0)) == {1}


def test_partial_last_line_is_ignored_on_resume(tmp_path: Path) -> None:
    store, _ = _store(tmp_path, pairs=2)
    _write_pair(store, 0, 0, 0, ("a_win", "a_win"))
    path = store.block_dir(0, 0) / "pair-0000001.jsonl"
    path.write_text(
        json.dumps(_game(1, 0, "a_win"), sort_keys=True) + '\n{"pair_id": 1, "gam',
        encoding="utf-8",
    )
    fresh = GameStore(store.root, {})
    fresh.load()
    assert fresh.games(0, 0)[1][0]["pair_id"] == 1
    assert set(fresh.scored_pairs(0, 0)) == {0}


def test_discard_pair_removes_the_partial_file(tmp_path: Path) -> None:
    store, _ = _store(tmp_path, pairs=2)
    _write_pair(store, 0, 0, 0, ("a_win", "a_win"))
    path = store.block_dir(0, 0) / "pair-0000000.jsonl"
    assert path.exists()
    store.discard_pair(0, 0, 0)
    assert not path.exists()
    store.clear_cache()
    assert store.scored_pairs(0, 0) == {}


def test_resume_header_must_match(tmp_path: Path) -> None:
    from splendor.ladder_search import _check_resume_header

    store, args = _store(tmp_path, pairs=2)
    store.load()
    _check_resume_header(store.header, args, Path("x.pt").resolve())
    with pytest.raises(ValueError):
        _check_resume_header(store.header, _args(pairs=3), Path("x.pt").resolve())
    with pytest.raises(ValueError):
        _check_resume_header(
            store.header, _args(pairs=2, threshold=0.6), Path("x.pt").resolve()
        )
    # Extending the plan bounds is allowed: it only changes how far the ladder
    # walks, not the semantics of the stages already in the log.
    _check_resume_header(
        store.header, _args(pairs=2, max_depth=9, min_depth=1), Path("x.pt").resolve()
    )


def test_require_scored_pair_rejects_unscored_games() -> None:
    from splendor.evaluation import ParallelPairResult
    from splendor.runner import MatchOutcome, MatchResult, RunStatus

    def match(status: RunStatus, outcome) -> MatchResult:
        return MatchResult(
            pair_id=0,
            game_in_pair=0,
            game_seed=1,
            policy_names=("a", "b"),
            policy_seeds=(1, 2),
            first_player=0,
            status=status,
            normal_end=False,
            terminal_result=None,
            outcome_for_a=outcome,
            winner_id=None,
            draw=False,
            scores=None,
            purchased_card_counts=None,
            completed_turns=(10, 10),
            completed_player_turns=20,
            decision_count=40,
            forced_skips=0,
            replay=None,  # type: ignore[arg-type]
            final_state_digest="x",
        )

    good = ParallelPairResult(
        matches=(
            match(RunStatus.NATURAL_TERMINAL, MatchOutcome.A_WIN),
            match(RunStatus.NATURAL_TERMINAL, MatchOutcome.A_WIN),
        ),
        policy_a_stats={},
        policy_b_stats={},
    )
    require_scored_pair(good)
    bad = ParallelPairResult(
        matches=(
            match(RunStatus.ENGINEERING_TRUNCATION, None),
            match(RunStatus.NATURAL_TERMINAL, MatchOutcome.A_WIN),
        ),
        policy_a_stats={},
        policy_b_stats={},
    )
    with pytest.raises(RuntimeError):
        require_scored_pair(bad)
