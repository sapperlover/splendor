from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import median

from .catalog import Catalog, load_default_catalog
from .engine import GameEngine, replay_game
from .invariants import InvariantViolation
from .model import DecisionPhase
from .replay import ReplayRecord, state_digest
from .runner import RunStatus, run_random_game, terminal_branch


REQUIRED_REPLAY_COVERAGE = frozenset(
    {
        DecisionPhase.MAIN_ACTION.value,
        DecisionPhase.PAYMENT.value,
        DecisionPhase.RETURN_TOKENS.value,
        DecisionPhase.CHOOSE_NOBLE.value,
        "TakeDistinct",
        "TakePair",
        "ReserveMarket",
        "ReserveDeck",
        "BuyMarket",
        "BuyReserved",
        "multiple_payment_options",
        "forced_skip",
        "final_round",
        "terminal",
    }
)
REQUIRED_SAVE_RESTORE_COVERAGE = frozenset(
    {
        DecisionPhase.MAIN_ACTION.value,
        DecisionPhase.PAYMENT.value,
        DecisionPhase.RETURN_TOKENS.value,
        DecisionPhase.CHOOSE_NOBLE.value,
        "final_round_pending",
        "forced_skip_pending",
    }
)


@dataclass(frozen=True, slots=True)
class ValidationReport:
    total_games: int
    seed_start: int
    seed_end_inclusive: int
    max_completed_turns: int
    natural_endings: int
    engineering_truncations: int
    exceptions: int
    player_0_wins: int
    player_1_wins: int
    draws: int
    mean_completed_player_turns: float
    median_completed_player_turns: float
    max_completed_player_turns_observed: int
    mean_policy_decisions: float
    phase_entries: dict[str, int]
    main_actions: dict[str, int]
    noble_claims: int
    final_round_triggers: int
    forced_skips: int
    forced_skip_draws: int
    terminal_branches: dict[str, int]
    no_legal_action_failures: int
    illegal_action_failures: int
    invariant_failures: int
    replay_samples_checked: int
    replay_mismatches: int
    save_restore_samples_checked: int
    save_restore_mismatches: int
    replay_coverage: tuple[str, ...]
    missing_replay_coverage: tuple[str, ...]
    save_restore_coverage: tuple[str, ...]
    missing_save_restore_coverage: tuple[str, ...]
    failure_scenes: tuple[str, ...]
    _completed_turn_samples: tuple[int, ...] = field(repr=False)
    _decision_samples: tuple[int, ...] = field(repr=False)

    @property
    def passed(self) -> bool:
        return not (
            self.engineering_truncations
            or self.exceptions
            or self.illegal_action_failures
            or self.invariant_failures
            or self.replay_mismatches
            or self.save_restore_mismatches
            or self.missing_replay_coverage
            or self.missing_save_restore_coverage
        )

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value.pop("_completed_turn_samples")
        value.pop("_decision_samples")
        value["passed"] = self.passed
        return value


def _external_actions(record: ReplayRecord) -> tuple[object, ...]:
    return tuple(event.action for event in record.events if not event.automatic)


def _validate_replay(record: ReplayRecord, catalog: Catalog, failure_dir: Path) -> None:
    replayed = replay_game(record, catalog=catalog, failure_dir=failure_dir)
    expected_digest = record.events[-1].state_digest_after if record.events else state_digest(replayed._debug_state())
    if state_digest(replayed._debug_state()) != expected_digest:
        replayed._fail(AssertionError("replay final authoritative state digest differs"))
    if replayed.replay_record != record:
        replayed._fail(AssertionError("replay record differs after deterministic replay"))


def _validate_save_restore(
    record: ReplayRecord,
    decision_index: int,
    catalog: Catalog,
    work_dir: Path,
) -> None:
    actions = _external_actions(record)
    if not 0 <= decision_index < len(actions):
        raise AssertionError(f"invalid checkpoint decision index: {decision_index}")
    original = GameEngine.new(
        record.header.game_seed,
        first_player=record.header.first_player,
        catalog=catalog,
        failure_dir=work_dir / "original-failures",
    )
    for action in actions[:decision_index]:
        original.apply(action)  # automatic events are reproduced inside apply()

    save_path = work_dir / "checkpoint.json"
    original.save(save_path)
    restored = GameEngine.restore(
        save_path,
        catalog=catalog,
        failure_dir=work_dir / "restored-failures",
    )

    def fail(detail: str) -> None:
        original.save(work_dir / "mismatch-original.json")
        restored.save(work_dir / "mismatch-restored.json")
        raise AssertionError(detail)

    if state_digest(original._debug_state()) != state_digest(restored._debug_state()):
        fail("authoritative state differs immediately after restore")
    if original._debug_state().rng != restored._debug_state().rng:
        fail("engine RNG differs immediately after restore")
    if original.decision() != restored.decision():
        fail("legal decision request differs immediately after restore")
    if original.replay_record != restored.replay_record:
        fail("replay prefix differs immediately after restore")

    for action in actions[decision_index:]:
        left = original.apply(action)
        right = restored.apply(action)
        if left != right:
            fail("decision/result differs after restored continuation")
        if state_digest(original._debug_state()) != state_digest(restored._debug_state()):
            fail("authoritative state differs after restored continuation")
        if original._debug_state().rng != restored._debug_state().rng:
            fail("engine RNG differs after restored continuation")
        if original.replay_record != restored.replay_record:
            fail("replay differs after restored continuation")

    if original.replay_record != record:
        fail("continued replay does not equal uninterrupted replay")


def run_stress_validation(
    game_count: int,
    *,
    seed_start: int = 0,
    max_completed_turns: int = 400,
    replay_sample_count: int = 64,
    catalog: Catalog | None = None,
    work_dir: Path | None = None,
) -> ValidationReport:
    if game_count <= 0:
        raise ValueError("game_count must be positive")
    if replay_sample_count <= 0:
        raise ValueError("replay_sample_count must be positive")
    catalog = catalog or load_default_catalog()
    work_dir = work_dir or Path.cwd() / "engine_validation"
    failure_dir = work_dir / "failure-scenes"
    checkpoint_dir = work_dir / "save-restore"

    statuses: Counter[RunStatus] = Counter()
    wins: Counter[int] = Counter()
    phase_entries: Counter[str] = Counter()
    main_actions: Counter[str] = Counter()
    terminal_branches: Counter[str] = Counter()
    turn_totals: list[int] = []
    decision_totals: list[int] = []
    noble_claims = 0
    trigger_count = 0
    forced_skips = 0
    forced_skip_draws = 0
    invariant_failures = 0
    illegal_failures = 0
    no_action_failures = 0
    failure_scenes: list[str] = []

    replay_samples: dict[int, ReplayRecord] = {}
    replay_tag_samples: dict[str, ReplayRecord] = {}
    checkpoint_samples: dict[str, tuple[ReplayRecord, int]] = {}
    stride = max(1, game_count // replay_sample_count)

    for offset in range(game_count):
        seed = seed_start + offset
        run = run_random_game(
            seed,
            max_completed_turns=max_completed_turns,
            catalog=catalog,
            failure_dir=failure_dir,
            raise_on_error=False,
        )
        statuses[run.status] += 1
        turn_totals.append(sum(run.completed_turns))
        decision_totals.append(run.decision_count)
        phase_entries.update(run.phase_entries)
        main_actions.update(run.main_actions)
        noble_claims += run.noble_claims
        trigger_count += run.final_round_triggers
        forced_skips += run.forced_skips
        forced_skip_draws += run.forced_skip_draw
        if run.failure_scene is not None:
            failure_scenes.append(str(run.failure_scene))
        if run.error_type == InvariantViolation.__name__:
            invariant_failures += 1
        if run.error_type == "IllegalActionError":
            illegal_failures += 1
        if run.error_detail == "a non-terminal state has no legal action":
            no_action_failures += 1

        if run.result is not None:
            terminal_branches[terminal_branch(run.result)] += 1
            if len(run.result.winner_ids) == 2:
                wins[2] += 1
            else:
                wins[run.result.winner_ids[0]] += 1

        # Determinism acceptance samples complete, naturally terminated games.
        # Error prefixes are separately retained as failure scenes and must not
        # be misreported as replay/save mismatches when they reproduce the same
        # expected exception at their final action.
        if run.status is RunStatus.NATURAL_TERMINAL:
            if offset % stride == 0 and len(replay_samples) < replay_sample_count:
                replay_samples[seed] = run.replay
            for tag in run.coverage_tags:
                if tag in REQUIRED_REPLAY_COVERAGE:
                    replay_tag_samples.setdefault(tag, run.replay)
            for tag, index in run.checkpoint_indices.items():
                if tag in REQUIRED_SAVE_RESTORE_COVERAGE:
                    checkpoint_samples.setdefault(tag, (run.replay, index))

    for record in replay_tag_samples.values():
        replay_samples.setdefault(record.header.game_seed, record)

    replay_mismatches = 0
    for record in replay_samples.values():
        try:
            _validate_replay(record, catalog, failure_dir / "replay")
        except Exception:
            replay_mismatches += 1

    save_restore_mismatches = 0
    checked_checkpoints = 0
    for tag, (record, index) in checkpoint_samples.items():
        checked_checkpoints += 1
        try:
            _validate_save_restore(
                record,
                index,
                catalog,
                checkpoint_dir / tag,
            )
        except Exception:
            save_restore_mismatches += 1

    replay_coverage = tuple(sorted(replay_tag_samples))
    restore_coverage = tuple(sorted(checkpoint_samples))
    total = len(turn_totals)
    return ValidationReport(
        total_games=game_count,
        seed_start=seed_start,
        seed_end_inclusive=seed_start + game_count - 1,
        max_completed_turns=max_completed_turns,
        natural_endings=statuses[RunStatus.NATURAL_TERMINAL],
        engineering_truncations=statuses[RunStatus.ENGINEERING_TRUNCATION],
        exceptions=statuses[RunStatus.ERROR],
        player_0_wins=wins[0],
        player_1_wins=wins[1],
        draws=wins[2],
        mean_completed_player_turns=sum(turn_totals) / total,
        median_completed_player_turns=float(median(turn_totals)),
        max_completed_player_turns_observed=max(turn_totals),
        mean_policy_decisions=sum(decision_totals) / total,
        phase_entries={phase.value: phase_entries[phase.value] for phase in DecisionPhase if phase is not DecisionPhase.TERMINAL},
        main_actions={name: main_actions[name] for name in _MAIN_ACTION_NAMES_FOR_REPORT},
        noble_claims=noble_claims,
        final_round_triggers=trigger_count,
        forced_skips=forced_skips,
        forced_skip_draws=forced_skip_draws,
        terminal_branches={
            branch: terminal_branches[branch]
            for branch in (
                "higher_score",
                "fewer_development_cards",
                "draw",
                "forced_skip_draw",
            )
        },
        no_legal_action_failures=no_action_failures,
        illegal_action_failures=illegal_failures,
        invariant_failures=invariant_failures,
        replay_samples_checked=len(replay_samples),
        replay_mismatches=replay_mismatches,
        save_restore_samples_checked=checked_checkpoints,
        save_restore_mismatches=save_restore_mismatches,
        replay_coverage=replay_coverage,
        missing_replay_coverage=tuple(sorted(REQUIRED_REPLAY_COVERAGE - set(replay_coverage))),
        save_restore_coverage=restore_coverage,
        missing_save_restore_coverage=tuple(
            sorted(REQUIRED_SAVE_RESTORE_COVERAGE - set(restore_coverage))
        ),
        failure_scenes=tuple(failure_scenes),
        _completed_turn_samples=tuple(turn_totals),
        _decision_samples=tuple(decision_totals),
    )


_MAIN_ACTION_NAMES_FOR_REPORT = (
    "TakeDistinct",
    "TakePair",
    "ReserveMarket",
    "ReserveDeck",
    "BuyMarket",
    "BuyReserved",
)


def _run_stress_shard(
    arguments: tuple[int, int, int, int, Path],
) -> ValidationReport:
    count, seed_start, max_completed_turns, replay_samples, work_dir = arguments
    return run_stress_validation(
        count,
        seed_start=seed_start,
        max_completed_turns=max_completed_turns,
        replay_sample_count=replay_samples,
        work_dir=work_dir,
    )


def _merge_reports(reports: tuple[ValidationReport, ...]) -> ValidationReport:
    ordered = tuple(sorted(reports, key=lambda report: report.seed_start))
    if not ordered:
        raise ValueError("at least one report is required")
    if len({report.max_completed_turns for report in ordered}) != 1:
        raise ValueError("cannot merge reports with different watchdog thresholds")
    for left, right in zip(ordered, ordered[1:]):
        if left.seed_end_inclusive + 1 != right.seed_start:
            raise ValueError("shard seed ranges must be contiguous")

    turn_samples = tuple(value for report in ordered for value in report._completed_turn_samples)
    decision_samples = tuple(value for report in ordered for value in report._decision_samples)
    phase_entries: Counter[str] = Counter()
    main_actions: Counter[str] = Counter()
    terminal_branches: Counter[str] = Counter()
    replay_coverage: set[str] = set()
    restore_coverage: set[str] = set()
    for report in ordered:
        phase_entries.update(report.phase_entries)
        main_actions.update(report.main_actions)
        terminal_branches.update(report.terminal_branches)
        replay_coverage.update(report.replay_coverage)
        restore_coverage.update(report.save_restore_coverage)

    total = len(turn_samples)
    return ValidationReport(
        total_games=sum(report.total_games for report in ordered),
        seed_start=ordered[0].seed_start,
        seed_end_inclusive=ordered[-1].seed_end_inclusive,
        max_completed_turns=ordered[0].max_completed_turns,
        natural_endings=sum(report.natural_endings for report in ordered),
        engineering_truncations=sum(report.engineering_truncations for report in ordered),
        exceptions=sum(report.exceptions for report in ordered),
        player_0_wins=sum(report.player_0_wins for report in ordered),
        player_1_wins=sum(report.player_1_wins for report in ordered),
        draws=sum(report.draws for report in ordered),
        mean_completed_player_turns=sum(turn_samples) / total,
        median_completed_player_turns=float(median(turn_samples)),
        max_completed_player_turns_observed=max(turn_samples),
        mean_policy_decisions=sum(decision_samples) / total,
        phase_entries=dict(phase_entries),
        main_actions=dict(main_actions),
        noble_claims=sum(report.noble_claims for report in ordered),
        final_round_triggers=sum(report.final_round_triggers for report in ordered),
        forced_skips=sum(report.forced_skips for report in ordered),
        forced_skip_draws=sum(report.forced_skip_draws for report in ordered),
        terminal_branches=dict(terminal_branches),
        no_legal_action_failures=sum(report.no_legal_action_failures for report in ordered),
        illegal_action_failures=sum(report.illegal_action_failures for report in ordered),
        invariant_failures=sum(report.invariant_failures for report in ordered),
        replay_samples_checked=sum(report.replay_samples_checked for report in ordered),
        replay_mismatches=sum(report.replay_mismatches for report in ordered),
        save_restore_samples_checked=sum(report.save_restore_samples_checked for report in ordered),
        save_restore_mismatches=sum(report.save_restore_mismatches for report in ordered),
        replay_coverage=tuple(sorted(replay_coverage)),
        missing_replay_coverage=tuple(sorted(REQUIRED_REPLAY_COVERAGE - replay_coverage)),
        save_restore_coverage=tuple(sorted(restore_coverage)),
        missing_save_restore_coverage=tuple(
            sorted(REQUIRED_SAVE_RESTORE_COVERAGE - restore_coverage)
        ),
        failure_scenes=tuple(path for report in ordered for path in report.failure_scenes),
        _completed_turn_samples=turn_samples,
        _decision_samples=decision_samples,
    )


def run_parallel_stress_validation(
    game_count: int,
    *,
    workers: int,
    seed_start: int = 0,
    max_completed_turns: int = 400,
    replay_sample_count: int = 64,
    work_dir: Path | None = None,
) -> ValidationReport:
    """Run deterministic, non-overlapping validation shards in processes."""

    if workers <= 0:
        raise ValueError("workers must be positive")
    if workers == 1:
        return run_stress_validation(
            game_count,
            seed_start=seed_start,
            max_completed_turns=max_completed_turns,
            replay_sample_count=replay_sample_count,
            work_dir=work_dir,
        )
    workers = min(workers, game_count)
    work_dir = work_dir or Path.cwd() / "engine_validation"
    quotient, remainder = divmod(game_count, workers)
    arguments: list[tuple[int, int, int, int, Path]] = []
    next_seed = seed_start
    for shard in range(workers):
        count = quotient + (1 if shard < remainder else 0)
        samples = max(1, replay_sample_count // workers)
        arguments.append(
            (
                count,
                next_seed,
                max_completed_turns,
                samples,
                work_dir / f"shard-{shard:02d}",
            )
        )
        next_seed += count
    with ProcessPoolExecutor(max_workers=workers) as executor:
        reports = tuple(executor.map(_run_stress_shard, arguments))
    return _merge_reports(reports)


def _write_report(path: Path, report: ValidationReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deterministic Splendor engine validation")
    parser.add_argument("--games", type=int, default=10_000)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--max-completed-turns", type=int, default=400)
    parser.add_argument("--replay-samples", type=int, default=64)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--work-dir", type=Path, default=Path("engine_validation"))
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("engine_validation/work-package-2d-stats.json"),
    )
    args = parser.parse_args()
    report = run_parallel_stress_validation(
        args.games,
        workers=args.workers,
        seed_start=args.seed_start,
        max_completed_turns=args.max_completed_turns,
        replay_sample_count=args.replay_samples,
        work_dir=args.work_dir,
    )
    _write_report(args.report, report)
    print(json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
