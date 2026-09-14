from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Sequence

from .actions import (
    Action,
    BuyMarket,
    BuyReserved,
    ClaimNoble,
    ReserveDeck,
    ReserveMarket,
    TakeDistinct,
    TakePair,
)
from .catalog import Catalog, load_default_catalog
from .contracts import DecisionRequest, Policy
from .engine import GameEngine
from .model import DecisionPhase
from .random_policy import RandomPolicy
from .replay import ForcedSkipEvent, ReplayRecord, state_digest, to_primitive
from .result import TerminalReason, TerminalResult
from .rng import DeterministicRng
from .rules import legal_actions


class RunStatus(str, Enum):
    NATURAL_TERMINAL = "natural_terminal"
    ENGINEERING_TRUNCATION = "engineering_truncation"
    ERROR = "error"


_MAIN_ACTION_NAMES: tuple[tuple[type[object], str], ...] = (
    (TakeDistinct, "TakeDistinct"),
    (TakePair, "TakePair"),
    (ReserveMarket, "ReserveMarket"),
    (ReserveDeck, "ReserveDeck"),
    (BuyMarket, "BuyMarket"),
    (BuyReserved, "BuyReserved"),
)


@dataclass(frozen=True, slots=True)
class GameRun:
    seed: int
    policy_seeds: tuple[int, int]
    first_player: int
    status: RunStatus
    replay: ReplayRecord
    result: TerminalResult | None
    completed_turns: tuple[int, int]
    decision_count: int
    phase_entries: dict[str, int]
    main_actions: dict[str, int]
    noble_claims: int
    final_round_triggers: int
    forced_skips: int
    forced_skip_draw: bool
    coverage_tags: frozenset[str]
    checkpoint_indices: dict[str, int]
    final_state_digest: str
    failure_scene: Path | None = None
    error_type: str | None = None
    error_detail: str | None = None


def _policy_seed(game_seed: int, player_id: int) -> int:
    # Fixed arithmetic, rather than process-randomized hash(), keeps the run
    # reproducible on every Python process and platform.
    return game_seed * 2 + 0xD1B54A32D192ED03 + player_id


def _action_name(action: Action) -> str | None:
    for action_type, name in _MAIN_ACTION_NAMES:
        if isinstance(action, action_type):
            return name
    return None


def _checkpoint_tags(engine: GameEngine, request: DecisionRequest) -> tuple[str, ...]:
    tags = [request.observation.phase.value]
    if engine._debug_state().consecutive_forced_skips:
        tags.append("forced_skip_pending")
    if engine._debug_state().final_round_triggered_by is not None:
        tags.append("final_round_pending")
    return tuple(tags)


def _write_failure_scene(
    directory: Path,
    *,
    reason: str,
    engine: GameEngine,
    policy_seeds: tuple[int, int],
    policy_snapshots: tuple[object, object] | None,
    error: Exception | None = None,
) -> Path:
    state = engine._debug_state()
    base = directory / (
        f"game-{engine.replay_record.header.game_seed}-"
        f"event-{len(engine.replay_record.events):06d}-{reason}.json"
    )
    path = base
    suffix = 1
    while path.exists():
        path = base.with_name(f"{base.stem}-{suffix}{base.suffix}")
        suffix += 1
    report = {
        "reason": reason,
        "error": None
        if error is None
        else {"type": type(error).__name__, "detail": str(error)},
        "seed": engine.replay_record.header.game_seed,
        "first_player": engine.replay_record.header.first_player,
        "phase": state.phase.value,
        # Recompute from the current authoritative state through the one rule
        # source, including failures raised while applying an action.
        "legal_actions": to_primitive(legal_actions(state, engine.catalog)),
        "engine_rng": to_primitive(state.rng),
        "policy_seeds": list(policy_seeds),
        "policy_rng": to_primitive(policy_snapshots),
        "state_digest": state_digest(state),
        "state": to_primitive(state),
        "replay_prefix": to_primitive(engine.replay_record),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def run_random_game(
    seed: int,
    *,
    first_player: int | None = None,
    max_completed_turns: int = 400,
    catalog: Catalog | None = None,
    failure_dir: Path | None = None,
    policies: Sequence[Policy] | None = None,
    raise_on_error: bool = True,
) -> GameRun:
    """Run one complete game or return an explicitly classified truncation.

    ``max_completed_turns`` is an engineering watchdog only.  Reaching it does
    not mutate the engine and never creates a ``TerminalResult``.
    """

    if max_completed_turns <= 0:
        raise ValueError("max_completed_turns must be positive")
    catalog = catalog or load_default_catalog()
    failure_dir = failure_dir or Path.cwd() / "validation_failures"
    policy_seeds = (_policy_seed(seed, 0), _policy_seed(seed, 1))
    if policies is None:
        policy_objects: tuple[Policy, Policy] = (
            RandomPolicy(policy_seeds[0]),
            RandomPolicy(policy_seeds[1]),
        )
    else:
        if len(policies) != 2:
            raise ValueError("exactly two policies are required")
        policy_objects = (policies[0], policies[1])

    engine = GameEngine.new(
        seed,
        first_player=first_player,
        catalog=catalog,
        failure_dir=failure_dir,
    )
    phase_entries: Counter[str] = Counter()
    main_actions: Counter[str] = Counter()
    coverage: set[str] = set()
    checkpoints: dict[str, int] = {}
    decision_count = 0
    noble_claims = 0
    trigger_count = 0
    forced_skips = 0
    checkpoint_rng = DeterministicRng(_policy_seed(seed, 2))
    checkpoint_seen: Counter[str] = Counter()

    def snapshots() -> tuple[object, object] | None:
        if all(isinstance(policy, RandomPolicy) for policy in policy_objects):
            return (
                policy_objects[0].rng_snapshot(),  # type: ignore[union-attr]
                policy_objects[1].rng_snapshot(),  # type: ignore[union-attr]
            )
        return None

    def result_for(
        status: RunStatus,
        terminal: TerminalResult | None,
        *,
        scene: Path | None = None,
        error: Exception | None = None,
    ) -> GameRun:
        state = engine._debug_state()
        return GameRun(
            seed=seed,
            policy_seeds=policy_seeds,
            first_player=engine.replay_record.header.first_player,
            status=status,
            replay=engine.replay_record,
            result=terminal,
            completed_turns=state.completed_turns,
            decision_count=decision_count,
            phase_entries=dict(phase_entries),
            main_actions=dict(main_actions),
            noble_claims=noble_claims,
            final_round_triggers=trigger_count,
            forced_skips=forced_skips,
            forced_skip_draw=(
                terminal is not None
                and terminal.reason is TerminalReason.CONSECUTIVE_FORCED_SKIPS
            ),
            coverage_tags=frozenset(coverage),
            checkpoint_indices=dict(checkpoints),
            final_state_digest=state_digest(state),
            failure_scene=scene,
            error_type=None if error is None else type(error).__name__,
            error_detail=None if error is None else str(error),
        )

    def record_new_events(before_count: int) -> None:
        nonlocal noble_claims, forced_skips
        for event in engine.replay_record.events[before_count:]:
            phase_entries[event.phase.value] += 1
            coverage.add(event.phase.value)
            if isinstance(event, ForcedSkipEvent):
                forced_skips += 1
                coverage.add("forced_skip")
                continue
            name = _action_name(event.action)
            if name is not None:
                main_actions[name] += 1
                coverage.add(name)
            if isinstance(event.action, ClaimNoble):
                noble_claims += 1

    try:
        while True:
            # GameEngine checks the initial state and every committed
            # transition.  Do not repeat the same full-state scan here.
            before_decision_events = len(engine.replay_record.events)
            decision_or_result = engine.decision()
            record_new_events(before_decision_events)
            if isinstance(decision_or_result, TerminalResult):
                coverage.add("terminal")
                return result_for(RunStatus.NATURAL_TERMINAL, decision_or_result)
            request = decision_or_result
            completed = sum(engine._debug_state().completed_turns)
            if completed >= max_completed_turns:
                scene = _write_failure_scene(
                    failure_dir,
                    reason="engineering-truncation",
                    engine=engine,
                    policy_seeds=policy_seeds,
                    policy_snapshots=snapshots(),
                )
                return result_for(RunStatus.ENGINEERING_TRUNCATION, None, scene=scene)

            for tag in _checkpoint_tags(engine, request):
                checkpoint_seen[tag] += 1
                if checkpoint_rng.randbelow(checkpoint_seen[tag]) == 0:
                    checkpoints[tag] = decision_count
            phase = request.observation.phase
            coverage.add(phase.value)
            if phase is DecisionPhase.PAYMENT and len(request.legal_actions) > 1:
                coverage.add("multiple_payment_options")

            action = policy_objects[engine.current_player].choose(request)
            decision_count += 1
            before_trigger = engine._debug_state().final_round_triggered_by
            before_events = len(engine.replay_record.events)
            outcome = engine.apply(action)
            after_trigger = engine._debug_state().final_round_triggered_by
            if before_trigger is None and after_trigger is not None:
                trigger_count += 1

            record_new_events(before_events)

            if isinstance(outcome, TerminalResult):
                coverage.update(("final_round", "terminal"))
                return result_for(RunStatus.NATURAL_TERMINAL, outcome)
    except Exception as error:
        scene = _write_failure_scene(
            failure_dir,
            reason="runner-error",
            engine=engine,
            policy_seeds=policy_seeds,
            policy_snapshots=snapshots(),
            error=error,
        )
        if raise_on_error:
            error.add_note(f"runner failure snapshot: {scene}")
            raise
        return result_for(RunStatus.ERROR, None, scene=scene, error=error)


def terminal_branch(result: TerminalResult) -> str:
    if result.reason is TerminalReason.CONSECUTIVE_FORCED_SKIPS:
        return "forced_skip_draw"
    if result.scores[0] != result.scores[1]:
        return "higher_score"
    if result.purchased_card_counts[0] != result.purchased_card_counts[1]:
        return "fewer_development_cards"
    return "draw"
