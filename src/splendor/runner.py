from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, fields
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
from .rules import legal_actions, player_bonuses, player_score


class RunStatus(str, Enum):
    NATURAL_TERMINAL = "natural_terminal"
    POLICY_CYCLE_DRAW = "policy_cycle_draw"
    ENGINEERING_TRUNCATION = "engineering_truncation"
    ERROR = "error"


class MatchOutcome(str, Enum):
    A_WIN = "a_win"
    DRAW = "draw"
    A_LOSS = "a_loss"


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


@dataclass(frozen=True, slots=True)
class MatchResult:
    """One formal game with policy A fixed to seat 0 and policy B to seat 1."""

    pair_id: int | None
    game_in_pair: int | None
    game_seed: int
    policy_names: tuple[str, str]
    policy_seeds: tuple[int, int]
    first_player: int
    status: RunStatus
    normal_end: bool
    terminal_result: TerminalResult | None
    outcome_for_a: MatchOutcome | None
    winner_id: int | None
    draw: bool
    scores: tuple[int, int] | None
    purchased_card_counts: tuple[int, int] | None
    completed_turns: tuple[int, int]
    completed_player_turns: int
    decision_count: int
    forced_skips: int
    replay: ReplayRecord
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


def canonical_state_signature(state: object, catalog: Catalog) -> str:
    """Canonical future-relevant state used only for deterministic evaluation.

    ``completed_turns`` is deliberately excluded by the 5C evaluation
    protocol.  Replay positions and logging counters do not live in
    ``GameState``.  RNG ``draws`` is likewise only an audit counter; the RNG
    algorithm and actual stream state remain part of the signature.

    Iterating the dataclass fields and excluding one named field makes new
    authoritative state fail closed: a future field is included unless the
    protocol explicitly classifies it as record-only.
    """

    from ._state import GameState

    if not isinstance(state, GameState):
        raise TypeError("canonical state signatures require a GameState")
    authoritative: dict[str, object] = {}
    for field in fields(state):
        if field.name == "completed_turns":
            continue
        value = getattr(state, field.name)
        if field.name == "rng":
            authoritative[field.name] = {
                "algorithm": value.algorithm,
                "state": value.state,
            }
        else:
            authoritative[field.name] = to_primitive(value)
    authoritative["catalog_data_fingerprint"] = catalog.data_fingerprint
    authoritative["derived_players"] = [
        {
            "score": player_score(player, catalog),
            "bonuses": to_primitive(player_bonuses(player, catalog)),
            "purchased_card_count": len(player.purchased_card_ids),
        }
        for player in state.players
    ]
    return json.dumps(
        authoritative,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


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
    context: dict[str, object] | None = None,
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
        "context": context,
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
    policy_seeds: tuple[int, int] | None = None,
    adjudicate_deterministic_policy_cycles: bool = False,
    raise_on_error: bool = True,
) -> GameRun:
    """Run one complete game or return an explicitly classified truncation.

    ``max_completed_turns`` is an engineering watchdog only.  Reaching it does
    not mutate the engine and never creates a ``TerminalResult``.

    Deterministic policy-cycle adjudication is opt-in because the proof only
    holds when both policies are fixed, stateless functions of their
    ``DecisionRequest``.  It is an evaluation result, never an engine result.
    """

    if max_completed_turns <= 0:
        raise ValueError("max_completed_turns must be positive")
    catalog = catalog or load_default_catalog()
    failure_dir = failure_dir or Path.cwd() / "validation_failures"
    if policy_seeds is None:
        policy_seeds = (_policy_seed(seed, 0), _policy_seed(seed, 1))
    elif len(policy_seeds) != 2 or not all(type(value) is int for value in policy_seeds):
        raise ValueError("policy_seeds must contain exactly two integers")
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
    seen_signatures: tuple[dict[str, int], dict[str, int]] = ({}, {})
    repeated_players: set[int] = set()

    def snapshots() -> tuple[object, object]:
        return tuple(
            policy.rng_snapshot() if isinstance(policy, RandomPolicy) else None
            for policy in policy_objects
        )  # type: ignore[return-value]

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
            if adjudicate_deterministic_policy_cycles:
                state = engine._debug_state()
                actor = state.current_player
                signature = canonical_state_signature(state, catalog)
                first_decision = seen_signatures[actor].get(signature)
                if first_decision is not None:
                    repeated_players.add(actor)
                else:
                    seen_signatures[actor][signature] = decision_count
                if len(repeated_players) == 2:
                    signature_digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()
                    scene = _write_failure_scene(
                        failure_dir,
                        reason="policy-cycle-draw",
                        engine=engine,
                        policy_seeds=policy_seeds,
                        policy_snapshots=snapshots(),
                        context={
                            "canonical_state_signature_sha256": signature_digest,
                            "first_seen_decision": first_decision,
                            "repeated_at_decision": decision_count,
                            "repeated_players": sorted(repeated_players),
                            "criterion": (
                                "each player has encountered a canonical state "
                                "signature for at least the second time"
                            ),
                            "excluded_fields": [
                                "completed_turns",
                                "rng.draws",
                                "replay/logging counters",
                            ],
                        },
                    )
                    return result_for(RunStatus.POLICY_CYCLE_DRAW, None, scene=scene)
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


def run_match(
    policy_a: Policy,
    policy_b: Policy,
    *,
    game_seed: int,
    policy_seeds: tuple[int, int],
    first_player: int,
    policy_names: tuple[str, str] = ("A", "B"),
    pair_id: int | None = None,
    game_in_pair: int | None = None,
    max_completed_turns: int = 400,
    catalog: Catalog | None = None,
    failure_dir: Path | None = None,
    adjudicate_deterministic_policy_cycles: bool = False,
    raise_on_error: bool = True,
) -> MatchResult:
    """Run one game through the same engine path used by every policy."""

    if first_player not in (0, 1):
        raise ValueError("first_player must be 0 or 1")
    if len(policy_names) != 2:
        raise ValueError("policy_names must contain exactly two names")
    run = run_random_game(
        game_seed,
        first_player=first_player,
        max_completed_turns=max_completed_turns,
        catalog=catalog,
        failure_dir=failure_dir,
        policies=(policy_a, policy_b),
        policy_seeds=policy_seeds,
        adjudicate_deterministic_policy_cycles=adjudicate_deterministic_policy_cycles,
        raise_on_error=raise_on_error,
    )
    terminal = run.result
    outcome = None
    winner_id = None
    draw = False
    if terminal is not None:
        if len(terminal.winner_ids) == 2:
            outcome = MatchOutcome.DRAW
            draw = True
        else:
            winner_id = terminal.winner_ids[0]
            outcome = MatchOutcome.A_WIN if winner_id == 0 else MatchOutcome.A_LOSS
    elif run.status is RunStatus.POLICY_CYCLE_DRAW:
        outcome = MatchOutcome.DRAW
        draw = True
    return MatchResult(
        pair_id=pair_id,
        game_in_pair=game_in_pair,
        game_seed=game_seed,
        policy_names=policy_names,
        policy_seeds=policy_seeds,
        first_player=run.first_player,
        status=run.status,
        normal_end=run.status is RunStatus.NATURAL_TERMINAL,
        terminal_result=terminal,
        outcome_for_a=outcome,
        winner_id=winner_id,
        draw=draw,
        scores=None if terminal is None else terminal.scores,
        purchased_card_counts=None if terminal is None else terminal.purchased_card_counts,
        completed_turns=run.completed_turns,
        completed_player_turns=sum(run.completed_turns),
        decision_count=run.decision_count,
        forced_skips=run.forced_skips,
        replay=run.replay,
        final_state_digest=run.final_state_digest,
        failure_scene=run.failure_scene,
        error_type=run.error_type,
        error_detail=run.error_detail,
    )
