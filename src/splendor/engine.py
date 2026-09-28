from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from ._state import DeckState, GameState, MarketState, PlayerState
from .actions import Action
from .catalog import Catalog, load_default_catalog
from .contracts import DecisionRequest, Observation
from .model import DecisionPhase, Tier, TokenVector
from .observation import build_observation
from .replay import (
    ActionEvent,
    ForcedSkipEvent,
    InvariantGuard,
    REPLAY_SCHEMA_VERSION,
    RULESET_ID,
    ReplayRecord,
    load_game,
    new_replay,
    save_game,
    state_digest,
)
from .rng import DeterministicRng, RNG_ALGORITHM
from .rules import apply_forced_skip, apply_legal_action, legal_actions
from .result import TerminalResult
from .state_machine import assert_phase_transition


class EngineError(RuntimeError):
    code = "engine_error"


class IllegalActionError(EngineError):
    code = "illegal_action"


class UnsupportedRuleError(EngineError):
    code = "unsupported_rule"


class ReplayMismatchError(EngineError):
    code = "replay_mismatch"


class GameEngine:
    """Complete two-player base-game engine with private authoritative state."""

    def __init__(
        self,
        state: GameState,
        catalog: Catalog,
        replay: ReplayRecord | None,
        failure_dir: Path,
        *,
        mode: str = "checked",
    ) -> None:
        if mode not in ("checked", "fast"):
            raise ValueError("engine mode must be checked or fast")
        if mode == "checked" and replay is None:
            raise ValueError("checked engine requires a replay record")
        self.mode = mode
        self.__state = state
        self.catalog = catalog
        self.__replay = replay
        self.__guard = InvariantGuard(catalog, replay, failure_dir) if mode == "checked" else None
        if self.__guard is not None:
            self.__guard.check(state)
        self.__state_digest = state_digest(state) if mode == "checked" else None
        self.__legal_actions_cache: tuple[Action, ...] | None = None
        self.__state_version = 0
        self.forced_skip_count = 0
        self.last_attempted_action: Action | None = None

    @classmethod
    def new(
        cls,
        seed: int,
        first_player: int | None = None,
        *,
        catalog: Catalog | None = None,
        failure_dir: Path | None = None,
        mode: str = "checked",
    ) -> GameEngine:
        catalog = catalog or load_default_catalog()
        rng = DeterministicRng(seed)

        # Always consume this draw so explicit and random-first-player games
        # with the same seed have identical shuffle streams and can replay from
        # the actual first player recorded in the header.
        sampled_first = rng.randbelow(2)
        if first_player is None:
            first_player = sampled_first
        elif first_player not in (0, 1):
            raise ValueError("first_player must be 0, 1, or None")

        decks: dict[Tier, tuple[str, ...]] = {}
        markets: dict[Tier, tuple[str, ...]] = {}
        for tier in Tier:
            cards = [card.card_id for card in catalog.cards if card.tier is tier]
            rng.shuffle(cards)
            face_up = tuple(cards.pop() for _ in range(catalog.constants.face_up_per_tier))
            decks[tier] = tuple(cards)
            markets[tier] = face_up

        nobles = [noble.noble_id for noble in catalog.nobles]
        rng.shuffle(nobles)
        selected_nobles = tuple(nobles.pop() for _ in range(catalog.constants.nobles_in_play))
        state = GameState(
            supply=catalog.constants.initial_tokens,
            decks=DeckState(decks[Tier.ONE], decks[Tier.TWO], decks[Tier.THREE]),
            market=MarketState(markets[Tier.ONE], markets[Tier.TWO], markets[Tier.THREE]),
            players=(PlayerState(TokenVector()), PlayerState(TokenVector())),
            nobles_in_play=selected_nobles,
            nobles_out_of_game=tuple(nobles),
            first_player=first_player,
            current_player=first_player,
            completed_turns=(0, 0),
            consecutive_forced_skips=0,
            phase=DecisionPhase.MAIN_ACTION,
            pending=None,
            final_round_triggered_by=None,
            result=None,
            rng=rng.snapshot(),
        )
        replay = new_replay(catalog, seed, first_player) if mode == "checked" else None
        return cls(state, catalog, replay, failure_dir or Path.cwd() / "failures", mode=mode)

    @classmethod
    def restore(
        cls,
        path: Path,
        *,
        catalog: Catalog | None = None,
        failure_dir: Path | None = None,
    ) -> GameEngine:
        catalog = catalog or load_default_catalog()
        state, replay = load_game(path, catalog)
        if state.first_player != replay.header.first_player:
            raise ValueError("save state and replay disagree about first player")
        return cls(state, catalog, replay, failure_dir or path.parent / "failures")

    @property
    def replay_record(self) -> ReplayRecord:
        if self.__replay is None:
            raise RuntimeError("fast engine does not record a full replay")
        return self.__replay

    @property
    def current_player(self) -> int:
        """Seat whose policy must answer the next decision request."""

        return self.__state.current_player

    @property
    def state_version(self) -> int:
        return self.__state_version

    @property
    def search_state_key(self) -> GameState:
        """Hashable content identity for search-owned simulator caches.

        Only search code that owns this engine may use the authoritative state
        as a cache key (determinized simulation engines are never handed to a
        live policy).  Live matches and UI consumers keep using
        :meth:`observation_for`.
        """

        return self.__state

    @property
    def current_state_digest(self) -> str:
        """Digest of the immutable current state, computed once per transition."""

        return self.__state_digest if self.__state_digest is not None else state_digest(self.__state)

    def fork(self, *, failure_dir: Path | None = None) -> GameEngine:
        """Fork an already-validated immutable state for controlled simulation.

        The parent state, replay, digest, and legal candidates are immutable and
        safe to share.  The fork skips only the redundant parent-state invariant
        scan; every state produced by the fork is validated normally.
        """

        forked = object.__new__(GameEngine)
        forked.mode = self.mode
        forked.__state = self.__state
        forked.catalog = self.catalog
        forked.__replay = self.__replay
        forked.__guard = (
            InvariantGuard(self.catalog, self.__replay, failure_dir or self.__guard.failure_dir)
            if self.__guard is not None else None
        )
        forked.__state_digest = self.__state_digest
        forked.__legal_actions_cache = self.__legal_actions_cache
        forked.__state_version = self.__state_version
        forked.forced_skip_count = self.forced_skip_count
        forked.last_attempted_action = None
        return forked

    def _fail(self, error: Exception, action: Action | None = None) -> None:
        if self.__guard is not None:
            self.__guard.fail(self.__state, error, action)
        raise error

    def _request(self, actions: tuple[Action, ...]) -> DecisionRequest:
        observation = build_observation(self.__state, self.__state.current_player, self.catalog)
        if self.mode == "checked":
            return DecisionRequest(observation, actions)
        # The rule core produced both values. Public callers still get validation.
        request = object.__new__(DecisionRequest)
        object.__setattr__(request, "observation", observation)
        object.__setattr__(request, "legal_actions", actions)
        return request

    def observation_for(self, observer: int) -> Observation:
        """Return the information-safe view for a seat, including at terminal.

        This is intended for non-policy consumers such as a local human UI.
        Policies continue to receive only :class:`DecisionRequest`.
        """

        return build_observation(self.__state, observer, self.catalog)

    def decision(self) -> DecisionRequest | TerminalResult:
        self._advance_automatic_transitions()
        if self.__state.phase is DecisionPhase.TERMINAL:
            if self.__state.result is None:
                self._fail(EngineError("terminal state has no result"))
            return self.__state.result
        actions = self._legal_actions()
        if self.__state.phase not in (
            DecisionPhase.MAIN_ACTION, DecisionPhase.PAYMENT,
            DecisionPhase.RETURN_TOKENS, DecisionPhase.CHOOSE_NOBLE,
        ):
            self._fail(UnsupportedRuleError(f"phase {self.__state.phase.value} has no policy decision"))
        if not actions:
            self._fail(EngineError("a non-terminal state has no legal action"))
        return self._request(actions)

    def apply(self, action: Action) -> DecisionRequest | TerminalResult:
        self._commit_action(action, automatic=False, trusted=False)
        return self._after_action()

    def apply_candidate(self, index: int, *, state_version: int) -> DecisionRequest | TerminalResult:
        """Execute an index from this engine's current authoritative candidates."""
        candidates = self.__legal_actions_cache
        if (state_version != self.__state_version or candidates is None
                or not 0 <= index < len(candidates)):
            raise IllegalActionError("candidate index is stale or out of range")
        self._commit_action(candidates[index], automatic=False, trusted=True)
        return self._after_action()

    def apply_candidate_folded(
        self, index: int, *, max_transitions: int,
        action_trace: list[Action] | None = None,
        state_version: int,
    ) -> tuple[DecisionRequest | TerminalResult, int, int]:
        """Apply a cached candidate and fold forced choices without intermediate views."""
        if max_transitions <= 0:
            raise ValueError("max_transitions must be positive")
        candidates = self.__legal_actions_cache
        if (state_version != self.__state_version or candidates is None
                or not 0 <= index < len(candidates)):
            raise IllegalActionError("candidate index is stale or out of range")
        self.last_attempted_action = candidates[index]
        self._commit_action(candidates[index], automatic=False, trusted=True)
        if action_trace is not None:
            action_trace.append(candidates[index])
        self.last_attempted_action = None
        transitions = 1
        folded = 0
        while True:
            self._advance_automatic_transitions()
            if self.__state.phase is DecisionPhase.TERMINAL:
                if self.__state.result is None:
                    self._fail(EngineError("terminal state has no result"))
                return self.__state.result, transitions, folded
            actions = self._legal_actions()
            if not actions:
                self._fail(EngineError("a non-terminal decision phase has no legal action"))
            if len(actions) != 1 or transitions >= max_transitions:
                return self._request(actions), transitions, folded
            self.last_attempted_action = actions[0]
            self._commit_action(actions[0], automatic=False, trusted=True)
            if action_trace is not None:
                action_trace.append(actions[0])
            self.last_attempted_action = None
            transitions += 1
            folded += 1

    def _after_action(self) -> DecisionRequest | TerminalResult:
        self._advance_automatic_transitions()
        if self.__state.phase is DecisionPhase.TERMINAL:
            if self.__state.result is None:
                self._fail(EngineError("terminal state has no result"))
            return self.__state.result
        actions = self._legal_actions()
        if not actions:
            self._fail(EngineError("a non-terminal decision phase has no legal action"))
        return self._request(actions)

    def _advance_automatic_transitions(self) -> None:
        while True:
            if self.__state.phase is DecisionPhase.CHOOSE_NOBLE:
                candidates = self._legal_actions()
                if len(candidates) == 1:
                    self._commit_action(candidates[0], automatic=True, trusted=True)
                    continue
                return
            if self.__state.phase is DecisionPhase.MAIN_ACTION:
                if self._legal_actions():
                    return
                self._commit_forced_skip()
                continue
            return

    def _commit_action(self, action: Action, *, automatic: bool, trusted: bool = False) -> None:
        if not trusted and action not in self._legal_actions():
            self._fail(IllegalActionError(f"action is not legal in {self.__state.phase.value}: {action!r}"), action)

        before = self.__state
        before_digest = self.__state_digest
        try:
            after = apply_legal_action(before, action, self.catalog)
            assert_phase_transition(before.phase, after.phase)
        except Exception as error:
            self._fail(error, action)
        if self.mode == "fast":
            self.__state = after
            self.__state_version += 1
            self.__legal_actions_cache = None
            return
        assert self.__guard is not None and self.__replay is not None
        self.__guard.check(after, action)
        after_digest = state_digest(after)

        event = ActionEvent(
            sequence=len(self.__replay.events),
            actor=before.current_player,
            phase=before.phase,
            action=action,
            automatic=automatic,
            state_digest_before=before_digest,
            state_digest_after=after_digest,
        )
        self.__state = after
        self.__state_version += 1
        self.__state_digest = after_digest
        self.__legal_actions_cache = None
        self.__replay = replace(
            self.__replay,
            events=self.__replay.events + (event,),
            result=after.result,
        )
        self.__guard.replay = self.__replay

    def _commit_forced_skip(self) -> None:
        before = self.__state
        before_digest = self.__state_digest
        try:
            after = apply_forced_skip(before, self.catalog)
            assert_phase_transition(before.phase, after.phase)
        except Exception as error:
            self._fail(error)
        self.forced_skip_count += 1
        if self.mode == "fast":
            self.__state = after
            self.__state_version += 1
            self.__legal_actions_cache = None
            return
        assert self.__guard is not None and self.__replay is not None
        self.__guard.check(after)
        after_digest = state_digest(after)
        event = ForcedSkipEvent(
            sequence=len(self.__replay.events),
            actor=before.current_player,
            state_digest_before=before_digest,
            state_digest_after=after_digest,
        )
        self.__state = after
        self.__state_version += 1
        self.__state_digest = after_digest
        self.__legal_actions_cache = None
        self.__replay = replace(
            self.__replay,
            events=self.__replay.events + (event,),
            result=after.result,
        )
        self.__guard.replay = self.__replay

    def save(self, path: Path) -> None:
        if self.__guard is None or self.__replay is None:
            raise RuntimeError("fast engine does not record a full replay")
        self.__guard.check(self.__state)
        save_game(path, self.__state, self.__replay)

    def _debug_state(self) -> GameState:
        """Tests and full debug replay only; never pass this value to a policy."""

        return self.__state

    def _legal_actions(self) -> tuple[Action, ...]:
        if self.__legal_actions_cache is None:
            self.__legal_actions_cache = legal_actions(self.__state, self.catalog)
        return self.__legal_actions_cache


def replay_game(
    record: ReplayRecord,
    *,
    catalog: Catalog | None = None,
    failure_dir: Path | None = None,
) -> GameEngine:
    catalog = catalog or load_default_catalog()
    header = record.header
    if header.schema_version != REPLAY_SCHEMA_VERSION:
        raise ValueError(f"unsupported replay schema: {header.schema_version}")
    if header.ruleset_id != RULESET_ID:
        raise ValueError(f"wrong ruleset: {header.ruleset_id}")
    if header.data_fingerprint != catalog.data_fingerprint:
        raise ValueError("replay uses different normative data")
    if header.rng_algorithm != RNG_ALGORITHM:
        raise ValueError(f"unsupported replay RNG: {header.rng_algorithm}")
    engine = GameEngine.new(
        record.header.game_seed,
        record.header.first_player,
        catalog=catalog,
        failure_dir=failure_dir,
    )
    cursor = 0
    while cursor < len(record.events):
        expected = record.events[cursor]
        if expected.action is None:
            before_count = len(engine.replay_record.events)
            engine.decision()
            produced = engine.replay_record.events[before_count:]
            expected_group = record.events[cursor : cursor + len(produced)]
            if not produced or produced != expected_group:
                engine._fail(ReplayMismatchError(f"event {expected.sequence} diverged"))
            cursor += len(produced)
            continue
        if expected.automatic:
            engine._fail(
                ReplayMismatchError("automatic event has no preceding player decision"),
                expected.action,
            )
        before_count = len(engine.replay_record.events)
        engine.apply(expected.action)
        produced = engine.replay_record.events[before_count:]
        expected_group = record.events[cursor : cursor + len(produced)]
        if produced != expected_group:
            engine._fail(
                ReplayMismatchError(f"event {expected.sequence} diverged"),
                expected.action,
            )
        cursor += len(produced)
    if engine.replay_record.result != record.result:
        engine._fail(ReplayMismatchError("terminal result diverged"))
    return engine
