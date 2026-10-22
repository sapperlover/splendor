"""Python control and model adapter for the C++ independent-world PUCT engine."""

from __future__ import annotations

import re
import math
import time
from collections import deque
from functools import cache
from dataclasses import dataclass
from itertools import count
from typing import Any, Mapping, Sequence

from .catalog import Catalog, load_default_catalog
from .contracts import DecisionRequest
from .features import EncodedDecision, EncodedState
from .mcts_policy import MCTSConfig, MCTSSearchResult, MCTSStats
from .search_policy import _request_public_word, _seed_from_public_word


def _extension():
    try:
        from . import _native_search
    except ImportError as error:
        raise RuntimeError(
            "Native search extension is unavailable. Install it with "
            "`conda run -n splendor python -m pip install -e . --no-deps`."
        ) from error
    return _native_search


@cache
def _default_catalog() -> Catalog:
    return load_default_catalog()


_CATALOGS: dict[str, object] = {}

def _native_catalog(catalog: Catalog):
    try:
        return _CATALOGS[catalog.data_fingerprint]
    except KeyError:
        native = _extension().Catalog(catalog)
        _CATALOGS[catalog.data_fingerprint] = native
        return native


@dataclass(frozen=True, slots=True)
class NativePendingEvaluation:
    request_id: str
    kind: str
    encoded: EncodedDecision | EncodedState
    world_index: int
    model_version: str
    request: None = None


_JOB_IDS = count()


@dataclass(frozen=True, slots=True)
class ChanceWideningConfig:
    """Bound each physical draw's admitted support by ceil(k * visits**alpha)."""

    k: float = 1.0
    alpha: float = 0.5

    def __post_init__(self) -> None:
        for name in ("k", "alpha"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"chance widening {name} must be finite")
        if self.k <= 0 or not 0 < self.alpha < 1:
            raise ValueError("chance widening requires k>0 and 0<alpha<1")


class NativeSearchJob:
    def __init__(
        self, policy: NativePUCTSearchPolicy, request: DecisionRequest, *,
        add_root_noise: bool, temperature: float, sampling_seed: int | None,
        root_encoded: EncodedDecision | None = None,
    ) -> None:
        self.policy = policy
        self.request = request
        self.started = time.perf_counter()
        self.job_id = next(_JOB_IDS)
        self.sequence = 0
        self.sampling_seed = policy.config.seed if sampling_seed is None else sampling_seed
        self.public_word = _request_public_word(request) if len(request.legal_actions) > 1 else 0
        widening = policy.chance_widening
        kwargs = {} if widening is None else {"chance_pw_k": widening.k, "chance_pw_alpha": widening.alpha}
        if policy.root_only:
            kwargs["root_only"] = True
        self._core = _extension().SearchJob(
            policy.native_catalog, request, policy.config, self.public_word,
            self.sampling_seed & ((1 << 64) - 1), add_root_noise, temperature, policy.stochastic,
            **kwargs,
        )
        self._pending: dict[str, NativePendingEvaluation] | None = None
        self._pending_dirty = True
        self._finished = False
        policy.stats.decisions += 1
        if len(request.legal_actions) == 1:
            policy.stats.single_candidate_decisions += 1
            self._collect_result()
        else:
            if not policy.root_only:
                policy.stats.searchable_decisions += 1
            policy._active_jobs.add(self)

    @property
    def done(self) -> bool:
        return self._core.done()

    @property
    def result(self) -> MCTSSearchResult:
        if not self.done:
            raise RuntimeError("search job has not finished")
        return self._result

    def pending_requests(self) -> tuple[NativePendingEvaluation, ...]:
        if self._pending_dirty:
            by_world = {} if self._pending is None else {
                item.world_index: item for item in self._pending.values()
            }
            pending = {}
            for item in self._core.pending():
                world = item["world"]
                if world in by_world:
                    evaluation = by_world[world]
                else:
                    cls = EncodedState if item["kind"] == "value_only" else EncodedDecision
                    encoded = cls(**item["encoded"])
                    request_id = f"native:{self.job_id}:{self.sequence}"
                    self.sequence += 1
                    evaluation = NativePendingEvaluation(
                        request_id, item["kind"], encoded, world, self.policy.model_version,
                    )
                pending[evaluation.request_id] = evaluation
            self._pending = pending
            self._pending_dirty = False
        return tuple(self._pending.values())

    def submit(self, request_id: str, logits: tuple[float, ...] | None,
               value: float, *, prior_is_normalized: bool = False) -> None:
        if self._pending is None or request_id not in self._pending:
            raise ValueError(f"unknown native search request {request_id}")
        if prior_is_normalized:
            raise ValueError("native search expects raw logits")
        world = self._pending[request_id].world_index
        self._core.submit(world, () if logits is None else logits, value)
        if world == -1:
            self._core.initialize_worlds(self.request)
        del self._pending[request_id]
        self._pending_dirty = True
        if self.done:
            self._collect_result()

    def _collect_result(self) -> None:
        if self._finished:
            return
        result = self._core.result()
        self._result = MCTSSearchResult(**result)
        self.policy.last_search_result = self._result
        stats = self._core.stats()
        for name, value in stats.items():
            if name == "native_compute_seconds":
                self.policy.native_compute_seconds += value
            else:
                setattr(self.policy.stats, name, getattr(self.policy.stats, name) + value)
        self.policy.stats.total_seconds += time.perf_counter() - self.started
        if result["searchable"]:
            raw = max(range(len(result["root_network_prior"])),
                      key=lambda i: (result["root_network_prior"][i], -i))
            self.policy.stats.changed_action_count += raw != result["chosen_candidate_index"]
        self._finished = True
        self.policy._active_jobs.discard(self)

    def cancel(self) -> None:
        self._pending = {}

    # -- contiguous batch interface ---------------------------------------

    @property
    def core(self):
        return self._core

    def pending_rows(self, state_only: bool) -> int:
        return self._core.pending_rows(state_only)

    def pending_width(self, state_only: bool) -> int:
        return self._core.pending_width(state_only)

    def root_pending(self) -> bool:
        return self._core.root_pending_now()

    def collect_if_done(self) -> None:
        """Archive the result when a batch submit completed the last simulation."""

        if self.done and not self._finished:
            self._collect_result()

    def initialize_worlds(self) -> None:
        self._core.initialize_worlds(self.request)

    def _action_from_native(self, raw):
        from .actions import (TakeDistinct, TakePair, ReserveMarket, ReserveDeck,
                              BuyMarket, BuyReserved, Pay, ReturnTokens, ClaimNoble)
        from .model import Color, Tier, TokenVector
        kind, card_or_noble, tier, tokens = raw
        colors = tuple(Color(name) for name in ("white", "blue", "green", "red", "black"))
        if kind == 0:
            return TakeDistinct(tuple(colors[i] for i in range(5) if tokens[i]))
        if kind == 1:
            return TakePair(colors[next(i for i in range(5) if tokens[i])])
        if kind == 2:
            return ReserveMarket(self.policy.catalog.cards[card_or_noble].card_id)
        if kind == 3:
            return ReserveDeck(Tier(tier + 1))
        if kind == 4:
            return BuyMarket(self.policy.catalog.cards[card_or_noble].card_id)
        if kind == 5:
            return BuyReserved(self.policy.catalog.cards[card_or_noble].card_id)
        if kind == 6:
            return Pay(TokenVector(*tokens))
        if kind == 7:
            return ReturnTokens(TokenVector(*tokens))
        return ClaimNoble(self.policy.catalog.nobles[card_or_noble].noble_id)

    def failure_context(self, request_id: str | None = None) -> dict[str, Any]:
        world = (-1 if request_id is None or self._pending is None or request_id not in self._pending
                 else self._pending[request_id].world_index)
        return self._failure_context(world, request_id)

    def failure_context_for_world(self, world: int | None) -> dict[str, Any]:
        """Failure context for a batch row identified by world (no request ids)."""

        return self._failure_context(-1 if world is None else world, None)

    def _failure_context(self, world: int, request_id: str | None) -> dict[str, Any]:
        native = self._core.failure_context(world)
        protocol = ("sampled-chance-pw-puct-native-v1" if self.policy.chance_widening is not None else
                    "sampled-chance-puct-native-v1" if self.policy.stochastic else "determinized-puct-native-v1")
        return {
            "algorithm": protocol,
            "search_protocol": protocol,
            **({"chance_widening": {"k": self.policy.chance_widening.k, "alpha": self.policy.chance_widening.alpha}}
               if self.policy.chance_widening is not None else {}),
            "root_request": self.request,
            "sampling_seed": self.sampling_seed,
            "world_index": native["world_index"],
            "determinization_seed": native.get("hidden_sample_seed", native["determinization_seed"]),
            **({"chance_seed": native["chance_seed"]} if self.policy.stochastic else {}),
            "simulation_index": native["simulation_index"],
            "branch_actions": [self._action_from_native(action) for action in native["branch_actions"]],
            "attempted_action": (None if native["attempted_action"] is None else
                                 self._action_from_native(native["attempted_action"])),
            "request_id": request_id,
        }


@dataclass(frozen=True, slots=True)
class NativeBatchRow:
    """Identity of one exported batch row, kept for failure attribution."""

    row: int
    world: int
    kind: int
    simulation: int
    candidates: int
    job: NativeSearchJob

    @property
    def kind_name(self) -> str:
        return {0: "root", 1: "policy_value", 2: "value_only"}[self.kind]


class NativeBatchRunner:
    """Drive native search jobs through reusable contiguous batch buffers.

    Rows keep the per-leaf order of the legacy adapter (policy rows first, then
    value-only rows, jobs in scheduling order) so batch composition matches the
    reference path; only the transport changes.  ``slots=1`` is the synchronous
    order, ``slots=2`` is a bounded pipeline: while one slot's input transfer and
    forward are in flight the CPU fills the next slot, and each slot's result is
    consumed in fill order before the slot is reused.
    """

    def __init__(self, evaluator, catalog: Catalog, *, capacity: int,
                 buckets: tuple[int, ...], slots: int = 1) -> None:
        if capacity <= 0:
            raise ValueError("native batch capacity must be positive")
        if not buckets or tuple(sorted(set(buckets))) != tuple(buckets):
            raise ValueError("native batch buckets must be sorted and unique")
        if slots not in (1, 2):
            raise ValueError("native batch pipeline supports one or two slots")
        self.evaluator = evaluator
        self.catalog = catalog
        self.native_catalog = _native_catalog(catalog)
        self.capacity = capacity
        self.buckets = tuple(buckets)
        self.slots = slots
        self._entries: dict[tuple[int, bool, int], tuple[Any, Any]] = {}
        self.rows = 0
        self.batches = 0
        self.wide_batches = 0
        self.root_rows = 0
        self.policy_rows = 0
        self.value_rows = 0
        self.inflight_batches = 0
        self.inflight_rows = 0
        self.peak_inflight_rows = 0
        self.fill_seconds = 0.0
        self.consume_seconds = 0.0
        self.batch_records: list[dict[str, Any]] = []
        self._error_record: dict[str, Any] | None = None
        self._building: dict[str, Any] | None = None

    # -- buffers -----------------------------------------------------------

    def _buffer_width(self, width: int) -> int:
        for bucket in self.buckets:
            if width <= bucket:
                return bucket
        self.wide_batches += 1
        return width

    def _buffer(self, slot: int, state_only: bool, width: int):
        bucket = self._buffer_width(width)
        key = (slot, state_only, bucket)
        entry = self._entries.get(key)
        if entry is None:
            probe = _extension().NativeBatch(self.native_catalog, state_only, 1, bucket)
            transfer = self.evaluator.new_native_transfer(
                state_only=state_only, capacity=self.capacity, width=bucket,
                float_stride=probe.float_stride(), mask_stride=probe.mask_stride())
            staging = getattr(transfer, "float_staging", None)
            if staging is None:
                buffer = _extension().NativeBatch(self.native_catalog, state_only, self.capacity, bucket)
            else:
                buffer = _extension().NativeBatch(self.native_catalog, state_only, self.capacity, bucket,
                                                  staging, transfer.mask_staging)
            entry = (buffer, transfer)
            self._entries[key] = entry
        return entry

    # -- scheduling --------------------------------------------------------

    def run(self, jobs: Sequence[NativeSearchJob], *,
            context: Mapping[NativeSearchJob, Any] | None = None) -> int:
        """Evaluate every pending row of ``jobs`` once; returns the row count."""

        self.batch_records = []
        self._error_record = None
        self._building = None
        processed = 0
        for state_only in (False, True):
            chunks = self._plan(jobs, state_only)
            if chunks:
                processed += self._run_phase(chunks, state_only, context)
        return processed

    def _plan(self, jobs: Sequence[NativeSearchJob], state_only: bool) -> list[tuple[NativeSearchJob, ...]]:
        handles = [job for job in jobs for _ in range(job.pending_rows(state_only))]
        return [tuple(dict.fromkeys(handles[start:start + self.capacity]))
                for start in range(0, len(handles), self.capacity)]

    def _run_phase(self, chunks, state_only: bool, context) -> int:
        free = list(range(self.slots))
        inflight: deque[dict[str, Any]] = deque()
        processed = 0
        try:
            for chunk in chunks:
                if not free:
                    record = inflight.popleft()
                    self._consume(record)
                    free.append(record["slot"])
                record = self._fill(free.pop(0), state_only, chunk, context)
                inflight.append(record)
                processed += record["rows"]
            while inflight:
                record = inflight.popleft()
                self._consume(record)
                free.append(record["slot"])
            return processed
        except Exception:
            # Outstanding CUDA copies must finish before their slot can die.
            # Their outputs are deliberately not submitted after another failure.
            for record in inflight:
                handle = record["handle"]
                if handle is not None:
                    try:
                        handle.discard()
                    except Exception:
                        # Keep the original batch error; this slot is abandoned.
                        pass
                record["status"] = "discarded"
            raise

    def _fill(self, slot: int, state_only: bool, chunk, context) -> dict[str, Any]:
        self._building = {"batch": len(self.batch_records), "slot": slot,
                          "state_only": state_only, "chunk": tuple(chunk),
                          "context": context}
        width = max(job.pending_width(state_only) for job in chunk)
        buffer, transfer = self._buffer(slot, state_only, width)
        assert buffer.row_count() == 0
        roots = [job for job in chunk if job.root_pending()]
        segments: list[tuple[int, int, NativeSearchJob]] = []
        fill_started = time.perf_counter()
        for job in chunk:
            start = buffer.row_count()
            added = buffer.fill(job.core)
            if added:
                segments.append((start, added, job))
        self.fill_seconds += time.perf_counter() - fill_started
        rows = buffer.row_count()
        if rows == 0:
            raise RuntimeError("native batch chunk exported no rows")
        # Lightweight identity: one array for the happy path, readable rows only
        # when a failure has to be logged.
        identity = buffer.row_ids_array()
        kinds = identity[:, 2]
        record = {"batch": len(self.batch_records), "slot": slot, "state_only": state_only,
                  "rows": rows, "width": buffer.padded_width(), "chunk": tuple(chunk),
                  "roots": tuple(roots), "identity": identity, "segments": tuple(segments),
                  "buffer": buffer, "transfer": transfer, "context": context, "handle": None,
                  "error": None, "status": "filled"}
        self.batch_records.append(record)
        self._building = None
        self.root_rows += int((kinds == 0).sum())
        self.policy_rows += int((kinds != 2).sum())
        self.value_rows += int((kinds == 2).sum())
        self.rows += rows
        self.batches += 1
        self.inflight_batches += 1
        self.inflight_rows += rows
        self.peak_inflight_rows = max(self.peak_inflight_rows, self.inflight_rows)
        try:
            record["handle"] = (transfer.begin_value_only(buffer) if state_only
                                else transfer.begin_policy_value(buffer))
        except Exception as error:
            self._mark_failed(record, "evaluate", error)
            raise
        return record

    def _consume(self, record: dict[str, Any]) -> None:
        started = time.perf_counter()
        try:
            record["handle"].finish()
            for job in record["roots"]:
                job.initialize_worlds()
            for job in dict.fromkeys(record["chunk"]):
                job.collect_if_done()
        except Exception as error:
            self._mark_failed(record, "submit", error)
            raise
        finally:
            self.inflight_batches -= 1
            self.inflight_rows -= record["rows"]
            self.consume_seconds += time.perf_counter() - started
        record["status"] = "submitted"

    def _mark_failed(self, record: dict[str, Any], stage: str, error: BaseException) -> None:
        record["status"] = "failed"
        record["error"] = {"stage": stage, "error_type": type(error).__name__, "error": str(error)}
        self._error_record = record

    # -- failure attribution ----------------------------------------------

    def row_context(self, report_row: NativeBatchRow) -> dict[str, Any]:
        job = report_row.job
        return {
            "row": report_row.row,
            "batch_world": report_row.world,
            "kind": report_row.kind_name,
            "simulation": report_row.simulation,
            "candidates": report_row.candidates,
            "job_id": job.job_id,
            "model_version": job.policy.model_version,
            "root_decision_index": getattr(job.request.observation, "own_completed_turns", None),
        }

    @staticmethod
    def _row_at(record: dict[str, Any], index: int) -> NativeBatchRow | None:
        """Expand one lightweight identity row; only used on failure paths."""

        identity = record.get("identity")
        if identity is None or not 0 <= index < record["rows"]:
            return None
        world, simulation, kind, candidates = (int(value) for value in identity[index])
        for start, count, job in record["segments"]:
            if start <= index < start + count:
                return NativeBatchRow(row=index, world=world, kind=kind,
                                      simulation=simulation, candidates=candidates, job=job)
        return None

    def rows_of(self, record: dict[str, Any]) -> tuple[NativeBatchRow, ...]:
        """Expand a record's cheap identity into readable rows (failure paths)."""

        return tuple(row for row in (self._row_at(record, index)
                                    for index in range(record["rows"])) if row is not None)

    def failure_report(self, error: BaseException) -> dict[str, Any]:
        """Locate the failed row when possible; otherwise report the batch scope."""

        record = self._error_record
        row = getattr(error, "row", None)
        if row is None:
            match = re.search(r"batch row (\d+)", str(error))
            row = int(match.group(1)) if match else None
        if record is not None and row is not None and 0 <= row < record["rows"]:
            entry = self._row_at(record, row)
            if entry is not None:
                context = None if record["context"] is None else record["context"].get(entry.job)
                return {"scope": "row", "row": row, "record": record, "entry": entry,
                        "job": entry.job, "context": context}
        pending = [item for item in self.batch_records if item["status"] != "submitted"]
        return {"scope": "batch", "row": None, "records": pending,
                "building": self._building,
                "context": (record or self._building or {}).get("context")}


class NativePUCTSearchPolicy:
    stochastic = False
    root_only = False
    chance_widening: ChanceWideningConfig | None = None

    def __init__(self, model, *, config: MCTSConfig | None = None,
                 device: str = "cpu", catalog: Catalog | None = None,
                 evaluator=None, model_version: str = "in-memory") -> None:
        from .search_inference import FrozenSearchInference
        self.config = config or MCTSConfig()
        self.model_version = model_version
        self.catalog = catalog or _default_catalog()
        self.native_catalog = _native_catalog(self.catalog)
        self.evaluator = evaluator or FrozenSearchInference(model, device=device)
        self.model = self.evaluator.model
        self.device = self.evaluator.device
        self.stats = MCTSStats()
        self.native_compute_seconds = 0.0
        self.last_search_result = None
        self._active_jobs: set[NativeSearchJob] = set()

    def live_simulations(self) -> int:
        """Completed simulations, including roots that have not been collected."""

        return self.stats.simulations + sum(
            int(job.core.stats()["simulations"]) for job in self._active_jobs
        )

    def start_job(self, request: DecisionRequest, *, add_root_noise: bool = False,
                  temperature: float = 0.0, sampling_seed: int | None = None,
                  root_encoded: EncodedDecision | None = None) -> NativeSearchJob:
        return NativeSearchJob(self, request, add_root_noise=add_root_noise,
                               temperature=temperature, sampling_seed=sampling_seed,
                               root_encoded=root_encoded)

    def search(self, request: DecisionRequest, *, add_root_noise: bool = False,
               temperature: float = 0.0, sampling_seed: int | None = None) -> MCTSSearchResult:
        job = self.start_job(request, add_root_noise=add_root_noise,
                             temperature=temperature, sampling_seed=sampling_seed)
        while not job.done:
            for item in job.pending_requests():
                if item.kind == "policy_value":
                    logits, value = self.evaluator.policy_value_encoded((item.encoded,))[0]
                    job.submit(item.request_id, logits, value)
                else:
                    value = self.evaluator.value_only_encoded((item.encoded,))[0]
                    job.submit(item.request_id, None, value)
        return job.result

    def choose(self, request: DecisionRequest):
        return request.legal_actions[self.search(request).chosen_candidate_index]


class NativeGreedyPolicy(NativePUCTSearchPolicy):
    """One root forward through native batches; argmax logits, zero simulations."""

    root_only = True

    def __init__(self, model, **kwargs) -> None:
        # The native batch job shares the transport descriptor with search jobs.
        # No worlds/budgets are instantiated for this policy.
        super().__init__(model, config=MCTSConfig(simulations=1, determinizations=1,
                         root_noise_fraction=0.), **kwargs)


class NativeChancePUCTSearchPolicy(NativePUCTSearchPolicy):
    """Sample chance at transitions, resample root hidden cards, share observer-history PUCT."""

    stochastic = True

    def __init__(self, model, *, config: MCTSConfig | None = None,
                 chance_widening: ChanceWideningConfig | None = None, **kwargs) -> None:
        config = config or MCTSConfig(determinizations=1)
        if config.determinizations != 1:
            raise ValueError("sampled chance search requires determinizations=1 for shared statistics")
        self.chance_widening = chance_widening
        super().__init__(model, config=config, **kwargs)
