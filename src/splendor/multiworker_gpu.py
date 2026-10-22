"""Spawned native search workers feeding one CUDA inference owner.

Only descriptors travel through queues.  The parent creates and unlinks every
shared segment; a worker owns each slot from fill until it consumes its reply.
Queue synchronization publishes writes before the peer reads them.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import queue
import resource
import threading
import time
import traceback
from collections import Counter, deque
from dataclasses import dataclass
from multiprocessing.connection import wait
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from .native_policy import _default_catalog, _extension, _native_catalog
from .opponent_pool import load_frozen_actor_critic_checkpoint
from .search_inference import FrozenSearchInference, InferenceConfig
from .selfplay_fast import (
    FastSelfPlayConfig, _GameSession, _atomic_json, _cpu_seconds,
)
from .replay import to_primitive


@dataclass(frozen=True)
class SlotNames:
    floats: str
    masks: str
    results: str
    float_count: int
    mask_count: int
    result_count: int


class _Slot:
    def __init__(self, names: SlotNames, *, create: bool = False) -> None:
        segments = []
        try:
            for name, count, size in (
                (names.floats, names.float_count, 4),
                (names.masks, names.mask_count, 1),
                (names.results, names.result_count, 4),
            ):
                segments.append(SharedMemory(
                    name=name, create=create, size=count * size if create else 0))
        except BaseException:
            for segment in segments:
                segment.close()
                if create:
                    segment.unlink()
            raise
        self.segments = tuple(segments)
        self.floats = np.ndarray((names.float_count,), np.float32, buffer=self.segments[0].buf)
        self.masks = np.ndarray((names.mask_count,), np.uint8, buffer=self.segments[1].buf)
        self.results = np.ndarray((names.result_count,), np.float32, buffer=self.segments[2].buf)

    def close(self) -> None:
        del self.floats, self.masks, self.results
        for segment in self.segments:
            segment.close()

    def unlink(self) -> None:
        for segment in self.segments:
            segment.unlink()


class _RemoteEvaluator:
    """Search-policy adapter; workers never construct a model or CUDA context."""
    model = None
    device = torch.device("cpu")


def _worker_failure(root: Path, worker_id: int, error: BaseException,
                    active: dict[int, _GameSession], outstanding: dict[int, Any],
                    failed_game_id: int | None) -> None:
    scenes = [session.failure_scene() for session in active.values()]
    def slot_report(slot: int, item: Any) -> dict[str, Any]:
        ids = item[1].row_ids_array()
        rows = [
            {
                "row": row, "game_id": game_id,
                "world": int(ids[row, 0]), "simulation": int(ids[row, 1]),
                "kind": int(ids[row, 2]), "candidates": int(ids[row, 3]),
            }
            for start, count, game_id in item[2]
            for row in range(start, start + count)
        ]
        return {"slot": slot, "generation": item[0], "rows": rows}

    _atomic_json(root / "failures" / f"worker-{worker_id}.json", {
        "protocol": "native-cuda-pool-failure-v1", "worker_id": worker_id,
        "error_type": type(error).__name__, "error_detail": str(error),
        "failed_game_id": failed_game_id,
        "traceback": traceback.format_exc(), "games": scenes,
        "outstanding": [slot_report(slot, item) for slot, item in outstanding.items()],
    })


def _checked_result(outstanding: dict[int, Any], worker_id: int,
                    slot_index: int, generation: int) -> Any:
    current = outstanding.get(slot_index)
    if current is None or current[0] != generation:
        raise RuntimeError(
            f"stale or duplicate result worker={worker_id} slot={slot_index} generation={generation}")
    return current


def _watch_owner(parent_pid: int) -> None:
    """Exit on owner death even if a queue put or its feeder is blocked."""
    parent = mp.parent_process()
    if parent is None or parent.pid != parent_pid:
        raise RuntimeError("search worker must be spawned by its GPU owner")

    def exit_when_owner_dies() -> None:
        wait((parent.sentinel,))
        # Queue finalizers can wait forever for a dead reader.  Closing the
        # process directly releases mappings and the resource-tracker pipe;
        # the shared tracker unlinks segments after the last client exits.
        os._exit(1)

    threading.Thread(target=exit_when_owner_dies, name="gpu-owner-lifetime",
                     daemon=True).start()


def _worker(config: FastSelfPlayConfig, root: Path, worker_id: int, quota: int,
            slots: tuple[SlotNames, ...], tasks: Any, ready: Any, replies: Any,
            fingerprint: str, checkpoint_sha: str, parent_pid: int,
            start_event: Any = None, stop_event: Any = None,
            session_factory: Callable = _GameSession, route_models: bool = False) -> None:
    _watch_owner(parent_pid)
    initialization_started = time.perf_counter()
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    attached = [_Slot(names) for names in slots]
    catalog = _native_catalog(_default_catalog())
    execution = config.execution
    micro = execution.local_microbatch
    width_limit = execution.shared_candidate_width
    evaluator = _RemoteEvaluator()
    active: dict[int, _GameSession] = {}
    completed: list[dict[str, Any]] = []
    outstanding: dict[int, tuple[int, Any, list[int], list[Any], list[Any]]] = {}
    generations = [0] * len(slots)
    buffers: dict[tuple[int, bool, int], Any] = {}
    peak_active = peak_pending = peak_inflight = exported_rows = batches = root_rows = policy_rows = value_rows = 0
    wide_batches = 0
    row_statistics = os.environ.get("SPLENDOR_SPEED_ROW_STATS") == "1"
    actual_widths = {stage: Counter() for stage in ("opening", "middle", "late")}
    descriptor_widths = Counter()
    fill_seconds = wait_seconds = backpressure_seconds = 0.0
    decision_latencies: list[float] = []
    kind_turn = 0
    model_turn = {False: 0, True: 0}
    exhausted = False
    current_game_id: int | None = None
    started = time.perf_counter()
    initialization_seconds = started - initialization_started
    cpu_started = _cpu_seconds()
    window_start = window_end = None

    def snapshot() -> dict[str, Any]:
        return {
            "at": time.perf_counter(), "cpu_seconds": _cpu_seconds(),
            "simulations": (sum(item["simulations"] for item in completed) +
                            sum(policy.live_simulations() for session in active.values()
                                for policy in session.policies)),
            "exported_rows": exported_rows, "batches": batches,
            "wait_result_seconds": wait_seconds,
            "queue_backpressure_seconds": backpressure_seconds,
            "completed_games": len(completed), "active_games": len(active),
            "root_decisions": (sum(item["decisions"] for item in completed) +
                               sum(len(session.records) for session in active.values())),
        }

    def advance() -> None:
        nonlocal peak_active, peak_pending, exhausted, current_game_id
        while len(active) < quota and not exhausted:
            try:
                game_id = tasks.get(timeout=0.02)
            except queue.Empty:
                break
            if game_id is None:
                exhausted = True
                break
            active[game_id] = session_factory(config, game_id, evaluator, checkpoint_sha)
        peak_active = max(peak_active, len(active))
        for game_id, session in tuple(active.items()):
            current_game_id = game_id
            if session.job is not None and session.job.done:
                session.take_result()
            session.ensure_job()
            if session.finished:
                summary = session.commit(root, fingerprint, checkpoint_sha)
                completed.append(summary)
                decision_latencies.extend(session.decision_latencies_ms)
                del active[game_id]
                ready.put(("complete", worker_id, game_id, summary), timeout=30)
            current_game_id = None
        peak_pending = max(
            peak_pending,
            sum(session.job.pending_rows(False) + session.job.pending_rows(True)
                for session in active.values() if session.job is not None),
        )

    def publish(slot_index: int) -> bool:
        nonlocal kind_turn, peak_inflight, exported_rows, batches, root_rows, policy_rows, value_rows, wide_batches
        nonlocal fill_seconds, backpressure_seconds
        for state_only in ((kind_turn % 2) == 1, (kind_turn % 2) == 0):
            jobs = [session.job for session in active.values()
                    if session.job is not None and session.job.pending_rows(state_only)]
            if not jobs:
                continue
            model_key = config.round_index
            if route_models:
                keys = sorted({job.policy.model_version for job in jobs})
                model_key = keys[model_turn[state_only] % len(keys)]
                model_turn[state_only] += 1
                jobs = [job for job in jobs if job.policy.model_version == model_key]
            job_set = set(jobs)
            kind_turn += 1
            width = 1 if state_only else max(job.pending_width(False) for job in jobs)
            if width > width_limit:
                raise ValueError(f"candidate width {width} exceeds configured shared_candidate_width={width_limit}")
            key = (slot_index, state_only, width)
            batch = buffers.get(key)
            if batch is None:
                batch = _extension().NativeBatch(catalog, state_only, micro, width,
                                                 attached[slot_index].floats,
                                                 attached[slot_index].masks)
                buffers[key] = batch
            if batch.row_count():
                raise RuntimeError("slot reused before result consumption")
            roots = [job for job in jobs if job.root_pending()]
            started_fill = time.perf_counter()
            used_jobs = []
            segments = []
            for session in active.values():
                if session.job not in job_set:
                    continue
                start_row = batch.row_count()
                added = batch.fill(session.job.core)
                if added:
                    used_jobs.append(session.job)
                    segments.append((start_row, added, session.game_id))
                if batch.row_count() == micro:
                    break
            fill_seconds += time.perf_counter() - started_fill
            rows = batch.row_count()
            if not rows:
                return False
            roots = [job for job in used_jobs if job in roots]
            ids = batch.row_ids_array()
            if row_statistics:
                descriptor_widths[str(0 if state_only else width)] += rows
                indices = {stage: [] for stage in actual_widths}
                for start, count, game_id in segments:
                    observation = active[game_id].job.request.observation
                    turns = observation.own_completed_turns + observation.opponent_completed_turns
                    stage = "opening" if turns < 10 else "middle" if turns < 50 else "late"
                    indices[stage].extend(range(start, start + count))
                for stage, selected in indices.items():
                    if selected:
                        counts = np.bincount(ids[selected, 3])
                        actual_widths[stage].update({str(int(i)): int(counts[i])
                                                   for i in np.flatnonzero(counts)})
            generations[slot_index] += 1
            generation = generations[slot_index]
            outstanding[slot_index] = (generation, batch, segments, roots, used_jobs)
            desc = (worker_id, slot_index, generation, rows,
                    "value" if state_only else "policy", width, model_key)
            before_put = time.perf_counter()
            ready.put(("batch", desc), timeout=30)
            backpressure_seconds += time.perf_counter() - before_put
            peak_inflight = max(peak_inflight, sum(item[1].row_count() for item in outstanding.values()))
            exported_rows += rows
            batches += 1
            wide_batches += int(not state_only and width > execution.candidate_buckets[-1])
            root_rows += int((ids[:, 2] == 0).sum())
            policy_rows += int((ids[:, 2] != 2).sum())
            value_rows += int((ids[:, 2] == 2).sum())
            return True
        return False

    try:
        ready.put(("hello", worker_id, checkpoint_sha))
        while True:
            if start_event is not None and start_event.is_set() and window_start is None:
                window_start = snapshot()
            stopping = stop_event is not None and stop_event.is_set()
            if stopping and window_end is None:
                window_end = snapshot()
            if not stopping:
                advance()
                for slot_index in range(len(slots)):
                    if slot_index not in outstanding:
                        publish(slot_index)
            if (not active and not outstanding and exhausted) or (stopping and not outstanding):
                break
            if not outstanding:
                # Forced roots can finish without inference. Advance them now;
                # a result queue cannot wake a worker with no exported request.
                continue
            ready.put(("blocked", worker_id,
                       tuple((slot, item[0]) for slot, item in outstanding.items())))
            before_wait = time.perf_counter()
            try:
                reply = replies.get(timeout=0.25)
            except queue.Empty:
                wait_seconds += time.perf_counter() - before_wait
                if os.getppid() != parent_pid:
                    raise RuntimeError("GPU owner exited while search worker was waiting")
                continue
            wait_seconds += time.perf_counter() - before_wait
            if reply[0] == "abort":
                raise RuntimeError(f"GPU owner failed: {reply[1]}")
            _, slot_index, generation = reply
            current = _checked_result(outstanding, worker_id, slot_index, generation)
            _, batch, _, roots, used_jobs = current
            rows, width = batch.row_count(), batch.padded_width()
            if batch.is_state_only():
                batch.submit(None, attached[slot_index].results[:rows])
            else:
                batch.submit_fused(attached[slot_index].results[:rows * (width + 1)].reshape(rows, width + 1))
            for job in roots:
                job.initialize_worlds()
            for job in used_jobs:
                job.collect_if_done()
            del outstanding[slot_index]
            del current, roots, used_jobs, job
    except BaseException as error:
        try:
            _worker_failure(root, worker_id, error, active, outstanding, current_game_id)
        except Exception:
            pass
        ready.put(("error", worker_id, error, traceback.format_exc()))
        return
    finally:
        buffers.clear()
        outstanding.clear()
        for slot in attached:
            slot.close()
    live_at_exit = snapshot()["simulations"]
    ready.put(("done", {
        "worker_id": worker_id, "completed": completed,
        "pid": os.getpid(), "cuda_initialized": torch.cuda.is_initialized(),
        "live_simulations_at_exit": live_at_exit,
        "in_progress_simulations_at_exit": (
            live_at_exit - sum(item["simulations"] for item in completed)),
        "wall_seconds": time.perf_counter() - started,
        "cpu_seconds": _cpu_seconds() - cpu_started,
        "initialization_seconds": initialization_seconds,
        "decision_latencies_ms": decision_latencies,
        "inference": {"physical_model_batches": 0, "logical_model_positions": 0,
                      "feature_seconds": 0.0, "forward_seconds": 0.0,
                      "h2d_seconds": 0.0, "d2h_seconds": 0.0, "wait_seconds": 0.0,
                      "batch_size_histogram": {}, "inference": {}, "compile": {}},
        "counters": {"native_batch_rows": exported_rows, "native_batch_batches": batches,
                     "native_root_rows": root_rows, "native_policy_rows": policy_rows,
                     "native_value_rows": value_rows, "native_wide_batches": wide_batches,
                     "native_peak_inflight_rows": peak_inflight,
                     "native_fill_seconds": fill_seconds, "native_consume_seconds": 0.0},
        "peak_active_games": peak_active, "peak_pending_requests": peak_pending,
        "cpu_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "gpu_peak_bytes": 0, "wait_result_seconds": wait_seconds,
        "queue_backpressure_seconds": backpressure_seconds,
        "actual_width_rows_by_root_stage": {stage: dict(counts) for stage, counts in actual_widths.items()},
        "descriptor_width_rows": dict(descriptor_widths),
        "window": {"start": window_start, "end": window_end or snapshot()}
                  if start_event is not None else None,
    }))


class _FixedPoolHandle:
    def __init__(self, owner: Any, sink: Any, width: int, state_only: bool) -> None:
        self.owner, self.sink, self.width, self.state_only = owner, sink, width, state_only

    def done(self) -> bool:
        return True

    def discard(self) -> None:
        pass

    def finish(self) -> None:
        rows = self.sink.row_count()
        if self.state_only:
            self.sink.submit(None, np.zeros(rows, dtype=np.float32))
        else:
            self.sink.submit_fused(np.zeros((rows, self.width + 1), dtype=np.float32))
        stats = self.owner.stats
        stats.batches += 1
        stats.positions += rows
        stats.batch_sizes[rows] += 1


class _FixedPoolTransfer:
    def __init__(self, owner: Any, capacity: int, width: int,
                 float_stride: int, mask_stride: int) -> None:
        self.owner, self.width = owner, width
        self.float_staging = np.empty(capacity * float_stride, dtype=np.float32)
        self.mask_staging = np.empty(capacity * mask_stride, dtype=np.bool_)

    def begin_policy_value(self, sink: Any) -> _FixedPoolHandle:
        return _FixedPoolHandle(self.owner, sink, self.width, False)

    def begin_value_only(self, sink: Any) -> _FixedPoolHandle:
        return _FixedPoolHandle(self.owner, sink, self.width, True)


class FixedPoolOwner:
    """Deterministic owner for bounded correctness tests and supply profiling."""
    device = torch.device("cpu")

    def __init__(self) -> None:
        from .search_inference import InferenceStats
        self.stats = InferenceStats(config={"mode": "fixed-value"})

    def warmup_native(self, widths: Any) -> float:
        return 0.0

    def new_native_transfer(self, *, state_only: bool, capacity: int, width: int,
                            float_stride: int, mask_stride: int) -> _FixedPoolTransfer:
        return _FixedPoolTransfer(self, capacity, width, float_stride, mask_stride)


class _OutputSink:
    def __init__(self, rows: int, layout: dict[str, Any]) -> None:
        self.rows = rows
        self.layout = layout
        self.output: np.ndarray | None = None

    def row_count(self) -> int:
        return self.rows

    def packed(self) -> dict[str, Any]:
        return self.layout

    def submit_fused(self, values: np.ndarray) -> None:
        self.output = values

    def submit(self, _logits: None, values: np.ndarray) -> None:
        self.output = values


def _gather(src: _Slot, src_width: int, dst: Any,
            kind: str, src_offset: int, dst_offset: int, rows: int,
            src_float_stride: int, src_mask_stride: int,
            dst_float_stride: int, dst_mask_stride: int,
            base_float: int, base_mask: int, action_dim: int) -> None:
    sf = src.floats[:(src_offset + rows) * src_float_stride].reshape(src_offset + rows, src_float_stride)[src_offset:]
    sm = src.masks[:(src_offset + rows) * src_mask_stride].reshape(src_offset + rows, src_mask_stride)[src_offset:]
    df = dst.float_staging.reshape(-1, dst_float_stride)[dst_offset:dst_offset + rows]
    dm = dst.mask_staging.reshape(-1, dst_mask_stride)[dst_offset:dst_offset + rows]
    if src_float_stride == dst_float_stride and src_mask_stride == dst_mask_stride:
        np.copyto(df, sf)
        np.copyto(dm, sm, casting="unsafe")
        return
    np.copyto(df[:, :base_float], sf[:, :base_float])
    np.copyto(dm[:, :base_mask], sm[:, :base_mask], casting="unsafe")
    if kind == "policy":
        count = src_width * action_dim
        df[:, base_float + count:].fill(0)
        dm[:, base_mask + src_width:].fill(0)
        np.copyto(df[:, base_float:base_float + count], sf[:, base_float:base_float + count])
        np.copyto(dm[:, base_mask:base_mask + src_width], sm[:, base_mask:base_mask + src_width], casting="unsafe")


def _batch_group(desc: tuple, compiled_width: int) -> tuple[str, bool, Any]:
    """Isolate models and keep wide eager requests out of compiled batches."""
    return desc[4], desc[4] == "policy" and desc[5] > compiled_width, desc[6]


def _batch_trigger(pending: deque, active_workers: set[int], blocked: dict,
                   max_batch: int, compiled_width: int,
                   now: float, deadline: float) -> str | None:
    if not pending:
        return None
    group = _batch_group(pending[0]["event"][1], compiled_width)
    if sum(item["event"][1][3] - item["offset"] for item in pending
           if _batch_group(item["event"][1], compiled_width) == group) >= max_batch:
        return "full"
    if not active_workers:
        return "producers_exhausted"
    supplied = {(item["event"][1][0], item["event"][1][1], item["event"][1][2])
                for item in pending}
    if all(wid in blocked and blocked[wid] and
           all((wid, slot, generation) in supplied for slot, generation in blocked[wid])
           for wid in active_workers):
        return "all_supply_arrived"
    if now >= deadline:
        return "deadline"
    return None


def collect(config: FastSelfPlayConfig, root: Path, missing: tuple[int, ...],
            fingerprint: str, checkpoint_sha: str,
            progress_callback: Callable[[int, int], None] | None = None,
            completed_before: int = 0, evaluator: Any = None,
            benchmark_seconds: float | None = None,
            warmup_seconds: float = 0.0,
            session_factory: Callable = _GameSession,
            model_evaluators: dict[str, Any] | None = None,
            completion_callback: Callable[[dict[str, Any]], None] | None = None,
            ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run bounded search sessions with one inference owner.

    Defaults to real selfplay games; a fixed-root source supplies the same
    job/completion interface without taking actions or loading a worker model.
    model_evaluators enables routing by job.policy.model_version; the default
    single-model descriptor retains the frozen collection round identity.
    """
    session_started = time.perf_counter()
    execution = config.execution
    if evaluator is not None and model_evaluators is not None:
        raise ValueError("provide evaluator or model_evaluators, not both")
    if benchmark_seconds is not None and benchmark_seconds <= 0:
        raise ValueError("benchmark_seconds must be positive")
    if warmup_seconds < 0:
        raise ValueError("warmup_seconds must be nonnegative")
    active_limit = min(execution.total_active_games, len(missing))
    worker_count = min(execution.search_workers, active_limit)
    quotas = [active_limit // worker_count + (i < active_limit % worker_count)
              for i in range(worker_count)]
    micro = execution.local_microbatch
    width_limit = execution.shared_candidate_width
    ctx = mp.get_context("spawn")
    tasks = ctx.Queue()
    ready = ctx.Queue(maxsize=worker_count * execution.shared_slots * 4)
    replies = [ctx.Queue(maxsize=execution.shared_slots * 2) for _ in range(worker_count)]
    start_event = ctx.Event() if benchmark_seconds is not None else None
    stop_event = ctx.Event() if benchmark_seconds is not None else None
    created: list[_Slot] = []
    slot_names: list[tuple[SlotNames, ...]] = []
    processes = []
    succeeded = False
    inflight = None
    try:
        unassigned = iter(missing)
        for _ in range(active_limit):
            tasks.put(next(unassigned))
        all_assigned = active_limit == len(missing)
        if all_assigned:
            for _ in range(worker_count):
                tasks.put(None)
        native_catalog = _native_catalog(_default_catalog())
        policy_probe = _extension().NativeBatch(native_catalog, False, 1, width_limit)
        float_count = micro * policy_probe.float_stride()
        mask_count = micro * policy_probe.mask_stride()
        result_count = micro * (width_limit + 1)
        for worker_id in range(worker_count):
            names_for_worker = []
            for slot_index in range(execution.shared_slots):
                prefix = f"spgpu_{os.getpid()}_{worker_id}_{slot_index}"
                names = SlotNames(prefix + "_f", prefix + "_m", prefix + "_r",
                                  float_count, mask_count, result_count)
                created.append(_Slot(names, create=True))
                names_for_worker.append(names)
            slot_names.append(tuple(names_for_worker))
        thread_variables = ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")
        previous_threads = {name: os.environ.get(name) for name in thread_variables}
        try:
            for name in thread_variables:
                os.environ[name] = "1"
            for worker_id in range(worker_count):
                process = ctx.Process(
                    target=_worker,
                    args=(config, root, worker_id, quotas[worker_id], slot_names[worker_id],
                          tasks, ready, replies[worker_id], fingerprint, checkpoint_sha, os.getpid(),
                          start_event, stop_event, session_factory, model_evaluators is not None),
                    name=f"native-search-{worker_id}",
                )
                process.start()
                processes.append(process)
        finally:
            for name, previous in previous_threads.items():
                if previous is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = previous
        torch.set_num_threads(1)
        if evaluator is None and model_evaluators is None:
            model = load_frozen_actor_critic_checkpoint(Path(config.checkpoint), device="cuda")
            evaluator = FrozenSearchInference(model, device="cuda", config=InferenceConfig(
                mode=execution.inference_mode, actor_projection=execution.actor_projection,
                candidate_buckets=execution.candidate_buckets, compile_mode=execution.compile_mode,
                compile_failure_policy=execution.compile_failure_policy, native_transport="reused",
                batch_buckets=execution.batch_buckets))
            torch.cuda.reset_peak_memory_stats()
        evaluators = model_evaluators if model_evaluators is not None else {config.round_index: evaluator}

        def inference_stats():
            if model_evaluators is None:
                return evaluator.stats.to_dict()
            from .search_inference import InferenceStats
            total = InferenceStats(config={"mode": "model-routed"})
            for item in evaluators.values():
                for name in ("batches", "positions", "feature_seconds", "forward_seconds",
                             "h2d_seconds", "d2h_seconds", "wait_seconds",
                             "native_graph_captures", "native_graph_replays"):
                    setattr(total, name, getattr(total, name) + getattr(item.stats, name))
                total.batch_sizes.update(item.stats.batch_sizes)
            return total.to_dict()

        init_done = time.perf_counter()
        compile_warmup_seconds = (sum(item.warmup_native(execution.candidate_buckets)
                                     for item in evaluators.values())
                                  if benchmark_seconds is not None else 0.0)
        transfers: dict[tuple[Any, int, str, int], tuple[Any, Any, dict[str, Any]]] = {}
        pending: deque[dict[str, Any]] = deque()
        results: list[dict[str, Any]] = []
        active_workers = set(range(worker_count))
        online_workers: set[int] = set()
        warmup_deadline = None
        window_start = window_end = None
        owner_cpu_start = _cpu_seconds()
        owner_wait = owner_coalesce_wait = owner_gather = owner_dispatch = 0.0
        peak_inflight = padding_positions = 0
        batch_padding_positions = 0
        next_lane = batches_prepared_while_inflight = 0
        overlapped_gather_seconds = 0.0
        width_hist: dict[str, int] = {}
        batches_waited = batches_full = 0
        wait_reasons = {name: 0 for name in ("full", "deadline", "all_supply_arrived", "producers_exhausted")}
        wait_detail = {name: {"batches": 0, "rows": 0, "waited_batches": 0, "wait_seconds": 0.}
                       for name in wait_reasons}
        blocked: dict[int, tuple] = {}
        returned = {}
        completed_count = 0
        supply_series = []
        last_supply_time = session_started
        tail_intervals = {name: {"batches": 0, "rows": 0, "coalesce_seconds": 0., "gather_seconds": 0.,
                                 "inference_seconds": 0., "dispatch_seconds": 0.}
                          for name in ("above100", "11to100", "9to10", "2to8", "last1")}

        def accept(event):
            nonlocal completed_count, all_assigned
            if event[0] == "batch":
                pending.append({"event": event, "offset": 0, "completed": 0})
            elif event[0] == "hello":
                if event[2] != checkpoint_sha:
                    raise RuntimeError(f"search worker {event[1]} checkpoint identity mismatch")
                online_workers.add(event[1])
            elif event[0] == "blocked":
                wid, slots = event[1:]
                if all(generation > returned.get((wid, slot), 0) for slot, generation in slots):
                    blocked[wid] = slots
            elif event[0] == "complete":
                completed_count += 1
                if completion_callback is not None:
                    completion_callback(event[3])
                if not all_assigned:
                    tasks.put(next(unassigned))
                    if completed_count + active_limit == len(missing):
                        all_assigned = True
                        for _ in range(worker_count):
                            tasks.put(None)
                if progress_callback is not None:
                    progress_callback(completed_before + completed_count, config.num_games)
            elif event[0] == "error":
                _, worker_id, error, trace = event
                raise type(error)(f"search worker {worker_id}: {error}\n{trace}")
            elif event[0] == "done":
                data = event[1]
                active_workers.remove(data["worker_id"])
                blocked.pop(data["worker_id"], None)
                results.append(data)

        def finish_batch(job):
            nonlocal owner_dispatch
            parts, kind, bucket = job["parts"], job["kind"], job["bucket"]
            used, waited, gather_elapsed = job["used"], job["waited"], job["gather_seconds"]
            handle, sink = job["handle"], job["sink"]
            finish_started = time.perf_counter()
            try:
                handle.finish()
                infer_elapsed = job["launch_seconds"] + time.perf_counter() - finish_started
                output = sink.output
                assert output is not None
                finite = np.isfinite(output)
                bad = np.flatnonzero(~finite if finite.ndim == 1 else ~finite.all(axis=1))
                if len(bad):
                    physical_row = int(bad[0])
                    affected = next(part for part in parts
                                    if part[5] <= physical_row < part[5] + part[6])
                    raise ValueError(
                        f"nonfinite pooled output worker={affected[1]} slot={affected[2]} "
                        f"generation={affected[3]} row={affected[4] + physical_row - affected[5]}")
                started_dispatch = time.perf_counter()
                for entry, wid, slot_index, generation, src_offset, dst_offset, take, src_width in parts:
                    dest = created[wid * execution.shared_slots + slot_index].results
                    if kind == "value":
                        dest[src_offset:src_offset + take] = output[dst_offset:dst_offset + take]
                    else:
                        dest_view = dest[:micro * (src_width + 1)].reshape(micro, src_width + 1)
                        dest_view[src_offset:src_offset + take, :src_width] = output[
                            dst_offset:dst_offset + take, :src_width]
                        dest_view[src_offset:src_offset + take, src_width] = output[
                            dst_offset:dst_offset + take, bucket]
                    entry["completed"] += take
                    if entry["completed"] == entry["event"][1][3]:
                        returned[wid, slot_index] = generation
                        blocked.pop(wid, None)
                        replies[wid].put(("result", slot_index, generation), timeout=30)
                owner_dispatch += time.perf_counter() - started_dispatch
                dispatch_elapsed = time.perf_counter() - started_dispatch
                remaining = len(missing) - completed_count
                interval = ("above100" if remaining > 100 else "11to100" if remaining > 10 else
                            "9to10" if remaining > 8 else "2to8" if remaining > 1 else "last1")
                data = tail_intervals[interval]
                data["batches"] += 1
                data["rows"] += used
                data["coalesce_seconds"] += waited
                data["gather_seconds"] += gather_elapsed
                data["inference_seconds"] += infer_elapsed
                data["dispatch_seconds"] += dispatch_elapsed
                width_hist[str(bucket)] = width_hist.get(str(bucket), 0) + used
            except BaseException as error:
                try:
                    _atomic_json(root / "failures" / "owner.json", {
                        "protocol": "native-cuda-pool-failure-v1", "origin": "inference",
                        "error_type": type(error).__name__, "error_detail": str(error),
                        "traceback": traceback.format_exc(),
                        "requests": [{"worker_id": p[1], "slot": p[2], "generation": p[3],
                                      "row_offset": p[4], "rows": p[6]} for p in parts],
                    })
                except Exception:
                    pass
                raise

        while active_workers or pending or inflight is not None:
            if inflight is not None:
                while True:
                    try:
                        accept(ready.get_nowait())
                    except queue.Empty:
                        break
                if not pending or inflight["handle"].done():
                    finish_batch(inflight)
                    inflight = None
            now = time.perf_counter()
            if now - last_supply_time >= 5:
                last_supply_time = now
                supply_series.append({"seconds": now - session_started,
                    "remaining_games": len(missing) - completed_count,
                    "active_workers": len(active_workers), "blocked_workers": len(blocked),
                    "pending_slots": len(pending),
                    "pending_rows": sum(i["event"][1][3] - i["offset"] for i in pending)})
            if benchmark_seconds is not None and len(online_workers) == worker_count:
                if warmup_deadline is None:
                    warmup_deadline = now + warmup_seconds
                if window_start is None and now >= warmup_deadline:
                    window_start = {
                        "at": now, "cpu_seconds": _cpu_seconds(),
                        "gather_seconds": owner_gather,
                        "dispatch_seconds": owner_dispatch,
                        "wait_input_seconds": owner_wait,
                        "wait_coalesce_seconds": owner_coalesce_wait,
                        "batch_padding_positions": batch_padding_positions,
                        "batches_prepared_while_inflight": batches_prepared_while_inflight,
                        "overlapped_gather_seconds": overlapped_gather_seconds,
                        "inference": inference_stats(),
                    }
                    start_event.set()
                if window_start is not None and window_end is None and (
                        now >= window_start["at"] + benchmark_seconds or not active_workers):
                    window_end = {
                        "at": now, "cpu_seconds": _cpu_seconds(),
                        "gather_seconds": owner_gather,
                        "dispatch_seconds": owner_dispatch,
                        "wait_input_seconds": owner_wait,
                        "wait_coalesce_seconds": owner_coalesce_wait,
                        "batch_padding_positions": batch_padding_positions,
                        "batches_prepared_while_inflight": batches_prepared_while_inflight,
                        "overlapped_gather_seconds": overlapped_gather_seconds,
                        "inference": inference_stats(),
                    }
                    stop_event.set()
            before_wait = time.perf_counter()
            if not pending:
                try:
                    event = ready.get(timeout=0.25)
                except queue.Empty:
                    owner_wait += time.perf_counter() - before_wait
                    for worker_id in active_workers:
                        if not processes[worker_id].is_alive():
                            raise RuntimeError(f"search worker {worker_id} exited without completion")
                    continue
                owner_wait += time.perf_counter() - before_wait
                accept(event)
            deadline = time.perf_counter() + execution.max_batch_wait_ms / 1000
            waited = 0.
            # Drain controls before deciding how much future supply is possible.
            while True:
                try:
                    accept(ready.get_nowait())
                except queue.Empty:
                    break
            if not pending:
                continue
            reason = _batch_trigger(pending, active_workers, blocked, execution.max_batch_size,
                                    execution.candidate_buckets[-1], time.perf_counter(), deadline)
            while reason is None:
                if inflight is not None and inflight["handle"].done():
                    finish_batch(inflight)
                    inflight = None
                coalesce_started = time.perf_counter()
                event = None
                try:
                    event = ready.get(timeout=max(0, deadline - coalesce_started))
                except queue.Empty:
                    pass
                elapsed = time.perf_counter() - coalesce_started
                owner_coalesce_wait += elapsed
                waited += elapsed
                if event is not None:
                    accept(event)
                reason = _batch_trigger(pending, active_workers, blocked, execution.max_batch_size,
                                        execution.candidate_buckets[-1], time.perf_counter(), deadline)
            if not pending:
                continue
            candidates = [entry for entry in pending if entry["event"][0] == "batch"]
            if not candidates:
                continue
            wait_reasons[reason] += 1
            batches_waited += int(waited > 0)
            kind = candidates[0]["event"][1][4]
            group = _batch_group(candidates[0]["event"][1], execution.candidate_buckets[-1])
            chosen = [entry for entry in candidates
                      if _batch_group(entry["event"][1], execution.candidate_buckets[-1]) == group]
            parts = []
            used = 0
            for entry in chosen:
                desc = entry["event"][1]
                wid, slot_index, generation, rows, _, src_width, version = desc
                if version not in evaluators:
                    raise RuntimeError("pooled inference model version mismatch")
                take = min(rows - entry["offset"], execution.max_batch_size - used)
                if not take:
                    break
                parts.append((entry, wid, slot_index, generation, entry["offset"], used, take, src_width))
                used += take
                if used == execution.max_batch_size:
                    break
            if used == execution.max_batch_size:
                batches_full += 1
            detail = wait_detail[reason]
            detail["batches"] += 1
            detail["rows"] += used
            detail["waited_batches"] += int(waited > 0)
            detail["wait_seconds"] += waited
            width = 0 if kind == "value" else max(part[7] for part in parts)
            bucket = (next((item for item in execution.candidate_buckets if width <= item),
                           width_limit) if kind == "policy" else 0)
            key = (version, next_lane, kind, bucket)
            cached = transfers.get(key)
            if cached is None:
                probe = _extension().NativeBatch(native_catalog, kind == "value", 1, bucket or 1)
                transfer = evaluators[version].new_native_transfer(
                    state_only=kind == "value", capacity=execution.max_batch_size,
                    width=bucket or 1, float_stride=probe.float_stride(),
                    mask_stride=probe.mask_stride())
                layout = probe.packed()
                cached = (probe, transfer, layout)
                transfers[key] = cached
            probe, transfer, layout = cached
            peak_inflight = max(peak_inflight, sum(
                item["event"][1][3] - item["offset"] for item in candidates))
            started_gather = time.perf_counter()
            float_fields = layout["float_fields"]
            mask_fields = layout["mask_fields"]
            base_float = next((field[1] for field in float_fields if field[0] == "candidate_features"),
                              probe.float_stride())
            base_mask = next((field[1] for field in mask_fields if field[0] == "candidate_mask"),
                             probe.mask_stride())
            action_dim = (next((field[2] // bucket for field in float_fields
                                if field[0] == "candidate_features"), 0) if bucket else 0)
            for entry, wid, slot_index, generation, src_offset, dst_offset, take, src_width in parts:
                src = created[wid * execution.shared_slots + slot_index]
                src_float_stride = base_float + (src_width * action_dim if kind == "policy" else 0)
                src_mask_stride = base_mask + (src_width if kind == "policy" else 0)
                _gather(src, src_width, transfer, kind, src_offset, dst_offset, take,
                        src_float_stride, src_mask_stride,
                        probe.float_stride(), probe.mask_stride(),
                        base_float, base_mask, action_dim)
                padding_positions += take * max(0, bucket - src_width)
            owner_gather += time.perf_counter() - started_gather
            gather_elapsed = time.perf_counter() - started_gather
            if inflight is not None:
                overlapped_gather_seconds += gather_elapsed
                batches_prepared_while_inflight += 1
                finish_batch(inflight)
                inflight = None
            try:
                # Reserve rows before launching: a split shared request may have
                # one prefix in flight and its suffix staged in the other bank.
                for entry, wid, slot_index, generation, src_offset, dst_offset, take, src_width in parts:
                    entry["offset"] += take
                    pending.remove(entry)
                    if entry["offset"] < entry["event"][1][3]:
                        pending.append(entry)
                sink = _OutputSink(used, layout)
                infer_started = time.perf_counter()
                handle = (transfer.begin_value_only(sink) if kind == "value"
                          else transfer.begin_policy_value(sink))
                if execution.batch_buckets:
                    batch_padding_positions += next(size for size in execution.batch_buckets if used <= size) - used
                inflight = dict(parts=parts, kind=kind, bucket=bucket, used=used,
                                waited=waited, gather_seconds=gather_elapsed,
                                handle=handle, sink=sink, lane=next_lane,
                                launch_seconds=time.perf_counter() - infer_started)
                next_lane = (next_lane + 1) % execution.pipeline_slots
                if execution.pipeline_slots == 1:
                    finish_batch(inflight)
                    inflight = None
            except BaseException as error:
                try:
                    _atomic_json(root / "failures" / "owner.json", {
                        "protocol": "native-cuda-pool-failure-v1", "origin": "inference",
                        "error_type": type(error).__name__, "error_detail": str(error),
                        "traceback": traceback.format_exc(),
                        "requests": [{"worker_id": p[1], "slot": p[2], "generation": p[3],
                                      "row_offset": p[4], "rows": p[6]} for p in parts],
                    })
                except Exception:
                    pass
                raise
        owner_stats = {
            "requested_search_workers": execution.search_workers, "actual_search_workers": worker_count,
            "requested_active_games": execution.total_active_games,
            "actual_active_limit": active_limit,
            "global_max_batch": execution.max_batch_size, "local_microbatch": micro,
            "shared_slots_per_worker": execution.shared_slots,
            "owner_pipeline_slots": execution.pipeline_slots,
            "batches_prepared_while_inflight": batches_prepared_while_inflight,
            "overlapped_gather_seconds": overlapped_gather_seconds,
            "shared_bytes": sum(sum(seg.size for seg in slot.segments) for slot in created),
            "owner_cpu_seconds": _cpu_seconds() - owner_cpu_start,
            "owner_wait_input_seconds": owner_wait,
            "owner_wait_coalesce_seconds": owner_coalesce_wait,
            "owner_gather_seconds": owner_gather,
            "owner_dispatch_seconds": owner_dispatch,
            "peak_ready_rows": peak_inflight, "padding_positions": padding_positions,
            "batch_padding_positions": batch_padding_positions,
            "width_histogram": width_hist, "batches_waited": batches_waited,
            "batches_full": batches_full, "batch_wait_reasons": wait_reasons,
            "batch_trigger_detail": wait_detail,
            "supply_series": supply_series, "tail_intervals_exclusive": tail_intervals,
            "owner_initialization_seconds": init_done - session_started,
            "inference": inference_stats(),
            "benchmark": ({
                "compile_warmup_seconds": compile_warmup_seconds,
                "warmup_seconds": warmup_seconds,
                "requested_seconds": benchmark_seconds,
                "owner_window": {"start": window_start, "end": window_end or {
                    "at": time.perf_counter(), "cpu_seconds": _cpu_seconds(),
                    "gather_seconds": owner_gather,
                    "dispatch_seconds": owner_dispatch,
                    "wait_input_seconds": owner_wait,
                    "wait_coalesce_seconds": owner_coalesce_wait,
                    "inference": inference_stats()}},
                "worker_windows": [
                    {"worker_id": item["worker_id"], **item["window"]}
                    for item in sorted(results, key=lambda result: result["worker_id"])
                ],
            } if benchmark_seconds is not None else None),
            "workers": [
                {"worker_id": item["worker_id"],
                 "pid": item["pid"], "cuda_initialized": item["cuda_initialized"],
                 "game_ids": [game["game_id"] for game in item["completed"]],
                 "cpu_seconds": item["cpu_seconds"],
                 "completed_simulations": sum(game["simulations"] for game in item["completed"]),
                 "in_progress_simulations_at_exit": item["in_progress_simulations_at_exit"],
                 "peak_active_games": item["peak_active_games"],
                 "wait_result_seconds": item["wait_result_seconds"],
                 "queue_backpressure_seconds": item["queue_backpressure_seconds"]}
                for item in sorted(results, key=lambda result: result["worker_id"])
            ],
            "worker_row_statistics": [{"worker_id": item["worker_id"],
                "actual_width_rows_by_root_stage": item["actual_width_rows_by_root_stage"],
                "descriptor_width_rows": item["descriptor_width_rows"]} for item in results],
            "gpu_peak_bytes": (torch.cuda.max_memory_allocated() if any(
                item.device.type == "cuda" for item in evaluators.values()) else 0),
            "owner_cpu_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        }
        succeeded = True
        return results, owner_stats
    except BaseException as error:
        failure_path = root / "failures" / "owner.json"
        if not failure_path.exists():
            try:
                _atomic_json(failure_path, {
                    "protocol": "native-cuda-pool-failure-v1", "origin": "collector",
                    "error_type": type(error).__name__, "error_detail": str(error),
                    "traceback": traceback.format_exc(),
                    "requests": [
                        {"worker_id": part[1], "slot": part[2], "generation": part[3],
                         "row_offset": part[4], "rows": part[6]}
                        for part in locals().get("parts", [])
                    ],
                })
            except Exception:
                pass
        for reply in replies:
            try:
                reply.put_nowait(("abort", f"{type(error).__name__}: {error}"))
            except queue.Full:
                pass
        raise
    finally:
        if inflight is not None:
            inflight["handle"].discard()
        if not succeeded or benchmark_seconds is not None:
            tasks.cancel_join_thread()
            for reply in replies:
                reply.cancel_join_thread()
        for process in processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        for slot in created:
            slot.close()
            slot.unlink()
        tasks.close()
        ready.close()
        for reply in replies:
            reply.close()
