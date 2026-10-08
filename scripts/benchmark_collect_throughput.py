"""Timed-window throughput benchmark over the real self-play workers.

Runs the production collection loop (legacy or pooled) for one common
wall-clock window and reports live increments (simulations, network positions,
physical batches, native batch rows) instead of whole-game totals, so a window
that ends mid-game still measures real work.  Phases are separated and each is
reported on its own clock:

* process start and worker initialization (model load),
* warm-up on the real configuration (covers the compiled graphs in use),
* one common measurement window started after every worker reported ready,
* tail/drain after the stop signal plus the parent's join time.

Counters and CPU time are sampled inside each worker at the window boundaries,
so CPU microseconds per simulation never mix in initialization, warm-up or
Manager overhead.  If ``--games-per-worker`` is too small the run stops early;
the actual (shorter) window is reported instead of being mislabelled.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import queue as queue_module
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from splendor.search_inference import FixedValueSearchInference
from splendor.selfplay_fast import FastSelfPlayConfig, _collect_group, build_manifest

_STAT_QUEUE: Any = None
_INT_COUNTERS = ("simulations", "network_positions", "physical_batches", "completed_games",
                 "native_batch_rows", "native_batch_batches", "native_wide_batches",
                 "native_root_rows", "native_policy_rows", "native_value_rows")
_FLOAT_COUNTERS = ("feature_seconds", "forward_seconds", "h2d_seconds", "d2h_seconds", "wait_seconds",
                   "native_fill_seconds", "native_consume_seconds")


def _worker_init(queue: Any) -> None:
    global _STAT_QUEUE
    _STAT_QUEUE = queue


def _emit_stat(sample: dict[str, Any]) -> None:
    if _STAT_QUEUE is not None:
        _STAT_QUEUE.put(sample)


class _WindowControl:
    """Worker-side view of the shared benchmark window."""

    def __init__(self, ready_queue: Any, start_event: Any, stop_event: Any,
                 warmup_seconds: float) -> None:
        self.warmup_seconds = warmup_seconds
        self._ready_queue = ready_queue
        self._start = start_event
        self._stop = stop_event

    def ready(self, payload: dict[str, Any]) -> None:
        self._ready_queue.put(payload)

    def wait_start(self) -> None:
        self._start.wait()

    def stop_requested(self) -> bool:
        return self._stop.is_set()


class _InProcessWindow:
    """Single-process window: optional warm-up, then one measured interval."""

    def __init__(self, warmup_seconds: float, window_seconds: float) -> None:
        self.warmup_seconds = warmup_seconds
        self.window_seconds = window_seconds
        self.ready_payload: dict[str, Any] | None = None
        self._stop_at: float | None = None

    def ready(self, payload: dict[str, Any]) -> None:
        self.ready_payload = payload

    def wait_start(self) -> None:
        self._stop_at = time.perf_counter() + self.window_seconds

    def stop_requested(self) -> bool:
        return self._stop_at is not None and time.perf_counter() >= self._stop_at


def _collect(config: FastSelfPlayConfig, root: Path, group: tuple[int, ...],
             fingerprint: str, checkpoint_sha: str, window: _WindowControl,
             evaluator_label: str) -> dict[str, Any]:
    evaluator = FixedValueSearchInference() if evaluator_label == "fixed" else None
    return _collect_group(config, root, group, fingerprint, checkpoint_sha,
                          stat_callback=_emit_stat, evaluator=evaluator,
                          window_control=window)


def _window_delta(window: dict[str, Any]) -> dict[str, Any]:
    start, end = window["start"], window["end"]
    result: dict[str, Any] = {name: end.get(name, 0) - start.get(name, 0) for name in _INT_COUNTERS}
    for name in _FLOAT_COUNTERS:
        result[name] = end.get(name, 0.0) - start.get(name, 0.0)
    histogram: dict[str, int] = {}
    for key, value in end["batch_sizes"].items():
        delta = value - start["batch_sizes"].get(key, 0)
        if delta:
            histogram[key] = delta
    result["batch_sizes"] = {key: value for key, value in sorted(histogram.items(), key=lambda item: int(item[0]))}
    result["batches"] = sum(result["batch_sizes"].values())
    result["in_window_compile_graphs"] = sorted(
        set(end.get("compile_seconds", {})) - set(start.get("compile_seconds", {})))
    result["cpu_seconds"] = window["cpu_seconds"]
    result["measured_seconds"] = window["ended_at"] - window["started_at"]
    result["tail_seconds"] = window["tail_seconds"]
    result["peak_inflight_rows"] = max(start.get("native_peak_inflight_rows", 0),
                                       end.get("native_peak_inflight_rows", 0))
    result["active_games_last"] = end["active_games"]
    return result


def _summarize(results: list[dict[str, Any]], groups: tuple[tuple[int, ...], ...], *,
               requested_seconds: float, warmup_seconds: float) -> dict[str, Any]:
    windows = [_window_delta(item["window"]) for item in results if item.get("window")]
    measured = [(group, result) for group, result in zip(groups, results) if result.get("window")]
    workers = [
        {
            "worker_id": group[0],
            "initialization_seconds": result["initialization_seconds"],
            "wall_seconds": result["wall_seconds"],
            "peak_active_games": result["peak_active_games"],
            "peak_pending_requests": result["peak_pending_requests"],
            **window,
        }
        for (group, result), window in zip(measured, windows)
    ]
    starts = [item["window"]["started_at"] for item in results if item.get("window")]
    ends = [item["window"]["ended_at"] for item in results if item.get("window")]
    common_seconds = max(0.0, min(ends) - max(starts)) if starts else 0.0
    totals: dict[str, float] = {}
    for name in _INT_COUNTERS + ("batches",):
        totals[name] = float(sum(window[name] for window in windows))
    for name in _FLOAT_COUNTERS:
        totals[name] = sum(window[name] for window in windows)
    cpu_seconds = sum(window["cpu_seconds"] for window in windows)
    rates = {f"{name}_per_second": totals[name] / common_seconds if common_seconds else 0.0
             for name in ("simulations", "network_positions", "physical_batches")}
    rates["batches_per_second"] = totals["batches"] / common_seconds if common_seconds else 0.0
    rates["mean_batch_size"] = totals["network_positions"] / totals["physical_batches"] if totals["physical_batches"] else 0.0
    rates["simulations_per_second_sum_of_workers"] = sum(
        window["simulations"] / window["measured_seconds"] for window in windows if window["measured_seconds"])
    histogram: dict[str, float] = {}
    for window in windows:
        for key, value in window["batch_sizes"].items():
            histogram[key] = histogram.get(key, 0) + value
    recompiled = sorted({name for window in windows for name in window["in_window_compile_graphs"]})
    return {
        "requested_window_seconds": requested_seconds,
        "warmup_seconds": warmup_seconds,
        "common_seconds": common_seconds,
        "workers_measured": len(windows),
        "workers_exhausted_games": sum(1 for item in results if not item.get("window")),
        "tails_seconds": [window["tail_seconds"] for window in windows],
        "recompiled_graphs_in_window": recompiled,
        "batch_sizes": {key: int(value) for key, value in sorted(histogram.items(), key=lambda item: int(item[0]))},
        "workers": workers,
        "totals": totals,
        "rates": rates,
        "cpu_seconds": cpu_seconds,
        "cpu_seconds_per_simulation": cpu_seconds / totals["simulations"] if totals["simulations"] else None,
        "cpu_us_per_simulation": 1e6 * cpu_seconds / totals["simulations"] if totals["simulations"] else None,
    }


def _run_pool(config: FastSelfPlayConfig, *, seconds: float, scratch: Path,
              warmup_seconds: float, label: str) -> dict[str, Any]:
    from splendor.multiworker_gpu import FixedPoolOwner, collect

    root = scratch / f"{config.run_id}-{time.time_ns()}"
    root.mkdir(parents=True)
    manifest = build_manifest(config)
    started = time.perf_counter()
    results, owner = collect(
        config, root, tuple(range(config.num_games)), manifest["semantic_fingerprint"],
        manifest["checkpoint_identity"]["sha256"],
        evaluator=FixedPoolOwner() if label == "fixed" else None,
        benchmark_seconds=seconds, warmup_seconds=warmup_seconds,
    )
    elapsed = time.perf_counter() - started
    benchmark = owner["benchmark"]
    owner_start = benchmark["owner_window"]["start"]
    owner_end = benchmark["owner_window"]["end"]
    if owner_start is None:
        measured = 0.0
        positions = batches = 0
        owner_cpu = 0.0
        histogram = {}
        owner_timings = {}
    else:
        measured = owner_end["at"] - owner_start["at"]
        first, last = owner_start["inference"], owner_end["inference"]
        positions = last["logical_model_positions"] - first["logical_model_positions"]
        batches = last["physical_model_batches"] - first["physical_model_batches"]
        owner_cpu = owner_end["cpu_seconds"] - owner_start["cpu_seconds"]
        histogram = {
            key: value - first["batch_size_histogram"].get(key, 0)
            for key, value in last["batch_size_histogram"].items()
            if value - first["batch_size_histogram"].get(key, 0)
        }
        owner_timings = {
            name: owner_end[name] - owner_start[name]
            for name in ("gather_seconds", "dispatch_seconds", "wait_input_seconds",
                         "wait_coalesce_seconds")
        }
        owner_timings.update({
            name: last[name] - first[name]
            for name in ("h2d_seconds", "forward_seconds", "d2h_seconds", "wait_seconds")
        })
    workers = []
    simulations = worker_cpu = 0.0
    for item in benchmark["worker_windows"]:
        first, last = item["start"], item["end"]
        if first is None:
            continue
        delta_sim = last["simulations"] - first["simulations"]
        delta_cpu = last["cpu_seconds"] - first["cpu_seconds"]
        simulations += delta_sim
        worker_cpu += delta_cpu
        workers.append({
            "worker_id": item["worker_id"],
            "started_at": first["at"], "ended_at": last["at"],
            "measured_seconds": last["at"] - first["at"],
            "simulations": delta_sim, "cpu_seconds": delta_cpu,
            "completed_games": last["completed_games"] - first["completed_games"],
            "network_requests": last["exported_rows"] - first["exported_rows"],
            "native_batches": last["batches"] - first["batches"],
            "wait_result_seconds": last["wait_result_seconds"] - first["wait_result_seconds"],
            "queue_backpressure_seconds": (
                last["queue_backpressure_seconds"] - first["queue_backpressure_seconds"]),
            "active_games_start": first["active_games"],
            "active_games_end": last["active_games"],
            "root_decisions": last["root_decisions"] - first["root_decisions"],
        })
    common_start = max([owner_start["at"]] + [item["started_at"] for item in workers]) if owner_start and workers else 0
    common_end = min([owner_end["at"]] + [item["ended_at"] for item in workers]) if owner_start and workers else 0
    common = max(0.0, common_end - common_start)
    worker_rate = sum(item["simulations"] / item["measured_seconds"]
                      for item in workers if item["measured_seconds"] > 0)
    return {
        "run_id": config.run_id, "backend": config.execution.backend,
        "checkpoint": config.checkpoint, "execution": asdict(config.execution),
        "simulations": config.search_config.simulations,
        "determinizations": config.search_config.determinizations,
        "evaluator": label, "process_seconds": elapsed,
        "worker_compile": [owner["inference"]["compile"]],
        "worker_wall_seconds": [item["wall_seconds"] for item in results],
        "worker_initialization_seconds": [item["initialization_seconds"] for item in results],
        "worker_gpu_peak_bytes": [0] * owner["actual_search_workers"],
        "gpu_owner": owner, "workers": workers,
        "clock": {
            "requested_window_seconds": seconds,
            "warmup_seconds": warmup_seconds,
            "common_seconds": common,
            "owner_measured_seconds": measured,
            "worker_measured_seconds": {str(item["worker_id"]): item["measured_seconds"]
                                        for item in workers},
            "rate_basis": "owner counts / owner interval; summed worker rates / individual worker intervals",
            "batch_sizes": histogram,
            "owner_timings": owner_timings,
            "cpu_seconds": worker_cpu + owner_cpu,
            "owner_cpu_seconds": owner_cpu,
            "worker_cpu_seconds": worker_cpu,
            "cpu_us_per_simulation": None,
            "worker_cpu_us_per_simulation": (1e6 * worker_cpu / simulations
                                              if simulations else None),
            "owner_cpu_us_per_network_position": (1e6 * owner_cpu / positions
                                                   if positions else None),
            "totals": {"simulations": simulations, "network_positions": positions,
                       "physical_batches": batches,
                       "completed_games": sum(item["completed_games"] for item in workers)},
            "rates": {
                "simulations_per_second": worker_rate,
                "network_positions_per_second": positions / measured if measured else 0.0,
                "physical_batches_per_second": batches / measured if measured else 0.0,
                "mean_batch_size": positions / batches if batches else 0.0,
            },
        },
    }


def run(config: FastSelfPlayConfig, *, seconds: float, scratch: Path,
        warmup_seconds: float = 0.0, dump_samples: bool = False,
        evaluator: Any = None, label: str = "real") -> dict[str, Any]:
    if config.execution.backend == "native_cuda_pool":
        return _run_pool(config, seconds=seconds, scratch=scratch,
                         warmup_seconds=warmup_seconds, label=label)
    workers = config.execution.workers
    root = scratch / config.run_id
    root.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(config)
    fingerprint = manifest["semantic_fingerprint"]
    checkpoint_sha = manifest["checkpoint_identity"]["sha256"]
    groups = tuple(tuple(range(index, config.num_games, workers)) for index in range(workers))
    groups = tuple(group for group in groups if group)

    process_started = time.perf_counter()
    samples: list[dict[str, Any]] = []
    ready_payloads: list[dict[str, Any]] = []
    ready_seconds: float | None = None
    stop_seconds: float | None = None
    if len(groups) == 1:
        # In-process single worker: shares the parent clock and allows a live
        # evaluator (used by the replay profiler).
        window = _InProcessWindow(warmup_seconds, seconds)
        results = [_collect_group(config, root, groups[0], fingerprint, checkpoint_sha,
                                  stat_callback=samples.append if dump_samples else None,
                                  evaluator=evaluator, window_control=window)]
        ready_payloads = [window.ready_payload or {}]
        ready_seconds = window.ready_payload.get("warmup_seconds", 0.0) if window.ready_payload else 0.0
    else:
        with mp.Manager() as manager:
            ready_queue = manager.Queue()
            stat_queue = manager.Queue()
            start_event = manager.Event()
            stop_event = manager.Event()
            with ProcessPoolExecutor(max_workers=len(groups), mp_context=mp.get_context("spawn"),
                                     initializer=_worker_init,
                                     initargs=(stat_queue if dump_samples else None,)) as pool:
                futures = [
                    pool.submit(_collect, config, root, group, fingerprint, checkpoint_sha,
                                _WindowControl(ready_queue, start_event, stop_event, warmup_seconds),
                                label if label in ("fixed",) else "real")
                    for group in groups
                ]
                while len(ready_payloads) < len(groups):
                    ready_payloads.append(ready_queue.get(timeout=1800.0))
                ready_seconds = time.perf_counter() - process_started
                start_event.set()
                time.sleep(seconds)
                stop_seconds = time.perf_counter()
                stop_event.set()
                results = [future.result(timeout=900.0) for future in futures]
            if dump_samples:
                while True:
                    try:
                        samples.append(stat_queue.get_nowait())
                    except queue_module.Empty:
                        break
    wall_seconds = time.perf_counter() - process_started
    summary = _summarize(results, groups, requested_seconds=seconds, warmup_seconds=warmup_seconds)
    if stop_seconds is not None and ready_seconds is not None:
        summary["parent_timeline"] = {
            "process_seconds": wall_seconds,
            "ready_seconds": ready_seconds,
            "requested_window_seconds": seconds,
            "stop_after_ready_seconds": stop_seconds - (process_started + ready_seconds),
            "join_seconds": wall_seconds - (stop_seconds - process_started),
        }
    return {
        "run_id": config.run_id,
        "backend": config.execution.backend,
        "checkpoint": str(Path(config.checkpoint).resolve()),
        "simulations": config.search_config.simulations,
        "determinizations": config.search_config.determinizations,
        "execution": asdict(config.execution),
        "evaluator": label,
        "torch_threads": torch.get_num_threads(),
        "games_per_worker": config.num_games // config.execution.workers,
        "process_seconds": wall_seconds,
        "worker_wall_seconds": [item["wall_seconds"] for item in results],
        "worker_initialization_seconds": [item["initialization_seconds"] for item in results],
        "worker_inference": [item["inference"] for item in results],
        "worker_peak_active_games": [item["peak_active_games"] for item in results],
        "worker_peak_pending_requests": [item["peak_pending_requests"] for item in results],
        "worker_gpu_peak_bytes": [item["gpu_peak_bytes"] for item in results],
        "worker_compile": [item["inference"].get("compile", {}) for item in results],
        "worker_ready": ready_payloads,
        "cpu_peak_rss_kib": max((item["cpu_peak_rss_kib"] for item in results), default=0),
        "clock": summary,
        **({"samples": samples} if dump_samples else {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, default=Path("runtime/inference_throughput/scratch"))
    parser.add_argument("--seconds", type=float, default=45.0)
    parser.add_argument("--warmup-seconds", type=float, default=5.0)
    parser.add_argument("--games-per-worker", type=int, default=24)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--active", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--torch-threads", type=int, default=None)
    parser.add_argument("--batch-interface", default=None, choices=("legacy", "native"))
    parser.add_argument("--inference-mode", default=None, choices=("eager", "compiled"))
    parser.add_argument("--actor-projection", default=None, choices=("on", "off"))
    parser.add_argument("--candidate-buckets", default=None,
                        help="comma separated candidate width buckets, for example 16,32,48")
    parser.add_argument("--compile-mode", default=None)
    parser.add_argument("--batch-buckets", default=None,
                        help="comma separated static batch sizes; requires compiled mode")
    parser.add_argument("--compile-failure-policy", default=None, choices=("fallback", "fail_fast"))
    parser.add_argument("--native-transport", default=None, choices=("reused", "packed"))
    parser.add_argument("--slots", type=int, default=None, help="native GPU owner pipeline slots (1 or 2)")
    parser.add_argument("--search-workers", type=int, default=None)
    parser.add_argument("--total-active", type=int, default=None)
    parser.add_argument("--local-microbatch", type=int, default=None)
    parser.add_argument("--shared-slots", type=int, default=None)
    parser.add_argument("--shared-candidate-width", type=int, default=None)
    parser.add_argument("--dump-samples", action="store_true")
    parser.add_argument("--evaluator", default="real", choices=("real", "fixed"),
                        help="real network or the instant fixed policy/value double")
    args = parser.parse_args()

    raw = json.loads(args.config.read_text(encoding="utf-8"))
    execution = raw.setdefault("execution", {})
    for name, value in (("backend", args.backend), ("workers", args.workers),
                        ("active_games_per_worker", args.active),
                        ("max_batch_size", args.batch),
                        ("torch_threads", args.torch_threads),
                        ("batch_interface", args.batch_interface),
                        ("inference_mode", args.inference_mode),
                        ("compile_mode", args.compile_mode),
                        ("compile_failure_policy", args.compile_failure_policy),
                        ("native_transport", args.native_transport),
                        ("pipeline_slots", args.slots),
                        ("search_workers", args.search_workers),
                        ("total_active_games", args.total_active),
                        ("local_microbatch", args.local_microbatch),
                        ("shared_slots", args.shared_slots),
                        ("shared_candidate_width", args.shared_candidate_width)):
        if value is not None:
            execution[name] = value
    if args.actor_projection is not None:
        execution["actor_projection"] = args.actor_projection == "on"
    if args.candidate_buckets is not None:
        execution["candidate_buckets"] = [int(part) for part in args.candidate_buckets.split(",")]
    if args.batch_buckets is not None:
        execution["batch_buckets"] = [int(part) for part in args.batch_buckets.split(",")]
    pooled = execution.get("backend") == "native_cuda_pool"
    workers = execution.get("search_workers", 1) if pooled else execution.get("workers", 1)
    active = execution.get("total_active_games", 1) if pooled else execution.get("active_games_per_worker", 1)
    supplied_games = workers * args.games_per_worker if pooled else args.games_per_worker
    if supplied_games < active:
        raise SystemExit(
            f"available games={supplied_games} is below active={active}; "
            "the run would never reach the configured concurrency"
        )
    raw["num_games"] = workers * args.games_per_worker
    config = FastSelfPlayConfig.from_dict(raw)

    evaluator = FixedValueSearchInference() if args.evaluator == "fixed" else None
    result = run(config, seconds=args.seconds, scratch=args.scratch, warmup_seconds=args.warmup_seconds,
                 dump_samples=args.dump_samples, evaluator=evaluator, label=args.evaluator)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    clock = result["clock"]
    print(json.dumps({**clock["rates"], "cpu_seconds": clock["cpu_seconds"],
                      "cpu_us_per_simulation": clock["cpu_us_per_simulation"],
                      **({"worker_cpu_us_per_simulation": clock["worker_cpu_us_per_simulation"],
                          "owner_cpu_us_per_network_position": clock["owner_cpu_us_per_network_position"]}
                         if "worker_cpu_us_per_simulation" in clock else {}),
                      "common_seconds": clock["common_seconds"],
                      "requested_window_seconds": clock["requested_window_seconds"]},
                     sort_keys=True))
    print(json.dumps({"process_seconds": result["process_seconds"],
                      "games_committed": clock["totals"]["completed_games"],
                      "worker_wall_seconds": result["worker_wall_seconds"],
                      "compile": [row.get("disabled_graphs", {}) for row in result["worker_compile"]]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
