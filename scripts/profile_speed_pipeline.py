"""Bounded, opt-in diagnostics for collection, AZ replay/learner and evaluation.

Production algorithms are called unchanged. Timing wrappers exist only in this
process (and explicitly spawned collection workers). CUDA stream intervals and
host wall intervals are reported separately; they must not be added together.
Heavy profiles and generated game/checkpoint files belong in --scratch.
"""
from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
import os
import platform
import pstats
import resource
import subprocess
import sys
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from splendor import actor_critic, alphazero_replay, alphazero_train, evaluation, search_inference
from splendor import multiworker_gpu as pool
from splendor.alphazero_loop import LoopConfig
from splendor.alphazero_replay import ReplayRun, build_replay_index
from splendor.alphazero_train import AlphazeroLearner, draw_training_batch
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.search_inference import FrozenSearchInference, InferenceConfig
from splendor.selfplay_fast import FastSelfPlayConfig, build_manifest, validate_run


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def provenance() -> dict[str, Any]:
    return {
        "command": sys.argv,
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], text=True),
        "script_sha256": sha256(Path(__file__)),
        "python": platform.python_version(), "torch": torch.__version__,
        "cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "cpu_count": os.cpu_count(), "torch_threads": torch.get_num_threads(),
        "meminfo": {line.split(":")[0]: line.split(":")[1].strip()
                    for line in Path("/proc/meminfo").read_text().splitlines()
                    if line.startswith(("MemTotal:", "MemAvailable:"))},
    }


def profile_rows(profile: cProfile.Profile, limit: int = 40) -> list[dict[str, Any]]:
    if not profile.getstats():
        return []
    rows = []
    for (filename, line, name), (primitive, calls, own, cumulative, _) in pstats.Stats(profile).stats.items():
        rows.append({"function": f"{filename}:{line}:{name}", "calls": calls,
                     "primitive_calls": primitive, "self_wall_seconds": own,
                     "inclusive_wall_seconds": cumulative})
    return sorted(rows, key=lambda row: row["self_wall_seconds"], reverse=True)[:limit]


class Timings:
    def __init__(self, cuda: bool = False) -> None:
        self.host: Counter[str] = Counter()
        self.calls: Counter[str] = Counter()
        self.events: list[tuple[str, Any, Any]] = []
        self.cuda = cuda

    @contextmanager
    def stage(self, name: str):
        device_stage = self.cuda and name in {
            "target_h2d", "feature_h2d", "forward", "loss", "backward", "gradient_clip", "optimizer",
        }
        if device_stage:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
        began = time.perf_counter()
        try:
            yield
        finally:
            self.host[name] += time.perf_counter() - began
            self.calls[name] += 1
            if device_stage:
                end.record()
                self.events.append((name, start, end))

    def result(self) -> dict[str, Any]:
        device: Counter[str] = Counter()
        if self.cuda:
            torch.cuda.synchronize()
            for name, start, end in self.events:
                device[name] += start.elapsed_time(end) / 1000
        return {"host_wall_seconds": dict(self.host), "calls": dict(self.calls),
                "cuda_stream_seconds": dict(device)}


@contextmanager
def wrapped(target: Any, name: str, timer: Timings, label: str):
    namespace = vars(target)
    had_attribute = name in namespace
    previous = namespace.get(name)
    original = getattr(target, name)

    def call(*args, **kwargs):
        with timer.stage(label):
            return original(*args, **kwargs)

    setattr(target, name, call)
    try:
        yield
    finally:
        if had_attribute:
            setattr(target, name, previous)
        else:
            delattr(target, name)


def learner_profile(args) -> dict[str, Any]:
    from contextlib import ExitStack
    from splendor.selfplay_data import collate_selfplay_samples

    # LoopRunner uses absolute run roots; match the persisted verification keys.
    args.run = args.run.resolve()
    config_raw = json.loads((args.run / "config.json").read_text())["config"]
    config = LoopConfig.from_dict(config_raw)
    learner_config = replace(config.learner, device=args.device)
    state_copy = args.scratch / "replay_index.json"
    # Keep the original derived index and every old checkpoint untouched.
    state_copy.write_bytes((args.run / "replay_index.json").read_bytes())
    started = time.perf_counter()
    index = build_replay_index(
        [ReplayRun(r, f"{config.run_id}-round-{r:03d}", args.run / "rounds" / f"round-{r:03d}" / "selfplay")
         for r in range(args.round + 1)], max_games=config.replay_max_games,
        cache_bytes=args.cache_mib * 1024**2, state_path=state_copy,
    )
    index_seconds = time.perf_counter() - started
    current = index.position_pool(split="train", round_index=args.round)
    history = index.position_pool(split="train", before_round=args.round)
    validation_games = [g for g in index.split_games("validation") if g.round_index == args.round]
    began = time.perf_counter()
    validation_samples = [index.cache.sample(g, p) for g in validation_games for p in range(g.decision_count)]
    validation_prepare_seconds = time.perf_counter() - began
    parent = (Path(config.initial_checkpoint) if args.round == 0 else
              args.run / "rounds" / f"round-{args.round - 1:03d}" / "checkpoints/final.pt")
    parameters = dict(sampling_seed=config.seeds.sampling_for(args.round), replay_identity=index.identity(),
                      parent_checkpoint={"path": str(parent), "sha256": sha256(parent)},
                      round_index=args.round, round_steps_target=learner_config.steps_for_round(len(current)))
    began = time.perf_counter()
    learner = (AlphazeroLearner.warm_start(parent, learner_config, **parameters) if args.round == 0 else
               AlphazeroLearner.continue_from(parent, learner_config, **parameters))
    if args.device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    initialization_seconds = time.perf_counter() - began
    cache_counts: Counter[str] = Counter()
    original_sample = index.cache.sample
    original_load = alphazero_replay.load_game
    load_seconds = 0.0

    def sample(game, position):
        cache_counts["hits" if game.key in index.cache._entries else "misses"] += 1
        return original_sample(game, position)

    def load(path):
        nonlocal load_seconds
        before = time.perf_counter()
        arrays, metadata = original_load(path)
        load_seconds += time.perf_counter() - before
        cache_counts["decoded_bytes"] += sum(a.nbytes for a in arrays.values())
        return arrays, metadata

    index.cache.sample = sample
    alphazero_replay.load_game = load
    records = []
    reference_hash = hashlib.sha256()
    prof = cProfile.Profile()
    cold_seconds = 0.0
    try:
        for step in range(args.warmup_steps + args.steps):
            timer = Timings(cuda=args.cprofile and args.device == "cuda" and step >= args.warmup_steps)
            before_counts = cache_counts.copy()
            before_load = load_seconds
            begin = time.perf_counter()
            with timer.stage("draw"):
                drawn = draw_training_batch(current, history, learner.sampling_rng, learner_config)
            reference_hash.update(json.dumps([r.to_dict() for r in drawn.references], sort_keys=True).encode())
            with timer.stage("collate"):
                batch = collate_selfplay_samples(drawn.samples)
            with ExitStack() as stack:
                for target, name, label in (
                    (alphazero_train, "_torch_batch", "target_h2d"),
                    (actor_critic, "tensorize_batch", "feature_h2d"),
                    (actor_critic.ActorCritic, "_forward_prepared", "forward"),
                    (alphazero_train, "alphazero_loss_sums", "loss"),
                    (torch.Tensor, "backward", "backward"),
                    (alphazero_train, "clip_grad_norm_", "gradient_clip"),
                    (learner.optimizer, "step", "optimizer"),
                ):
                    stack.enter_context(wrapped(target, name, timer, label))
                if args.cprofile and step >= args.warmup_steps:
                    prof.enable()
                with timer.stage("train_step"):
                    metric = learner.train_step(batch, references=drawn.references)
                prof.disable()
            with timer.stage("completion_wait"):
                if args.device == "cuda":
                    torch.cuda.synchronize()
            with timer.stage("metrics_write"):
                alphazero_train._append_jsonl(args.scratch / "training_metrics.jsonl", metric)
            wall = time.perf_counter() - begin
            if step < args.warmup_steps:
                cold_seconds += wall
            else:
                result = timer.result()
                # train_step is inclusive; expose its unwrapped remainder explicitly.
                h = result["host_wall_seconds"]
                wrapped_sum = sum(h.get(k, 0) for k in ("target_h2d", "feature_h2d", "forward", "loss",
                                                      "backward", "gradient_clip", "optimizer"))
                result["train_step_remainder_seconds"] = h["train_step"] - wrapped_sum
                result.update(step=step, wall_seconds=wall, decode_seconds=load_seconds - before_load,
                              cache_delta=dict(cache_counts - before_counts), cache_bytes=index.cache.bytes,
                              cache_entries=index.cache.entries, candidate_width=batch.decisions.candidate_mask.shape[1],
                              candidate_positions=int(batch.decisions.candidate_mask.sum()),
                              physical_candidate_positions=int(batch.decisions.candidate_mask.size),
                              policy_samples=int((batch.policy_weights > 0).sum()), total_loss=metric["total_loss"])
                records.append(result)
    finally:
        alphazero_replay.load_game = original_load
        index.cache.sample = original_sample
    timer = Timings(cuda=args.device == "cuda")
    with timer.stage("checkpoint"):
        learner.save(args.scratch / "diagnostic.pt", metadata={"diagnostic_only": True})
    with timer.stage("validation"):
        validation = learner.evaluate_samples(validation_samples)
    costs = timer.result()
    prof.dump_stats(str(args.scratch / "train_step.pstats"))
    summed: Counter[str] = Counter()
    stream: Counter[str] = Counter()
    for r in records:
        summed.update(r["host_wall_seconds"])
        stream.update(r["cuda_stream_seconds"])
    total = sum(r["wall_seconds"] for r in records)
    exclusive = {k:v for k,v in summed.items() if k != "train_step"}
    exclusive["train_step_remainder"] = sum(r["train_step_remainder_seconds"] for r in records)
    exclusive["measurement_and_loop_remainder"] = total - sum(exclusive.values())
    return {
        "round": args.round, "cache_mib": args.cache_mib, "device": args.device,
        "learner_config": asdict(learner_config), "sampling_seed": config.seeds.sampling_for(args.round),
        "parent_checkpoint": {"path": str(parent), "sha256": sha256(parent)}, "replay_identity": index.identity(),
        "index_seconds": index_seconds, "validation_prepare_seconds": validation_prepare_seconds,
        "initialization_seconds": initialization_seconds, "warmup_steps": args.warmup_steps,
        "warmup_seconds": cold_seconds, "measured_steps": args.steps, "measured_wall_seconds": total,
        "exclusive_host_wall_seconds": exclusive, "cuda_stream_seconds": dict(stream),
        "cache_total": dict(cache_counts), "reference_sequence_sha256": reference_hash.hexdigest(),
        "production_cache_stats": index.cache.stats() if hasattr(index.cache, "stats") else None,
        "profiler_enabled": args.cprofile,
        "checkpoint_validation": costs, "validation": validation,
        "top_train_step_functions": profile_rows(prof), "steps": records,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "gpu_peak_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else 0,
        "scope": "isolated updates, actual frozen replay and parent optimizer; no production run modified",
    }


def replay_profile(args) -> dict[str, Any]:
    """Separate function profile; its overhead is excluded from learner ratios."""
    from splendor.rng import DeterministicRng
    args.run = args.run.resolve()
    config = LoopConfig.from_dict(json.loads((args.run / "config.json").read_text())["config"])
    state_copy = args.scratch / "replay_index.json"
    state_copy.write_bytes((args.run / "replay_index.json").read_bytes())
    index = build_replay_index(
        [ReplayRun(r, f"{config.run_id}-round-{r:03d}", args.run / "rounds" / f"round-{r:03d}" / "selfplay")
         for r in range(args.round + 1)], max_games=config.replay_max_games,
        cache_bytes=args.cache_mib * 1024**2, state_path=state_copy,
    )
    current = index.position_pool(split="train", round_index=args.round)
    history = index.position_pool(split="train", before_round=args.round)
    rng = DeterministicRng(config.seeds.sampling_for(args.round))
    for _ in range(args.warmup_steps):
        drawn = draw_training_batch(current, history, rng, config.learner)
    prof = cProfile.Profile()
    started = time.perf_counter()
    for _ in range(args.steps):
        drawn = prof.runcall(draw_training_batch, current, history, rng, config.learner)
    wall = time.perf_counter() - started
    prof.dump_stats(str(args.scratch / "replay.pstats"))
    return {"round": args.round, "cache_mib": args.cache_mib, "steps": args.steps,
            "warmup_steps": args.warmup_steps, "wall_seconds": wall, "replay_identity": index.identity(),
            "top_functions": profile_rows(prof, 60), "scope": "separate read-only replay cProfile; no learner updates"}


_ORIGINAL_WORKER = pool._worker


def profiled_worker(*args, **kwargs):
    """Spawn-safe worker wrapper: count real leaf widths without changing rows."""
    root, worker_id = args[1], args[2]
    native = pool._extension()
    original_batch = native.NativeBatch
    original_ensure = pool._GameSession.ensure_job
    stages = {}
    rows_by_stage = defaultdict(Counter)
    descriptor_widths = Counter()
    prof = cProfile.Profile()

    def ensure(session):
        previous_job = session.job
        result = original_ensure(session)
        if session.job is not None and session.job is not previous_job:
            turns = session.stepper.completed_player_turns
            stages[id(session.job.core)] = "opening" if turns < 10 else "middle" if turns < 50 else "late"
        return result

    class Batch:
        def __init__(self, *a, **kw):
            self.inner = original_batch(*a, **kw)
            self.row_stages = []
            self.counted = False

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def fill(self, core):
            if self.inner.row_count() == 0:
                self.row_stages = []
                self.counted = False
            added = self.inner.fill(core)
            self.row_stages.extend([stages[id(core)]] * added)
            return added

        def row_ids_array(self):
            ids = self.inner.row_ids_array()
            if not self.counted:
                self.counted = True
                width = 0 if self.inner.is_state_only() else self.inner.padded_width()
                descriptor_widths[str(width)] += len(ids)
                for stage in set(self.row_stages):
                    widths = ids[np.asarray(self.row_stages) == stage, 3]
                    unique, counts = np.unique(widths, return_counts=True)
                    rows_by_stage[stage].update({str(int(w)):int(n) for w,n in zip(unique, counts)})
            return ids

    pool._GameSession.ensure_job = ensure
    native.NativeBatch = Batch
    try:
        if os.environ.get("SPLENDOR_SPEED_CPROFILE") == "1":
            prof.runcall(_ORIGINAL_WORKER, *args, **kwargs)
        else:
            _ORIGINAL_WORKER(*args, **kwargs)
    finally:
        pool._GameSession.ensure_job = original_ensure
        native.NativeBatch = original_batch
        path = root / "profiling" / f"worker-{worker_id}.json"
        write_json(path, {"rows_by_root_stage": dict(rows_by_stage), "descriptor_width_rows": dict(descriptor_widths),
                          "top_functions": profile_rows(prof)})
        prof.dump_stats(str(path.with_suffix(".pstats")))


def collection_profile(args) -> dict[str, Any]:
    raw = json.loads(args.config.read_text())
    if args.games is not None:
        raw["num_games"] = args.games
    if args.active is not None:
        raw["execution"]["total_active_games"] = args.active
    if args.batch is not None:
        raw["execution"]["max_batch_size"] = args.batch
    config = FastSelfPlayConfig.from_dict(raw)
    if config.execution.backend != "native_cuda_pool":
        raise ValueError("collection profiling requires native_cuda_pool")
    manifest = build_manifest(config)
    write_json(args.scratch / "manifest.json", manifest)
    began = time.perf_counter()
    model = load_frozen_actor_critic_checkpoint(Path(config.checkpoint), device="cuda")
    evaluator = FrozenSearchInference(model, device="cuda", config=InferenceConfig(
        mode=config.execution.inference_mode, actor_projection=config.execution.actor_projection,
        candidate_buckets=config.execution.candidate_buckets, compile_mode=config.execution.compile_mode,
        compile_failure_policy=config.execution.compile_failure_policy, native_transport="reused"))
    load_seconds = time.perf_counter() - began
    completed = 0
    completion_times = []
    physical_batches = []
    new_transfer = evaluator.new_native_transfer
    collect_started = 0.0

    def progress(done, total):
        nonlocal completed
        completed = done
        completion_times.append({"completed_games": done, "at_seconds": time.perf_counter() - collect_started})

    class Transfer:
        def __init__(self, inner, width):
            self.inner, self.width = inner, width

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def begin(self, sink, state_only):
            started = time.perf_counter()
            remaining = config.num_games - completed
            inner = (self.inner.begin_value_only(sink) if state_only else self.inner.begin_policy_value(sink))

            class Handle:
                def finish(handle):
                    inner.finish()
                    physical_batches.append({"rows": sink.row_count(), "width": self.width,
                                             "kind": "value" if state_only else "policy",
                                             "remaining_games": remaining,
                                             "inference_wall_seconds": time.perf_counter() - started})
            return Handle()

        def begin_policy_value(self, sink):
            return self.begin(sink, False)

        def begin_value_only(self, sink):
            return self.begin(sink, True)

    def transfer(**kwargs):
        return Transfer(new_transfer(**kwargs), 0 if kwargs["state_only"] else kwargs["width"])

    evaluator.new_native_transfer = transfer
    prof = cProfile.Profile()
    pool._worker = profiled_worker
    previous_profile = os.environ.get("SPLENDOR_SPEED_CPROFILE")
    os.environ["SPLENDOR_SPEED_CPROFILE"] = "1" if args.cprofile else "0"
    try:
        began = time.perf_counter()
        collect_started = began
        execute = (lambda *a, **kw: prof.runcall(pool.collect, *a, **kw)) if args.cprofile else pool.collect
        workers, owner = execute(
            config, args.scratch, tuple(range(config.num_games)),
            manifest["semantic_fingerprint"], manifest["checkpoint_identity"]["sha256"],
            evaluator=evaluator, benchmark_seconds=args.seconds,
            warmup_seconds=args.warmup_seconds,
            progress_callback=progress,
        )
        wall = time.perf_counter() - began
    finally:
        pool._worker = _ORIGINAL_WORKER
        if previous_profile is None:
            os.environ.pop("SPLENDOR_SPEED_CPROFILE", None)
        else:
            os.environ["SPLENDOR_SPEED_CPROFILE"] = previous_profile
    prof.dump_stats(str(args.scratch / "owner.pstats"))
    validation = validate_run(args.scratch, replay=True) if args.seconds is None else None
    worker_profiles = [json.loads(p.read_text()) for p in sorted((args.scratch / "profiling").glob("worker-*.json"))]
    write_json(args.scratch / "physical_batches.json", physical_batches)
    remaining_groups = defaultdict(Counter)
    width_groups = defaultdict(Counter)
    for b in physical_batches:
        group = ("last_10_percent" if b["remaining_games"] <= max(1, (config.num_games + 9) // 10) else
                 "last_half" if b["remaining_games"] <= config.num_games // 2 else "first_half")
        for counter in (remaining_groups[group], width_groups[str(b["width"])]):
            counter.update(batches=1, rows=b["rows"], inference_wall_seconds=b["inference_wall_seconds"])
    return {"config": asdict(config), "checkpoint_identity": manifest["checkpoint_identity"],
            "model_load_seconds": load_seconds, "collect_wall_seconds": wall,
            "owner": owner, "worker_profiles": worker_profiles,
            "worker_results": [{k:v for k,v in w.items() if k not in ("completed", "decision_latencies_ms")}
                               | {"completed_games": len(w["completed"])} for w in workers],
            "top_owner_functions": profile_rows(prof), "validation": validation,
            "completion_times": completion_times, "batches_by_remaining_games": dict(remaining_groups),
            "batches_by_physical_width": dict(width_groups),
            "cprofile": args.cprofile,
            "scope": "instrumented pool; stages follow root game turns; cProfile overhead applies only when enabled"}


def evaluation_profile(args) -> dict[str, Any]:
    from contextlib import ExitStack
    from splendor.engine import GameEngine
    from splendor.native_policy import NativePUCTSearchPolicy

    config = evaluation.FixedEvaluationConfig.from_dict(json.loads(args.config.read_text()))
    config = replace(config, workers=1, device=args.device, game_seeds=config.game_seeds[:args.pairs])
    if len(config.game_seeds) != args.pairs:
        raise ValueError("evaluation config has fewer paired seeds than requested")
    timer = Timings()
    prof = cProfile.Profile()
    with ExitStack() as stack:
        for target, name, label in (
            (evaluation, "_create_serializable_policy", "policy_create"),
            (evaluation, "_append_game_log", "sidecar"),
            (evaluation, "_evaluation_summary_from_records", "summary"),
            (GameEngine, "new", "engine_create"),
            (GameEngine, "decision", "real_decision"),
            (GameEngine, "apply", "real_apply"),
            (NativePUCTSearchPolicy, "search", "search"),
            (FrozenSearchInference, "policy_value_encoded", "policy_value_inference"),
            (FrozenSearchInference, "value_only_encoded", "value_only_inference"),
            (FrozenSearchInference, "_forward", "inference_forward"),
            (FrozenSearchInference, "_forward_value", "inference_value_forward"),
            (FrozenSearchInference, "_tensor_batch", "inference_tensorize"),
            (search_inference, "collate_encoded_decisions", "inference_collate"),
            (search_inference, "collate_encoded_states", "inference_state_collate"),
        ):
            stack.enter_context(wrapped(target, name, timer, label))
        began = time.perf_counter()
        execute = (lambda *a, **kw: prof.runcall(evaluation.run_fixed_evaluation, *a, **kw)) if args.cprofile else evaluation.run_fixed_evaluation
        run = execute(config, args.scratch / "evaluation.json")
        wall = time.perf_counter() - began
    prof.dump_stats(str(args.scratch / "evaluation.pstats"))
    timings = timer.result()
    h = timings["host_wall_seconds"]
    exclusive = {k:v for k,v in h.items() if k not in ("search", "policy_value_inference", "value_only_inference")}
    inference_total = h.get("policy_value_inference", 0) + h.get("value_only_inference", 0)
    children = sum(v for k,v in h.items() if k.startswith("inference_"))
    exclusive["inference_output_and_other"] = inference_total - children
    exclusive["search_except_inference"] = h.get("search", 0) - h.get("policy_value_inference", 0) - h.get("value_only_inference", 0)
    exclusive["runner_store_and_other"] = wall - sum(exclusive.values())
    return {"config": config.to_dict(), "wall_seconds": wall, "timings": timings,
            "exclusive_host_wall_seconds": exclusive, "summary": run.summary,
            "top_functions": profile_rows(prof), "cprofile": args.cprofile,
            "scope": "single-worker actual native paired evaluation; not an 8-worker panel throughput"}


def historical_profile(args) -> dict[str, Any]:
    rounds = []
    for root in sorted(Path("artifacts/az_loop_seed1/run_snapshot").glob("round-*")):
        d = json.loads((root / "round_summary.json").read_text())
        wall = d["wall_seconds_this_session"]
        parts = {"collect": d["collect"]["wall_seconds"], "train_including_validation": d["train"]["training_seconds"],
                 **{f"evaluation_{k}": v["wall_seconds_this_session"] for k,v in d["evaluation"].items()}}
        parts["setup_and_other"] = wall - sum(parts.values())
        rounds.append({"source": str(root / "round_summary.json"), "source_sha256": sha256(root / "round_summary.json"),
                       "wall_seconds": wall, "parts_seconds": parts,
                       "parts_fraction": {k:v/wall for k,v in parts.items()}})
    status = Path("runtime/module_b/full_pool_b500/status.json")
    d = json.loads(status.read_text())
    owner = d["gpu_owner"]
    names = ("owner_wait_input_seconds", "owner_wait_coalesce_seconds", "owner_gather_seconds", "owner_dispatch_seconds")
    parts = {k:owner[k] for k in names}
    # These inference counters are host launch/copy/event-wait intervals.
    parts.update({f"inference_{k}":owner["inference"][k] for k in ("h2d_seconds", "forward_seconds", "d2h_seconds", "wait_seconds")})
    physical = sum(int(w)*n for w,n in owner["width_histogram"].items())
    return {"historical_az_rounds": rounds,
            "historical_pool": {"source": str(status), "source_sha256": sha256(status),
                                "wall_seconds": d["wall_seconds"], "owner_intervals_seconds": parts,
                                "owner_intervals_fraction_of_wall": {k:v/d["wall_seconds"] for k,v in parts.items()},
                                "owner": {k:v for k,v in owner.items() if k not in ("workers", "benchmark")},
                                "worker_cpu_seconds": d["worker_cpu_seconds"],
                                "physical_candidate_positions": physical,
                                "owner_padding_fraction": owner["padding_positions"]/physical,
                                "note": "owner intervals are host wall counters, not GPU utilization; worker cumulative times overlap"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--cprofile", action="store_true", help="whole-run function profile for collection/evaluation; adds overhead")
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("history")
    p = sub.add_parser("learner")
    p.add_argument("--run", type=Path, default=Path("runs/az_loop_seed1"))
    p.add_argument("--round", type=int, default=4)
    p.add_argument("--cache-mib", type=int, default=256)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--warmup-steps", type=int, default=8)
    p.add_argument("--steps", type=int, default=24)
    p = sub.add_parser("replay")
    p.add_argument("--run", type=Path, default=Path("runs/az_loop_seed1"))
    p.add_argument("--round", type=int, default=4)
    p.add_argument("--cache-mib", type=int, default=256)
    p.add_argument("--warmup-steps", type=int, default=8)
    p.add_argument("--steps", type=int, default=2)
    p = sub.add_parser("collection")
    p.add_argument("--config", type=Path, default=Path("configs/native_search/selfplay_cuda_pool_500_b500.json"))
    p.add_argument("--games", type=int)
    p.add_argument("--active", type=int)
    p.add_argument("--batch", type=int)
    p.add_argument("--seconds", type=float)
    p.add_argument("--warmup-seconds", type=float, default=2)
    p = sub.add_parser("evaluation")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--pairs", type=int, default=1)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.output.exists() or args.scratch.exists():
        raise ValueError("use new output and scratch paths; existing measurements are never overwritten")
    args.scratch.mkdir(parents=True)
    (args.scratch / "profile_speed_pipeline.py").write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)
    before = provenance()
    began = time.perf_counter()
    result = {"history": historical_profile, "learner": learner_profile, "replay": replay_profile,
              "collection": collection_profile, "evaluation": evaluation_profile}[args.mode](args)
    write_json(args.output, {"protocol": "speed-pipeline-profile-v1", "provenance": before,
                             "total_command_body_seconds": time.perf_counter() - began, "result": result})
    print(json.dumps({"output": str(args.output), "mode": args.mode,
                      "seconds": time.perf_counter() - began}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
