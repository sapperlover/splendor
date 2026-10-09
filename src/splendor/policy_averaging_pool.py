"""Fixed-root tasks on the existing shared-memory search-worker pool."""
from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from .alphazero_recording import write_json
from .mcts_policy import MCTSConfig
from .features import FEATURE_SCHEMA_VERSION
from .multiworker_gpu import collect
from .native_policy import NativePUCTSearchPolicy
from .replay import to_primitive
from .selfplay import derive_seed
from .selfplay_data import load_game
from .selfplay_fast import ExecutionConfig, FastSelfPlayConfig
from .search_inference import NativeForwardHandle


class _CPUTransfer:
    """CPU owner's shared-memory input, using the same native packed layout."""

    def __init__(self, evaluator, capacity, width, float_stride, mask_stride):
        self.evaluator, self.width = evaluator, width
        self.float_staging = np.empty(capacity * float_stride, dtype=np.float32)
        self.mask_staging = np.empty(capacity * mask_stride, dtype=np.bool_)

    def _arrays(self, sink):
        rows, layout = sink.row_count(), sink.packed()
        arrays = {"feature_schema_version": FEATURE_SCHEMA_VERSION}
        for fields, blob, stride in (("float_fields", self.float_staging, layout["float_stride"]),
                                    ("mask_fields", self.mask_staging, layout["mask_stride"])):
            data = blob.reshape(-1, stride)[:rows]
            for name, offset, count, shape in layout[fields]:
                arrays[name] = data[:, offset:offset + count].reshape(rows, *shape)
        return arrays

    def begin_policy_value(self, sink):
        logits, values = self.evaluator.policy_value_native(self._arrays(sink))
        output = np.column_stack((logits, values))
        return NativeForwardHandle(sink.row_count(), self.width, False, lambda: sink.submit_fused(output))

    def begin_value_only(self, sink):
        values = self.evaluator.value_only_native(self._arrays(sink))
        return NativeForwardHandle(sink.row_count(), 0, True, lambda: sink.submit(None, values))


class _CPUOwner:
    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.device, self.stats = evaluator.device, evaluator.stats

    def new_native_transfer(self, *, state_only, **kwargs):
        return _CPUTransfer(self.evaluator, **kwargs)


def task_path(root: Path, task_id: int) -> Path:
    return root / "roots.games" / f"task-{task_id:07d}.npz"


@dataclass
class RootTaskSource:
    archives: tuple[Path, ...]
    # Archive ordinal, decision index, repeat, world seed, noise seed, candidates.
    tasks: tuple[tuple[int, int, int, int, int, int], ...]
    search: dict
    add_root_noise: bool
    cache: OrderedDict = field(default_factory=OrderedDict, init=False, repr=False)

    def __call__(self, config, task_id, evaluator, checkpoint_sha):
        archive, decision, repeat, world, noise, width = self.tasks[task_id]
        requests = self.cache.get(archive)
        if requests is None:
            from .policy_averaging import replay_roots
            arrays, metadata = load_game(self.archives[archive])
            if metadata["checkpoint_sha256"] != checkpoint_sha:
                raise ValueError("fixed-root archive teacher changed")
            requests = dict(replay_roots(arrays, metadata))
            self.cache[archive] = requests
            if len(self.cache) > 2:
                self.cache.popitem(last=False)
        self.cache.move_to_end(archive)
        request = requests[decision]
        if len(request.legal_actions) != width:
            raise ValueError("fixed-root candidate ordering changed")
        return RootSession(task_id, request, (archive, decision, repeat, world, noise, width),
                           self.search, self.add_root_noise, evaluator)


class RootSession:
    """One search, immutable real decision; completion never advances the game."""

    def __init__(self, task_id, request, spec, search, add_noise, evaluator):
        self.game_id = task_id  # Internal pool task ID; no new selfplay game.
        self.request, self.spec = request, spec
        self.policy = NativePUCTSearchPolicy(None, evaluator=evaluator,
                                            config=MCTSConfig(**{**search, "seed": spec[3]}))
        self.policies = (self.policy,)
        self.job = self.policy.start_job(request, add_root_noise=add_noise,
                                        temperature=0., sampling_seed=spec[4])
        self.result = None
        self.records = []
        self.decision_latencies_ms = []
        self.started = time.perf_counter()

    @property
    def finished(self):
        return self.result is not None

    def ensure_job(self):
        if self.job is not None and self.job.done:
            self.take_result()

    def take_result(self):
        result = self.job.result
        if not result.searchable or sum(result.root_visit_counts) != self.policy.config.simulations:
            raise ValueError("fixed-root search budget mismatch")
        self.result = result
        self.records.append(1)
        self.decision_latencies_ms.append((time.perf_counter() - self.started) * 1000)
        self.job = None

    def commit(self, root, fingerprint, checkpoint_sha):
        from .policy_averaging import _npz
        if not self.finished:
            raise RuntimeError("cannot save an incomplete fixed-root search")
        started = time.perf_counter()
        _npz(task_path(root, self.game_id), targets=np.asarray(self.result.policy_target, dtype=np.float64),
             counts=np.asarray(self.result.root_visit_counts, dtype=np.int32),
             task=np.asarray(self.spec, dtype=np.uint64), identity=np.asarray(fingerprint))
        return {"game_id": self.game_id, "decisions": 1, "simulations": self.policy.config.simulations,
                "write_seconds": time.perf_counter() - started}

    def failure_scene(self):
        return {"task_id": self.game_id, "archive_index": self.spec[0], "decision_index": self.spec[1],
                "repeat": self.spec[2], "root_request": to_primitive(self.request),
                "search_context": None if self.job is None else
                                  to_primitive(self.job.failure_context_for_world(None))}


def make_source(files, raw, config):
    tasks = []
    lookup = {}
    for rep in range(3):
        for ordinal, file in enumerate(files):
            arrays, meta = load_game(file)
            for decision in np.flatnonzero(arrays["searchable"]):
                i = int(decision)
                world = derive_seed(config.world_seed, raw["round_index"], meta["game_id"], "m4_extra_world", rep, i)
                noise = derive_seed(config.noise_seed, raw["round_index"], meta["game_id"], "m4_extra_noise", rep, i)
                width = int(arrays["candidate_offsets"][i + 1] - arrays["candidate_offsets"][i])
                lookup[ordinal, i, rep] = len(tasks)
                tasks.append((ordinal, i, rep, world, noise, width))
    return RootTaskSource(tuple(files), tuple(tasks), raw["search_config"], raw["add_root_noise"]), lookup


def read_task(output, source, task_id, fingerprint):
    spec = source.tasks[task_id]
    with np.load(task_path(output, task_id), allow_pickle=False) as z:
        if (str(z["identity"]) != fingerprint or
                not np.array_equal(z["task"], np.asarray(spec, dtype=np.uint64)) or
                z["targets"].shape != (spec[5],) or z["counts"].shape != (spec[5],) or
                int(z["counts"].sum()) != source.search["simulations"]):
            raise ValueError("saved fixed-root task identity/budget mismatch")
        return z["targets"].copy(), z["counts"].copy()


def run_pool(files, raw, output, config, evaluator, fingerprint, checkpoint_sha, progress_callback):
    source, lookup = make_source(files, raw, config)
    missing = tuple(i for i in range(len(source.tasks)) if not task_path(output, i).exists())
    # Validation of every committed task happens when assembling the old chunk format.
    if missing:
        if evaluator.device.type == "cpu" and evaluator.model is not None:
            evaluator = _CPUOwner(evaluator)
        execution = ExecutionConfig(backend="native_cuda_pool", engine_mode="fast", batch_interface="native",
            inference_mode="eager", actor_projection=False, max_batch_size=config.max_batch_size,
            search_workers=config.search_workers, total_active_games=config.active_roots,
            local_microbatch=config.local_microbatch, shared_slots=config.shared_slots,
            shared_candidate_width=256, candidate_buckets=(16, 32, 48, 64), max_batch_wait_ms=5,
            pipeline_slots=1, torch_threads=1)
        pool_config = replace(FastSelfPlayConfig.from_dict(raw), execution=execution, num_games=len(source.tasks))
        started = time.perf_counter()
        def completed(done, total):
            if done % 64 == 0 or done == total:
                if progress_callback:
                    progress_callback(-1, 0, done, total)
                import json
                print(json.dumps({"event": "policy_averaging_search_progress", "completed_searches": done,
                                  "total_searches": total, "seconds": time.perf_counter() - started}), flush=True)
        _, owner = collect(pool_config, output, missing, fingerprint, checkpoint_sha,
                           progress_callback=completed, completed_before=len(source.tasks) - len(missing),
                           evaluator=evaluator, session_factory=source)
        write_json(output / "pool_statistics" / f"session-{time.time_ns()}.json", {
            "scope": "fixed-root tasks searched during this invocation; pool game IDs are task IDs",
            "completed_before": len(source.tasks) - len(missing), "searched_tasks": len(missing),
            "total_tasks": len(source.tasks), "seconds": time.perf_counter() - started, "owner": owner})
    return source, lookup
