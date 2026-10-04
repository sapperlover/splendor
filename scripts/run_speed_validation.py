"""Production-path speed experiments with bounded aggregate instrumentation.

Run from the repository root. PYTHONPATH selects the preserved baseline or
current source. Heavy games/checkpoints stay in runtime; --record is durable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from splendor import multiworker_gpu as pool
from splendor.search_inference import FrozenSearchInference


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


_worker = pool._worker
from splendor import evaluation as evaluation_module
_evaluation_init = evaluation_module._parallel_worker_init
_numeric_stats = evaluation_module._numeric_policy_stats


def measured_evaluation_init():
    """Spawn-safe host timers; per-policy totals are persisted in sidecars."""
    _evaluation_init()
    for name in ('policy_value_encoded', 'value_only_encoded', '_tensor_batch', '_forward', '_forward_value'):
        original = getattr(FrozenSearchInference, name)
        def timed(self, *a, _original=original, _name=name, **kw):
            before = time.perf_counter()
            try:
                return _original(self, *a, **kw)
            finally:
                if not hasattr(self, '_speed_times'):
                    self._speed_times = Counter()
                self._speed_times[_name] += time.perf_counter() - before
        setattr(FrozenSearchInference, name, timed)

    def stats(policy):
        result = _numeric_stats(policy)
        evaluator = getattr(policy, 'evaluator', None)
        timing = getattr(evaluator, '_speed_times', {})
        result['inference_wall_seconds'] = timing.get('policy_value_encoded', 0.) + timing.get('value_only_encoded', 0.)
        result['input_tensor_seconds'] = timing.get('_tensor_batch', 0.)
        result['core_forward_seconds'] = timing.get('_forward', 0.) + timing.get('_forward_value', 0.)
        result['inference_other_seconds'] = result['inference_wall_seconds'] - result['input_tensor_seconds'] - result['core_forward_seconds']
        if evaluator is not None:
            result['physical_batches'] = evaluator.stats.batches
            result['inference_positions'] = evaluator.stats.positions
        return result
    evaluation_module._numeric_policy_stats = stats


def measured_worker(*args, **kwargs):
    """Only aggregate exported row identities; no device synchronization."""
    root, wid = args[1:3]
    native = pool._extension()
    original_batch, original_ensure = native.NativeBatch, pool._GameSession.ensure_job
    stages, hist, descriptors = {}, defaultdict(Counter), Counter()

    def ensure(session):
        previous = session.job
        result = original_ensure(session)
        if session.job is not None and previous is not session.job:
            turns = session.stepper.completed_player_turns
            stages[id(session.job.core)] = "opening" if turns < 10 else "middle" if turns < 50 else "late"
        return result

    class Batch:
        def __init__(self, *a, **kw):
            self.inner = original_batch(*a, **kw)
            self.parts = []
            self.counted = False

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def fill(self, core):
            if self.inner.row_count() == 0:
                self.parts = []
                self.counted = False
            start = self.inner.row_count()
            count = self.inner.fill(core)
            if count:
                self.parts.append((start, count, stages[id(core)]))
            return count

        def row_ids_array(self):
            ids = self.inner.row_ids_array()
            if not self.counted:
                self.counted = True
                width = 0 if self.inner.is_state_only() else self.inner.padded_width()
                descriptors[str(width)] += len(ids)
                # Stage segments are contiguous and contain one row for N=1.
                for start, count, stage in self.parts:
                    hist[stage].update(map(str, ids[start:start + count, 3].tolist()))
            return ids

    native.NativeBatch, pool._GameSession.ensure_job = Batch, ensure
    try:
        _worker(*args, **kwargs)
    finally:
        native.NativeBatch, pool._GameSession.ensure_job = original_batch, original_ensure
        write(root / "measurement" / f"worker-{wid}.json", {
            "actual_width_rows_by_root_stage": dict(hist), "descriptor_width_rows": dict(descriptors)})


class Resources:
    def __init__(self):
        self.stop = threading.Event()
        self.rows = []
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        while not self.stop.is_set():
            row = {"at_unix": time.time(), "tree_cpu_seconds": 0., "tree_rss_bytes": 0,
                   "tree_pss_bytes": 0, "processes": 0}
            processes = {}
            for path in Path('/proc').iterdir():
                if not path.name.isdigit():
                    continue
                try:
                    fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
                    processes[int(path.name)] = (int(fields[1]), fields, path)
                except (FileNotFoundError, PermissionError, ProcessLookupError):
                    pass
            selected = {os.getpid()}
            while True:
                more = {pid for pid, (ppid, _, _) in processes.items() if ppid in selected}
                if more.issubset(selected):
                    break
                selected.update(more)
            for pid in selected:
                if pid not in processes:
                    continue
                _, fields, path = processes[pid]
                row['tree_cpu_seconds'] += (int(fields[11]) + int(fields[12])) / os.sysconf('SC_CLK_TCK')
                row['tree_rss_bytes'] += int(fields[21]) * os.sysconf('SC_PAGE_SIZE')
                try:
                    lines = (path / 'smaps_rollup').read_text().splitlines()
                    row['tree_pss_bytes'] += next(int(line.split()[1]) * 1024 for line in lines if line.startswith('Pss:'))
                except (FileNotFoundError, PermissionError, ProcessLookupError):
                    pass
                row['processes'] += 1
            try:
                row["gpu"] = subprocess.check_output([
                    "nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw,clocks.sm,clocks.mem",
                    "--format=csv,noheader,nounits"], text=True).strip()
            except (OSError, subprocess.CalledProcessError):
                row["gpu"] = None
            self.rows.append(row)
            if hasattr(self, 'live_path'):
                with self.live_path.open('a') as stream:
                    stream.write(json.dumps(row) + '\n')
            self.stop.wait(5)


def collection(args):
    from splendor.selfplay_fast import FastSelfPlayConfig, generate, validate_run
    config = FastSelfPlayConfig.from_dict(json.loads(args.config.read_text()))
    totals = defaultdict(lambda: {"rows": 0, "batches": 0, "inference_wall_seconds": 0.})
    milestones = []
    completed = 0
    started = time.perf_counter()
    original = FrozenSearchInference.new_native_transfer

    def progress(done, total):
        nonlocal completed
        completed = done
        milestones.append({"completed_games": done, "seconds": time.perf_counter() - started,
                           "inference": {k: dict(v) for k, v in totals.items()}})
        if done % 25 == 0 or done == total:
            print(json.dumps({"completed": done, "total": total, "seconds": milestones[-1]["seconds"]}), flush=True)

    def transfer(evaluator, **kw):
        inner = original(evaluator, **kw)
        class Transfer:
            def __getattr__(self, name):
                return getattr(inner, name)

            def begin(self, sink, state_only):
                before = time.perf_counter()
                remaining = config.num_games - completed
                handle = inner.begin_value_only(sink) if state_only else inner.begin_policy_value(sink)
                class Handle:
                    def finish(self):
                        handle.finish()
                        elapsed = time.perf_counter() - before
                        keys = [f"width:{0 if state_only else kw['width']}", "session"]
                        for limit in (100, 10, 8, 1):
                            if remaining <= limit:
                                keys.append(f"last:{limit}")
                        for key in keys:
                            item = totals[key]
                            item["rows"] += sink.row_count()
                            item["batches"] += 1
                            item["inference_wall_seconds"] += elapsed
                return Handle()

            def begin_policy_value(self, sink):
                return self.begin(sink, False)

            def begin_value_only(self, sink):
                return self.begin(sink, True)
        return Transfer()

    if args.instrument:
        FrozenSearchInference.new_native_transfer = transfer
    try:
        if args.window_seconds:
            args.run.mkdir(parents=True, exist_ok=False)
            from splendor.selfplay_fast import build_manifest
            manifest = build_manifest(config)
            workers, owner = pool.collect(config, args.run, tuple(range(config.num_games)),
                manifest["semantic_fingerprint"], manifest["checkpoint_identity"]["sha256"],
                evaluator=None, benchmark_seconds=args.window_seconds, warmup_seconds=2.)
            return {"main_seconds": time.perf_counter() - started, "scope": "opening diagnostic window",
                    "owner": owner, "workers": workers, "aggregate_inference_session": dict(totals)}
        summary = generate(config, args.run, resume=args.resume, progress_callback=progress)
        main = time.perf_counter() - started
    finally:
        pool._worker = _worker
        FrozenSearchInference.new_native_transfer = original
    before = time.perf_counter()
    checked = validate_run(args.run, replay=True)
    replay_seconds = time.perf_counter() - before
    return {"main_seconds": main, "checked_replay_seconds": replay_seconds,
            "delivery_seconds": main + replay_seconds, "summary": summary, "checked_replay": checked,
            "milestones": milestones, "aggregate_inference": dict(totals),
            "status": json.loads((args.run / "status.json").read_text()),
            "worker_widths": [json.loads(p.read_text()) for p in sorted((args.run / "measurement").glob("worker-*.json"))]}


def learner_smoke(args):
    from dataclasses import asdict
    from splendor.alphazero_replay import ReplayRun, build_replay_index
    from splendor.alphazero_train import AlphazeroLearner, LearnerConfig, run_training_round
    from splendor.selfplay_fast import FastSelfPlayConfig
    args.run.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.collected_run / 'manifest.json').read_text())
    config = FastSelfPlayConfig.from_dict(manifest['config'])
    index = build_replay_index([ReplayRun(config.round_index, config.run_id, args.collected_run.resolve())],
        max_games=32, cache_bytes=2 * 1024**3, state_path=args.run / 'replay_index.json')
    current = index.position_pool(split='train', round_index=config.round_index)
    validation_games = index.split_games('validation')
    if not validation_games:
        raise ValueError('the retained real-data smoke panel needs a validation game')
    samples = [index.cache.sample(g, p) for g in validation_games for p in range(min(4, g.decision_count))]
    learner_config = LearnerConfig(device='cuda', batch_size=16, history_batch_size=16, max_steps_per_round=2)
    learner = AlphazeroLearner.warm_start(config.checkpoint, learner_config, sampling_seed=12349000,
        replay_identity=index.identity(), parent_checkpoint=manifest['checkpoint_identity'],
        round_index=0, round_steps_target=2)
    result = run_training_round(learner, steps_target=2, current_pool=current, history_pool=None,
        validation_samples=samples, validation_games=len(validation_games),
        checkpoint_dir=args.run / 'checkpoints', metrics_path=args.run / 'metrics.jsonl')
    result.update(learner_config=asdict(learner_config), replay_identity=index.identity(),
                  source_run=str(args.collected_run), scope='two updates on retained real games; no strength evaluation')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("collection", "evaluation", "learner", "learner_smoke"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--instrument", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--window-seconds", type=float, default=0.)
    parser.add_argument("--replay-run", type=Path, default=Path('runs/az_loop_seed1'))
    parser.add_argument("--round", type=int, default=4)
    parser.add_argument("--cache-mib", type=int, default=256)
    parser.add_argument("--warmup-steps", type=int, default=8)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument('--collected-run', type=Path, default=Path('runtime/end_to_end_speed_2026-10-03/collection_optimized'))
    args = parser.parse_args()
    previous = json.loads(args.record.read_text()) if args.resume and args.record.exists() else None
    session_number = 1 + len(previous.get('sessions', [previous])) if previous else 1
    torch.set_num_threads(1)
    from splendor import native_policy
    extension = Path(native_policy._extension().__file__)
    meta = {"command": __import__('sys').argv, "start_unix": time.time(), "session_number": session_number,
            "python": platform.python_version(), "torch": torch.__version__, "cuda_build": torch.version.cuda,
            "source": pool.__file__, "extension": str(extension),
            "extension_sha256": hashlib.sha256(extension.read_bytes()).hexdigest(),
            "affinity": sorted(os.sched_getaffinity(0)), "cpu_count": os.cpu_count(),
            "threads": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
            "meminfo": {line.split(':')[0]: line.split(':')[1].strip() for line in Path('/proc/meminfo').read_text().splitlines()
                        if line.startswith(('MemTotal:', 'MemAvailable:'))},
            "gpu_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "row_statistics": os.environ.get('SPLENDOR_SPEED_ROW_STATS'),
            "config": json.loads(args.config.read_text()), "instrument": args.instrument,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write(args.record.with_suffix('.started.json'), meta)
    snapshot = Path('runtime/end_to_end_speed_2026-10-03/scripts') / (args.record.stem + '_runner.py')
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_bytes(Path(__file__).read_bytes())
    resource = Resources()
    resource.live_path = Path('runtime/end_to_end_speed_2026-10-03') / (args.record.stem + f'.session-{session_number}.resources.jsonl')
    resource.thread.start()
    invocation_started = time.perf_counter()
    try:
        if args.mode == "collection":
            result = collection(args)
        elif args.mode == 'learner':
            from profile_speed_pipeline import learner_profile
            args.scratch = args.run.resolve()
            args.scratch.mkdir(parents=True, exist_ok=False)
            args.run = args.replay_run
            args.device, args.cprofile = 'cuda', False
            before = time.perf_counter()
            result = learner_profile(args)
            result['main_seconds'] = time.perf_counter() - before
            result['diagnostic_script_sha256'] = hashlib.sha256(Path('scripts/profile_speed_pipeline.py').read_bytes()).hexdigest()
        elif args.mode == 'learner_smoke':
            before = time.perf_counter()
            result = learner_smoke(args)
            result['main_seconds'] = time.perf_counter() - before
        else:
            from splendor.evaluation import FixedEvaluationConfig, run_fixed_evaluation
            config = FixedEvaluationConfig.from_dict(meta["config"])
            if args.instrument and getattr(config, 'execution', None) is None:
                evaluation_module._parallel_worker_init = measured_evaluation_init
            before = time.perf_counter()
            run = run_fixed_evaluation(config, args.run, resume=args.resume,
                progress_callback=lambda n, total: print(json.dumps({"pairs": n, "total": total}), flush=True))
            result = {"main_seconds": time.perf_counter() - before, "summary": run.summary}
        if torch.cuda.is_available():
            result['torch_gpu_allocator'] = {'max_allocated_bytes': torch.cuda.max_memory_allocated(),
                'max_reserved_bytes': torch.cuda.max_memory_reserved(),
                'allocated_at_end_bytes': torch.cuda.memory_allocated(),
                'reserved_at_end_bytes': torch.cuda.memory_reserved()}
        meta.update(end_unix=time.time(), result=result)
    except BaseException as error:
        meta.update(end_unix=time.time(), exception_type=type(error).__name__, exception_detail=str(error),
                    interrupted_wall_seconds=time.perf_counter() - invocation_started)
        raise
    finally:
        resource.stop.set()
        resource.thread.join()
        resource_path = args.record.with_suffix(f'.session-{session_number}.resources.json')
        write(resource_path, resource.rows)
        meta['resources'] = str(resource_path)
        old_sessions = previous.get('sessions', [previous]) if previous else []
        meta['sessions'] = old_sessions + [dict(meta)]
        write(args.record, meta)
    print(json.dumps({"record": str(args.record), "main_seconds": result["main_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
