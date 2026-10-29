"""Two disjoint fixed-root workers, reusing the frozen semantics in run.py."""
from __future__ import annotations

import argparse
import fcntl
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from splendor.alphazero_recording import environment_identity, read_json, sha256, write_json
from splendor.catalog import load_default_catalog
from splendor.compact_trace import decode_request
from splendor.mcts_policy import MCTSConfig
from splendor.native_policy import NativeBatchRunner, NativePUCTSearchPolicy
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.search_inference import FrozenSearchInference, InferenceConfig
import run as exp


def worker(index):
    plan = read_json(exp.ART / 'plan.json'); exp.identity_check(plan)
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    model = load_frozen_actor_critic_checkpoint(exp.CHECKPOINT, device='cuda')
    cfg = dict(plan['inference'])
    for key in ('batch_buckets', 'candidate_buckets'):
        cfg[key] = tuple(cfg[key])
    evaluator = FrozenSearchInference(model, device='cuda', config=InferenceConfig(**cfg))
    runner = NativeBatchRunner(evaluator, load_default_catalog(), capacity=exp.CAPACITY, buckets=exp.BUCKETS, slots=2)
    roots = exp.requests()
    started = time.perf_counter()
    completed = 0
    for panel_index, panel in enumerate(plan['panels']):
        config = MCTSConfig(**plan['search_common'], simulations=panel['simulations'], determinizations=panel['determinizations'], seed=exp.SEARCH_SEED)
        policy = NativePUCTSearchPolicy(model, evaluator=evaluator, config=config, catalog=load_default_catalog())
        for chunk_index, start in enumerate(range(0, exp.COUNT, exp.CHUNK)):
            if (panel_index * 32 + chunk_index) % 2 != index:
                continue
            group = roots[start:start + exp.CHUNK]
            path = exp.RUNTIME / 'search' / panel['name'] / f'{start:04d}.json'
            if path.exists():
                saved = read_json(path)
                assert saved['panel'] == panel and saved['plan_sha256'] == sha256(exp.ART / 'plan.json')
                assert [r['root_id'] for r in saved['rows']] == [r['root_id'] for r in group]
                continue
            tick = time.perf_counter()
            jobs = [policy.start_job(decode_request(r['request']), add_root_noise=panel['noise'],
                                     temperature=0., sampling_seed=r['sampling_seed']) for r in group]
            while any(not j.done for j in jobs):
                assert runner.run([j for j in jobs if not j.done]) > 0
            result = [exp.search_row(j, r, panel, panel['simulations']) for j, r in zip(jobs, group, strict=True)]
            elapsed = time.perf_counter() - tick
            write_json(path, dict(panel=panel, plan_sha256=sha256(exp.ART / 'plan.json'), rows=result,
                                  seconds=elapsed, finished_unix=time.time(), execution_worker=index))
            completed += len(group)
            status = dict(stage='parallel_search', worker=index, pid=os.getpid(), panel=panel['name'],
                          root_start=start, roots_completed_this_worker=completed, chunk_seconds=elapsed,
                          simulations_per_second=panel['simulations'] * len(group) / elapsed,
                          updated_unix=time.time(), seconds=time.perf_counter() - started,
                          inference=evaluator.stats.to_dict(), runner_rows=runner.rows, runner_batches=runner.batches,
                          runner_wide_batches=runner.wide_batches, cuda_peak_bytes=torch.cuda.max_memory_allocated())
            write_json(exp.RUNTIME / f'worker{index}_live.json', status)
            print(json.dumps({k: v for k, v in status.items() if k != 'inference'}), flush=True)
            del jobs
            gc.collect()
        del policy
    exp.identity_check(plan)
    write_json(exp.RUNTIME / f'worker{index}_complete.json', dict(seconds=time.perf_counter() - started,
               inference=evaluator.stats.to_dict(), runner_rows=runner.rows, runner_batches=runner.batches,
               runner_wide_batches=runner.wide_batches, cuda_peak_bytes=torch.cuda.max_memory_allocated()))


def coordinate():
    with (exp.RUNTIME / 'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = read_json(exp.RUNTIME / 'live.json')
        assert previous['stage'] == 'search'
        write_json(exp.RUNTIME / 'serial_session_last_completed_chunk.json', previous)
        started = time.time()
        write_json(exp.ART / 'parallel_execution.json', dict(environment=environment_identity(), command=sys.argv,
                   driver_sha256=sha256(Path(__file__)), started_unix=started, workers=2,
                   assignment='(panel_index*32+chunk_index)%2; disjoint full chunks, existing files checked and skipped',
                   unchanged='model, safe requests, seeds, native search semantics, chunk composition, FP32/buckets/slots',
                   previous_session='runtime/' + exp.ART.name + '/serial_session_last_completed_chunk.json'))
        handles, children = [], []
        for index in (0, 1):
            handle = (exp.RUNTIME / f'worker{index}.log').open('w')
            handles.append(handle)
            children.append(subprocess.Popen([sys.executable, str(Path(__file__)), '--worker', str(index)], stdout=handle, stderr=subprocess.STDOUT))
        codes = [child.wait() for child in children]
        for handle in handles:
            handle.close()
        assert codes == [0, 0], codes
        sessions = [previous] + [read_json(exp.RUNTIME / f'worker{i}_complete.json') for i in (0, 1)]
        compile_stats = [s['inference']['compile'] for s in sessions]
        first = min(read_json(p)['started_unix'] for p in exp.RUNTIME.glob('execution_*.json'))
        completed = dict(seconds=time.time() - first, parallel_wall_seconds=time.time() - started,
                         execution_sessions=sessions, workers_after_resume=2,
                         inference={'compile': dict(compile_failures=sum([s['compile_failures'] for s in compile_stats], []),
                             eager_fallbacks=sum(s['eager_fallbacks'] for s in compile_stats),
                             compile_failure_fallbacks=sum(s['compile_failure_fallbacks'] for s in compile_stats))},
                         runner_rows=sum(s['runner_rows'] for s in sessions),
                         runner_batches=sum(s['runner_batches'] for s in sessions),
                         runner_wide_batches=sum(s['runner_wide_batches'] for s in sessions))
        write_json(exp.RUNTIME / 'complete.json', completed)
        write_json(exp.RUNTIME / 'live.json', dict(stage='complete', updated_unix=time.time(), **completed))
        print(json.dumps({'complete': True, 'seconds': completed['seconds'], 'parallel_seconds': completed['parallel_wall_seconds']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=int, choices=[0, 1])
    args = parser.parse_args()
    if args.worker is None:
        coordinate()
    else:
        worker(args.worker)
