"""Bounded S1024/N1 execution benchmark on the existing 200 frozen roots."""
from __future__ import annotations

import argparse
import gc
from pathlib import Path
import sys
import time
import weakref

import torch

from splendor.alphazero_recording import environment_identity, read_json, sha256, write_json
from splendor.compact_trace import decode_request
from splendor.mcts_policy import MCTSConfig
from splendor.native_policy import NativeBatchRunner, NativePUCTSearchPolicy
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.search_inference import FrozenSearchInference, InferenceConfig

ROOT = Path(__file__).resolve().parents[2]
ART = Path(__file__).resolve().parent


def memory(runner, references):
    rss = next(line.split()[1] for line in Path('/proc/self/status').read_text().splitlines()
               if line.startswith('VmRSS:'))
    return {'rss_bytes': int(rss) * 1024,
            'cuda_allocated_bytes': torch.cuda.memory_allocated(),
            'cuda_reserved_bytes': torch.cuda.memory_reserved(),
            'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'native_buffers': len(runner._entries),
            'cuda_graphs': sum(len(getattr(transfer, '_graphs', {}))
                               for _, transfer in runner._entries.values()),
            'live_previous_jobs': sum(ref() is not None for ref in references)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=['eager64', 'eager200', 'graph200'], required=True)
    parser.add_argument('--repeats', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output = ART / 'speed' / f'{args.case}.json'
    if output.exists():
        raise FileExistsError(output)
    plan = read_json(ART / 'plan.json')
    checkpoint = ROOT / plan['checkpoint']
    if sha256(checkpoint) != plan['checkpoint_sha256']:
        raise ValueError('frozen checkpoint changed')
    bank_path = ROOT / 'artifacts/entity_attention_chance_search_20261007/root_bank.json'
    bank = read_json(bank_path)['positions']
    indices = read_json(ROOT / 'artifacts/entity_attention_chance_pw_20261007/selection.json')['confirmation_indices']
    requests = [decode_request(bank[i]['request']) for i in indices]
    capacity = 64 if args.case == 'eager64' else 200
    inference = InferenceConfig(mode='compiled' if args.case == 'graph200' else 'eager',
        actor_projection=False, candidate_buckets=(16, 32, 48, 64),
        compile_mode='native-cudagraph' if args.case == 'graph200' else None,
        batch_buckets=(16, 32, 64, 128, 200) if args.case == 'graph200' else (),
        compile_failure_policy='fail_fast', native_transport='reused')
    model = load_frozen_actor_critic_checkpoint(checkpoint, device='cuda')
    evaluator = FrozenSearchInference(model, device='cuda', config=inference)
    policy = NativePUCTSearchPolicy(model, device='cuda', evaluator=evaluator,
        config=MCTSConfig(**dict(plan['search_config'], seed=16710000)))
    runner = NativeBatchRunner(evaluator, policy.catalog, capacity=capacity,
                               buckets=(16, 32, 48, 64), slots=1)
    result = {'command': sys.argv, 'case': args.case, 'capacity': capacity,
        'checkpoint_sha256': sha256(checkpoint), 'bank_sha256': sha256(bank_path),
        'benchmark_sha256': sha256(Path(__file__)),
        'code_hashes': {str(p.relative_to(ROOT)): sha256(p) for p in
                       [*sorted((ROOT / 'src/splendor').glob('*.py')),
                        ROOT / 'src/splendor/native_search.cpp',
                        *sorted((ROOT / 'src/splendor').glob('_native_search*.so'))]},
        'environment': environment_identity(), 'inference_config': inference.to_dict(),
        'sampling_seed': 16720000, 'search_config': dict(plan['search_config'], seed=16710000),
        'tf32_matmul': False, 'tf32_cudnn': False, 'windows': []}
    previous = []
    for repeat in range(args.repeats):
        records, references = [], []
        before = evaluator.stats.to_dict()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        for offset in range(0, len(requests), capacity):
            jobs = [policy.start_job(request, sampling_seed=16720000)
                    for request in requests[offset:offset + capacity]]
            references.extend(weakref.ref(job) for job in jobs)
            while any(not job.done for job in jobs):
                if runner.run([job for job in jobs if not job.done]) == 0:
                    raise RuntimeError('native search made no progress')
            for index, job in zip(indices[offset:offset + capacity], jobs):
                search = job.result
                if search.simulations != 1024 or sum(search.root_visit_counts) != 1024:
                    raise ValueError('the benchmark changed the S1024 budget')
                records.append({'bank_index': index, 'stage': bank[index]['stage'],
                    'chosen_candidate_index': search.chosen_candidate_index,
                    'root_counts': list(search.root_visit_counts),
                    'target': list(search.policy_target), 'prior': list(search.root_network_prior),
                    'stats': dict(job.core.stats())})
            del jobs, job
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        gc.collect()
        after = evaluator.stats.to_dict()
        window = {'repeat': repeat, 'cold': repeat == 0, 'wall_seconds': seconds,
            'roots': len(records), 'simulations_per_second': 1024 * len(records) / seconds,
            'forward_seconds': after['forward_seconds'] - before['forward_seconds'],
            'batches': after['physical_model_batches'] - before['physical_model_batches'],
            'positions': after['logical_model_positions'] - before['logical_model_positions'],
            'active_jobs': len(policy._active_jobs), 'memory': memory(runner, previous),
            'records': records}
        result['windows'].append(window)
        previous = references
        write_json(output, result)
        print({key: value for key, value in window.items() if key != 'records'}, flush=True)
    if policy._active_jobs:
        raise RuntimeError('finished searches retained active jobs')
    runner.batch_records.clear()
    runner._error_record = None
    gc.collect()
    result['after_last_batch_release'] = memory(runner, previous)
    result['inference_stats'] = evaluator.stats.to_dict()
    write_json(output, result)


if __name__ == '__main__':
    main()
