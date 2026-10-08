"""Small real S1024/N1 CUDA panels for the stage evaluator execution change."""
import argparse
from pathlib import Path
import sys
import time

import torch

import run as experiment
from splendor.alphazero_recording import environment_identity, read_json, sha256, write_json
from splendor.evaluation import run_fixed_evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['late_only', 'early_only'], required=True)
    parser.add_argument('--execution', choices=['eager', 'fast'], required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    plan_path = Path(__file__).with_name('plan_fast.json' if args.execution == 'fast' else 'plan.json')
    plan = read_json(plan_path)
    plan['pairs'] = 8
    # Independent diagnostic seeds: these games are not formal strength results.
    plan['game_seed_start'] = 22400000
    checkpoint = experiment.ROOT / plan['checkpoint']
    if sha256(checkpoint) != plan['checkpoint_sha256']:
        raise ValueError('frozen checkpoint changed')
    config = experiment.evaluation_config(plan, args.mode, 'bounded-cuda-smoke')
    output = Path(__file__).with_name('speed') / f'{args.mode}_{args.execution}.json'
    if output.exists():
        raise FileExistsError(output)
    write_json(output.with_suffix('.command.json'), {'command': sys.argv,
        'plan': plan, 'config': config.to_dict(), 'environment': environment_identity(),
        'checkpoint_sha256': sha256(checkpoint), 'script_sha256': sha256(Path(__file__)),
        'runner_sha256': sha256(Path(experiment.__file__)),
        'code_hashes': {str(p.relative_to(experiment.ROOT)): sha256(p) for p in
                       [*sorted((experiment.ROOT / 'src/splendor').glob('*.py')),
                        experiment.ROOT / 'src/splendor/native_search.cpp',
                        *sorted((experiment.ROOT / 'src/splendor').glob('_native_search*.so'))]}})
    memory = []

    def progress(n, total):
        rss_kib = next(line.split()[1] for line in Path('/proc/self/status').read_text().splitlines()
                       if line.startswith('VmRSS:'))
        row = {'pairs': n, 'rss_bytes': int(rss_kib) * 1024,
               'cuda_allocated_bytes': torch.cuda.memory_allocated(),
               'cuda_reserved_bytes': torch.cuda.memory_reserved(),
               'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated()}
        memory.append(row)
        print(row, flush=True)

    started = time.perf_counter()
    with experiment.install_stage_policy(args.mode, plan['late_score_threshold']):
        result = run_fixed_evaluation(config, output,
            failure_dir=experiment.ROOT / plan['runtime_root'] / 'smoke_failures',
            progress_every=2, summary_every=2, progress_callback=progress)
    stats = result.summary['policy_a_stats']
    forbidden = 'early' if args.mode == 'late_only' else 'late'
    if not result.summary['passed'] or stats[f'{forbidden}_simulations'] != 0:
        raise RuntimeError('invalid stage panel')
    if stats['simulations'] != 1024 * stats['searchable_decisions']:
        raise RuntimeError('S1024 budget changed')
    if result.summary['policy_b_stats'] or len(result.summary['execution_stats']['models']) != 1:
        raise RuntimeError('same-weight greedy failed to share its evaluator')
    write_json(output.with_suffix('.memory.json'), {'memory': memory,
        'wall_including_startup_seconds': time.perf_counter() - started,
        'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'stage_budget_passed': True})
    print({'mode': args.mode, 'execution': args.execution,
           'wall_including_startup_seconds': time.perf_counter() - started,
           'searchable_decisions': stats['searchable_decisions'], 'passed': True}, flush=True)


if __name__ == '__main__':
    main()
