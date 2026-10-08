"""Full native S1024/N1 vs same-weight greedy, using the existing paired seeds."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from dataclasses import replace

import torch

import run as stage
from splendor.alphazero_recording import environment_identity, read_json, sha256, source_identity, write_json
from splendor.evaluation import run_fixed_evaluation
from splendor.native_policy import NativePUCTSearchPolicy

ROOT = stage.ROOT
PROTOCOL = 'entity-attention-full-search-v1'


class FullSearchPolicy(NativePUCTSearchPolicy):
    """Unchanged native search on every root; the score gate only labels costs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stats = stage.StageStats()
        self.entered_late = False

    def start_job(self, request, *, add_root_noise=False, temperature=0.,
                  sampling_seed=None, root_encoded=None):
        if add_root_noise or temperature != 0:
            raise ValueError('this frozen experiment uses zero root noise and temperature')
        phase = stage.stage_for(request, 10)
        for suffix in ('decisions', 'search_decisions'):
            name = f'{phase}_{suffix}'
            setattr(self.stats, name, getattr(self.stats, name) + 1)
        if phase == 'late' and not self.entered_late:
            self.entered_late = True
            self.stats.late_entry_count += 1
            self.stats.late_entry_completed_turns_sum += (
                request.observation.own_completed_turns + request.observation.opponent_completed_turns)
        return stage.StageSearchJob(self, request, stage=phase, add_root_noise=False,
            temperature=0., sampling_seed=sampling_seed, root_encoded=root_encoded)


@contextmanager
def install_full_policy():
    import splendor.evaluation_batch as batch
    original = batch.NativePUCTSearchPolicy
    batch.NativePUCTSearchPolicy = FullSearchPolicy
    try:
        yield
    finally:
        batch.NativePUCTSearchPolicy = original


def evaluation_config(plan, fingerprint):
    config = stage.evaluation_config(plan, 'full_search', fingerprint)
    return replace(config, metadata={**config.metadata, 'protocol': PROTOCOL,
        'required_runner': str(Path(__file__).relative_to(ROOT)),
        'stage_rule': 'search every root; public maxscore>=10 labels statistics only'})


def prepare(plan, config_path):
    if (plan['protocol'] != PROTOCOL or plan['arms'] != ['full_search']
            or plan['late_score_threshold'] != 10):
        raise ValueError('this experiment is full search with score10 statistics')
    checkpoint = ROOT / plan['checkpoint']
    if sha256(checkpoint) != plan['checkpoint_sha256']:
        raise ValueError('frozen checkpoint changed')
    artifact, runtime = ROOT / plan['artifact_root'], ROOT / plan['runtime_root']
    paths = [Path(__file__), Path(stage.__file__), *sorted((ROOT / 'src/splendor').glob('*.py')),
             ROOT / 'src/splendor/native_search.cpp', *sorted((ROOT / 'src/splendor').glob('_native_search*.so'))]
    hashes = {str(p.relative_to(ROOT)): sha256(p) for p in paths}
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    manifest_path = artifact / 'manifest.json'
    if manifest_path.exists():
        existing = read_json(manifest_path)
        if existing['plan'] != plan or existing['code_fingerprint'] != fingerprint:
            raise ValueError('plan or execution code changed; use a new output directory')
    else:
        snapshot = runtime / 'source'
        source = source_identity(ROOT, snapshot)
        for p in [Path(__file__), Path(stage.__file__), ROOT / 'src/splendor/native_search.cpp',
                  *sorted((ROOT / 'src/splendor').glob('_native_search*.so'))]:
            target = snapshot / p.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, target)
        comparisons = {mode: {'path': str(ROOT / plan['comparison_root'] / 'evaluations' / f'{mode}.json'),
                              'sha256': sha256(ROOT / plan['comparison_root'] / 'evaluations' / f'{mode}.json')}
                       for mode in ['late_only', 'early_only']}
        write_json(manifest_path, {'protocol': PROTOCOL, 'plan': plan,
            'checkpoint': {'path': str(checkpoint), 'sha256': plan['checkpoint_sha256']},
            'code_hashes': hashes, 'code_fingerprint': fingerprint, 'source': source,
            'comparisons': comparisons, 'command': sys.argv, 'config_path': str(config_path.resolve()),
            'environment': environment_identity()})
        write_json(artifact / 'configuration_snapshot.json', plan)
    write_json(artifact / 'config.json', evaluation_config(plan, fingerprint).to_dict())
    return artifact, runtime, fingerprint


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('plan_full.json'))
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    plan = read_json(args.config)
    artifact, runtime, fingerprint = prepare(plan, args.config)
    if args.prepare_only:
        print(json.dumps({'status': 'prepared', 'pairs': plan['pairs'], 'checkpoint': plan['checkpoint']}), flush=True)
        return 0
    write_json(artifact / 'invocations' / f'{time.time_ns()}.json',
               {'command': sys.argv, 'environment': environment_identity()})

    def progress(n, total):
        rss = next(line.split()[1] for line in Path('/proc/self/status').read_text().splitlines()
                   if line.startswith('VmRSS:'))
        row = {'event': 'progress', 'arm': 'full_search', 'completed_pairs': n, 'pairs': total,
               'rss_bytes': int(rss) * 1024}
        if plan['device'] == 'cuda':
            row.update(cuda_allocated_bytes=torch.cuda.memory_allocated(),
                       cuda_reserved_bytes=torch.cuda.memory_reserved())
        print(json.dumps(row), flush=True)
        with (runtime / 'memory.jsonl').open('a') as stream:
            stream.write(json.dumps(row) + '\n')

    output = artifact / 'evaluation.json'
    runtime.mkdir(parents=True, exist_ok=True)
    with install_full_policy():
        result = run_fixed_evaluation(evaluation_config(plan, fingerprint), output,
            resume=Path(str(output) + '.games').exists(), failure_dir=runtime / 'failures',
            progress_every=25, summary_every=25, progress_callback=progress)
    if not result.summary['passed'] or result.summary['completed_pairs'] != plan['pairs']:
        raise RuntimeError('full search contains invalid/incomplete games; inspect retained failure records')
    stats = result.summary['policy_a_stats']
    if (stats['simulations'] != plan['search_config']['simulations'] * stats['searchable_decisions']
            or stats['early_greedy_decisions'] + stats['late_greedy_decisions'] != 0
            or stats['early_simulations'] + stats['late_simulations'] != stats['simulations']
            or result.summary['policy_b_stats']):
        raise RuntimeError('full search stage counters or search budget do not match the plan')
    print(json.dumps({'event': 'full_search_done', 'score': result.summary['win_rate_score'],
                      'ci95': result.summary['paired_bootstrap']}, ensure_ascii=False), flush=True)
    del result
    gc.collect()
    if plan['device'] == 'cuda':
        torch.cuda.empty_cache()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
