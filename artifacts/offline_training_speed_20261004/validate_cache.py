"""Bounded CPU cache check. Run from the repository root; no formal training."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from splendor.alphazero_checkpoint import load_alphazero_checkpoint
from splendor.alphazero_recording import read_json, write_json
from splendor.alphazero_replay import PositionPool
from splendor.offline_train import make_index
from splendor.rng import DeterministicRng


def assert_same(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_same(a, b)
    else:
        assert left == right


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = read_json(Path('configs/entity_attention_v1/offline_smoke.json'))
    config['learner']['device'] = 'cpu'
    config['max_steps'] = 4
    config['learner']['checkpoint_every_steps'] = 2
    config['replay_cache_bytes'] = 1024**2
    config_path = args.output / 'config.json'
    write_json(config_path, config)
    continuous, resumed = args.output / 'continuous', args.output / 'resumed'
    command = [sys.executable, '-m', 'splendor.offline_train', '--config', str(config_path)]
    subprocess.run(command + ['--output', str(continuous)], check=True)
    subprocess.run(command + ['--output', str(resumed), '--max-steps-this-session', '2'], check=True)
    subprocess.run(command + ['--output', str(resumed), '--resume', '--replay-cache-mib', '8'], check=True)
    a = load_alphazero_checkpoint(continuous / 'checkpoints/final.pt')
    b = load_alphazero_checkpoint(resumed / 'checkpoints/final.pt')
    assert_same(a.model_state_dict, b.model_state_dict)
    assert_same(a.optimizer_state_dict, b.optimizer_state_dict)
    assert_same(a.torch_cpu_rng_state, b.torch_cpu_rng_state)
    assert a.sampling_rng_state == b.sampling_rng_state
    assert a.round_train_step == b.round_train_step == 4
    assert read_json(continuous / 'config.json') == read_json(resumed / 'config.json') == config
    assert read_json(resumed / 'training.json')['replay_cache']['capacity_bytes'] == 8 * 1024**2
    assert read_json(continuous / 'usage.json')['sequence_sha256'] == read_json(resumed / 'usage.json')['sequence_sha256']
    del a, b
    gc.collect()
    dataset = read_json(Path('runs/entity_attention_offline/dataset.json'))
    # Small real working set reproduces eviction vs fitting entirely; not a full 8 GiB run.
    dataset['games'] = [g for g in dataset['games'] if g['round_index'] == 0 and g['split'] == 'train'][:24]
    records = []
    for capacity_mib in (8, 32):
        index = make_index(dataset, capacity_mib * 1024**2)
        pool = PositionPool(index.split_games('train'), index.cache, 'cache-check')
        rng = DeterministicRng(13400000)
        for _ in range(2048):
            pool.sample(rng)
        before = index.cache.stats()
        sequence = hashlib.sha256()
        tick = time.perf_counter()
        for _ in range(4096):
            _, ref = pool.sample(rng)
            sequence.update(np.asarray((ref.round_index, ref.game_id, ref.decision_index), dtype='<i8').tobytes())
        elapsed = time.perf_counter() - tick
        after = index.cache.stats()
        records.append({'capacity_mib': capacity_mib, 'games': len(pool.games), 'measured_draws': 4096,
                        'warmup_draws': 2048, 'seconds': elapsed, 'sequence_sha256': sequence.hexdigest(),
                        'delta': {k: after[k] - before[k] for k in ('hits', 'misses', 'decode_count', 'load_seconds')},
                        'cache_bytes': after['cache_bytes']})
        del pool, index
        gc.collect()
    assert records[0]['sequence_sha256'] == records[1]['sequence_sha256']
    assert records[1]['delta']['misses'] == 0
    write_json(args.output / 'result.json', {
        'cpu_resume_equivalence': '4 continuous updates vs 2+2 resumed with larger execution-only cache: bitwise model/optimizer/RNG and identical reference sequence',
        'frozen_config_unchanged': True, 'records': records,
        'sampling_speedup_small_working_set': records[0]['seconds'] / records[1]['seconds'],
        'scope': 'CPU only; 24 real train games; draw-only speedup is not measured full training speedup'})
    print(json.dumps(records), flush=True)


if __name__ == '__main__':
    main()
