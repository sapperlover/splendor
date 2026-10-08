"""Four tiny GPU rounds in an independent directory; no strength conclusion."""
from __future__ import annotations

import argparse
import gc
import time
from dataclasses import replace
from pathlib import Path

import torch

from splendor.alphazero_loop import LoopSeeds
from splendor.alphazero_recording import environment_identity, read_json, sha256, write_json
from splendor.hybrid_train import HybridConfig, run_hybrid
from splendor.selfplay import derive_seed


def rss_kib():
    return int(next(line.split()[1] for line in Path('/proc/self/status').read_text().splitlines()
                    if line.startswith('VmRSS:')))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    art = Path(__file__).resolve().parent
    config = HybridConfig.from_dict(read_json(Path('configs/entity_attention_v1/hybrid_cpu_smoke.json')))
    execution = read_json(Path('configs/entity_attention_v1/hybrid_fork_s1024_n1.json'))['selfplay']['execution']
    execution.update(total_active_games=4, search_workers=2, local_microbatch=4,
                     max_batch_size=16, batch_buckets=[16])
    split = next(base for base in range(14610000, 14612000) if all(
        len({derive_seed(LoopSeeds(0, base, 0).split_for(r), r, gid, 'split') < (1 << 63)
             for gid in range(4)}) == 2 for r in range(4)))
    evaluation = read_json(Path('configs/entity_attention_v1/hybrid_fork_s1024_n1.json'))['evaluation']
    evaluation['game_seeds'] = [14709000]
    evaluation['execution']['active_games'] = 2
    evaluation['execution']['max_batch_size'] = 16
    for side in ('policy_a', 'policy_b'):
        evaluation[side]['search_config']['simulations'] = 2
    config = replace(config, run_id='entity_attention_memory_gpu_smoke',
        offline=replace(config.offline, learner=replace(config.offline.learner, device='cuda')),
        selfplay=replace(config.selfplay, execution=execution, validation_fraction=.5),
        seeds=replace(config.seeds, split=split), new_games_per_round=4, replay_max_games=4,
        online_validation_positions=16, mix_schedule=((1, 6), (2, 4)),
        evaluation_every_rounds=2, evaluation=evaluation)
    write_json(art / 'gpu_smoke_config.json', config.to_dict())
    started = time.perf_counter()
    memory = []
    # Keep one Python process across rounds so cross-round retention is exercised.
    state = run_hybrid(config, args.output, online_rounds=4, replay_cache_bytes_override=32 * 1024**2)
    for human in range(1, 5):
        stage = args.output / 'rounds' / f'round-{human - 1:03d}'
        status = read_json(stage / 'selfplay/status.json')
        metrics = [r for r in map(__import__('json').loads, (stage / 'training_metrics.jsonl').read_text().splitlines())
                   if r['event'] == 'train_step']
        memory.append({'round': human, 'owner_peak_rss_kib': status['gpu_owner']['owner_cpu_peak_rss_kib'],
                       'online_cache': metrics[-1]['replay_cache'],
                       'completed_games': read_json(stage / 'selfplay/summary.json')['completed_games'],
                       'native_graph_captures': status['gpu_owner']['inference']['native_graph_captures'],
                       'compile': status['gpu_owner']['inference']['compile']})
        assert memory[-1]['completed_games'] == 4
    for human in (2, 4):
        result = read_json(args.output / 'rounds' / f'round-{human - 1:03d}' / 'evaluation/mcts_vs_theta5000.json')
        assert result['passed'] and result['valid_games'] == 2
    gc.collect()
    torch.cuda.empty_cache()
    write_json(art / 'gpu_smoke_result.json', {'state': state, 'rounds': memory,
        'final_rss_kib': rss_kib(), 'elapsed_seconds': time.perf_counter() - started,
        'environment': environment_identity(), 'config_sha256': sha256(art / 'gpu_smoke_config.json'),
        'script_sha256': sha256(Path(__file__)), 'scope': '4 rounds x 4 games x 2 updates, S2/N1; '
        '2 search workers, shared_slots=2, owner pipeline_slots=1, B16. No long-run RAM or strength guarantee.'})


if __name__ == '__main__':
    main()
