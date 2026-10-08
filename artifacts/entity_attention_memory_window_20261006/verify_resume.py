"""Exercise the real resume CLI without collecting or updating formal weights."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from splendor.alphazero_recording import read_json, sha256, write_json


def main():
    run = Path('runs/entity_attention_hybrid_s1024_n1_from_round9_seed1')
    art = Path(__file__).resolve().parent
    state = read_json(run / 'hybrid_state.json')
    paths = [run / 'config.json', run / 'hybrid_state.json', run / 'fork_manifest.json',
             Path(state['latest_checkpoint']['path']), Path(state['anchor_checkpoint']['path']),
             run / 'rounds/round-029/replay_snapshot.json', run / 'rounds/round-030/selfplay/manifest.json']
    before = {str(path): sha256(path) for path in paths}
    sessions = set((run / 'source').iterdir())
    command = [sys.executable, '-u', '-m', 'splendor.hybrid_train', '--config',
               'configs/entity_attention_v1/hybrid_fork_s1024_n1.json', '--output', str(run),
               '--resume', '--online-rounds', '30', '--replay-cache-mib', '8192']
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    new_sessions = set((run / 'source').iterdir()) - sessions
    after = {str(path): sha256(path) for path in paths}
    new_state = read_json(run / 'hybrid_state.json')
    identities = [read_json(path / 'identity.json') for path in new_sessions]
    data = {'command': command, 'exit_code': result.returncode, 'stdout': result.stdout,
            'stderr': result.stderr, 'before_sha256': before, 'after_sha256': after,
            'preserved_files_equal': before == after, 'state': new_state,
            'effective_cache_bytes': [item['replay_cache_capacity_bytes'] for item in identities],
            'inductor_compile_threads_env': [item['inductor_compile_threads_env'] for item in identities],
            'formal_optimizer_updates': 0, 'formal_new_games': 0}
    write_json(art / 'resume_verification.json', data)
    assert result.returncode == 0, result.stderr
    assert before == after and new_state == state
    assert data['effective_cache_bytes'] == [8 * 1024**3]
    print({'preserved_files_equal': True, 'total_train_step': new_state['latest_checkpoint']['total_train_step'],
           'completed_rounds': len(new_state['completed_rounds']), 'cache_bytes': data['effective_cache_bytes'][0]})


if __name__ == '__main__':
    main()
