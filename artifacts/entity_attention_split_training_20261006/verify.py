"""Bounded production-path verification: frozen real fork, four GPU updates, S8 pair.

Run from the repository root in conda splendor with PYTHONPATH=src. This does
not collect new games or initialize the formal output directory.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from splendor.alphazero_checkpoint import load_alphazero_checkpoint
from splendor.alphazero_recording import environment_identity, read_json, sha256, source_identity, write_json
from splendor.alphazero_train import AlphazeroLearner, run_training_round
from splendor.hybrid_train import (HybridConfig, HybridReferenceWriter, _evaluation, draw_split_batch,
                                   run_hybrid, split_replay_description, split_replay_pools)
from splendor.offline_train import _digest, make_index


CONFIG = Path('configs/entity_attention_v1/hybrid_split_policy_value.json')
PARENT = Path('runs/entity_attention_hybrid_s1024_n1_from_round9_seed1')
ARTIFACT = Path('artifacts/entity_attention_split_training_20261006')


def same_checkpoint(a, b):
    assert a.sampling_rng_state == b.sampling_rng_state
    assert torch.equal(a.torch_cpu_rng_state, b.torch_cpu_rng_state)
    assert len(a.torch_cuda_rng_states) == len(b.torch_cuda_rng_states)
    assert all(torch.equal(x, y) for x, y in zip(a.torch_cuda_rng_states, b.torch_cuda_rng_states))
    for name in a.model_state_dict:
        torch.testing.assert_close(a.model_state_dict[name], b.model_state_dict[name], rtol=0, atol=0)
    assert a.optimizer_state_dict['param_groups'] == b.optimizer_state_dict['param_groups']
    for key, state in a.optimizer_state_dict['state'].items():
        for name in state:
            torch.testing.assert_close(state[name], b.optimizer_state_dict['state'][key][name], rtol=0, atol=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=Path('runs/entity_attention_split_training_validation_20261006'))
    args = parser.parse_args()
    output = args.output.resolve()
    assert not output.exists(), 'use a fresh private verification output'
    assert torch.cuda.is_available(), 'GPU verification needs CUDA'
    config = HybridConfig.from_dict(read_json(CONFIG))
    torch.set_num_threads(1)
    inputs = [PARENT / 'hybrid_state.json', PARENT / 'config.json',
              PARENT / 'rounds/round-049/checkpoints/final.pt',
              PARENT / 'rounds/round-049/replay_snapshot.json', Path(config.evaluation_anchor['path'])]
    hashes = {str(p): sha256(p) for p in inputs}
    state = run_hybrid(config, output / 'fork', fork_from=PARENT, fork_round=50,
                       online_rounds=0, replay_cache_bytes_override=8 * 1024**3)
    parent_path = PARENT / 'rounds/round-049/checkpoints/final.pt'
    parent = load_alphazero_checkpoint(parent_path)
    copied = load_alphazero_checkpoint(state['latest_checkpoint']['path'])
    same_checkpoint(parent, copied)
    manifest = read_json(output / 'fork/fork_manifest.json')
    assert len(manifest['games']) == 3000 and state['completed_rounds'] == []
    index = make_index({'games': manifest['games']}, 256 * 1024**2)
    recent, older, rounds = split_replay_pools(index, config)
    description = split_replay_description(index, config, (recent, older), rounds)
    identity = {'digest': _digest(description), 'split_training': description}
    identity.update({name: {'digest': identity['digest'], 'run_ids': sorted({g.run_id for g in index.games})}
                     for name in ('policy_recent', 'value_recent', 'value_history')})
    initial_learner = AlphazeroLearner.resume(parent_path, config.offline.learner,
        replay_identity=parent.replay_identity, round_index=parent.round_index,
        round_steps_target=parent.round_steps_target)
    initial_learner.round_index += 1
    initial_learner.round_train_step = 0
    initial_learner.round_steps_target = 4
    initial_learner.replay_identity = identity
    initial = output / 'training/initial.pt'
    initial_learner.save(initial)
    del initial_learner
    inherited_ids = {g.run_id for g in index.games}
    split = config.split_training
    def train(path, session, limit, checkpoint):
        learner = AlphazeroLearner.resume(checkpoint, config.offline.learner,
            replay_identity=identity, round_index=parent.round_index + 1, round_steps_target=4)
        writer = HybridReferenceWriter(path / 'usage', session, identity, (256, 128, 128),
                                       ('policy_recent', 'value_recent', 'value_history'))
        try:
            result = run_training_round(learner, steps_target=4, current_pool=recent, history_pool=None,
                validation_samples=(), validation_games=0, checkpoint_dir=path / 'checkpoints',
                metrics_path=path / 'training_metrics.jsonl', reference_writer=writer,
                batch_drawer=lambda rng: draw_split_batch(recent, older, rng, 256, 128,
                                                          inherited_run_ids=inherited_ids),
                split_grad_norms=(split.policy_max_grad_norm, split.value_max_grad_norm),
                max_steps_this_session=limit)
        finally:
            writer.close()
        return result
    continuous = output / 'continuous'
    segmented = output / 'segmented'
    timing = train(continuous, 'continuous', None, initial)
    train(segmented, 'first', 2, initial)
    train(segmented, 'resumed', 2, segmented / 'checkpoints/step-000000002.pt')
    final_path = continuous / 'checkpoints/step-000000004.pt'
    same_checkpoint(load_alphazero_checkpoint(final_path),
                    load_alphazero_checkpoint(segmented / 'checkpoints/step-000000004.pt'))
    rows = [json.loads(line) for line in (continuous / 'training_metrics.jsonl').read_text().splitlines()
            if json.loads(line)['event'] == 'train_step']
    assert len(rows) == 4 and all(r['policy_batch_size'] == r['value_batch_size'] == 256 for r in rows)
    evaluation = json.loads(json.dumps(config.evaluation))
    evaluation['pairs'] = 1
    evaluation['seed_start'] = 19900000
    evaluation['execution']['active_games'] = 2
    for name in ('policy_a', 'policy_b'):
        evaluation[name]['search_config']['simulations'] = 8
    smoke_config = replace(config, evaluation=evaluation)
    _evaluation(smoke_config, final_path, Path(config.evaluation_anchor['path']).resolve(), output / 'smoke')
    smoke_path = output / 'smoke/evaluation/mcts_vs_fixed_anchor.json'
    smoke = read_json(smoke_path)
    assert smoke['passed'] and smoke['completed_games'] == 2
    assert all(sha256(Path(path)) == digest for path, digest in hashes.items())
    # All inherited archives were read-only; verify their frozen hashes as well.
    assert all(sha256(Path(g['path'])) == g['archive_sha256'] for g in manifest['games'])
    write_json(ARTIFACT / 'verification.json', {
        'status': 'passed', 'formal_training_started': False, 'new_collection_games': 0,
        'parent_inputs_sha256': hashes, 'config_sha256': sha256(CONFIG),
        'parent_total_steps': parent.total_train_step, 'inherited_games': len(manifest['games']),
        'model_optimizer_rng_inherited_exactly': True,
        'four_gpu_updates_equal_two_plus_two_resume_exactly': True,
        'initial_recent_round_indices': rounds, 'split_replay': description,
        'gpu_training': timing, 'gpu_step_records': rows,
        'smoke_scored_games': 2, 'smoke_simulations': 8,
        'smoke_used_for_strength_conclusions': False, 'smoke_summary_sha256': sha256(smoke_path),
        'environment': environment_identity(),
        'source_identity': source_identity(Path.cwd(), output / 'source'),
        'verification_script_sha256': sha256(Path(__file__)),
        'command': ['conda', 'run', '-n', 'splendor', 'env', 'PYTHONPATH=src',
                    'TORCHINDUCTOR_COMPILE_THREADS=2', 'python', str(Path(__file__)), '--output', str(output)],
        'private_output': str(output), 'parent_inputs_and_archives_unchanged': True,
    })
    print(json.dumps({'status': 'passed', 'gpu_updates_per_arm': 4, 'smoke_games': 2,
                      'artifact': str(ARTIFACT / 'verification.json')}))


if __name__ == '__main__':
    main()
