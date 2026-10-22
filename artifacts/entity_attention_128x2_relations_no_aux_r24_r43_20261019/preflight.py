"""Four real samples and at most two updates, isolated from the formal run."""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import torch

from splendor.alphazero_recording import environment_identity, read_json, sha256, source_identity, write_json
from splendor.alphazero_replay import ReplayGame, ReplayIndex
from splendor.alphazero_train import AlphazeroLearner
from splendor.entity_attention import EntityAttention
from splendor.offline_scratch import ScratchOfflineConfig, initialize_learner
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.selfplay_data import collate_selfplay_samples

ART = Path(__file__).resolve().parent
ROOT = ART.parents[1]
CONFIG = ROOT / 'configs/entity_attention_v1/offline_128x2_relations_no_aux_r24_r43_20261019.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    formal = ScratchOfflineConfig.from_dict(read_json(CONFIG))
    config = replace(formal, learner=replace(formal.learner, device=args.device, batch_size=4,
                                            history_batch_size=2, checkpoint_every_steps=1))
    runtime = ROOT / 'runtime' / ART.name / 'preflight' / args.device
    runtime.mkdir(parents=True, exist_ok=True)
    assert sha256(Path(config.manifest)) == config.manifest_sha256
    with Path(config.manifest).open() as handle:
        record = next(row for line in handle if (row := json.loads(line))['round_index'] == 23
                      and row['split'] == 'train')
    archive = Path(record['path'])
    assert sha256(archive) == record['archive_sha256']
    fields = ('round_index', 'run_id', 'game_id', 'split', 'decision_count', 'policy_samples')
    game = ReplayGame(**{name: record[name] for name in fields}, path=archive)
    index = ReplayIndex([game], cache_bytes=64 * 1024**2)
    batch = collate_selfplay_samples([index.cache.sample(game, i) for i in range(4)])
    identity = {'digest': record['archive_sha256'], 'value_auxiliary': None, 'policy_surprise_weighting': None}
    learner = initialize_learner(config, identity, 2)
    assert type(learner.model) is EntityAttention and learner.model.trainable_parameter_count() == 583682
    assert not hasattr(learner.model, 'score_head') and not hasattr(learner.model, 'remaining_turns_head')
    assert not learner.optimizer.state and learner.total_train_step == 0
    assert learner.config.learning_rate == 1e-4
    first = learner.train_step(batch, references=[])
    assert all(math.isfinite(first[key]) for key in ('total_loss', 'policy_loss', 'value_loss'))
    assert 'score_loss' not in first and 'remaining_turns_loss' not in first
    checkpoint = runtime / 'step-1.pt'
    learner.save(checkpoint)
    resumed = AlphazeroLearner.resume(checkpoint, config.learner, replay_identity=identity,
                                     round_index=0, round_steps_target=2)
    assert resumed.total_train_step == 1
    frozen = load_frozen_actor_critic_checkpoint(checkpoint, device=args.device)
    with torch.inference_mode():
        for left, right in zip(learner.model(batch.decisions), frozen(batch.decisions)):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
    exact = None
    if args.device == 'cpu':
        learner.train_step(batch, references=[])
        resumed.train_step(batch, references=[])
        for name, value in learner.model.state_dict().items():
            torch.testing.assert_close(value, resumed.model.state_dict()[name], atol=0, rtol=0)
        for key, state in learner.optimizer.state_dict()['state'].items():
            for name, value in state.items():
                torch.testing.assert_close(value, resumed.optimizer.state_dict()['state'][key][name], atol=0, rtol=0)
        exact = True
    result = {'status': 'passed', 'scope': __doc__, 'device': args.device,
              'command': [sys.executable, *sys.argv], 'config_sha256': sha256(CONFIG),
              'preflight_sha256': sha256(Path(__file__)), 'parameters': 583682,
              'fresh_optimizer': True, 'auxiliary_heads_absent': True, 'cpu_resume_exact': exact,
              'first_step': first, 'checkpoint_sha256': sha256(checkpoint),
              'archive': str(archive), 'archive_sha256': record['archive_sha256'], 'decision_indices': [0, 1, 2, 3],
              'batch_size': 4, 'source': source_identity(ROOT, runtime / 'source'), 'environment': environment_identity()}
    write_json(ART / f'{args.device}_preflight.json', result)
    print({k: result[k] for k in ('status', 'device', 'parameters', 'auxiliary_heads_absent', 'cpu_resume_exact')}, flush=True)


if __name__ == '__main__':
    main()
