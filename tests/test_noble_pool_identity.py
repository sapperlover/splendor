"""Catalog changes must not silently extend a frozen collection/evaluation."""
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

from splendor.actor_critic import ActorCritic
from splendor.catalog import load_default_catalog
from splendor.features import FEATURE_SCHEMA_VERSION
from splendor.mcts_policy import MCTSConfig
from splendor.ppo import TRAINING_CHECKPOINT_FORMAT_VERSION
from splendor.replay import RULESET_ID


@pytest.fixture
def checkpoint(tmp_path):
    model=ActorCritic()
    path=tmp_path/'model.pt'
    torch.save({'format_version':TRAINING_CHECKPOINT_FORMAT_VERSION,
        'feature_schema_version':FEATURE_SCHEMA_VERSION,'metadata':{'rules_version':RULESET_ID},
        'model_config':asdict(model.config),'model_state_dict':model.state_dict()},path)
    return path


@pytest.mark.parametrize('fast', [False, True])
def test_selfplay_manifest_pins_catalog_identity(checkpoint, monkeypatch, fast):
    if fast:
        import splendor.selfplay_fast as module
        config=module.FastSelfPlayConfig(protocol='mcts-selfplay-v2',run_id='noble-pool',
            round_index=0,checkpoint=str(checkpoint),num_games=1,seed=37,
            search_config=MCTSConfig(simulations=1,determinizations=1),
            execution=module.ExecutionConfig(backend='native_cpu',engine_mode='fast',inference_mode='eager'))
    else:
        import splendor.selfplay as module
        config=module.SelfPlayConfig(protocol='mcts-selfplay-v1',run_id='noble-pool',
            round_index=0,checkpoint=str(checkpoint),num_games=1,seed=37,
            search_config=MCTSConfig(simulations=1,determinizations=1))
    catalog=load_default_catalog()
    original=module.build_manifest(config)
    assert original['catalog_fingerprint']==catalog.data_fingerprint
    monkeypatch.setattr(module,'load_default_catalog',lambda:replace(catalog,data_fingerprint='different-pool'))
    changed=module.build_manifest(config)
    assert changed['semantic_fingerprint']!=original['semantic_fingerprint']


def test_evaluation_resume_rejects_changed_catalog(tmp_path, monkeypatch):
    import splendor.evaluation as module
    config=module.FixedEvaluationConfig(
        policy_a=module.SerializablePolicySpec.baseline('score'),
        policy_b=module.SerializablePolicySpec.baseline('development'),game_seeds=(37,))
    path=tmp_path/'evaluation.json'
    result=module.run_fixed_evaluation(config,path)
    catalog=load_default_catalog()
    assert result.summary['catalog_fingerprint']==catalog.data_fingerprint
    monkeypatch.setattr(module,'load_default_catalog',lambda:replace(catalog,data_fingerprint='different-pool'))
    with pytest.raises(ValueError,match='different configuration'):
        module.run_fixed_evaluation(config,path,resume=True)


def test_legacy_resume_cannot_bypass_missing_catalog_identity(tmp_path):
    import splendor.evaluation as module
    config=module.FixedEvaluationConfig(
        policy_a=module.SerializablePolicySpec.baseline('score'),
        policy_b=module.SerializablePolicySpec.baseline('development'),game_seeds=(37,))
    path=tmp_path/'evaluation.json'
    module.run_fixed_evaluation(config,path)
    (Path(str(path)+'.games')/'header.json').unlink()
    with pytest.raises(ValueError,match='catalog fingerprint'):
        module.run_fixed_evaluation(config,path,resume=True,allow_legacy_resume=True)
