from dataclasses import asdict, replace

import pytest
import torch

from splendor.actor_critic import ActorCritic, TorchStateBatch, build_actor_critic, tensorize_batch
from splendor.alphazero_checkpoint import load_alphazero_checkpoint
from splendor.alphazero_train import AlphazeroLearner, LearnerConfig
from splendor.contracts import DecisionRequest
from splendor.entity_attention import EntityAttention
from splendor.features import (FEATURE_SCHEMA_VERSION, collate_encoded_decisions, encode_decision)
from splendor.observation import build_observation
from splendor.offline_train import OfflineConfig, freeze_validation, make_index
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.ppo import TRAINING_CHECKPOINT_FORMAT_VERSION
from splendor.replay import RULESET_ID
from splendor.rng import DeterministicRng
from splendor.rules import legal_actions
from splendor.search_inference import FrozenSearchInference, InferenceConfig


@pytest.fixture(autouse=True, scope="module")
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def batch(initial_state, catalog):
    request = DecisionRequest(build_observation(initial_state, 0, catalog), legal_actions(initial_state, catalog))
    return tensorize_batch(collate_encoded_decisions([encode_decision(request)]))


def test_shape_state_only_and_candidate_permutation(batch):
    model = EntityAttention().eval()
    with torch.inference_mode():
        logits, value = model(batch)
        perm = torch.arange(logits.shape[1] - 1, -1, -1)
        other_logits, other_value = model(replace(batch, candidate_features=batch.candidate_features[:, perm],
                                                candidate_mask=batch.candidate_mask[:, perm]))
        state = TorchStateBatch(**{k: getattr(batch, k) for k in TorchStateBatch.__dataclass_fields__})
        torch.testing.assert_close(model.value_from_state(state), value)
        torch.testing.assert_close(model.value_only(batch), value)
        torch.testing.assert_close(other_value, value)
        torch.testing.assert_close(other_logits, logits[:, perm])
        torch.testing.assert_close(model.policy_logits(batch), logits)
    assert logits.shape == batch.candidate_mask.shape
    assert value.shape == (1,)


def test_set_permutations_and_padding_do_not_change_outputs(batch):
    model = EntityAttention().eval()
    # Populate both reserves so their permutations exercise real, nonempty entities.
    batch = replace(batch, self_reserved=batch.market_t1[:, :3], opponent_reserved=batch.market_t2[:, :3],
                    self_reserved_mask=batch.market_t1_mask[:, :3], opponent_reserved_mask=batch.market_t2_mask[:, :3])
    changed = {}
    for name in ('market_t1', 'market_t2', 'market_t3', 'self_reserved', 'opponent_reserved', 'nobles'):
        changed[name] = getattr(batch, name).flip(1)
        changed[name + '_mask'] = getattr(batch, name + '_mask').flip(1)
    padded = replace(batch, candidate_features=torch.nn.functional.pad(batch.candidate_features, (0, 0, 0, 4), value=1e30),
                     candidate_mask=torch.nn.functional.pad(batch.candidate_mask, (0, 4)))
    padded = replace(padded, pending_card=torch.full_like(batch.pending_card, float('nan')))
    with torch.inference_mode():
        reference = model(batch)
        actual = model(replace(batch, **changed))
        padding = model(padded)
    for x, y in zip(reference, actual):
        torch.testing.assert_close(x, y, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(reference[0], padding[0][:, :reference[0].shape[1]])
    torch.testing.assert_close(reference[1], padding[1])


def test_unknown_reserve_is_valid_and_all_global_fields_survive(batch):
    model = EntityAttention().eval()
    partition = model.self_indices + model.opponent_indices + model.context_indices
    assert sorted(partition) == list(range(batch.global_features.shape[1]))
    unknown = batch.opponent_reserved.clone()
    unknown[:, 0, 0] = 1  # valid=1, known=0
    unknown[:, 0, 2] = 1  # tier one; no identity or cost is supplied
    mask = batch.opponent_reserved_mask.clone()
    mask[:, 0] = True
    with torch.inference_mode():
        baseline = model(batch)[1]
        observed = model(replace(batch, opponent_reserved=unknown, opponent_reserved_mask=mask))[1]
    assert not torch.allclose(baseline, observed, atol=1e-7, rtol=0)
    with pytest.raises(ValueError, match='actor_projection=false'):
        FrozenSearchInference(model, config=InferenceConfig(actor_projection=True))


def test_new_and_legacy_checkpoint_dispatch_is_strict(tmp_path, batch):
    model = EntityAttention().eval()
    learner = AlphazeroLearner(model, LearnerConfig(device='cpu'), sampling_rng=DeterministicRng(7))
    path = tmp_path / 'attention.pt'
    learner.save(path)
    restored = load_frozen_actor_critic_checkpoint(path)
    assert isinstance(restored, EntityAttention)
    assert isinstance(load_alphazero_checkpoint(path).build_model(), EntityAttention)
    with torch.inference_mode():
        for a, b in zip(model(batch), restored(batch)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    pure = tmp_path / 'pure.pt'
    restored.save_weights(pure)
    assert isinstance(ActorCritic.load_weights(pure), EntityAttention)
    legacy = ActorCritic()
    torch.save({'format_version': TRAINING_CHECKPOINT_FORMAT_VERSION,
                'feature_schema_version': FEATURE_SCHEMA_VERSION, 'metadata': {'rules_version': RULESET_ID},
                'model_config': asdict(legacy.config), 'model_state_dict': legacy.state_dict()}, tmp_path / 'legacy.pt')
    assert type(load_frozen_actor_critic_checkpoint(tmp_path / 'legacy.pt')) is ActorCritic
    payload = torch.load(path, weights_only=True)
    payload['model_config']['architecture'] = 'unknown'
    torch.save(payload, path)
    with pytest.raises(ValueError, match='unknown model architecture'):
        load_frozen_actor_critic_checkpoint(path)
    payload['model_config']['architecture'] = 'entity_attention_v1'
    payload['model_state_dict'].pop(next(iter(payload['model_state_dict'])))
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match='Missing key'):
        load_frozen_actor_critic_checkpoint(path)


def test_offline_keeps_all_rounds_and_original_validation_split(tmp_path):
    games = [{'round_index': r, 'run_id': f'round-{r}', 'game_id': i, 'split': split,
              'decision_count': 20, 'policy_samples': 15, 'path': str(tmp_path / f'{r}-{i}.npz')}
             for r in range(10) for i, split in enumerate(('train','validation'))]
    index = make_index({'games': games}, 1024)
    config = OfflineConfig(source_run='unused', source_artifacts='unused',
                           model_config={'architecture':'entity_attention_v1'}, max_steps=8, validation_positions=50)
    fixed = freeze_validation(index, config)
    assert index.total_games == 20
    assert index.evicted_games == 0
    assert fixed == freeze_validation(index, config)
    assert fixed['composition_by_round'] == {str(r): 5 for r in range(10)}
    assert all(ref['game_id'] == 1 for ref in fixed['references'])
    assert len({(ref['round_index'], ref['game_id'], ref['decision_index']) for ref in fixed['references']}) == 50
