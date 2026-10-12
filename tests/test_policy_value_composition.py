from dataclasses import asdict

import pytest
import torch

from splendor.actor_critic import ActorCritic, MODEL_WEIGHTS_FORMAT_VERSION, TorchStateBatch, build_actor_critic, tensorize_batch
from splendor.contracts import DecisionRequest
from splendor.entity_attention import EntityAttention, EntityAttentionPolicyValue
from splendor.features import FEATURE_SCHEMA_VERSION, collate_encoded_decisions, encode_decision
from splendor.observation import build_observation
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.replay import RULESET_ID
from splendor.rules import legal_actions
from splendor.search_inference import CompiledForward


def test_frozen_branch_routing_and_compiled_tensor_core(tmp_path, initial_state, catalog):
    torch.set_num_threads(1)
    request = DecisionRequest(build_observation(initial_state, 0, catalog), legal_actions(initial_state, catalog))
    batch = tensorize_batch(collate_encoded_decisions([encode_decision(request)]))
    torch.manual_seed(17)
    policy = EntityAttention().eval()
    torch.manual_seed(23)
    value = EntityAttention().eval()
    model = EntityAttentionPolicyValue().eval()
    model.policy_model.load_state_dict(policy.state_dict())
    model.value_model.load_state_dict(value.state_dict())
    path = tmp_path / 'combined.pt'
    torch.save({'checkpoint_schema': 'splendor-frozen-actor-critic-v1',
        'format_version': MODEL_WEIGHTS_FORMAT_VERSION, 'feature_schema_version': FEATURE_SCHEMA_VERSION,
        'config': asdict(model.config), 'state_dict': model.state_dict(),
        'metadata': {'rules_version': RULESET_ID}}, path)
    model = load_frozen_actor_critic_checkpoint(path)
    assert isinstance(model, EntityAttentionPolicyValue)
    assert isinstance(ActorCritic.load_weights(path), EntityAttentionPolicyValue)
    assert isinstance(build_actor_critic(asdict(model.config)), EntityAttentionPolicyValue)
    state = TorchStateBatch(**{k: getattr(batch, k) for k in TorchStateBatch.__dataclass_fields__})
    core = CompiledForward(model, projection=None, buckets=(64,), compile_mode=None)
    tensor_names = tuple(k for k in batch.__dataclass_fields__ if k != 'feature_schema_version')
    with torch.inference_mode():
        expected_policy, expected_value = policy(batch)[0], value(batch)[1]
        torch.testing.assert_close(model(batch)[0], expected_policy, rtol=0, atol=0)
        torch.testing.assert_close(model(batch)[1], expected_value, rtol=0, atol=0)
        torch.testing.assert_close(model.policy_logits(batch), expected_policy, rtol=0, atol=0)
        torch.testing.assert_close(model.value_only(batch), expected_value, rtol=0, atol=0)
        torch.testing.assert_close(model.value_from_state(state), expected_value, rtol=0, atol=0)
        actual = core._policy_value_impl(64, *[getattr(batch, k) for k in tensor_names])
        torch.testing.assert_close(actual[0], expected_policy, rtol=0, atol=0)
        torch.testing.assert_close(actual[1], expected_value, rtol=0, atol=0)
        torch.testing.assert_close(core._policy_only_impl(64, *[getattr(batch, k) for k in tensor_names]), expected_policy, rtol=0, atol=0)
        torch.testing.assert_close(core._value_only_impl(None, *[getattr(state, k) for k in tensor_names[:-2]]), expected_value, rtol=0, atol=0)
    with pytest.raises(ValueError, match='actor_projection=false'):
        model.build_actor_projection()
    payload = torch.load(path, weights_only=True)
    payload['metadata']['rules_version'] = 'wrong-rules'
    torch.save(payload, path)
    with pytest.raises(ValueError, match='rules version mismatch'):
        load_frozen_actor_critic_checkpoint(path)
