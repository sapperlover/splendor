from copy import deepcopy

import numpy as np
import pytest
import torch

from splendor.alphazero_train import AlphazeroLearner, LearnerConfig, alphazero_loss_sums
from splendor.contracts import DecisionRequest
from splendor.entity_attention import EntityAttention
from splendor.features import collate_encoded_decisions, encode_decision
from splendor.observation import build_observation
from splendor.rng import DeterministicRng
from splendor.rules import legal_actions
from splendor.selfplay_data import SelfPlayBatch


@pytest.fixture
def batch(initial_state, catalog):
    request = DecisionRequest(build_observation(initial_state, 0, catalog), legal_actions(initial_state, catalog))
    decisions = collate_encoded_decisions([encode_decision(request)] * 2)
    targets = np.zeros(decisions.candidate_mask.shape, dtype=np.float32)
    targets[:, 0] = 1
    return SelfPlayBatch(decisions, targets, targets.copy(), np.array([1., -1.], dtype=np.float32),
                         np.ones(2, dtype=np.float32), np.ones(2, dtype=np.float32), ())


def learner(model):
    return AlphazeroLearner(model, LearnerConfig(device="cpu", batch_size=2, history_batch_size=1,
                            learning_rate=2e-5, max_grad_norm=0.01), sampling_rng=DeterministicRng(17))


def test_joint_same_batch_matches_original_update_and_resumes(batch, tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(17)
    original = learner(EntityAttention())
    separate = learner(deepcopy(original.model))
    a = original.train_step(batch, references=())
    b = separate.train_step(batch, value_batch=batch, references=())
    for key in ("policy_loss", "value_loss", "grad_norm", "value_zero_mse"):
        assert b[key] == pytest.approx(a[key], rel=2e-6)
    for name, param in original.model.state_dict().items():
        torch.testing.assert_close(separate.model.state_dict()[name], param, rtol=1e-5, atol=1e-7)
    path = tmp_path / "joint.pt"
    separate.save(path)
    separate.train_step(batch, value_batch=batch, references=())
    resumed = AlphazeroLearner.resume(path, separate.config, replay_identity=separate.replay_identity,
                                     round_index=0, round_steps_target=0)
    resumed.train_step(batch, value_batch=batch, references=())
    for name, param in separate.model.state_dict().items():
        torch.testing.assert_close(resumed.model.state_dict()[name], param, rtol=0, atol=0)


def test_joint_value_batch_reaches_encoder_with_global_clip(batch):
    torch.set_num_threads(1)
    torch.manual_seed(23)
    actual = learner(EntityAttention())
    expected = learner(deepcopy(actual.model))
    # Distinct labels, including unequal value weights, detect accidental use
    # of the policy batch's value labels or denominators.
    value_batch = SelfPlayBatch(batch.decisions, batch.policy_targets, batch.behavior_policies,
                                np.array([-0.5, 0.25], dtype=np.float32), batch.policy_weights,
                                np.array([1., 2.], dtype=np.float32), ())
    batch.policy_weights[:] = 0
    before = {k: p.clone() for k, p in actual.model.state_dict().items()}
    logits, values = expected.model(value_batch.decisions)
    terms = alphazero_loss_sums(logits, values,
        candidate_mask=torch.as_tensor(batch.decisions.candidate_mask),
        policy_targets=torch.as_tensor(batch.policy_targets), value_targets=torch.as_tensor(value_batch.value_targets),
        policy_weights=torch.as_tensor(batch.policy_weights), value_weights=torch.as_tensor(value_batch.value_weights))
    expected.optimizer.zero_grad(set_to_none=True)
    terms.total(expected.config).backward()
    norm = float(torch.nn.utils.clip_grad_norm_(expected.model.parameters(), expected.config.max_grad_norm))
    expected.optimizer.step()
    result = actual.train_step(batch, value_batch=value_batch, references=())
    assert result["grad_norm"] == pytest.approx(norm, rel=2e-6)
    assert result["value_weight_sum"] == 3
    assert result["value_target_sum"] == 0
    assert result["value_zero_mse"] == pytest.approx(0.125)
    assert result["grad_clipped"]
    assert any(not torch.equal(p, before[k]) for k, p in actual.model.state_dict().items()
               if not k.startswith(("critic.", "actor.")))
    for name, param in expected.model.state_dict().items():
        torch.testing.assert_close(actual.model.state_dict()[name], param, rtol=1e-5, atol=1e-7)
