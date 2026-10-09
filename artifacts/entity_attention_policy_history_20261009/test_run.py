"""Finite checks for sampling controls, real gradients and checkpoint resume."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from splendor.alphazero_replay import ReplayGame, ReplayIndex
from splendor.alphazero_train import AlphazeroLearner, LearnerConfig, alphazero_loss_sums, _torch_batch
from splendor.engine import GameEngine
from splendor.entity_attention import EntityAttention
from splendor.features import encode_decision
from splendor.hybrid_train import draw_split_batch
from splendor.rng import DeterministicRng
from splendor.selfplay_data import STATE_FIELDS, SelfPlaySample, collate_selfplay_samples

spec = importlib.util.spec_from_file_location('policy_history_probe', Path(__file__).with_name('run.py'))
probe = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = probe
spec.loader.exec_module(probe)


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_plans_reproduce_original_sampler_and_keep_value_shared():
    games = [ReplayGame(r, f'r{r}', 0, 'train', 11+r, 11+r, Path(str(r))) for r in range(5)]
    index = ReplayIndex(games)
    index.cache.sample = lambda game, decision: (game.key, decision)
    pools = probe.pools(index, [1, 2, 3, 4])
    rng = DeterministicRng(65)
    plans = probe.batch_plans(pools, rng.snapshot(), 110, steps=8)
    for i in range(8):
        batch = draw_split_batch(pools['recent'], pools['history'], rng, 256, 128)
        refs = np.asarray([(r.round_index, r.game_id, r.decision_index) for r in batch.references])
        np.testing.assert_array_equal(plans['policy_A'][i], refs[:256])
        np.testing.assert_array_equal(plans['value'][i], refs[256:])
        assert int(plans['rng_state'][i]) == rng.snapshot().state
        assert int(plans['rng_draws'][i]) == rng.snapshot().draws
    assert np.all(plans['policy_B'][:, :128, 0] == 4)
    assert set(plans['policy_B'][:, 128:, 0].ravel()) == {1, 2, 3}
    assert np.all(plans['policy_C'][:, :, 0] == 4)
    assert np.all(plans['value'][:, 128:, 0] == 0)


def samples():
    result = []
    for seed, weight in [(111, 1.), (112, 2.), (113, 0.)]:
        decision = GameEngine.new(seed).decision()
        encoded = encode_decision(decision)
        n = len(decision.legal_actions)
        arrays = {name: np.stack([getattr(encoded, name)]) for name in STATE_FIELDS}
        target = np.zeros(n, dtype=np.float32); target[seed % n] = 1.
        arrays.update(candidate_offsets=np.asarray([0, n]), candidate_features=encoded.candidate_features,
                      policy_target=target, behavior_policy=target, value_target=np.asarray([1 if seed%2 else -1]),
                      policy_weight=np.asarray([weight], dtype=np.float32), value_weight=np.asarray([1.], dtype=np.float32),
                      decision_index=np.asarray([0]), player_id=np.asarray([0]))
        result.append(SelfPlaySample(arrays, {'game_id': seed}, 0))
    return result


def test_policy_gradient_matches_weighted_autograd_and_leaves_critic_out():
    data = samples(); lookup = {(1, i): SimpleNamespace(round_index=1, game_id=i, run_id='r1') for i in range(3)}
    index = SimpleNamespace(cache=SimpleNamespace(sample=lambda game, decision: data[game.game_id]))
    store = SimpleNamespace(transform=lambda sample: sample)
    refs = np.asarray([(1, i, 0) for i in [0, 1, 2, 1, 0, 2]])
    torch.manual_seed(15)
    learner = AlphazeroLearner(EntityAttention(), LearnerConfig(device='cpu'), sampling_rng=DeterministicRng(9))
    got, metrics = probe.policy_gradient(learner, index, lookup, store, refs)
    batch = collate_selfplay_samples([data[i] for i in [0, 1, 2, 1, 0, 2]])
    tensors = _torch_batch(batch, learner.device)
    logits, values = learner.model(batch.decisions)
    terms = alphazero_loss_sums(logits, values, **tensors)
    terms.policy_loss.backward()
    expected = probe.vector(probe.policy_parameters(learner.model), grad=True)
    np.testing.assert_allclose(got, expected, rtol=2e-5, atol=2e-7)
    assert metrics['policy_weight_sum'] == 6.
    assert all(p.grad is None for p in learner.model.critic.parameters())


def test_inherited_adamw_can_oppose_latest_descent_and_checkpoint_resume_is_exact(tmp_path):
    parameter = torch.nn.Parameter(torch.tensor([1.]))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3, weight_decay=1e-4)
    for _ in range(10):
        parameter.grad = torch.tensor([1.]); optimizer.step()
    before = parameter.detach().numpy().copy()
    parameter.grad = torch.tensor([-1.]); optimizer.step()
    delta = parameter.detach().numpy()-before
    assert probe.geometry(delta, np.asarray([1.]))['cosine'] < 0  # Latest descent is +1.
    assert probe.geometry(np.asarray([-1.]), delta)['dot'] > 0
    batch = collate_selfplay_samples(samples())
    torch.manual_seed(61)
    learner = AlphazeroLearner(EntityAttention(), LearnerConfig(device='cpu'), sampling_rng=DeterministicRng(12))
    learner.replay_identity = {'digest': 'finite-test'}
    update = lambda obj: obj.train_split_step(batch, batch, references=(), policy_max_grad_norm=1., value_max_grad_norm=1.)
    update(learner)
    cp = tmp_path/'resume.pt'; learner.save(cp)
    update(learner)
    resumed = probe.restore(cp, 'cpu', {'digest': 'finite-test'}, require_identity=True)
    update(resumed)
    for name, value in learner.model.state_dict().items():
        assert torch.equal(value, resumed.model.state_dict()[name])
    for k, state in learner.optimizer.state_dict()['state'].items():
        for name, value in state.items():
            assert torch.equal(value, resumed.optimizer.state_dict()['state'][k][name])
    with pytest.raises(ValueError, match='identity'):
        probe.restore(cp, 'cpu', {'digest': 'different'}, require_identity=True)
