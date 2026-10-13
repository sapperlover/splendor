from dataclasses import replace

import numpy as np
import pytest

from splendor.alphazero_replay import ReplayReference
from splendor.policy_surprise import PolicySurpriseConfig, PolicySurpriseWeighting, game_policy_surprise
from splendor.selfplay_data import SelfPlaySample


def arrays():
    return {"candidate_offsets": np.array([0, 2, 4, 5]),
            "policy_target": np.array([.5, .5, 1., 0., 1.], dtype=np.float32),
            "root_network_prior": np.array([.5, .5, .25, .75, 1.], dtype=np.float32),
            "root_used_prior": np.array([.1, .9, .9, .1, 1.], dtype=np.float32),
            "policy_weight": np.array([1., 1., 0.], dtype=np.float32)}


def test_per_game_mixture_uses_unnoised_teacher_and_preserves_mass():
    raw = arrays()
    surprise, weights = game_policy_surprise(raw, PolicySurpriseConfig(.5))
    np.testing.assert_allclose(surprise, [0., np.log(4.), 0.])
    np.testing.assert_allclose(weights, [.5, 1.5, 0.])
    assert weights.sum() == raw['policy_weight'].sum()
    np.testing.assert_array_equal(raw['policy_weight'], [1., 1., 0.])
    raw['root_used_prior'][:] = 0
    np.testing.assert_array_equal(game_policy_surprise(raw, PolicySurpriseConfig(.5))[1], weights)


def test_zero_surprise_zero_prior_and_disabled_fraction_are_finite():
    raw = arrays()
    raw['root_network_prior'] = raw['policy_target'].copy()
    np.testing.assert_array_equal(game_policy_surprise(raw, PolicySurpriseConfig())[1], raw['policy_weight'])
    raw['root_network_prior'][2:4] = [0., 1.]
    surprise, weights = game_policy_surprise(raw, PolicySurpriseConfig())
    assert np.isfinite(surprise).all() and np.isfinite(weights).all()
    assert surprise[1] == pytest.approx(100 * np.log(10))
    np.testing.assert_array_equal(game_policy_surprise(raw, PolicySurpriseConfig(0))[1], raw['policy_weight'])
    raw['policy_weight'][:] = 0
    np.testing.assert_array_equal(game_policy_surprise(raw, PolicySurpriseConfig())[1], 0)


@pytest.mark.parametrize('fraction', [-1, 1.01, True, float('nan'), float('inf')])
def test_invalid_fraction(fraction):
    with pytest.raises(ValueError, match='fraction'):
        PolicySurpriseConfig(fraction)


def test_weight_cache_isolates_runs_and_changes_only_policy_batch_fields():
    from splendor.selfplay_data import SelfPlayBatch
    raw = arrays()
    samples = [SelfPlaySample(raw, {}, i) for i in range(3)]
    refs = [ReplayReference(1, 'a', 0, i) for i in range(3)]
    batch = SelfPlayBatch(None, None, None, np.array([1, -1, 1]), raw['policy_weight'],
                         np.ones(3), ())
    weighter = PolicySurpriseWeighting(PolicySurpriseConfig())
    weighted, stats = weighter.apply(batch, samples, refs)
    assert weighted.value_weights is batch.value_weights and weighted.value_targets is batch.value_targets
    assert weighted.policy_targets is batch.policy_targets
    np.testing.assert_array_equal(weighted.policy_weights, [.5, 1.5, 0])
    assert stats['effective_policy_positions'] == pytest.approx(1.6)
    other = {**raw, 'root_network_prior': raw['policy_target'].copy()}
    weighted, _ = weighter.apply(batch, [replace(s, arrays=other) for s in samples],
                                [replace(r, run_id='b') for r in refs])
    np.testing.assert_array_equal(weighted.policy_weights, raw['policy_weight'])
