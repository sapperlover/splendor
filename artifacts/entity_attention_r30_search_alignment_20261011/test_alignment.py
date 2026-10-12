import runpy
from pathlib import Path

import numpy as np
import pytest

module = runpy.run_path(str(Path(__file__).with_name("analyze.py")))


def test_top1_ties_padding_and_kl_direction():
    visits = np.array([[5, 5, 0], [8, 2, 0]])
    logp = np.array([[np.log(.25), np.log(.75), -np.inf], [np.log(.5), np.log(.5), -np.inf]])
    result = module["policy_metrics"](visits, logp)
    assert result["top1_strict"].tolist() == [0., 1.]
    assert result["top1_tie_aware"].tolist() == [1., 1.]
    assert result["search_top1_tied"].tolist() == [1., 0.]
    assert result["visit_kl"][0] == pytest.approx(.5*np.log(.5/.25)+.5*np.log(.5/.75))
    assert result["visit_kl"][1] == pytest.approx(.8*np.log(.8/.5)+.2*np.log(.2/.5))


def test_two_seed_decomposition_and_no_reverse_kl_smoothing():
    a, b = np.array([[10, 0], [8, 2]]), np.array([[0, 10], [4, 6]])
    lp = np.log([[.2, .8], [.5, .5]])
    x = module["policy_metrics"](a, lp)
    y = module["policy_metrics"](b, lp)
    variation = module["search_variation"](a, b, lp)
    np.testing.assert_allclose((x["visit_kl"]+y["visit_kl"])/2,
        variation["mean_two_targets_kl"]+variation["repeat_visit_jsd"], rtol=1e-12, atol=1e-12)
    assert variation["repeat_top1_agreement"].tolist() == [0., 0.]
    assert variation["repeat_visit_jsd"][0] == pytest.approx(np.log(2))


def test_game_bootstrap_preserves_position_weights_and_duplicate_games():
    result = module["cluster_mean"]([0, 0, 1], [1, 1, 2], 33520000)
    assert result["estimate"] == pytest.approx(1/3)
    assert result["lower"] == 0
    assert result["upper"] == 1
    assert result["games"] == 2
