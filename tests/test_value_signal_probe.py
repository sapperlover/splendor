"""Statistical identities and perspective/seed checks for the bounded signal probe."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

PATH=Path(__file__).resolve().parents[1]/"artifacts/entity_attention_r43_value_signal_64_20261014/run.py"
spec=importlib.util.spec_from_file_location("value_signal_probe",PATH)
probe=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=probe
spec.loader.exec_module(probe)


def test_signal_decomposition_and_finite_mean_correction():
    values=np.array([.2,-.4,.9])
    outcomes=np.array([[1,-1,1,-1],[-1,-1,-1,1],[1,1,1,1]])
    m=probe.signal_metrics(values,outcomes)
    assert m["single_return_mse"]==pytest.approx(m["within_root_variance"]+m["corrected_expected_value_mse"])
    assert m["mean_target_mc_variance"]==pytest.approx(outcomes.var(axis=1,ddof=1).mean()/4)
    assert m["between_root_signal_variance"]==pytest.approx(outcomes.mean(axis=1).var(ddof=1)-m["mean_target_mc_variance"])
    assert m["mixed_win_loss_roots"]==2


def test_deterministic_targets_have_zero_noise():
    values=np.array([-1,0,1.])
    outcomes=np.repeat(values[:,None],16,axis=1)
    m=probe.signal_metrics(values,outcomes)
    assert m["within_root_variance"]==0
    assert m["corrected_expected_value_mse"]==0
    assert m["signal_fraction_of_return_variance"]==1


def test_finite_sample_negative_correction_is_not_clipped():
    values=np.zeros(3)
    outcomes=np.array([[1,-1]]*3)
    m=probe.signal_metrics(values,outcomes)
    assert m["corrected_expected_value_mse"]==-1
    assert m["noise_fraction_of_model_mse"]==2


def test_seed_streams_are_distinct_and_reproducible():
    streams=[probe.rollout_seeds(i,j) for i in range(64) for j in range(16)]
    all_seeds=[seed for row in streams for seed in row.values()]
    assert len(set(all_seeds))==3072
    assert probe.rollout_seeds(3,4)==probe.rollout_seeds(3,4)


def test_rejects_invalid_outcome_population():
    with pytest.raises(ValueError):probe.signal_metrics([0],[[2,1]])
    with pytest.raises(ValueError):probe.signal_metrics([0],[[1]])


def test_bootstrap_resamples_independent_roots_within_stages():
    values=np.array([-.2,.2,.8])
    outcomes=np.array([[-1,-1,1,1],[-1,1,1,1],[1,1,1,1]])
    ci=probe.bootstrap(values,outcomes,["early","middle","late"],resamples=50)
    m=probe.signal_metrics(values,outcomes)
    for key,(lo,hi) in ci.items():
        assert lo==pytest.approx(m[key])
        assert hi==pytest.approx(m[key])
