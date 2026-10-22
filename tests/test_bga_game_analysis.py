"""Small hand-calculated checks for BGA root evidence aggregation."""
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture(scope='module')
def analysis():
    path = Path(__file__).resolve().parents[1] / 'artifacts/bga_927500015_r20_analysis_20261020/analyze.py'
    spec = importlib.util.spec_from_file_location('bga_analysis', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(visits, q, best):
    return dict(visits=visits, q=q, choice=best, prior=[.2, .8], max_tree_depth=3, transition_cap_hits=0)


def test_seed_average_does_not_turn_unvisited_q_into_zero(analysis):
    out = analysis.combine([row([100, 0], [.2, None], 0), row([1, 99], [.8, -.4], 1)])
    assert out['q'] == pytest.approx([.5, -.4])  # equal seed weights, not pooled visit weights
    assert out['q_seed_coverage'] == [2, 1]
    assert out['minimum_visits'] == [1, 0]
    assert out['visit_share'] == pytest.approx([.505, .495])
    assert out['choice'] == 0


def test_native_visit_ties_use_prior_before_action_order(analysis):
    assert analysis.choice([12, 12], [.1, .9]) == 1
    assert analysis.choice([12, 12], [.5, .5]) == 0
    out = analysis.combine([row([50, 50], [None, None], 1)])
    assert out['choice'] == 1
    assert out['q'] == [None, None]


def test_budget_and_distribution_comparison(analysis):
    panels = analysis.panels()
    assert len(panels) == 8
    assert sum(p['simulations'] * 76 * 3 for p in panels) == 15876096
    assert {p['simulations'] // p['determinizations'] for p in panels if p['determinizations'] == 4} == {256, 4096}
    assert analysis.js([2, 8], [20, 80]) == pytest.approx(0.)
    assert analysis.js([1, 0], [0, 1]) == pytest.approx(.6931471805599453)


@pytest.fixture(scope='module')
def summary_analysis():
    path = Path(__file__).resolve().parents[1] / 'artifacts/bga_927500015_r20_analysis_20261020/summarize.py'
    spec = importlib.util.spec_from_file_location('bga_summary_analysis', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def search(visits, q):
    return dict(visits=visits, q=q, choice=visits.index(max(visits)), max_tree_depth=3, transition_cap_hits=0,
                value_sums=[n * v if v is not None else 0. for n, v in zip(visits, q)])


def test_value_difference_is_paired_before_averaging(summary_analysis):
    estimate = summary_analysis.estimate([.3, .7], 0, [search([2, 8], [-.2, .2]),
        search([0, 10], [None, .9]), search([4, 6], [.4, .6])])
    assert estimate['actual_probability'] == pytest.approx(.2)
    assert estimate['delta_score_pp'] == pytest.approx(-15.)
    assert estimate['value_pair_seeds'] == 2
    assert estimate['actual_q_seed_sd'] is None


def test_human_can_have_higher_q_than_visit_choice(summary_analysis):
    estimate = summary_analysis.estimate([.9, .1], 1, [search([9, 1], [.2, .4])] * 3)
    assert estimate['choice'] == 0
    assert estimate['delta_score_pp'] == pytest.approx(10.)


def test_steps_have_equal_weight_despite_missing_seed_q(summary_analysis):
    a = summary_analysis.estimate([.3, .7], 0, [search([1, 9], [0., .4]),
        search([0, 10], [None, .4]), search([0, 10], [None, .4])])
    b = summary_analysis.estimate([.3, .7], 0, [search([1, 9], [.4, .6])] * 3)
    rows = [dict(r, bga_move=i, agree=False, greedy_choice=1) for i, r in enumerate([a, b])]
    assert summary_analysis.aggregate(rows)['mean_delta_score_pp'] == pytest.approx(-15.)
