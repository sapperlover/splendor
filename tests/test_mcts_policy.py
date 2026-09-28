from __future__ import annotations

import inspect
import math
import random
from dataclasses import replace

import pytest
import torch

from splendor.actor_critic import ActorCritic
from splendor.contracts import DecisionRequest
from splendor.engine import GameEngine
from splendor.evaluation import SerializablePolicySpec, _create_serializable_policy
from splendor.mcts_policy import (
    MCTSConfig,
    PUCTSearchPolicy,
    _Edge,
    _Node,
    _mean_world_visits,
    _temperature_policy,
)
from splendor.result import TerminalResult


def zero_model() -> ActorCritic:
    model = ActorCritic().eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    return model


@pytest.mark.parametrize(('field', 'value'), [
    ('simulations', 0), ('simulations', True), ('determinizations', 0),
    ('determinizations', 3), ('c_puct', 0), ('c_puct', math.nan),
    ('c_puct', math.inf), ('seed', True), ('max_simulation_transitions', False),
    ('root_noise_fraction', -0.1), ('root_noise_fraction', math.inf),
    ('dirichlet_alpha', 0), ('dirichlet_alpha', math.nan),
])
def test_bad_config(field, value):
    with pytest.raises(ValueError):
        MCTSConfig(**({'simulations': 2, 'determinizations': 2} | {field: value}))


def test_budget_and_equal_world_target(catalog):
    request = GameEngine.new(42, catalog=catalog).decision()
    assert isinstance(request, DecisionRequest)
    policy = PUCTSearchPolicy(zero_model(), config=MCTSConfig(simulations=5, determinizations=2), catalog=catalog)
    result = policy.search(request)
    assert result.simulation_budgets == (3, 2)
    assert tuple(map(sum, result.per_world_root_visit_counts)) == (3, 2)
    assert sum(result.root_visit_counts) == 5
    assert len(result.root_visit_counts) == len(request.legal_actions)
    assert result.policy_target == _mean_world_visits(result.per_world_root_visit_counts, result.simulation_budgets)
    assert sum(result.policy_target) == pytest.approx(1)
    assert sum(result.behavior_policy) == pytest.approx(1)
    assert policy.stats.model_batches >= 2
    assert _mean_world_visits(((2, 1), (0, 2)), (3, 2)) == pytest.approx((1/3, 2/3))
    assert (2/5, 3/5) != pytest.approx((1/3, 2/3))


def test_select_uses_root_perspective_and_unvisited_q(catalog):
    request = GameEngine.new(3, catalog=catalog).decision()
    assert isinstance(request, DecisionRequest)
    request = replace(request, legal_actions=request.legal_actions[:2])
    policy = PUCTSearchPolicy(zero_model(), config=MCTSConfig(simulations=2, determinizations=1), catalog=catalog)
    engine = GameEngine.new(3, catalog=catalog)
    node = _Node(engine, request, (0.5, 0.5), [_Edge(10, 9), _Edge(10, -9)])
    assert policy._select(node, engine.current_player) == 0
    assert policy._select(node, 1-engine.current_player) == 1
    node.edges = [_Edge(), _Edge()]
    node.prior = (0.1, 0.9)
    assert policy._select(node, engine.current_player) == 1


def test_noise_temperature_and_rng_isolation(catalog):
    request = GameEngine.new(11, catalog=catalog).decision()
    assert isinstance(request, DecisionRequest)
    policy = PUCTSearchPolicy(zero_model(), config=MCTSConfig(simulations=7, determinizations=2), catalog=catalog)
    global_before = random.getstate()
    torch_before = torch.random.get_rng_state().clone()
    noisy = policy.search(request, add_root_noise=True, temperature=0.8, sampling_seed=123)
    again = policy.search(request, add_root_noise=True, temperature=0.8, sampling_seed=123)
    assert noisy == again
    assert noisy.root_network_prior != noisy.root_used_prior
    assert noisy.policy_target == again.policy_target
    assert sum(noisy.behavior_policy) == pytest.approx(1)
    clean = policy.search(request)
    assert policy.choose(request) is request.legal_actions[clean.chosen_candidate_index]
    assert policy.last_search_result == clean
    assert random.getstate() == global_before
    assert torch.equal(torch.random.get_rng_state(), torch_before)
    assert _temperature_policy((0.25, 0.75), 0.5) == pytest.approx((0.1, 0.9))
    assert _temperature_policy((0.25, 0.75), 1e-320) == (0.0, 1.0)
    tiny_alpha = PUCTSearchPolicy(zero_model(), config=MCTSConfig(simulations=2, determinizations=1, dirichlet_alpha=1e-300), catalog=catalog)
    assert sum(tiny_alpha.search(request, add_root_noise=True).root_used_prior) == pytest.approx(1)
    huge_alpha = PUCTSearchPolicy(zero_model(), config=MCTSConfig(simulations=2, determinizations=1, dirichlet_alpha=1e308), catalog=catalog)
    assert sum(huge_alpha.search(request, add_root_noise=True).root_used_prior) == pytest.approx(1)
    for bad in (-1, math.nan, math.inf, True):
        with pytest.raises(ValueError):
            policy.search(request, temperature=bad)


def test_forced_root_fast_path_and_stable_result(catalog):
    engine = GameEngine.new(31, catalog=catalog)
    request = engine.decision()
    assert isinstance(request, DecisionRequest)
    policy = PUCTSearchPolicy(zero_model(), config=MCTSConfig(simulations=2, determinizations=1), catalog=catalog)
    previous = policy.search(request)
    forced = replace(request, legal_actions=request.legal_actions[:1])
    result = policy.search(forced)
    assert result.root_visit_counts == (0,)
    assert result.policy_target == result.behavior_policy == (1.0,)
    assert result.simulations == 0 and not result.searchable
    assert policy.last_search_result == result and previous.searchable
    assert tuple(inspect.signature(PUCTSearchPolicy.choose).parameters) == ('self', 'request')


def _fake_job(policy, request, engine, prior, monkeypatch):
    monkeypatch.setattr('splendor.mcts_policy.determinize_request',
                        lambda *args, **kwargs: engine)
    job = policy.start_job(request)
    pending = job.pending_requests()
    assert len(pending) == 1 and pending[0].kind == 'policy_value'
    job.submit(pending[0].request_id, prior, 0.0, prior_is_normalized=True)
    return job


def test_terminal_edge_reuse_backs_up_once_per_simulation(catalog, monkeypatch):
    request = GameEngine.new(5, catalog=catalog).decision()
    request = replace(request, legal_actions=request.legal_actions[:2])
    terminal = TerminalResult((15, 0), (3, 0), (0,), (10, 10))
    class TerminalEngine:
        current_player = 0
        state_version = 0
        def fork(self):
            return self
        def apply_candidate_folded(self, index, *, max_transitions, action_trace=None, state_version):
            return terminal, 1, 0
    policy = PUCTSearchPolicy(zero_model(), config=MCTSConfig(simulations=2, determinizations=1), catalog=catalog)
    job = _fake_job(policy, request, TerminalEngine(), (1.0, 0.0), monkeypatch)
    assert job.done and job.result.root_visit_counts == (2, 0)
    assert policy.stats.terminal_hits == 2


def test_evaluation_spec_and_default_seed(tmp_path, monkeypatch):
    cfg = {'simulations': 3, 'determinizations': 1}
    spec = SerializablePolicySpec.mcts_checkpoint('m', tmp_path / 'model.pt', search_config=cfg)
    assert SerializablePolicySpec.from_dict(spec.to_dict()) == spec
    from splendor import evaluation
    monkeypatch.setattr(evaluation, '_PARALLEL_MODEL_CACHE', {(spec.checkpoint, 'cpu'): zero_model()})
    policy = _create_serializable_policy(spec, seed=17)
    assert isinstance(policy, PUCTSearchPolicy)
    assert policy.config.seed == 17
    with pytest.raises((TypeError, ValueError)):
        SerializablePolicySpec.mcts_checkpoint('bad', tmp_path / 'model.pt', search_config={'surprise': 1})
    legacy = SerializablePolicySpec.search_checkpoint('old', tmp_path / 'model.pt', search_config={'depth': 1})
    assert _create_serializable_policy(legacy, seed=17).config.seed == 17


class ScriptEngine:
    state_version = 0
    def __init__(self, request, *, player=0, next_player=0, terminal=None):
        self.request = request
        self.current_player = player
        self.next_player = next_player
        self.terminal = terminal

    def fork(self):
        return ScriptEngine(self.request, player=self.current_player,
                            next_player=self.next_player, terminal=self.terminal)

    def apply_candidate(self, index, *, state_version):
        self.current_player = self.next_player
        return self.request if self.terminal is None else self.terminal

    def apply_candidate_folded(self, index, *, max_transitions, action_trace=None, state_version):
        if action_trace is not None:
            action_trace.append(self.request.legal_actions[index])
        return self.apply_candidate(index, state_version=state_version), 1, 0


@pytest.mark.parametrize('leaf_player,expected', [(0, 0.4), (1, -0.4)])
def test_new_leaf_stops_and_backs_up_once(catalog, monkeypatch, leaf_player, expected):
    request = GameEngine.new(4, catalog=catalog).decision()
    request = replace(request, legal_actions=request.legal_actions[:2])
    policy = PUCTSearchPolicy(zero_model(), config=MCTSConfig(simulations=1, determinizations=1,
                              fold_forced_phases=False), catalog=catalog)
    job = _fake_job(policy, request, ScriptEngine(request, next_player=leaf_player),
                    (1.0, 0.0), monkeypatch)
    root = job._worlds[0].root
    pending = job.pending_requests()
    assert len(pending) == 1 and pending[0].kind == 'policy_value'
    job.submit(pending[0].request_id, (1.0, 0.0), 0.4, prior_is_normalized=True)
    assert job.done and root.edges[0].visits == 1
    assert root.edges[0].value_sum == pytest.approx(expected)
    assert root.edges[1].visits == 0 and root.edges[0].child is not None


def test_transition_cap_uses_critic_and_not_terminal(catalog, monkeypatch):
    request = GameEngine.new(6, catalog=catalog).decision()
    request = replace(request, legal_actions=request.legal_actions[:2])
    policy = PUCTSearchPolicy(zero_model(), config=MCTSConfig(simulations=1,
                              determinizations=1, max_simulation_transitions=1), catalog=catalog)
    job = _fake_job(policy, request, ScriptEngine(request), (1.0, 0.0), monkeypatch)
    root = job._worlds[0].root
    pending = job.pending_requests()
    assert len(pending) == 1 and pending[0].kind == 'value_only'
    job.submit(pending[0].request_id, None, 0.35)
    assert root.edges[0].value_sum == pytest.approx(0.35)
    assert root.edges[0].child is None and root.edges[0].terminal is None
    assert policy.stats.transition_cap_hits == 1 and policy.stats.terminal_hits == 0


def test_paired_serial_parallel_resume_and_fingerprint(tmp_path):
    from dataclasses import asdict
    from splendor.evaluation import FixedEvaluationConfig, run_fixed_evaluation
    from splendor.features import FEATURE_SCHEMA_VERSION
    from splendor.ppo import TRAINING_CHECKPOINT_FORMAT_VERSION
    from splendor.replay import RULESET_ID

    checkpoint = tmp_path / 'zero.pt'
    model = zero_model()
    torch.save({
        'format_version': TRAINING_CHECKPOINT_FORMAT_VERSION,
        'feature_schema_version': FEATURE_SCHEMA_VERSION,
        'metadata': {'rules_version': RULESET_ID},
        'model_config': asdict(model.config),
        'model_state_dict': model.state_dict(),
    }, checkpoint)
    spec_a = SerializablePolicySpec.mcts_checkpoint('m1', checkpoint, search_config={'simulations': 3, 'determinizations': 2})
    spec_b = SerializablePolicySpec.mcts_checkpoint('m2', checkpoint, search_config={'simulations': 2, 'determinizations': 1})
    serial = FixedEvaluationConfig(spec_a, spec_b, (9001,), workers=1, bootstrap_resamples=0)
    parallel = replace(serial, workers=2)
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        first = run_fixed_evaluation(serial, tmp_path / 'serial.json')
        second = run_fixed_evaluation(parallel, tmp_path / 'parallel.json')
    finally:
        torch.set_num_threads(previous_threads)
    assert first.summary['passed'] and second.summary['passed']
    assert first.summary['valid_games'] == second.summary['valid_games'] == 2
    for left, right in zip(first.pair_records[0], second.pair_records[0]):
        for key in ('status', 'outcome_for_a', 'final_state_digest', 'decision_count'):
            assert left.get(key) == right.get(key)
        for side in ('policy_a_stats', 'policy_b_stats'):
            stable = lambda row: {k: v for k, v in row[side].items() if not k.endswith('_seconds')}
            assert stable(left) == stable(right)
    resumed = run_fixed_evaluation(serial, tmp_path / 'serial.json', resume=True)
    assert resumed.summary['valid_games'] == 2
    with pytest.raises(ValueError):
        run_fixed_evaluation(replace(serial, policy_a=SerializablePolicySpec.mcts_checkpoint('m1', checkpoint, search_config={'simulations': 4, 'determinizations': 2})), tmp_path / 'serial.json', resume=True)


def test_low_prior_winning_terminal_can_overtake_high_prior_loss(catalog, monkeypatch):
    request = GameEngine.new(12, catalog=catalog).decision()
    request = replace(request, legal_actions=request.legal_actions[:2])
    losing = TerminalResult((0, 15), (0, 3), (1,), (10, 10))
    winning = TerminalResult((15, 0), (3, 0), (0,), (10, 10))

    class TerminalEngine:
        current_player = 0
        state_version = 0
        def fork(self):
            return self
        def apply_candidate_folded(self, index, *, max_transitions, action_trace=None, state_version):
            return (losing if index == 0 else winning), 1, 0

    policy = PUCTSearchPolicy(zero_model(), config=MCTSConfig(simulations=20,
                              determinizations=1), catalog=catalog)
    job = _fake_job(policy, request, TerminalEngine(), (0.9, 0.1), monkeypatch)
    assert job.done and job.result.root_visit_counts[1] > job.result.root_visit_counts[0]
    assert policy.stats.terminal_hits == 20


def test_forced_chain_uses_resulting_actor(catalog, monkeypatch):
    request = GameEngine.new(13, catalog=catalog).decision()
    root_request = replace(request, legal_actions=request.legal_actions[:2])
    forced_request = replace(request, legal_actions=request.legal_actions[:1])

    class ForcedEngine:
        state_version = 0
        def __init__(self, step=0):
            self.step = step
            self.current_player = 0 if step < 2 else 1
        def fork(self):
            return ForcedEngine(self.step)
        def apply_candidate_folded(self, index, *, max_transitions, action_trace=None, state_version):
            self.step += 2
            self.current_player = 1
            if action_trace is not None:
                action_trace.extend((root_request.legal_actions[index], forced_request.legal_actions[0]))
            return root_request, 2, 1

    policy = PUCTSearchPolicy(zero_model(), config=MCTSConfig(simulations=1,
                              determinizations=1), catalog=catalog)
    job = _fake_job(policy, root_request, ForcedEngine(), (1.0, 0.0), monkeypatch)
    root = job._worlds[0].root
    pending = job.pending_requests()[0]
    job.submit(pending.request_id, (0.5, 0.5), 0.4, prior_is_normalized=True)
    assert root.edges[0].value_sum == pytest.approx(-0.4)
    assert root.edges[0].child.engine.current_player == 1
    assert policy.stats.folded_actions == 1 and policy.stats.state_transitions == 2
