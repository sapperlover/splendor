"""Batched FP32 search runner: equality with the scalar path and guard rails."""

from __future__ import annotations

import pytest
import torch

from splendor.batched_search import BatchedSearchRunner
from splendor.benchmark_search import fixed_requests
from splendor.mcts_policy import MCTSConfig, PUCTSearchPolicy
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.search_policy import PolicyValueSearchPolicy, SearchConfig

# A batch-shape change may move float32 outputs by a few ulps; batched and
# scalar searches must still pick the same action with the same logical work.
BATCH_TOLERANCE = 1e-4


@pytest.fixture(scope="module")
def model():
    torch.set_num_threads(1)
    return load_frozen_actor_critic_checkpoint(
        "runs/5c_main_seed1/checkpoints/local_005000000_lifetime_005512112.pt",
        device="cpu",
    )


def test_batched_d6_matches_sequential(model) -> None:
    requests = [request for _, request, _ in fixed_requests()]
    config = SearchConfig(depth=3, top_k=3, seed=11, fold_forced_phases=True)
    policy = PolicyValueSearchPolicy(model, config=config, device="cpu", engine_mode="fast")
    sequential = []
    logical = []
    seen = 0
    for request in requests:
        sequential.append(policy.choose(request))
        logical.append(policy.stats.model_positions - seen)
        seen = policy.stats.model_positions

    runner = BatchedSearchRunner(policy, max_batch_size=4)
    batched = runner.choose_many(policy, requests)

    assert batched == sequential
    assert len(runner.last_sessions) == len(requests)
    for session, expected_action, expected_logical in zip(
        runner.last_sessions, sequential, logical
    ):
        assert session.chosen == expected_action
        assert session.policy.stats.model_positions == expected_logical
    assert runner.stats.physical_positions > 0
    assert runner.stats.batch_sizes


def test_batched_d6_root_values_track_scalar_path(model) -> None:
    requests = [request for _, request, _ in fixed_requests()][:3]
    config = SearchConfig(depth=2, top_k=3, seed=23, fold_forced_phases=True)
    policy = PolicyValueSearchPolicy(model, config=config, device="cpu", engine_mode="fast")
    reference = []
    for request in requests:
        policy.choose(request)
        reference.append(dict(policy.last_root_values or {}))
    runner = BatchedSearchRunner(policy, max_batch_size=3)
    runner.choose_many(policy, requests)
    for session, values in zip(runner.last_sessions, reference):
        current = session.policy.last_root_values or {}
        assert set(current) == set(values)
        for action, value in current.items():
            assert abs(value - values[action]) <= BATCH_TOLERANCE


def test_batched_mcts_matches_sequential(model) -> None:
    requests = [request for _, request, _ in fixed_requests()][:3]
    policy = PUCTSearchPolicy(
        model,
        config=MCTSConfig(simulations=8, determinizations=2, seed=29),
        device="cpu", engine_mode="fast",
    )
    sequential = [policy.search(request) for request in requests]
    runner = BatchedSearchRunner(policy, max_batch_size=6)
    batched = runner.search_many(policy, requests)
    for request, expected, actual in zip(requests, sequential, batched):
        assert (
            request.legal_actions[expected.chosen_candidate_index]
            == request.legal_actions[actual.chosen_candidate_index]
        )
        assert expected.simulations == actual.simulations
        l1 = sum(abs(x - y) for x, y in zip(expected.policy_target, actual.policy_target))
        assert l1 <= BATCH_TOLERANCE
    assert runner.stats.policy_batches >= 1


def test_batched_runner_requires_shared_evaluator(model) -> None:
    policy = PolicyValueSearchPolicy(
        model, config=SearchConfig(depth=2, top_k=2, seed=5), device="cpu",
        engine_mode="fast",
    )
    other = PolicyValueSearchPolicy(
        model, config=SearchConfig(depth=2, top_k=2, seed=5), device="cpu",
        engine_mode="fast",
    )
    runner = BatchedSearchRunner(policy, max_batch_size=2)
    request = fixed_requests()[0][1]
    with pytest.raises(ValueError, match="evaluator"):
        runner.choose_many(other, [request])
