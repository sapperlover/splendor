"""Tests for the search-vs-raw-policy decision metrics.

``chosen_policy_rank`` is the 1-based rank of the action the search picked
inside the raw policy's own ordering over all legal actions;
``search_changed_action_rate`` is the share of decisions whose final action
differs from the raw argmax.
"""

from __future__ import annotations

import pytest
import torch

from splendor.actor_critic import ActorCritic
from splendor.contracts import DecisionRequest
from splendor.engine import GameEngine
from splendor.features import collate_encoded_decisions, encode_decision
from splendor.heuristic_policy import DevelopmentPolicy, ScorePolicy
from splendor.result import TerminalResult
from splendor.search_policy import (
    PolicyValueSearchPolicy,
    SearchConfig,
    decision_quality_metrics,
)


@pytest.fixture(scope="module")
def model() -> ActorCritic:
    torch.set_num_threads(1)
    torch.manual_seed(23)
    return ActorCritic().eval()


def _requests(seed_count: int, limit: int) -> list[DecisionRequest]:
    requests: list[DecisionRequest] = []
    for seed in range(seed_count):
        engine = GameEngine.new(seed, first_player=seed % 2)
        policies = (ScorePolicy(), DevelopmentPolicy())
        for step in range(120):
            response = engine.decision()
            if isinstance(response, TerminalResult):
                break
            if step % 2 == 0:
                requests.append(response)
            engine.apply(policies[engine.current_player].choose(response))
            if len(requests) >= limit:
                return requests
    return requests


def _raw_ranking(model: ActorCritic, request: DecisionRequest) -> tuple[int, ...]:
    batch = collate_encoded_decisions((encode_decision(request),))
    with torch.inference_mode():
        logits, _ = model(batch)
        mask = torch.as_tensor(batch.candidate_mask, device=logits.device)
        masked = model.masked_logits(logits, mask)
    row = masked.detach().cpu()[0, : len(request.legal_actions)].tolist()
    return tuple(sorted(range(len(row)), key=lambda index: (-row[index], index)))


def test_ranked_indices_matches_the_reference_ordering() -> None:
    logits = [0.5, -1.0, 0.5, 2.0, 2.0, -1.0]
    expected = tuple(sorted(range(len(logits)), key=lambda i: (-logits[i], i)))
    assert PolicyValueSearchPolicy._ranked_indices(logits) == expected


def test_decision_quality_metrics_derives_rates() -> None:
    stats = {
        "decisions": 10,
        "determinizations": 8,
        "searchable_decisions": 8,
        "expanded_nodes": 400,
        "leaf_nodes": 250,
        "model_positions": 420,
        "folded_decisions": 40,
        "pruned_branches": 100,
        "total_seconds": 2.0,
        "inference_seconds": 1.0,
        "changed_action_count": 3,
        "chosen_rank_sum": 20,
        "chosen_rank_top1_count": 4,
        "chosen_rank_top2_count": 6,
        "chosen_rank_top3_count": 7,
    }
    metrics = decision_quality_metrics(stats)
    assert metrics["chosen_policy_rank"] == pytest.approx(20 / 8)
    assert metrics["search_changed_action_rate"] == pytest.approx(0.3)
    assert metrics["search_changed_action_rate_searchable"] == pytest.approx(3 / 8)
    assert metrics["chosen_rank_top1_rate"] == pytest.approx(0.5)
    assert metrics["chosen_rank_top2_rate"] == pytest.approx(6 / 8)
    assert metrics["chosen_rank_top3_rate"] == pytest.approx(7 / 8)
    assert metrics["searchable_decision_share"] == pytest.approx(0.8)
    assert metrics["search_calls"] == 10
    assert metrics["search_expansions"] == 8
    assert metrics["nodes_per_search"] == pytest.approx(50.0)
    assert metrics["leaf_nodes_per_search"] == pytest.approx(31.25)
    assert metrics["model_positions_per_search"] == pytest.approx(52.5)
    assert metrics["folded_plies_per_search"] == pytest.approx(5.0)
    assert metrics["wall_ms_per_search"] == pytest.approx(250.0)
    assert metrics["inference_ms_per_search"] == pytest.approx(125.0)
    assert metrics["alpha_beta_cutoffs"] == 100
    assert metrics["cutoff_rate"] == pytest.approx(0.25)
    assert metrics["search_changed_raw_action_count"] == 3
    assert metrics["search_changed_raw_action_rate"] == pytest.approx(0.3)
    assert metrics["search_changed_raw_action_rate_searchable"] == pytest.approx(3 / 8)


def test_choice_metrics_match_an_independent_recomputation(model) -> None:
    requests = _requests(4, 60)
    assert requests
    policy = PolicyValueSearchPolicy(
        model, config=SearchConfig(top_k=5, depth=3, seed=4)
    )
    expected_ranks: list[int] = []
    expected_changed = 0
    for request in requests:
        ranking = _raw_ranking(model, request)
        action = policy.choose(request)
        if len(request.legal_actions) == 1:
            assert policy.last_root_values is None
            continue
        chosen_index = request.legal_actions.index(action)
        expected_ranks.append(ranking.index(chosen_index) + 1)
        if chosen_index != ranking[0]:
            expected_changed += 1

    stats = policy.stats
    assert stats.searchable_decisions == len(expected_ranks)
    assert stats.chosen_rank_sum == sum(expected_ranks)
    assert stats.changed_action_count == expected_changed
    assert stats.chosen_rank_top1_count == sum(1 for r in expected_ranks if r == 1)
    assert stats.chosen_rank_top2_count == sum(1 for r in expected_ranks if r <= 2)
    assert stats.chosen_rank_top3_count == sum(1 for r in expected_ranks if r <= 3)
    assert stats.chosen_rank_beyond3_count == sum(
        1 for r in expected_ranks if r > 3
    )
    metrics = decision_quality_metrics(stats.to_dict())
    assert metrics["chosen_policy_rank"] == pytest.approx(
        sum(expected_ranks) / len(expected_ranks)
    )
    assert metrics["search_changed_action_rate_searchable"] == pytest.approx(
        expected_changed / len(expected_ranks)
    )


def test_trivial_decisions_are_excluded_from_searchable_metrics(model) -> None:
    requests = _requests(6, 120)
    policy = PolicyValueSearchPolicy(
        model, config=SearchConfig(top_k=2, depth=2, seed=9)
    )
    multi_candidate = 0
    for request in requests:
        policy.choose(request)
        if len(request.legal_actions) > 1:
            multi_candidate += 1
    assert policy.stats.searchable_decisions == multi_candidate
    assert policy.stats.decisions == len(requests)
    assert policy.stats.searchable_decisions < policy.stats.decisions
    metrics = decision_quality_metrics(policy.stats.to_dict())
    assert metrics["search_changed_action_rate"] < metrics[
        "search_changed_action_rate_searchable"
    ]


def test_search_never_changes_a_degenerate_zero_model_choice(model) -> None:
    zero = ActorCritic().eval()
    with torch.no_grad():
        for parameter in zero.parameters():
            parameter.zero_()
    requests = _requests(2, 20)
    policy = PolicyValueSearchPolicy(
        zero, config=SearchConfig(top_k=3, depth=2, seed=1)
    )
    for request in requests:
        chosen = policy.choose(request)
        if len(request.legal_actions) > 1:
            # All logits tie, so the raw argmax is the lowest action index.
            assert chosen == request.legal_actions[0]
    assert policy.stats.changed_action_count == 0
    assert policy.stats.chosen_rank_top1_count == policy.stats.searchable_decisions
