"""Strict-equivalence checks for the rewritten search policy.

``tests/reference_search_policy_legacy.py`` is a frozen copy of the search
before alpha-beta, move ordering, state-only leaf encoding and macro folding.
The rewritten policy must agree with it on the same state and determinization:

* the chosen action is identical;
* the chosen (exact) root value is identical up to float32 forward noise;
* every root action the new search evaluated exactly agrees as well;
* actions that were pruned at the root are *strictly* worse than the choice,
  which is what makes pruning unable to change the winner.

Set ``SPLENDOR_SEARCH_EQUIV_STATES`` to raise the per-configuration state budget
for a deep soak run (thousands to tens of thousands of states).
"""

from __future__ import annotations

import os

import pytest
import torch

import reference_search_policy_legacy as legacy
from splendor.actor_critic import ActorCritic
from splendor.contracts import DecisionRequest
from splendor.engine import GameEngine
from splendor.heuristic_policy import DevelopmentPolicy, ScorePolicy
from splendor.random_policy import RandomPolicy
from splendor.result import TerminalResult
from splendor.search_policy import PolicyValueSearchPolicy, SearchConfig, determinize_request


HEAVY_STATES = int(os.environ.get("SPLENDOR_SEARCH_EQUIV_STATES", "0"))
# (config, default state budget, heavy-mode divisor): the heavier configurations
# are proportionally more expensive, so the soak scales them down.
CONFIGS = (
    ({"top_k": 2, "depth": 2}, 40, 1),
    ({"top_k": 3, "depth": 3}, 20, 5),
    ({"top_k": 5, "depth": 4}, 8, 40),
)
# Single-row and batched forwards differ only by float32 rounding; the search
# semantics (node set, ordering, minimax, tie-break) must still be identical.
VALUE_TOLERANCE = 1e-5


def _budget(default: int, divisor: int) -> int:
    if HEAVY_STATES <= 0:
        return default
    return max(default, HEAVY_STATES // divisor)


@pytest.fixture(scope="module")
def model() -> ActorCritic:
    # Real evaluation always pins CPU inference to one thread (`_parallel_worker_init`);
    # multi-threaded reductions change float32 rounding, so the equivalence
    # comparison must run under the same setting.
    torch.set_num_threads(1)
    torch.manual_seed(17)
    return ActorCritic().eval()


@pytest.fixture(scope="module")
def zero_model() -> ActorCritic:
    """Degenerate critic: every leaf ties, so no branch can be pruned."""

    torch.manual_seed(17)
    critic = ActorCritic().eval()
    with torch.no_grad():
        for parameter in critic.parameters():
            parameter.zero_()
    return critic


def _sample_requests(count: int) -> list[DecisionRequest]:
    """Deterministic mix of random and heuristic decisions from real games."""

    requests: list[DecisionRequest] = []
    for seed in range(600):
        engine = GameEngine.new(seed, first_player=seed % 2)
        policies = (
            (RandomPolicy(seed * 7 + 1), RandomPolicy(seed * 7 + 2))
            if seed % 2
            else (ScorePolicy(), DevelopmentPolicy())
        )
        for step in range(120):
            response = engine.decision()
            if isinstance(response, TerminalResult):
                break
            if step % 2 == 0:
                requests.append(response)
            engine.apply(policies[engine.current_player].choose(response))
            if len(requests) >= count:
                return requests
    return requests


def _compare_state(request: DecisionRequest, model: ActorCritic, kwargs: dict) -> float:
    reference = legacy.LegacyPolicyValueSearchPolicy(
        model, config=legacy.LegacySearchConfig(**kwargs)
    )
    policy = PolicyValueSearchPolicy(model, config=SearchConfig(**kwargs))
    reference_action = reference.choose(request)
    action = policy.choose(request)

    assert action == reference_action, request.observation.phase
    if len(request.legal_actions) == 1:
        assert reference.last_root_values is None
        assert policy.last_root_values is None
        return 0.0
    if action != reference_action:
        # Admissible only when the reference decision itself lived inside
        # float32 forward noise: single-row and batched leaf evaluation round
        # differently (measured up to ~4e-7), so a margin below the tolerance
        # is not a reproducible decision.
        ordered = sorted(reference.last_root_values.values(), reverse=True)
        assert ordered[0] - ordered[1] <= VALUE_TOLERANCE
    reference_value = reference.last_root_values[reference_action]
    policy_value = policy.last_root_values[action]
    assert abs(policy_value - reference_value) <= VALUE_TOLERANCE

    for candidate, value in policy.last_root_values.items():
        expected = reference.last_root_values.get(candidate)
        if expected is None:
            continue
        assert abs(value - expected) <= VALUE_TOLERANCE, candidate
    for bound in (policy.last_root_bound_values or {}).values():
        assert bound < policy_value
    return abs(policy_value - reference_value)


@pytest.mark.parametrize(
    ("kwargs", "default_states", "divisor"),
    CONFIGS,
    ids=("top2-depth2", "top3-depth3", "top5-depth4"),
)
def test_new_search_matches_legacy_on_random_states(
    model, kwargs, default_states, divisor
) -> None:
    budget = _budget(default_states, divisor)
    requests = _sample_requests(budget)
    assert len(requests) == budget
    worst = max(_compare_state(request, model, kwargs) for request in requests)
    assert worst <= VALUE_TOLERANCE


def test_determinization_roots_agree_between_implementations() -> None:
    for request in _sample_requests(40):
        for seed in (0, 101, 987_654_321):
            theirs = legacy.determinize_request(request, seed=seed)
            ours = determinize_request(request, seed=seed)
            assert ours.current_state_digest == theirs.current_state_digest
            assert ours.decision() == theirs.decision()


def test_alpha_beta_never_expands_more_nodes_than_the_reference(model) -> None:
    requests = _sample_requests(12)
    kwargs = {"top_k": 5, "depth": 3}
    new_nodes = 0
    old_nodes = 0
    pruned = 0
    for request in requests:
        reference = legacy.LegacyPolicyValueSearchPolicy(
            model, config=legacy.LegacySearchConfig(**kwargs)
        )
        policy = PolicyValueSearchPolicy(model, config=SearchConfig(**kwargs))
        reference.choose(request)
        policy.choose(request)
        new_nodes += policy.stats.expanded_nodes
        old_nodes += reference.stats.expanded_nodes
        pruned += policy.stats.pruned_branches
    assert new_nodes <= old_nodes
    assert pruned > 0


class _RecordingPolicy(PolicyValueSearchPolicy):
    """Records the candidate count of every request it evaluates."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.seen_candidate_counts: list[int] = []

    def _infer_logits(self, request: DecisionRequest) -> list[float]:
        self.seen_candidate_counts.append(len(request.legal_actions))
        return super()._infer_logits(request)


def _requests_with_forced_payment(limit: int) -> list[DecisionRequest]:
    """States where one of the first three legal actions forces the reply.

    A zeroed model orders candidates by index, so ``top_k <= 3`` explores
    exactly those actions and the folded branch is guaranteed to be visited.
    """

    found: list[DecisionRequest] = []
    for seed in range(20):
        engine = GameEngine.new(seed, first_player=seed % 2)
        policies = (ScorePolicy(), DevelopmentPolicy())
        for _ in range(120):
            response = engine.decision()
            if isinstance(response, TerminalResult):
                break
            if len(response.legal_actions) >= 2:
                matched = False
                for action in response.legal_actions[:3]:
                    child = engine.fork()
                    reply = child.apply(action)
                    if (
                        isinstance(reply, DecisionRequest)
                        and len(reply.legal_actions) == 1
                    ):
                        matched = True
                        break
                if matched:
                    found.append(response)
                    break
            engine.apply(policies[engine.current_player].choose(response))
            if len(found) >= limit:
                return found
    return found


def test_folding_never_evaluates_a_forced_decision(zero_model) -> None:
    requests = _requests_with_forced_payment(4)
    assert requests, "no state with a forced follow-up decision was found"
    for request in requests:
        policy = _RecordingPolicy(
            zero_model,
            config=SearchConfig(top_k=3, depth=3, seed=5, fold_forced_phases=True),
        )
        action = policy.choose(request)
        assert action in request.legal_actions
        assert policy.stats.folded_decisions > 0
        # Only the root itself may be a forced decision; every interior node the
        # search spends model calls on must be a real choice.
        assert policy.seen_candidate_counts[0] == len(request.legal_actions)
        assert all(count >= 2 for count in policy.seen_candidate_counts[1:])


def test_folding_spends_depth_on_real_choices(zero_model) -> None:
    requests = _requests_with_forced_payment(4)
    assert requests
    for request in requests:
        plain = _RecordingPolicy(
            zero_model,
            config=SearchConfig(top_k=3, depth=3, seed=5, fold_forced_phases=False),
        )
        folded = _RecordingPolicy(
            zero_model,
            config=SearchConfig(top_k=3, depth=3, seed=5, fold_forced_phases=True),
        )
        assert plain.choose(request) in request.legal_actions
        assert folded.choose(request) in request.legal_actions
        assert folded.stats.folded_decisions > 0
        # The plain search spends whole plies on forced decisions; the folded
        # one spends every ply on a request with a real choice, so it reaches
        # strictly more strategic decisions inside the same depth budget.
        plain_forced = sum(1 for count in plain.seen_candidate_counts if count == 1)
        folded_forced = sum(1 for count in folded.seen_candidate_counts if count == 1)
        assert plain_forced > 0
        assert folded_forced == 0
        assert len(folded.seen_candidate_counts) > len(plain.seen_candidate_counts)


def test_folding_reaches_the_same_terminal_outcome_with_top_k_one(model) -> None:
    requests = _sample_requests(3)
    for request in requests:
        results = []
        for fold in (False, True):
            policy = PolicyValueSearchPolicy(
                model,
                config=SearchConfig(
                    top_k=1,
                    depth=400,
                    seed=3,
                    fold_forced_phases=fold,
                ),
            )
            action = policy.choose(request)
            results.append((action, policy.last_root_values[action]))
        assert results[0][0] == results[1][0]
        # Both searches run to a real terminal in this line of play.
        assert results[0][1] == results[1][1]
        assert any(
            abs(results[0][1] - terminal) == 0.0 for terminal in (-1.0, 0.0, 1.0)
        )


def test_stats_expose_the_new_counters(model) -> None:
    policy = PolicyValueSearchPolicy(
        model, config=SearchConfig(top_k=2, depth=2, seed=1)
    )
    request = _sample_requests(1)[0]
    policy.choose(request)
    payload = policy.stats.to_dict()
    assert payload["folded_decisions"] == 0
    assert payload["pruned_branches"] >= 0
    assert set(payload) >= {
        "expanded_nodes",
        "leaf_nodes",
        "model_positions",
        "folded_decisions",
        "pruned_branches",
    }
