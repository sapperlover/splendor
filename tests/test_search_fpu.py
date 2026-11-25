"""First-play urgency: unvisited edges scored at parent Q minus a reduction."""

from __future__ import annotations

import math
import random
from dataclasses import replace

import pytest

from splendor.actor_critic import ActorCritic
from splendor.catalog import load_default_catalog
from splendor.contracts import DecisionRequest
from splendor.engine import GameEngine
from splendor.mcts_policy import MCTSConfig, PUCTSearchPolicy, _Edge, _Node
from splendor.search_policy import _request_public_word
from splendor.selfplay_fast import ExecutionConfig, FastSelfPlayConfig

native = pytest.importorskip("splendor._native_search")


@pytest.mark.parametrize("value", [-0.1, math.nan, math.inf, True, "0.2"])
def test_fpu_reduction_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="fpu_reduction"):
        MCTSConfig(fpu_reduction=value)


def test_unvisited_score_is_parent_q_minus_reduction():
    catalog = load_default_catalog()
    engine = GameEngine.new(3, catalog=catalog)
    request = engine.decision()
    assert isinstance(request, DecisionRequest)
    request = replace(request, legal_actions=request.legal_actions[:3])
    actor = engine.current_player
    config = MCTSConfig(simulations=2, determinizations=1, c_puct=1.0, fpu_reduction=0.2)
    policy = PUCTSearchPolicy(ActorCritic(), config=config, catalog=catalog)
    # Root perspective: edge 0 backed up +1.5 over 3 visits; node value 0.3 from its actor.
    node = _Node(engine, request, (0.2, 0.3, 0.5), [_Edge(3, 1.5), _Edge(), _Edge()], value=0.3)
    for root_player, sign in ((actor, 1), (1 - actor, -1)):
        parent_q = (0.3 + sign * 1.5) / 4
        factor = math.sqrt(4)
        scores = [sign * 0.5 + factor * 0.2 / 4,
                  parent_q - 0.2 + factor * 0.3, parent_q - 0.2 + factor * 0.5]
        expected = max(range(3), key=lambda i: scores[i])
        assert policy._select(node, root_player) == expected
    # The node's own actor sees a strong visited edge (Q=+0.5): parent Q 0.45 - 0.2 + 1.0
    # beats 0.5 + 0.1, so the high-prior unvisited edge is tried; at a large reduction it is not.
    assert policy._select(node, actor) == 2
    policy.config = replace(config, fpu_reduction=1.0)
    assert policy._select(node, actor) == 0
    policy.config = replace(config, fpu_reduction=None)
    assert policy._select(node, actor) == 2


def _run_pair(config, seed):
    """Drive native and Python jobs with identical pseudo-random network outputs."""

    request = GameEngine.new(seed, first_player=0, mode="fast").decision()
    catalog = load_default_catalog()
    native_job = native.SearchJob(native.Catalog(catalog), request, config,
                                  _request_public_word(request), config.seed, False, 0.0)
    python_job = PUCTSearchPolicy(ActorCritic(), config=config).start_job(request)
    rng = random.Random(seed)
    while not native_job.done():
        incoming = native_job.pending()
        reference = python_job.pending_requests()
        assert len(incoming) == len(reference)
        for item, other in zip(incoming, reference):
            assert item["kind"] == other.kind
            width = len(other.request.legal_actions)
            logits = None if other.kind == "value_only" else tuple(rng.gauss(0, 1) for _ in range(width))
            value = rng.uniform(-1, 1)
            native_job.submit(item["world"], () if logits is None else logits, value)
            if item["world"] == -1:
                native_job.initialize_worlds(request)
            python_job.submit(other.request_id, logits, value)
    actual = native_job.result()
    expected = python_job.result
    assert actual["per_world_root_visit_counts"] == [list(row) for row in expected.per_world_root_visit_counts]
    assert actual["chosen_candidate_index"] == expected.chosen_candidate_index
    return actual["per_world_root_visit_counts"]


@pytest.mark.parametrize("seed", [42, 7])
@pytest.mark.parametrize("worlds", [1, 2])
def test_native_fpu_matches_python_reference(seed, worlds):
    base = MCTSConfig(simulations=48, determinizations=worlds, seed=155, root_noise_fraction=0)
    historical = _run_pair(base, seed)
    reduced = [_run_pair(replace(base, fpu_reduction=r), seed) for r in (0.0, 0.2)]
    assert reduced[0] != historical or reduced[1] != historical


def test_chance_search_applies_fpu():
    catalog = load_default_catalog()
    request = GameEngine.new(42, first_player=0, mode="fast").decision()

    def visits(config):
        job = native.SearchJob(native.Catalog(catalog), request, config,
                               _request_public_word(request), 155, False, 0.0, True)
        rng = random.Random(3)
        while not job.done():
            for item in job.pending():
                width = item["encoded"]["candidate_mask"].shape[0] if item["kind"] == "policy_value" else 0
                job.submit(item["world"], tuple(rng.gauss(0, 1) for _ in range(width)), rng.uniform(-1, 1))
                if item["world"] == -1:
                    job.initialize_worlds(request)
        result = job.result()
        assert sum(result["per_world_root_visit_counts"][0]) == config.simulations
        return result["per_world_root_visit_counts"]

    base = MCTSConfig(simulations=48, determinizations=1, seed=155, root_noise_fraction=0)
    assert visits(replace(base, fpu_reduction=0.2)) != visits(base)


def test_selfplay_config_omits_unset_fpu(tmp_path):
    checkpoint = tmp_path / "unused.pt"
    checkpoint.touch()

    def config(search):
        return FastSelfPlayConfig(
            protocol="mcts-selfplay-v2", run_id="fpu", round_index=0, checkpoint=str(checkpoint),
            num_games=1, seed=1, search_config=search,
            execution=ExecutionConfig(backend="batched_cpu", engine_mode="fast", workers=1,
                                      active_games_per_worker=1, max_batch_size=1),
        )

    assert "fpu_reduction" not in config(MCTSConfig()).to_dict()["search_config"]
    raw = config(MCTSConfig(fpu_reduction=0.2)).to_dict()
    assert raw["search_config"]["fpu_reduction"] == 0.2
    assert FastSelfPlayConfig.from_dict(raw).search_config.fpu_reduction == 0.2


def test_legacy_v3_search_training_matches_only_disabled_fpu():
    from splendor.selfplay_data import same_search_training

    legacy = {"search_config": {"c_puct": 1.5}, "add_root_noise": True, "budget_randomization": None}
    disabled = {**legacy, "search_config": {"c_puct": 1.5, "fpu_reduction": None}}
    enabled = {**legacy, "search_config": {"c_puct": 1.5, "fpu_reduction": 0.0}}
    assert same_search_training(legacy, disabled)
    assert not same_search_training(legacy, enabled)
    assert not same_search_training(legacy, {**disabled, "add_root_noise": False})
