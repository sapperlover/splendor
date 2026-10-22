from __future__ import annotations

import math
import random
from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

from splendor.actor_critic import ActorCritic
from splendor.catalog import load_default_catalog
from splendor.engine import GameEngine
from splendor.features import FEATURE_SCHEMA_VERSION
from splendor.mcts_policy import MCTSConfig, PUCTSearchPolicy, _Edge, _Node
from splendor.ppo import TRAINING_CHECKPOINT_FORMAT_VERSION
from splendor.replay import RULESET_ID
from splendor.search_policy import _request_public_word
from splendor.search_training import SearchBudgetRandomization, prune_policy_counts


@pytest.mark.parametrize("kwargs", [
    {"low_simulations": 0}, {"low_simulations": True}, {"low_probability": math.nan},
    {"low_probability": -0.1}, {"low_probability": 1.1}, {"low_probability": True},
    {"low_determinizations": 0}, {"low_determinizations": True},
])
def test_invalid_randomization(kwargs):
    with pytest.raises(ValueError):
        SearchBudgetRandomization(**kwargs)


def test_budget_draw_is_reproducible_independent_and_validated():
    cfg = SearchBudgetRandomization(low_simulations=8, low_probability=0.75)
    before = random.getstate()
    decisions = [cfg.is_low(i) for i in range(1000)]
    assert decisions == [cfg.is_low(i) for i in range(1000)]
    assert 700 < sum(decisions) < 800
    assert random.getstate() == before
    assert all(SearchBudgetRandomization(low_probability=1).is_low(i) for i in range(10))
    assert not any(SearchBudgetRandomization(low_probability=0).is_low(i) for i in range(10))
    cfg.validate_search(16, 2)
    for high, worlds in ((8, 2), (16, 9)):
        with pytest.raises(ValueError):
            cfg.validate_search(high, worlds)
    assert cfg.to_dict() == {"low_simulations": 8, "low_probability": 0.75}
    SearchBudgetRandomization(8, 0.75, 1).validate_search(16, 4)
    with pytest.raises(ValueError):
        SearchBudgetRandomization(8, 0.75, 4).validate_search(16, 2)


@pytest.mark.parametrize("kwargs", [
    {"forced_playouts": -1}, {"forced_playouts": True}, {"forced_playouts": math.nan},
    {"policy_target_pruning": True}, {"policy_target_pruning": 1},
])
def test_invalid_search_options(kwargs):
    with pytest.raises(ValueError):
        MCTSConfig(**kwargs)


def test_forced_playouts_only_root_and_only_visited_children():
    engine = GameEngine.new(3, mode="fast")
    request = engine.decision()
    policy = PUCTSearchPolicy(ActorCritic(), config=MCTSConfig(forced_playouts=2))
    node = _Node(engine, request, (0.8, 0.2), [_Edge(20, 18), _Edge(1, -1)])
    assert policy._select(node, engine.current_player) == 0
    assert policy._select(node, engine.current_player, is_root=True) == 1
    assert node.edges[1].forced_playouts == 1
    node.edges[1] = _Edge()
    assert policy._select(node, engine.current_player, is_root=True) == policy._select(node, engine.current_player)
    assert node.edges[1].forced_playouts == 0


def test_pruning_retains_good_moves_and_preserves_winner():
    args = {"c_puct": 1.5, "forced_playouts": 2}
    assert prune_policy_counts((90, 10), (72, -8), (0.8, 0.2), **args) == (90, 4)
    assert prune_policy_counts((99, 1), (79.2, -0.8), (0.8, 0.2), **args) == (99, 0)
    assert prune_policy_counts((90, 10), (9, 9), (0.8, 0.2), **args) == (90, 10)
    assert prune_policy_counts((1, 1), (0, 0), (0.5, 0.5), **args) == (1, 0)


def test_binary_pruning_matches_literal_visit_removal():
    rng = random.Random(123)
    for _ in range(100):
        counts = [rng.randrange(50) for _ in range(5)]
        values = [n * rng.uniform(-1, 1) for n in counts]
        prior = [rng.random() for _ in counts]
        prior = [p / sum(prior) for p in prior]
        best = max(range(5), key=lambda i: (counts[i], prior[i], -i))
        factor = 1.5 * math.sqrt(1 + sum(counts))
        threshold = values[best] / counts[best] + factor * prior[best] / (1 + counts[best])
        expected = counts.copy()
        for i, n in enumerate(counts):
            if i == best or n == 0:
                continue
            for _ in range(min(n, math.floor(math.sqrt(2 * prior[i] * sum(counts))))):
                if values[i] / n + factor * prior[i] / expected[i] >= threshold:
                    break
                expected[i] -= 1
            if expected[i] <= 1:
                expected[i] = 0
        assert prune_policy_counts(counts, values, prior, c_puct=1.5, forced_playouts=2) == tuple(expected)


@pytest.mark.parametrize("worlds", [1, 2])
def test_native_forcing_and_pruning_matches_python_and_preserves_behavior(worlds):
    from splendor import _native_search as native
    request = GameEngine.new(42, first_player=0, mode="fast").decision()
    cfg = MCTSConfig(simulations=129, determinizations=worlds, seed=155,
                     forced_playouts=2, policy_target_pruning=True)
    core = native.SearchJob(native.Catalog(load_default_catalog()), request, cfg,
                            _request_public_word(request), 155, False, 1.0)
    python_job = PUCTSearchPolicy(ActorCritic(), config=cfg).start_job(request, temperature=1.0)
    while not core.done():
        incoming, reference = core.pending(), python_job.pending_requests()
        assert len(incoming) == len(reference)
        for item, other in zip(incoming, reference):
            logits = None if other.kind == "value_only" else (0.0,) * len(other.request.legal_actions)
            core.submit(item["world"], () if logits is None else logits, 0.375)
            if item["world"] == -1:
                core.initialize_worlds(request)
            python_job.submit(other.request_id, logits, 0.375)
    actual, expected = core.result(), python_job.result
    for name in ("per_world_root_visit_counts", "per_world_forced_playouts", "per_world_policy_target_counts",
                 "per_world_root_value_sums", "policy_target", "behavior_policy"):
        np.testing.assert_allclose(actual[name], getattr(expected, name), atol=1e-12, rtol=0)
    assert sum(map(sum, actual["per_world_forced_playouts"])) > 0
    assert sum(map(sum, actual["per_world_policy_target_counts"])) <= cfg.simulations
    raw = np.asarray(actual["per_world_root_visit_counts"])
    np.testing.assert_allclose(actual["behavior_policy"], (raw / raw.sum(axis=1)[:, None]).mean(axis=0))
    pruned = np.asarray(actual["per_world_policy_target_counts"])
    np.testing.assert_allclose(actual["policy_target"], (pruned / pruned.sum(axis=1)[:, None]).mean(axis=0))


@pytest.fixture
def checkpoint(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(11)
    model = ActorCritic().eval()
    path = tmp_path / "teacher.pt"
    torch.save({"format_version": TRAINING_CHECKPOINT_FORMAT_VERSION,
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "metadata": {"rules_version": RULESET_ID}, "model_config": asdict(model.config),
                "model_state_dict": model.state_dict()}, path)
    return path


@pytest.mark.parametrize("backend,interface,low_worlds", [
    ("native_cpu", "native", None), ("native_cpu", "legacy", None),
    ("batched_cpu", "legacy", None), ("native_cpu", "native", 1)])
def test_collection_archive_replay_and_training(tmp_path, checkpoint, backend, interface, low_worlds):
    from splendor.alphazero_replay import ReplayRun, ReplayReference, build_replay_index
    from splendor.alphazero_train import AlphazeroLearner, LearnerConfig
    from splendor.selfplay_data import (DATA_SCHEMA_V3, SelfPlayDataset, collate_selfplay_samples,
                                       game_path, load_game, validate_game_arrays)
    from splendor.selfplay_fast import FastSelfPlayConfig, generate, validate_run

    raw = {"protocol": "mcts-selfplay-v2", "run_id": "random-budget-smoke", "round_index": 0,
           "checkpoint": str(checkpoint), "num_games": 1, "seed": 2000,
           "search_config": {"simulations": 17, "determinizations": 2, "seed": 90,
                             "forced_playouts": 2, "policy_target_pruning": True},
           "budget_randomization": {"low_simulations": 4, "low_probability": 0.75},
           "execution": {"backend": backend, "batch_interface": interface,
                         "active_games_per_worker": 1, "max_batch_size": 4},
           "validation_fraction": 0}
    cfg = FastSelfPlayConfig.from_dict(raw)
    if low_worlds is not None:
        cfg = replace(cfg, budget_randomization=replace(cfg.budget_randomization, low_determinizations=low_worlds))
    assert FastSelfPlayConfig.from_dict(cfg.to_dict()) == cfg
    root = tmp_path / "collection"
    summary = generate(cfg, root)
    assert summary["completed_games"] == 1
    assert summary["low_budget_decisions"] > 0 and summary["full_budget_decisions"] > 0
    assert summary["policy_samples"] == summary["full_budget_decisions"]
    assert summary["pruned_visits"] > 0
    assert validate_run(root, replay=True)["simulations"] == summary["simulations"]
    original = game_path(root, 0).read_bytes()
    assert generate(cfg, root, resume=True)["completed_games"] == 1
    assert game_path(root, 0).read_bytes() == original
    with pytest.raises(ValueError, match="fingerprint"):
        generate(replace(cfg, budget_randomization=SearchBudgetRandomization(4, 0.5)), root, resume=True)
    arrays, metadata = load_game(game_path(root, 0))
    assert metadata["data_schema"] == DATA_SCHEMA_V3
    assert np.all(arrays["policy_weight"][arrays["low_budget"]] == 0)
    assert np.all(arrays["value_weight"] == 1)
    if low_worlds is not None:
        low_mask = arrays["low_budget"]
        assert np.all(arrays["valid_world_count"][low_mask] == low_worlds)
        assert np.all(arrays["simulation_budgets"][low_mask, low_worlds:] == 0)
        for i in np.flatnonzero(low_mask):
            start, end = arrays["candidate_offsets"][i:i+2]
            for name in ["per_world_root_visit_counts", "per_world_policy_target_counts", "per_world_root_value_sums", "per_world_forced_playouts"]:
                assert not arrays[name][low_worlds:, start:end].any()
    common = {"simulations": 17, "worlds": 2}
    wrong = {name: a.copy() for name, a in arrays.items()}
    wrong["low_budget"][0] = not wrong["low_budget"][0]
    with pytest.raises(ValueError, match="weights|budget draw"):
        validate_game_arrays(wrong, metadata, **common)
    wrong = {name: a.copy() for name, a in arrays.items()}
    wrong["per_world_policy_target_counts"][0, 0] += 1
    with pytest.raises(ValueError, match="training statistics|pruned policy"):
        validate_game_arrays(wrong, metadata, **common)
    dataset = SelfPlayDataset(root)
    low = next(i for i in range(len(dataset)) if arrays["low_budget"][i])
    full = next(i for i in range(len(dataset)) if arrays["policy_weight"][i])
    batch = collate_selfplay_samples([dataset[low], dataset[full]])
    np.testing.assert_array_equal(batch.policy_weights, [0, 1])
    index = build_replay_index([ReplayRun(0, cfg.run_id, root)], max_games=2)
    assert index.games[0].policy_samples == summary["policy_samples"]
    learner = AlphazeroLearner.warm_start(checkpoint,
        LearnerConfig(device="cpu", batch_size=2, history_batch_size=1), sampling_seed=7,
        replay_identity=index.identity(), parent_checkpoint={"path": str(checkpoint), "sha256": "smoke"},
        round_index=0, round_steps_target=2)
    references = [ReplayReference(0, cfg.run_id, 0, i) for i in (low, full)]
    metrics = learner.train_step(batch, references=references)
    assert math.isfinite(metrics["total_loss"])
    low_batch = collate_selfplay_samples([dataset[low]])
    metrics = learner.train_step(low_batch, references=references[:1])
    assert metrics["policy_loss"] == 0 and math.isfinite(metrics["value_loss"])


def test_old_config_serialization_and_loop_budget_forwarding(checkpoint):
    from splendor.alphazero_loop import SelfPlaySection
    from splendor.selfplay_fast import FastSelfPlayConfig
    raw = {"protocol": "mcts-selfplay-v2", "run_id": "old", "round_index": 0,
           "checkpoint": str(checkpoint), "num_games": 1, "seed": 1,
           "search_config": {"simulations": 1024, "determinizations": 1}, "execution": {}}
    cfg = FastSelfPlayConfig.from_dict(raw)
    assert cfg.data_schema == "splendor-mcts-selfplay-v2"
    assert "budget_randomization" not in cfg.to_dict()
    assert "forced_playouts" not in cfg.to_dict()["search_config"]
    section = SelfPlaySection(search_config=raw["search_config"], execution={}, add_root_noise=True,
                              temperature_schedule={}, validation_fraction=0.1, max_completed_turns=400,
                              budget_randomization={"low_simulations": 128, "low_probability": 0.75})
    assert FastSelfPlayConfig.from_dict(raw | section.to_dict()).budget_randomization.low_simulations == 128
