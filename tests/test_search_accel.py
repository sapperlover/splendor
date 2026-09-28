from __future__ import annotations

import json
import random
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from splendor.actions import BuyMarket, BuyReserved, Pay
from splendor.actor_critic import ActorCritic
from splendor.compact_trace import decode_request, replay_trace
from splendor.engine import GameEngine, IllegalActionError
from splendor.contracts import DecisionRequest
from splendor.features import collate_encoded_decisions, encode_decision
from splendor.match_stepper import MatchStepper
from splendor.mcts_policy import MCTSConfig, PUCTSearchPolicy
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.replay import to_primitive
from splendor.reproduce_failure import reproduce
from splendor.search_inference import FrozenSearchInference
from splendor.selfplay_fast import FastSelfPlayConfig, generate, validate_run
from splendor.selfplay_data import SelfPlayDataset, collate_selfplay_samples
from splendor.result import TerminalResult


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "runs/5c_main_seed1/checkpoints/local_005000000_lifetime_005512112.pt"


def test_fast_engine_matches_checked_and_trace_replays(monkeypatch):
    import splendor.engine as engine_module
    calls = {"digest": 0, "check": 0}
    digest = engine_module.state_digest
    check = engine_module.InvariantGuard.check

    def digest_spy(state):
        calls["digest"] += 1
        return digest(state)

    def check_spy(self, state, action=None):
        calls["check"] += 1
        return check(self, state, action)

    monkeypatch.setattr(engine_module, "state_digest", digest_spy)
    monkeypatch.setattr(engine_module.InvariantGuard, "check", check_spy)
    for seed in (7, 42, 107):
        fast = MatchStepper(seed, seed, seed % 2, mode="fast")
        checked = MatchStepper(seed, seed, seed % 2, mode="checked")
        assert calls == {"digest": 1, "check": 1}
        rng = random.Random(seed)
        for _ in range(400):
            assert fast.current == checked.current
            assert fast.engine._debug_state() == checked.engine._debug_state()
            if isinstance(fast.current, TerminalResult):
                break
            index = rng.randrange(len(fast.request.legal_actions))
            before = calls.copy()
            fast.apply(index)
            assert calls == before
            checked.apply(index)
            assert calls["digest"] > before["digest"]
            assert calls["check"] > before["check"]
            assert fast.engine.forced_skip_count == checked.engine.forced_skip_count
        assert fast.terminal is not None
        assert replay_trace(fast.trace) == fast.terminal
        assert decode_request(to_primitive(GameEngine.new(seed, first_player=seed % 2).decision())) == GameEngine.new(seed, first_player=seed % 2).decision()
        calls = {"digest": 0, "check": 0}


def _fake_evaluation(job):
    while not job.done:
        for item in reversed(job.pending_requests()):
            if item.kind == "policy_value":
                logits = tuple((index + 1) * 0.01 for index in range(len(item.request.legal_actions)))
                job.submit(item.request_id, logits, 0.125)
            else:
                job.submit(item.request_id, None, 0.125)
    return job.result


def test_mcts_job_interleaving_preserves_visits_and_targets():
    torch.set_num_threads(1)
    model = load_frozen_actor_critic_checkpoint(CHECKPOINT, device="cpu")
    request = GameEngine.new(42, first_player=0).decision()
    config = MCTSConfig(simulations=257, determinizations=2, seed=53)
    first = PUCTSearchPolicy(model, config=config)
    second = PUCTSearchPolicy(model, config=config)
    one = _fake_evaluation(first.start_job(request, add_root_noise=True, temperature=1.0, sampling_seed=9))
    job = second.start_job(request, add_root_noise=True, temperature=1.0, sampling_seed=9)
    while not job.done:
        pending = job.pending_requests()
        for item in pending:
            if item.kind == "policy_value":
                logits = tuple((index + 1) * 0.01 for index in range(len(item.request.legal_actions)))
                job.submit(item.request_id, logits, 0.125)
            else:
                job.submit(item.request_id, None, 0.125)
    assert job.result == one
    assert sum(job.result.root_visit_counts) == 257
    assert job.result.simulation_budgets == (129, 128)
    assert second.stats.simulations == 257
    assert second.stats.model_positions > 0
    with pytest.raises(RuntimeError, match="complete"):
        job.submit("bad", None, 0.0)


def test_revisited_folded_edges_preserve_replayable_branch():
    from splendor.benchmark_search import fixed_requests

    request = fixed_requests()[1][1]
    job = PUCTSearchPolicy(ActorCritic().eval(), config=MCTSConfig(
        simulations=128, determinizations=1, seed=53)).start_job(request)
    saw_revisited_payment = False
    while not job.done:
        for item in job.pending_requests():
            context = job.failure_context(item.request_id)
            branch = context["branch_actions"]
            saw_revisited_payment |= any(isinstance(action, Pay) for action in branch[:-2])
            scene = {
                "root_request": to_primitive(request),
                "search_context": to_primitive(context),
                "batch_requests": [{"request_id": item.request_id,
                                    "request": to_primitive(item.request)}],
            }
            reproduce(scene)  # Also checks the reconstructed leaf request exactly.
            logits = tuple(10.0 if isinstance(action, (BuyMarket, BuyReserved)) else 0.0
                           for action in item.request.legal_actions)
            job.submit(item.request_id, logits if item.kind == "policy_value" else None, 0.0)
    assert saw_revisited_payment
    assert job.result.simulations == 128


@pytest.mark.parametrize("invalid", ["value", "missing_logits", "logit_count", "logit_nan", "prior_nan"])
def test_invalid_submission_preserves_pending_world_context(invalid):
    request = GameEngine.new(42, first_player=0).decision()
    job = PUCTSearchPolicy(ActorCritic().eval(), config=MCTSConfig(
        simulations=8, determinizations=2, seed=53)).start_job(request)
    root = job.pending_requests()[0]
    job.submit(root.request_id, (0.0,) * len(request.legal_actions), 0.0)
    item = job.pending_requests()[0]
    before = job.failure_context(item.request_id)
    assert before["world_index"] == 0
    assert job.failure_context()["world_index"] == 1
    logits = (0.0,) * len(item.request.legal_actions)
    bad_logits = logits
    if invalid == "missing_logits":
        bad_logits = None
    elif invalid == "logit_count":
        bad_logits = ()
    elif invalid in ("logit_nan", "prior_nan"):
        bad_logits = (float("nan"),) + logits[1:]
    with pytest.raises(ValueError):
        job.submit(item.request_id, bad_logits,
                   float("nan") if invalid == "value" else 0.0,
                   prior_is_normalized=invalid == "prior_nan")
    assert job.failure_context(item.request_id) == before
    assert item in job.pending_requests()
    reproduce({"root_request": to_primitive(request), "search_context": to_primitive(before),
               "batch_requests": [{"request_id": item.request_id,
                                   "request": to_primitive(item.request)}]})
    job.submit(item.request_id, logits, 0.0)
    assert item not in job.pending_requests()
    assert _fake_evaluation(job).simulations == 8


def test_mcts_job_cancel_releases_pending():
    model = load_frozen_actor_critic_checkpoint(CHECKPOINT, device="cpu")
    request = GameEngine.new(42, first_player=0).decision()
    job = PUCTSearchPolicy(model, config=MCTSConfig(simulations=2)).start_job(request)
    pending = job.pending_requests()[0]
    job.cancel()
    assert job.done
    assert not job.pending_requests()
    with pytest.raises(RuntimeError, match="cancelled"):
        _ = job.result
    with pytest.raises(RuntimeError, match="cancelled"):
        job.submit(pending.request_id, (0.0,), 0.0)


def test_frozen_inference_scalar_batch_and_heads():
    torch.set_num_threads(1)
    model = load_frozen_actor_critic_checkpoint(CHECKPOINT, device="cpu")
    evaluator = FrozenSearchInference(model)
    requests = tuple(GameEngine.new(seed, first_player=0).decision() for seed in range(20, 28))
    scalar = tuple(evaluator.policy_value((request,))[0] for request in requests)
    batched = evaluator.policy_value(requests)
    for (single_logits, single_value), (batch_logits, batch_value) in zip(scalar, batched):
        np.testing.assert_allclose(single_logits, batch_logits, atol=1e-5, rtol=1e-5)
        assert single_value == pytest.approx(batch_value, abs=1e-5)
    states = evaluator.value_only(requests)
    policies = evaluator.policy_only(requests)
    for index in range(len(requests)):
        assert states[index] == pytest.approx(batched[index][1], abs=1e-5)
        np.testing.assert_allclose(policies[index], batched[index][0], atol=1e-5, rtol=1e-5)
    assert evaluator.stats.batch_sizes[8] >= 3


def test_v2_collector_resume_dataset_and_reconstruction(tmp_path):
    config = FastSelfPlayConfig.from_dict({
        "protocol": "mcts-selfplay-v2", "run_id": "test-fast", "round_index": 0,
        "checkpoint": str(CHECKPOINT), "num_games": 2, "seed": 2000,
        "search_config": {"simulations": 4, "determinizations": 2, "seed": 90},
        "execution": {"backend": "batched_cpu", "engine_mode": "fast", "workers": 1,
                      "active_games_per_worker": 2, "max_batch_size": 8},
    })
    root = tmp_path / "run"
    summary = generate(config, root)
    assert summary["completed_games"] == 2
    assert summary["inference"]["mean_batch_size"] > 1
    assert summary["statistics_scope"] == "collection_session"
    assert summary["statistics_complete"] is True
    assert summary["statistics_game_ids"] == [0, 1]
    assert summary["logical_model_positions"] == summary["inference"]["logical_model_positions"]
    assert summary["games_per_wall_second"] == pytest.approx(2 / summary["wall_seconds"])
    assert validate_run(root, replay=True)["samples"] == summary["samples"]
    assert generate(config, root, resume=True)["completed_games"] == 2
    dataset = SelfPlayDataset(root)
    sample = next(iter(dataset))
    batch = collate_selfplay_samples((sample,))
    assert batch.value_targets.shape == (1,)
    trace = json.loads((root / "manifest.json").read_text())
    assert trace["versions"]["data"] == "splendor-mcts-selfplay-v2"
    from splendor.selfplay_data import load_game, game_path
    _, metadata = load_game(game_path(root, 0))
    scene = {"trace": metadata["trace"], "root_decision_index": metadata["decision_count"]}
    assert reproduce(scene)["real_actions_replayed"] == metadata["decision_count"]
    broken = json.loads(json.dumps(metadata["trace"]))
    broken["actions"][0]["candidate_index"] = 99999
    with pytest.raises(ValueError, match="trace action 0"):
        replay_trace(broken)


def test_fast_candidate_rejects_old_state_version():
    engine = GameEngine.new(42, first_player=0, mode="fast")
    request = engine.decision()
    assert isinstance(request, DecisionRequest)
    old_version = engine.state_version
    engine.apply_candidate(0, state_version=old_version)
    with pytest.raises(IllegalActionError, match="stale"):
        engine.apply_candidate(0, state_version=old_version)
    with pytest.raises(IllegalActionError, match="not legal"):
        engine.apply(BuyMarket("not-a-market-card"))
    fork = engine.fork()
    assert fork._debug_state() == engine._debug_state()
    assert fork.state_version == engine.state_version


def test_failure_scene_replays_without_network(tmp_path, monkeypatch):
    config = FastSelfPlayConfig.from_dict({
        "protocol": "mcts-selfplay-v2", "run_id": "failure-test", "round_index": 0,
        "checkpoint": str(CHECKPOINT), "num_games": 1, "seed": 2011,
        "search_config": {"simulations": 2, "determinizations": 1, "seed": 7},
        "execution": {"backend": "batched_cpu", "engine_mode": "fast", "workers": 1,
                      "active_games_per_worker": 1, "max_batch_size": 4},
    })
    root = tmp_path / "failed"
    original = FrozenSearchInference.policy_value

    def broken(self, requests, *, encoded=None):
        raise ValueError("injected model failure")

    monkeypatch.setattr(FrozenSearchInference, "policy_value", broken)
    with pytest.raises(ValueError, match="injected model failure"):
        generate(config, root)
    scene = json.loads((root / "failures" / "game-000000.json").read_text())
    assert scene["batch_requests"][0]["request_id"]
    assert reproduce(scene)["real_actions_replayed"] == 0
    monkeypatch.setattr(FrozenSearchInference, "policy_value", original)
    assert generate(config, root, resume=True)["completed_games"] == 1


def test_partial_v2_resume_skips_committed_game(tmp_path):
    from splendor.selfplay_fast import build_manifest, _atomic_json, _collect_group
    config = FastSelfPlayConfig.from_dict({
        "protocol": "mcts-selfplay-v2", "run_id": "partial", "round_index": 0,
        "checkpoint": str(CHECKPOINT), "num_games": 2, "seed": 3100,
        "search_config": {"simulations": 2, "determinizations": 1},
        "execution": {"backend": "batched_cpu", "engine_mode": "fast", "workers": 1,
                      "active_games_per_worker": 2, "max_batch_size": 4},
    })
    root = tmp_path / "partial"
    manifest = build_manifest(config)
    _atomic_json(root / "manifest.json", manifest)
    _collect_group(config, root, (0,), manifest["semantic_fingerprint"],
                   manifest["checkpoint_identity"]["sha256"])
    first = (root / "games" / "game-000000.npz")
    before = first.stat().st_mtime_ns
    interrupted = validate_run(root)
    assert interrupted["statistics_scope"] == "unavailable"
    assert interrupted["statistics_missing_game_ids"] == [0]
    assert interrupted["games_per_wall_second"] is None
    summary = generate(config, root, resume=True)
    assert summary["completed_games"] == 2
    assert first.stat().st_mtime_ns == before
    assert validate_run(root, replay=True)["missing_games"] == 0
    from splendor.selfplay_data import load_game, game_path
    metadata = [load_game(game_path(root, i))[1] for i in (0, 1)]
    positions = [sum(s["model_positions"] for s in game["mcts_stats"]) for game in metadata]
    assert summary["logical_model_positions"] == sum(positions)
    assert summary["inference"]["logical_model_positions"] == positions[1]
    assert summary["statistics_scope"] == "collection_session"
    assert summary["statistics_complete"] is False
    assert summary["statistics_game_ids"] == [1]
    assert summary["statistics_missing_game_ids"] == [0]
    assert summary["decision_latency"]["count"] == metadata[1]["decision_count"]
    wall = summary["wall_seconds"]
    assert summary["games_per_wall_second"] == pytest.approx(1 / wall)
    assert summary["policy_samples_per_wall_second"] == pytest.approx(metadata[1]["policy_samples"] / wall)
    simulations = sum(s["simulations"] for s in metadata[1]["mcts_stats"])
    assert summary["simulations_per_wall_second"] == pytest.approx(simulations / wall)
    assert generate(config, root, resume=True) == summary

    # Old status files have no reliable coverage and must not publish run rates.
    status_path = root / "status.json"
    status = json.loads(status_path.read_text())
    del status["statistics_scope"], status["statistics_game_ids"]
    _atomic_json(status_path, status)
    legacy = validate_run(root)
    assert legacy["statistics_scope"] == "unknown"
    assert legacy["statistics_complete"] is False
    assert legacy["games_per_wall_second"] is None
    assert legacy["policy_samples_per_wall_second"] is None
    assert legacy["simulations_per_wall_second"] is None


def test_search_branch_scene_decodes_and_replays(tmp_path):
    request = GameEngine.new(42, first_player=0).decision()
    model = load_frozen_actor_critic_checkpoint(CHECKPOINT, device="cpu")
    policy = PUCTSearchPolicy(model, config=MCTSConfig(simulations=1,
                              determinizations=1, seed=123))
    job = policy.start_job(request)
    root_pending = job.pending_requests()[0]
    job.submit(root_pending.request_id, tuple(range(len(request.legal_actions))), 0.0)
    leaf = job.pending_requests()[0]
    context = to_primitive(job.failure_context(leaf.request_id))
    scene = {"protocol": "search-accel-failure-v1", "root_request": to_primitive(request),
             "search_context": context,
             "batch_requests": [{"request_id": leaf.request_id, "request": to_primitive(leaf.request)}]}
    path = tmp_path / "failure.json"
    path.write_text(json.dumps(scene))
    assert reproduce(json.loads(path.read_text()))["branch_actions_replayed"] >= 1
    completed = subprocess.run([sys.executable, "-m", "splendor.reproduce_failure",
                                "--input", str(path)], cwd=ROOT, capture_output=True,
                               text=True, check=True)
    assert json.loads(completed.stdout)["branch_actions_replayed"] >= 1
    scene["search_context"]["branch_actions"][0] = {"__type__": "BuyMarket", "card_id": "not-present"}
    with pytest.raises(ValueError, match="illegal"):
        reproduce(scene)


def test_cuda_archive_configuration_is_readable_without_gpu(tmp_path, monkeypatch):
    raw = {
        "protocol": "mcts-selfplay-v2", "run_id": "cuda-portable", "round_index": 0,
        "checkpoint": str(CHECKPOINT), "num_games": 1, "seed": 7,
        "search_config": {"simulations": 2, "determinizations": 1},
        "execution": {"backend": "batched_cuda", "engine_mode": "fast", "workers": 1,
                      "active_games_per_worker": 2, "max_batch_size": 4},
    }
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config = FastSelfPlayConfig.from_dict(raw)
    from splendor.selfplay_fast import build_manifest, _atomic_json
    root = tmp_path / "cuda-archive"
    _atomic_json(root / "manifest.json", build_manifest(config))
    assert validate_run(root)["missing_games"] == 1
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        generate(config, root, resume=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_batched_inference_matches_cpu_with_fp32_tolerance():
    from splendor.benchmark_search import micro_requests
    requests = micro_requests()[:16]
    cpu = FrozenSearchInference(load_frozen_actor_critic_checkpoint(CHECKPOINT, device="cpu"))
    cuda = FrozenSearchInference(load_frozen_actor_critic_checkpoint(CHECKPOINT, device="cuda"), device="cuda")
    cpu_rows = cpu.policy_value(requests)
    cuda_rows = cuda.policy_value(requests)
    for (cpu_logits, cpu_value), (cuda_logits, cuda_value) in zip(cpu_rows, cuda_rows):
        np.testing.assert_allclose(cpu_logits, cuda_logits, atol=1e-5, rtol=1e-5)
        assert cpu_value == pytest.approx(cuda_value, abs=1e-5)
