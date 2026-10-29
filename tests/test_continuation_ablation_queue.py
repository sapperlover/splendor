from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path
import threading
import fcntl

import pytest
import torch

from test_continuation_auxiliary import continuation_pool, opponent_timeline
from test_continuation_diagnostics import diagnostic_batch
from splendor.actor_critic import build_actor_critic
from splendor.alphazero_checkpoint import load_alphazero_checkpoint
from splendor.alphazero_recording import read_json, sha256, write_json
from splendor.alphazero_train import AlphazeroLearner, LearnerConfig
from splendor.continuation_auxiliary import (ContinuationAuxiliaryConfig, add_continuation_heads,
    game_continuation_targets)
from splendor.offline_continuation import run, select_checkpoint
from splendor.rng import DeterministicRng

ROOT = Path(__file__).resolve().parents[1]
QUEUE_ART = ROOT / "artifacts/entity_attention_auxiliary_queue_20261024"


@pytest.fixture(autouse=True, scope="module")
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def load_entry(filename):
    spec = importlib.util.spec_from_file_location("queue_" + filename, QUEUE_ART / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_policy_only_has_no_shortterm_labels_head_or_diagnostics_and_keeps_opponent_initialization():
    arrays, meta = opponent_timeline()  # No archived MCTS value sums are needed.
    cfg = ContinuationAuxiliaryConfig(shortterm_value_weight=0., opponent_policy_weight=.15)
    targets = game_continuation_targets(arrays, meta, cfg)
    assert targets and all(not key.startswith("shortterm_") for key in targets)
    base = build_actor_critic({"architecture": "entity_attention_v1", "hidden": 16, "heads": 2,
                              "layers": 1, "ffn_hidden": 32})
    torch.manual_seed(123)
    both = add_continuation_heads(base, predict_opponent_policy=True)
    both_rng = torch.get_rng_state()
    torch.manual_seed(123)
    policy = add_continuation_heads(base, predict_opponent_policy=True, predict_shortterm_value=False)
    assert not hasattr(policy, "shortterm_value_head")
    assert all(not key.startswith("shortterm_") for key in policy.state_dict())
    torch.testing.assert_close(torch.get_rng_state(), both_rng, rtol=0, atol=0)
    for key, value in policy.state_dict().items():
        torch.testing.assert_close(value, both.state_dict()[key], rtol=0, atol=0)
    batch, labels = diagnostic_batch()
    labels = {k: v for k, v in labels.items() if not k.startswith("shortterm_")}
    learner = AlphazeroLearner(policy, LearnerConfig(device="cpu"), sampling_rng=DeterministicRng(7))
    metrics = learner.train_step(batch, references=[], value_batch=batch, continuation_auxiliary=cfg,
                                 continuation_labels=labels, record_encoder_grad_norms=True)
    assert metrics["encoder_grad_norm_opponent_policy"] > 0
    assert not any("shortterm" in key for key in metrics)
    assert policy.context_encoder.weight.grad.abs().sum() > 0
    restored = build_actor_critic(asdict(policy.config))
    restored.load_state_dict(policy.state_dict(), strict=True)
    with torch.inference_mode():
        for expected, actual in zip(policy(batch.decisions), restored(batch.decisions)):
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)


@pytest.mark.parametrize("weights", [(0., 0.), (-.1, .15), (.25, -.1)])
def test_disabled_or_negative_auxiliary_configuration_is_rejected(weights):
    with pytest.raises(ValueError):
        ContinuationAuxiliaryConfig(shortterm_value_weight=weights[0], opponent_policy_weight=weights[1])


def test_policy_only_full_validation_selection_lightweight_recording_and_exact_resume(continuation_pool, tmp_path):
    cfg = replace(continuation_pool, continuation_auxiliary={"shortterm_value_weight": 0., "opponent_policy_weight": .15})
    full, resumed = tmp_path / "policy_full", tmp_path / "policy_resumed"
    run(cfg, full)
    run(cfg, resumed, max_steps_this_session=1)
    run(cfg, resumed)
    a, b = [load_alphazero_checkpoint(p / "checkpoints/final.pt") for p in (full, resumed)]
    assert not a.model_config["predict_shortterm_value"] and a.model_config["predict_opponent_policy"]
    assert a.sampling_rng_state == b.sampling_rng_state
    for key, value in a.model_state_dict.items():
        torch.testing.assert_close(value, b.model_state_dict[key], rtol=0, atol=0)
    for pid, state in a.optimizer_state_dict["state"].items():
        for key, value in state.items():
            torch.testing.assert_close(value, b.optimizer_state_dict["state"][pid][key], rtol=0, atol=0)
    metrics = read_json(resumed / "full_validation/step-000000005.json")["metrics"]
    assert metrics["opponent_positions"] > 0 and "shortterm_loss" not in metrics
    assert select_checkpoint(cfg, resumed)["selected_step"] in [2, 3, 4, 5]
    assert not list(resumed.rglob("*.bin")) and not list(resumed.rglob("*.steps.jsonl"))


def test_formal_three_arms_change_only_requested_fields_and_keep_r37_panels():
    entry = load_entry("run_experiment.py")
    base = read_json(entry.PARENT_CONFIG)
    for key, arm in entry.VARIANTS.items():
        cfg = read_json(ROOT / "configs/entity_attention_v1" / arm["config"])
        assert cfg["continuation_auxiliary"] == dict(base["continuation_auxiliary"],
            shortterm_value_weight=arm["value"], opponent_policy_weight=arm["policy"])
        for name in ("continuation_auxiliary", "manifest", "manifest_sha256", "expected_games_per_round", "last_round"):
            cfg.pop(name)
        reference = {k: v for k, v in base.items() if k in cfg}
        assert cfg == reference
        art = ROOT / "artifacts" / arm["name"]
        plan = read_json(art / "plan.json")
        assert plan["opponents"] == read_json(entry.PARENT_ART / "plan.json")["opponents"]
        assert [p["pairs"] for p in plan["panels"]] == [1000, 500]
        assert read_json(art / "evaluation_templates.json") == read_json(entry.PARENT_ART / "evaluation_templates.json")
        assert plan["steps_target"] == (148632 if arm["games"] == 44000 else 350552)


def test_complete_boundary_rejects_training_only_or_failed_panels(tmp_path, monkeypatch):
    q = load_entry("serial_queue.py")
    monkeypatch.setattr(q, "ROOT", tmp_path)
    name = "fixture"
    art, output = tmp_path / "artifacts" / name, tmp_path / "runs" / name
    checkpoint = output / "candidate.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"candidate")
    selection = {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint), "selected_step": 2}
    write_json(art / "execution.json", {"sessions": [{"status": "running"}]})
    write_json(output / "progress.json", {"status": "complete"})
    write_json(art / "summary.json", {"status": "complete"})
    write_json(output / "selection.json", selection)
    write_json(art / "selected_candidate.json", selection)
    with pytest.raises(ValueError, match="entire"):
        q.require_complete(name)
    write_json(art / "execution.json", {"sessions": [{"status": "complete"}]})
    for identifier, pairs in [("greedy_vs_R37", 1000), ("search_S1024N1_vs_R37", 500)]:
        config = {"test": identifier}
        result = {"passed": True, "completed_pairs": pairs, "valid_games": pairs*2,
                  "config": config, "checkpoint_identity": {"policy_a": {"sha256": sha256(checkpoint)}},
                  **{k: 0 for k in ["exceptions", "illegal_actions", "engineering_truncations", "invariant_failures"]}}
        write_json(art / "evaluation_configs" / f"{identifier}.json", config)
        write_json(art / "panels" / f"{identifier}.json", result)
    assert len(q.require_complete(name)["panels"]) == 2
    bad = art / "panels/search_S1024N1_vs_R37.json"
    result = read_json(bad);result["completed_pairs"] = 499;write_json(bad, result)
    with pytest.raises(ValueError, match="incomplete"):
        q.require_complete(name)


def test_wait_uses_full_supervisor_lock_without_starting_child(tmp_path, monkeypatch):
    q = load_entry("serial_queue.py")
    monkeypatch.setattr(q, "ROOT", tmp_path)
    monkeypatch.setattr(q, "RUNTIME", tmp_path / "queue_runtime")
    lock_path = tmp_path / "runtime/current/chain.lock"
    lock_path.parent.mkdir(parents=True)
    entered = threading.Event()
    monkeypatch.setattr(q, "require_complete", lambda name: entered.set() or {"experiment": name})
    with lock_path.open("a") as owner:
        fcntl.flock(owner, fcntl.LOCK_EX)
        worker = threading.Thread(target=q.wait_for_predecessor, args=("current", {}))
        worker.start()
        assert not entered.wait(.1)  # Child/trainer exit alone cannot release this supervisor lock.
        fcntl.flock(owner, fcntl.LOCK_UN)
        worker.join(2)
    assert entered.is_set() and not worker.is_alive()


def test_queue_runs_three_whole_scripts_sequentially_and_stops_on_error(monkeypatch, tmp_path):
    q = load_entry("serial_queue.py")
    monkeypatch.setattr(q, "ROOT", tmp_path)
    events = []
    monkeypatch.setattr(q, "write", lambda *_: None)
    monkeypatch.setattr(q, "wait_for_predecessor", lambda name, state: events.append(("wait", name)) or {})
    monkeypatch.setattr(q, "require_complete", lambda name: events.append(("verify", name)) or {"experiment": name})
    class Child:
        pid = 42
        def __init__(self, command, **_): events.append(("start", command[1]));self.command = command
        def wait(self): events.append(("finish", self.command[1]));return 0
    monkeypatch.setattr(q.subprocess, "Popen", Child)
    state = {"completed": [], "predecessor_checks": []}
    q.run_queue(state)
    assert state["status"] == "complete" and len(state["completed"]) == 3
    assert [event[0] for event in events] == ["wait", "start", "finish", "verify"]*3
    class Failure(Child):
        def wait(self): return 1
    events.clear();monkeypatch.setattr(q.subprocess, "Popen", Failure)
    with pytest.raises(q.subprocess.CalledProcessError):
        q.run_queue({"completed": [], "predecessor_checks": []})
    assert len([e for e in events if e[0] == "start"]) == 1
