from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

from artifacts.entity_attention_r37_104k_offline_lr_20261027.train import (
    R37OfflineConfig, initialize_learner, prepare_inputs, run, select_checkpoint)
from artifacts.entity_attention_r37_104k_offline_lr_20261027.run import configured_entry, TRAIN_MODULE
from splendor.alphazero_checkpoint import load_alphazero_checkpoint
from splendor.alphazero_recording import read_json, sha256
from splendor.offline_scratch import initialize_learner as scratch_initializer
from splendor.rng import DeterministicRng
from test_offline_mixed_budget import mixed, one_thread

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def warm(mixed, tmp_path, monkeypatch):
    import splendor.offline_objectives as core
    monkeypatch.setattr(core, "source_identity", lambda *_: {"engineering_test": True})
    base = replace(mixed, model_config={"architecture":"entity_attention_v1", "hidden":16,
        "heads":2, "layers":1, "ffn_hidden":32})
    # A genuine updated optimizer in the parent must not leak into this new experiment.
    from splendor.offline_mixed_budget import run as scratch_run
    scratch_run(base, tmp_path / "parent", max_steps_this_session=2)
    path = tmp_path / "parent/checkpoints/step-000000002.pt"
    cfg = R37OfflineConfig.from_dict(asdict(base) | {"initial_checkpoint":str(path),
        "initial_checkpoint_sha256":sha256(path), "early_stopping_increases":3})
    return cfg


def test_initializes_parent_weights_but_fresh_optimizer_offline_rng_and_local_steps(warm):
    parent = load_alphazero_checkpoint(warm.initial_checkpoint)
    assert parent.optimizer_state_dict["state"]
    learner = initialize_learner(warm, {"test":True}, 8)
    assert not learner.optimizer.state and learner.total_train_step == learner.round_train_step == 0
    assert learner.sampling_rng.snapshot() == DeterministicRng(warm.sampling_seed).snapshot()
    for key, value in learner.model.state_dict().items():
        torch.testing.assert_close(value, parent.model_state_dict[key], rtol=0, atol=0)
    actual_rng = torch.get_rng_state().clone()
    scratch_initializer(warm, {"test":True}, 8)
    torch.testing.assert_close(actual_rng, torch.get_rng_state(), rtol=0, atol=0)


def test_formal_two_variants_only_change_initial_weights_and_requested_lr():
    baseline = read_json(ROOT / "configs/entity_attention_v1/offline_noaux_r21_r37_20261026.json")
    for tag, lr in (("lr1e4", 1e-4), ("lr2e5", 2e-5)):
        cfg = read_json(ROOT / f"configs/entity_attention_v1/offline_r37_104k_{tag}_20261027.json")
        assert cfg.pop("initial_checkpoint").endswith("round-005/checkpoints/final.pt")
        assert cfg.pop("initial_checkpoint_sha256") == "611f4f4ed18f397b03c79bb3a2beae51e7a77a747883680d9189eed3052a1045"
        expected = {**baseline, "learner": baseline["learner"] | {"learning_rate":lr}}
        assert cfg == expected
        plan = read_json(ROOT / f"artifacts/entity_attention_r37_104k_{tag}_20261027/plan.json")
        assert plan["steps_target"] == 350552 and plan["training_module"] == TRAIN_MODULE
        assert not plan["initialization"]["preserve_optimizer"]
        assert plan["panels"][0]["pairs"] == 1000 and plan["panels"][1]["pairs"] == 500


def test_checkpoint_architecture_mismatch_is_rejected(warm):
    with pytest.raises(ValueError, match="architecture"):
        initialize_learner(replace(warm, model_config=warm.model_config | {"ffn_hidden":64}), {}, 8)


@pytest.mark.parametrize("tag", ["lr1e4", "lr2e5"])
def test_supervisor_routes_training_and_selection_to_the_weights_only_entry(tag, tmp_path, monkeypatch):
    import subprocess
    pipeline = configured_entry(tag)
    pipeline.RUNTIME = tmp_path
    calls = []
    class Child:
        pid = 12345
        stdout = ()
        def __init__(self, command, **kwargs):
            calls.append(command)
        def wait(self):
            return 0
    monkeypatch.setattr(subprocess, "Popen", Child)
    pipeline.run_child(["python", "-u", "-m", "splendor.offline_continuation"], "train", tmp_path)
    pipeline.run_child(["python", "-u", "-m", "splendor.offline_continuation"], "select", tmp_path)
    pipeline.run_child(["python", "-u", "-m", "splendor.evaluation"], "greedy_vs_R37", tmp_path)
    assert [command[3] for command in calls] == [TRAIN_MODULE, TRAIN_MODULE, "splendor.evaluation"]
