from dataclasses import asdict, replace
import json

import numpy as np
import pytest
import torch

from test_alphazero_training import az_runs
from splendor.actor_critic import build_actor_critic
from splendor.alphazero_checkpoint import load_alphazero_checkpoint
from splendor.alphazero_recording import sha256, write_json
from splendor.alphazero_replay import build_replay_index
from splendor.alphazero_train import AlphazeroLearner, LearnerConfig, draw_training_batch
from splendor.entity_attention import EntityAttention
from splendor.offline_objectives import OfflineObjectiveConfig, freeze_inputs, initialize_learner, run
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.rng import DeterministicRng
from splendor.selfplay_data import SelfPlaySample, collate_selfplay_samples
from splendor.value_auxiliary import ValueAuxiliaryConfig, add_auxiliary_heads, auxiliary_targets


@pytest.fixture(autouse=True, scope="module")
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_auxiliary_labels_use_actor_score_and_unfinished_full_turns():
    arrays = {"player_id": np.array([0, 1, 1]), "completed_player_turns": np.array([4, 4, 5])}
    metadata = {"terminal_result": {"scores": [18, 15], "completed_turns": [3, 3]}}
    samples = [SelfPlaySample(arrays, metadata, i) for i in range(3)]
    score, turns = auxiliary_targets(samples, ValueAuxiliaryConfig())
    np.testing.assert_allclose(score, [.2, -.2, -.2])
    np.testing.assert_allclose(turns, [.05, .05, .025])
    # Multiple decisions inside one unfinished turn have the same countdown.
    arrays["completed_player_turns"][0] = 7
    with pytest.raises(ValueError, match="terminal turn count"):
        auxiliary_targets(samples, ValueAuxiliaryConfig())


@pytest.fixture
def setup(tmp_path, az_runs):
    _, first, second = az_runs
    index = build_replay_index([first, second], max_games=6, cache_bytes=64 * 1024**2)
    samples = [index.cache.sample(index.games[0], i) for i in range(3)]
    cfg = LearnerConfig(device="cpu", batch_size=4, history_batch_size=2,
                        learning_rate=2e-5, checkpoint_every_steps=1, validation_batch_size=4)
    parent = AlphazeroLearner(EntityAttention(), cfg, sampling_rng=DeterministicRng(7))
    parent.train_step(collate_selfplay_samples(samples), references=[])
    path = tmp_path / "parent.pt"
    parent.save(path)
    records = []
    for game in index.games:
        stat = game.path.stat()
        records.append({**asdict(game), "path": str(game.path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                        "archive_sha256": sha256(game.path)})
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in records))
    config = OfflineObjectiveConfig(manifest=str(manifest), manifest_sha256=sha256(manifest),
                                    initial_checkpoint=str(path), initial_checkpoint_sha256=sha256(path),
                                    first_round=2, last_round=2, expected_games_per_round=3,
                                    draw_exposures=.02, cache_mib=64, validation_positions=4,
                                    value_auxiliary=asdict(ValueAuxiliaryConfig()), learner=cfg)
    return config, parent, samples


def test_upgrade_preserves_policy_value_and_auxiliary_gradients_reach_encoder(setup):
    _, parent, samples = setup
    model = add_auxiliary_heads(parent.model)
    batch = collate_selfplay_samples(samples).decisions
    expected = parent.model(batch)
    actual = model.forward_with_auxiliary(batch)
    for a, b in zip(expected, actual[:2]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert torch.all(actual[3] > 0)
    (actual[2].square().mean() + actual[3].square().mean()).backward()
    assert model.context_encoder.weight.grad.abs().sum() > 0
    assert model.critic.network[1].weight.grad is None
    assert model.actor[1].weight.grad is None
    restored = build_actor_critic(asdict(model.config))
    restored.load_state_dict(model.state_dict(), strict=True)
    for a, b in zip(model(batch), restored(batch)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_optimizer_upgrade_retains_old_moments_and_empty_new_head_state(setup, tmp_path):
    config, parent, samples = setup
    learner = initialize_learner(config, {"digest": "test"}, 3)
    for (_, old), (_, new) in zip(parent.model.named_parameters(), learner.model.named_parameters()):
        torch.testing.assert_close(old, new, rtol=0, atol=0)
        for name, value in parent.optimizer.state[old].items():
            torch.testing.assert_close(value, learner.optimizer.state[new][name], rtol=0, atol=0)
    for head in (learner.model.score_head, learner.model.remaining_turns_head):
        assert all(not learner.optimizer.state.get(p) for p in head.parameters())
    labels = auxiliary_targets(samples, ValueAuxiliaryConfig())
    record = learner.train_step(collate_selfplay_samples(samples), references=[],
                                value_auxiliary=ValueAuxiliaryConfig(), auxiliary_labels=labels)
    assert record["total_loss"] == pytest.approx(record["policy_loss"] + record["value_loss"] +
                                                .25 * record["score_loss"] + .25 * record["remaining_turns_loss"], abs=1e-6)
    checkpoint = tmp_path / "aux.pt"
    learner.save(checkpoint)
    frozen = load_frozen_actor_critic_checkpoint(checkpoint)
    for a, b in zip(learner.model(collate_selfplay_samples(samples).decisions), frozen(collate_selfplay_samples(samples).decisions)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_branches_use_identical_frozen_pool_and_sample_stream(setup, tmp_path):
    config, _, _ = setup
    psw = replace(config, value_auxiliary=None, policy_surprise_weighting={"fraction": .5})
    _, pool_a, identity_a, fixed_a, plan_a = freeze_inputs(config, tmp_path / "aux")
    _, pool_b, identity_b, fixed_b, plan_b = freeze_inputs(psw, tmp_path / "psw")
    assert fixed_a == fixed_b and plan_a["steps_target"] == plan_b["steps_target"]
    a = initialize_learner(config, identity_a, plan_a["steps_target"])
    b = initialize_learner(psw, identity_b, plan_b["steps_target"])
    for name, value in b.model.state_dict().items():
        torch.testing.assert_close(a.model.state_dict()[name], value, rtol=0, atol=0)
    for _ in range(3):
        da = draw_training_batch(pool_a, None, a.sampling_rng, a.config)
        db = draw_training_batch(pool_b, None, b.sampling_rng, b.config)
        assert da.references == db.references


@pytest.mark.parametrize("branch", ["policy_surprise", "value_auxiliary"])
def test_offline_entry_freezes_selected_round_and_resumes_exactly(setup, tmp_path, monkeypatch, branch):
    import splendor.offline_objectives as entry
    config, _, _ = setup
    if branch == "policy_surprise":
        config = replace(config, value_auxiliary=None, policy_surprise_weighting={"fraction": .5})
    monkeypatch.setattr(entry, "source_identity", lambda *_: {"test": True})
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    index, pool, _, _, plan = freeze_inputs(config, full)
    assert len(index.games) == 3 and all(g.round_index == 1 for g in index.games)
    assert plan["steps_target"] >= 2
    run(config, full)
    run(config, resumed, max_steps_this_session=1)
    run(config, resumed)
    a = load_alphazero_checkpoint(full / "checkpoints/final.pt")
    b = load_alphazero_checkpoint(resumed / "checkpoints/final.pt")
    for key in a.model_state_dict:
        torch.testing.assert_close(a.model_state_dict[key], b.model_state_dict[key], rtol=0, atol=0)
    assert a.sampling_rng_state == b.sampling_rng_state
    for key, state in a.optimizer_state_dict["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(value, b.optimizer_state_dict["state"][key][name], rtol=0, atol=0)
    assert json.loads((full / "full_validation.json").read_text())["positions"] == sum(g.decision_count for g in index.split_games("validation"))
    bad = replace(config, sampling_seed=config.sampling_seed + 1)
    with pytest.raises(ValueError, match="identical frozen configuration"):
        freeze_inputs(bad, full)
