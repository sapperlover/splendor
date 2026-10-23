from dataclasses import asdict

import numpy as np
import pytest
import torch

from splendor.actor_critic import TorchStateBatch, build_actor_critic, tensorize_batch
from splendor.engine import GameEngine
from splendor.features import GLOBAL_FEATURE_NAMES as G, collate_encoded_decisions, encode_decision
from splendor.progress_auxiliary import (ProgressAuxiliaryConfig, add_progress_head,
    game_progress_targets, progress_loss_sums)


@pytest.fixture(autouse=True, scope="module")
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_retained_progress_head_preserves_base_forward_and_state_only_inputs():
    torch.manual_seed(19)
    base = build_actor_critic({"architecture": "entity_attention_v1", "hidden": 16, "heads": 2,
        "layers": 1, "ffn_hidden": 32, "market_relations": True,
        "value_readout": "global_players_market"})
    model = add_progress_head(base)
    batch = tensorize_batch(collate_encoded_decisions([encode_decision(GameEngine.new(42).decision())]))
    for name, value in base.state_dict().items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
    for expected, actual in zip(base(batch), model(batch)):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    state = TorchStateBatch(**{k: getattr(batch, k) for k in TorchStateBatch.__dataclass_fields__})
    torch.testing.assert_close(model.value_from_state(state), model(batch)[1])
    _, _, progress = model.forward_with_progress(batch)
    labels = (np.zeros((1, 2, 2), dtype=np.int8), np.zeros((1, 2, 2, 5), dtype=np.int8), np.ones(1, dtype=bool))
    sums = progress_loss_sums(progress, labels, torch.ones(1))
    (sums["progress_score_ce_sum"] + sums["progress_bonus_ce_sum"]).backward()
    assert model.context_encoder.weight.grad.abs().sum() > 0
    assert model.critic.network[1].weight.grad is None
    assert model.actor[1].weight.grad is None
    restored = build_actor_critic(asdict(model.config))
    restored.load_state_dict(model.state_dict(), strict=True)
    for expected, actual in zip(model(batch), restored(batch)):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)


def timeline(final_phase="payment"):
    # p0 buys on turn0; p1 buys on turn1. Each payment is a subdecision.
    players = np.array([0, 0, 1, 1])
    g = np.zeros((4, 148), dtype=np.float32)
    for i in (0, 2):
        g[i, G.index("phase_main_action")] = 1
    g[1, G.index("phase_payment")] = 1
    g[3, G.index("phase_" + final_phase)] = 1
    g[2:, G.index("opponent_score_scaled")] = 1/15
    g[2:, G.index("opponent_bonus_white_scaled")] = 1/18
    if final_phase == "choose_noble":
        g[3, G.index("self_score_scaled")] = 2/15
        g[3, G.index("self_bonus_blue_scaled")] = 1/18
    pending = np.zeros((4, 29), dtype=np.float32)
    pending[3, :2] = 1
    pending[3, 6] = 1  # blue bonus applied by the last Pay
    arrays = {"global_features": g, "player_id": players, "completed_player_turns": np.array([0, 0, 1, 1]),
              "pending_card": pending, "pending_card_mask": np.array([False, False, False, True])}
    return arrays, {"status": "natural_terminal", "terminal_result": {"scores": [1, 2 if final_phase == "payment" else 5], "completed_turns": [1, 1]}}


@pytest.mark.parametrize("phase", ["payment", "choose_noble"])
def test_progress_turn_units_perspective_terminal_purchase_and_noble(phase):
    arrays, meta = timeline(phase)
    labels = game_progress_targets(arrays, meta)
    assert labels["progress_mask"].tolist() == [True, False, True, False]
    score = 2 if phase == "payment" else 5
    np.testing.assert_array_equal(labels["progress_score"][0], [[1, score], [1, score]])
    np.testing.assert_array_equal(labels["progress_score"][2], [[score, 0], [score, 0]])
    np.testing.assert_array_equal(labels["progress_bonus"][0, 0], [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]])
    np.testing.assert_array_equal(labels["progress_bonus"][2, 0], [[0, 1, 0, 0, 0], [0]*5])
    # Final forced skips increase the clock but not the resources.
    meta["terminal_result"]["completed_turns"] = [2, 2]
    meta["terminal_result"]["reason"] = "consecutive_forced_skips"
    for key, value in game_progress_targets(arrays, meta).items():
        np.testing.assert_array_equal(value, labels[key])


def test_masked_progress_loss_has_no_gradient_and_outputs_are_averaged():
    logits = torch.zeros(2, 128, requires_grad=True)
    labels = (np.zeros((2, 2, 2), dtype=np.int8), np.zeros((2, 2, 2, 5), dtype=np.int8), np.array([True, False]))
    sums = progress_loss_sums(logits, labels, torch.ones(2))
    assert float(sums["progress_score_ce_sum"].detach()) == pytest.approx(np.log(17))
    assert float(sums["progress_bonus_ce_sum"].detach()) == pytest.approx(np.log(3))
    (sums["progress_score_ce_sum"] + sums["progress_bonus_ce_sum"]).backward()
    assert logits.grad[0].abs().sum() > 0 and logits.grad[1].abs().sum() == 0
    for weight in (-1, 0, float("nan"), True):
        with pytest.raises(ValueError):
            ProgressAuxiliaryConfig(loss_weight=weight)


