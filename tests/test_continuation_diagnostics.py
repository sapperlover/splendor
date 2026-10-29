from copy import deepcopy
import math

import numpy as np
import pytest
import torch

from splendor.actor_critic import build_actor_critic
from splendor.alphazero_train import AlphazeroLearner, LearnerConfig, encoder_gradient_metrics
from splendor.continuation_auxiliary import ContinuationAuxiliaryConfig, add_continuation_heads
from splendor.opponent_policy import CARD_CLASSES, UNKNOWN_SLOT, opponent_loss_sums, opponent_metrics
from splendor.rng import DeterministicRng
from splendor.selfplay_data import SelfPlaySample, collate_selfplay_samples
from test_continuation_auxiliary import opponent_timeline


@pytest.fixture(autouse=True, scope="module")
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_analytic_encoder_grad_norms_weighting_unused_parameters_and_no_accumulation():
    p = torch.nn.Parameter(torch.tensor([3., -4.]))
    unused = torch.nn.Parameter(torch.tensor([9.]))
    head = torch.nn.Parameter(torch.tensor([2.]))
    p.grad, unused.grad = torch.ones_like(p), torch.ones_like(unused)
    before = torch.get_rng_state().clone()
    result = encoder_gradient_metrics((p, unused), {
        "policy": (2*p[0] - p[1], .5), "value": (p.square().sum(), .3), "head_only": (head.sum(), 1.)})
    assert result["encoder_grad_norm_policy"] == pytest.approx(math.sqrt(5))
    assert result["encoder_grad_norm_policy_weighted"] == pytest.approx(.5*math.sqrt(5))
    assert result["encoder_grad_norm_value"] == 10
    assert result["encoder_grad_norm_value_weighted"] == 3
    assert result["encoder_grad_norm_head_only"] == 0
    torch.testing.assert_close(p.grad, torch.ones_like(p), rtol=0, atol=0)
    torch.testing.assert_close(unused.grad, torch.ones_like(unused), rtol=0, atol=0)
    torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)
    # The original graph remains usable for the actual joint update.
    p.grad = None
    (2*p[0] - p[1] + p.square().sum()).backward()
    torch.testing.assert_close(p.grad, torch.tensor([8., -9.]))


def diagnostic_batch():
    from splendor.opponent_policy import game_opponent_policy_targets
    arrays, meta = opponent_timeline()
    meta["game_id"] = 0
    arrays.update(decision_index=np.arange(4), value_target=np.array([1, 1, -1, -1], np.int8),
        value_weight=np.array([1, .5, 2, 0], np.float32), behavior_policy=arrays["policy_target"])
    batch = collate_selfplay_samples([SelfPlaySample(arrays, meta, i) for i in range(4)])
    labels = {"shortterm_value_target": np.full((4, 3), .4, np.float32),
        "shortterm_value_mask": np.array([1, 0, 1, 0], bool), "shortterm_main_mask": np.array([1, 0, 1, 0], bool)}
    labels.update(game_opponent_policy_targets(arrays, meta))
    return batch, labels


def test_state_only_continuation_matches_full_batch_outputs_gradients_and_schema():
    from dataclasses import replace
    from splendor.features import EncodedStateBatch
    from splendor.alphazero_train import _cpu_scalar_sums
    torch.manual_seed(91024)
    base = build_actor_critic({"architecture": "entity_attention_v1", "hidden": 16, "heads": 2,
        "layers": 1, "ffn_hidden": 32, "market_relations": True, "value_readout": "global_players_market"})
    model = add_continuation_heads(base, predict_opponent_policy=True)
    twin = deepcopy(model)
    batch, _ = diagnostic_batch()
    state = EncodedStateBatch(**{name: getattr(batch.decisions, name) for name in EncodedStateBatch.__dataclass_fields__})
    a, aa = model.value_with_continuation(batch.decisions)
    b, bb = twin.value_with_continuation_from_state(state)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert aa.keys() == bb.keys()
    for name in aa:
        torch.testing.assert_close(aa[name], bb[name], rtol=0, atol=0)
    (a.sum() + sum(x.sum() for x in aa.values())).backward()
    (b.sum() + sum(x.sum() for x in bb.values())).backward()
    for pa, pb in zip(model.parameters(), twin.parameters()):
        if pa.grad is None:
            assert pb.grad is None
        else:
            torch.testing.assert_close(pa.grad, pb.grad, rtol=0, atol=0)
    with pytest.raises(ValueError, match="schema"):
        twin.value_with_continuation_from_state(replace(state, feature_schema_version="invalid"))
    sums = {"a": a.sum(), "b": aa["shortterm_value"].sum()}
    assert _cpu_scalar_sums(sums) == {key: float(value.detach()) for key, value in sums.items()}


def test_state_only_progress_matches_full_batch_outputs_and_gradients():
    from splendor.features import EncodedStateBatch
    from splendor.progress_auxiliary import add_progress_head
    base = build_actor_critic({"architecture": "entity_attention_v1", "hidden": 16, "heads": 2,
        "layers": 1, "ffn_hidden": 32, "market_relations": True, "value_readout": "global_players_market"})
    model = add_progress_head(base)
    twin = deepcopy(model)
    batch, _ = diagnostic_batch()
    state = EncodedStateBatch(**{name: getattr(batch.decisions, name) for name in EncodedStateBatch.__dataclass_fields__})
    a = model.value_with_progress(batch.decisions)
    b = twin.value_with_progress_from_state(state)
    for expected, actual in zip(a, b):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    sum(x.sum() for x in a).backward()
    sum(x.sum() for x in b).backward()
    for pa, pb in zip(model.parameters(), twin.parameters()):
        if pa.grad is None:
            assert pb.grad is None
        else:
            torch.testing.assert_close(pa.grad, pb.grad, rtol=0, atol=0)


@pytest.mark.parametrize("failure", ["logits", "values", "auxiliary_loss", "gradient"])
def test_nonfinite_failure_precedes_optimizer_update(monkeypatch, failure):
    import splendor.alphazero_train as training
    base = build_actor_critic({"architecture": "entity_attention_v1", "hidden": 16, "heads": 2,
        "layers": 1, "ffn_hidden": 32})
    model = add_continuation_heads(base, predict_opponent_policy=True)
    learner = AlphazeroLearner(model, LearnerConfig(device="cpu"), sampling_rng=DeterministicRng(7))
    before = deepcopy(model.state_dict())
    if failure == "logits":
        original = model.policy_logits
        monkeypatch.setattr(model, "policy_logits", lambda batch: original(batch) * float("nan"))
    elif failure == "values":
        original = model.value_with_continuation_from_state
        def broken_values(batch):
            values, predictions = original(batch)
            return values * float("inf"), predictions
        monkeypatch.setattr(model, "value_with_continuation_from_state", broken_values)
    elif failure == "auxiliary_loss":
        original = training.continuation_loss
        monkeypatch.setattr(training, "continuation_loss", lambda *args: original(*args) * float("nan"))
    else:
        monkeypatch.setattr(training, "clip_grad_norm_", lambda *args: float("inf"))
    batch, labels = diagnostic_batch()
    with pytest.raises(training.NonFiniteTrainingError):
        learner.train_step(batch, references=[], value_batch=batch,
            continuation_auxiliary=ContinuationAuxiliaryConfig(opponent_policy_weight=.15), continuation_labels=labels)
    assert learner.total_train_step == learner.round_train_step == 0
    assert not learner.optimizer.state_dict()["state"]
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_all_task_diagnostics_leave_joint_updates_optimizer_gradients_and_rng_bitwise_identical():
    torch.manual_seed(91023)
    base = build_actor_critic({"architecture": "entity_attention_v1", "hidden": 16, "heads": 2,
        "layers": 1, "ffn_hidden": 32, "market_relations": True, "value_readout": "global_players_market"})
    model = add_continuation_heads(base, predict_opponent_policy=True)
    cfg = LearnerConfig(device="cpu", policy_loss_weight=1.7, value_loss_weight=.4, max_grad_norm=.1)
    ordinary = AlphazeroLearner(model, cfg, sampling_rng=DeterministicRng(7))
    diagnosed = AlphazeroLearner(deepcopy(model), cfg, sampling_rng=DeterministicRng(7))
    auxiliary = ContinuationAuxiliaryConfig(opponent_policy_weight=.15)
    batch, labels = diagnostic_batch()
    encoder = diagnosed.model.shared_encoder_parameters()
    encoder_ids = {id(p) for p in encoder}
    assert not encoder_ids & {id(p) for module in (diagnosed.model.critic, diagnosed.model.actor,
        diagnosed.model.action_encoder, diagnosed.model.cross_attention, diagnosed.model.memory_norm,
        diagnosed.model.shortterm_value_head, diagnosed.model.opponent_policy_head) for p in module.parameters()}
    assert id(diagnosed.model.market_relation_encoder.weight) in encoder_ids
    tasks = ("policy", "value", "shortterm_h2", "shortterm_h6", "shortterm_h14", "shortterm_value",
             "opponent_action", "opponent_buy", "opponent_reserve", "opponent_policy")
    for _ in range(3):
        before = torch.get_rng_state().clone()
        a = ordinary.train_step(batch, references=[], value_batch=batch,
            continuation_auxiliary=auxiliary, continuation_labels=labels)
        b = diagnosed.train_step(batch, references=[], value_batch=batch,
            continuation_auxiliary=auxiliary, continuation_labels=labels, record_encoder_grad_norms=True)
        for task in tasks:
            assert b[f"encoder_grad_norm_{task}"] > 0
            coefficient = (1.7 if task == "policy" else .4 if task == "value" else
                           .25/3 if task.startswith("shortterm_h") else .25 if task == "shortterm_value" else .15)
            assert b[f"encoder_grad_norm_{task}_weighted"] == pytest.approx(coefficient*b[f"encoder_grad_norm_{task}"])
        assert 0 < b["encoder_grad_norm_total"] <= b["grad_norm"]
        assert b["grad_norm"] == a["grad_norm"] and b["grad_clipped"]
        for pa, pb in zip(ordinary.model.parameters(), diagnosed.model.parameters()):
            torch.testing.assert_close(pa, pb, rtol=0, atol=0)
            torch.testing.assert_close(pa.grad, pb.grad, rtol=0, atol=0)
        for pid, state in ordinary.optimizer.state_dict()["state"].items():
            for key, value in state.items():
                torch.testing.assert_close(value, diagnosed.optimizer.state_dict()["state"][pid][key], rtol=0, atol=0)
        torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)
        assert ordinary.sampling_rng.snapshot() == diagnosed.sampling_rng.snapshot()


def weighted_opponent_case():
    action = np.zeros((4, 4), np.float32)
    action[:2] = [[.6, .2, .2, 0], [.1, .2, .7, 0]]
    buy = np.zeros((4, CARD_CLASSES), np.float32)
    reserve = np.zeros_like(buy)
    buy[:2, 0], buy[:2, UNKNOWN_SLOT] = [.2, .1], [.4, 0]
    reserve[:2, 0], reserve[:2, UNKNOWN_SLOT] = [.15, .35], [.05, .35]
    labels = {"opponent_action_target": action, "opponent_buy_target": buy,
        "opponent_reserve_target": reserve, "opponent_policy_weight": np.array([.25, 1, 0, 0], np.float32)}
    predictions = {"opponent_action": torch.zeros(4, 4), "opponent_buy": torch.zeros(4, CARD_CLASSES),
                   "opponent_reserve": torch.zeros(4, CARD_CLASSES)}
    predictions["opponent_buy"][1, 0] = 1
    predictions["opponent_reserve"][1, UNKNOWN_SLOT] = 2
    return predictions, labels, torch.tensor([2., 1., 3., 4.])


def test_label_coverage_unknown_mass_and_conditional_ce_use_distinct_correct_denominators():
    predictions, labels, weight = weighted_opponent_case()
    sums = opponent_loss_sums(predictions, labels, weight)
    result = opponent_metrics(sums)
    assert result["opponent_eligible_positions"] == 10
    assert result["opponent_label_positions"] == 3
    assert result["opponent_label_coverage"] == .3
    assert result["opponent_positions"] == 1.5
    buy_ce = .3*math.log(CARD_CLASSES) + .1*(math.log(math.e + CARD_CLASSES-1) - 1)
    reserve_ce = .1*math.log(CARD_CLASSES) + .7*math.log(math.exp(2) + CARD_CLASSES-1) - .35*2
    assert result["opponent_buy_ce"] == pytest.approx(buy_ce/1.5)
    assert result["opponent_buy_conditional_ce"] == pytest.approx(buy_ce/.4)
    assert result["opponent_reserve_conditional_ce"] == pytest.approx(reserve_ce/.8)
    assert result["opponent_buy_unknown_ratio"] == pytest.approx(.2/.4)
    assert result["opponent_reserve_unknown_ratio"] == pytest.approx(.375/.8)
    assert result["opponent_unknown_ratio"] == pytest.approx(.575/1.2)
    # Full-population ratios aggregate sums, independent of validation batch boundaries.
    chunks = [opponent_loss_sums({k:v[part] for k,v in predictions.items()},
        {k:v[part] for k,v in labels.items()}, weight[part]) for part in (slice(0, 1), slice(1, 4))]
    combined = opponent_metrics({key: chunks[0][key] + chunks[1][key] for key in sums})
    for key, value in result.items():
        assert combined[key] == pytest.approx(value)


def test_zero_label_or_action_mass_is_explicitly_undefined_for_ratios_and_conditional_ce():
    predictions, labels, weight = weighted_opponent_case()
    labels["opponent_policy_weight"][:] = 0
    result = opponent_metrics(opponent_loss_sums(predictions, labels, weight))
    assert result["opponent_label_coverage"] == 0
    assert result["opponent_buy_mass"] == 0 and result["opponent_buy_ce"] == 0
    for key in ("opponent_buy_conditional_ce", "opponent_reserve_conditional_ce",
                "opponent_unknown_ratio", "opponent_buy_unknown_ratio", "opponent_reserve_unknown_ratio"):
        assert result[key] is None
    result = opponent_metrics(opponent_loss_sums(predictions, labels, weight*0))
    assert result["opponent_label_coverage"] is None


def test_zero_auxiliary_mass_has_zero_encoder_gradient_in_every_auxiliary_task():
    base = build_actor_critic({"architecture": "entity_attention_v1", "hidden": 16, "heads": 2, "layers": 1, "ffn_hidden": 32})
    learner = AlphazeroLearner(add_continuation_heads(base, predict_opponent_policy=True),
        LearnerConfig(device="cpu"), sampling_rng=DeterministicRng(7))
    batch, labels = diagnostic_batch()
    labels["shortterm_value_mask"][:] = False
    labels["opponent_policy_weight"][:] = 0
    record = learner.train_step(batch, references=[], value_batch=batch,
        continuation_auxiliary=ContinuationAuxiliaryConfig(opponent_policy_weight=.15),
        continuation_labels=labels, record_encoder_grad_norms=True)
    for key, value in record.items():
        if key.startswith(("encoder_grad_norm_shortterm", "encoder_grad_norm_opponent")):
            assert value == 0
    assert record["encoder_grad_norm_total"] > 0
