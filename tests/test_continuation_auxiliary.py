from dataclasses import asdict, replace
import json

import numpy as np
import pytest
import torch

from splendor.actor_critic import TorchStateBatch, build_actor_critic, tensorize_batch
from splendor.alphazero_checkpoint import load_alphazero_checkpoint
from splendor.alphazero_recording import read_json, sha256
from splendor.alphazero_train import LearnerConfig
from splendor.continuation_auxiliary import (ContinuationAuxiliaryConfig, HORIZONS, LAMBDAS,
    add_continuation_heads, continuation_loss, continuation_loss_sums, game_shortterm_value_targets)
from splendor.engine import GameEngine
from splendor.features import GLOBAL_FEATURE_NAMES as G, collate_encoded_decisions, encode_decision
from splendor.offline_continuation import ContinuationOfflineConfig, run, select_checkpoint


@pytest.fixture(autouse=True, scope="module")
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def timeline():
    # Two main roots and two payment subdecisions; world visits deliberately unequal.
    g = np.zeros((4, len(G)), dtype=np.float32)
    g[[0, 2], G.index("phase_main_action")] = 1
    g[[1, 3], G.index("phase_payment")] = 1
    visits = np.array([[1, 0, 0, 0, 1, 1, 0, 0], [3, 0, 0, 0, 2, 0, 0, 0], [0]*8])
    values = visits * np.array([[.2]*4 + [.4]*4, [.6]*4 + [.8]*4, [0]*8])
    return {"global_features": g, "player_id": np.array([0, 0, 1, 1]),
        "completed_player_turns": np.array([0, 0, 1, 1]), "candidate_offsets": np.arange(5)*2,
        "per_world_root_visit_counts": visits, "per_world_root_value_sums": values,
        "valid_world_count": np.full(4, 2), "value_target": np.array([1, 1, -1, -1])}, {
        "status": "natural_terminal", "terminal_result": {"completed_turns": [1, 1]}}


def test_katago_ema_current_root_equal_worlds_perspective_and_terminal_tail():
    arrays, meta = timeline()
    result = game_shortterm_value_targets(arrays, meta)
    np.testing.assert_allclose(LAMBDAS, [2/3, 6/7, 14/15])
    # Root0=(.2+.6)/2=.4; root1=(.4+.8)/2=.6 from p1's perspective.
    for j, lam in enumerate(LAMBDAS):
        assert result["shortterm_value_target"][0, j] == pytest.approx((1-lam)*.4 + lam*(1-lam)*(-.6) + lam**2)
        assert result["shortterm_value_target"][2, j] == pytest.approx((1-lam)*.6 - lam)
    assert result["shortterm_value_mask"].tolist() == [True, False, True, False]
    # Flipping both seat IDs and payoff retains every actor-relative label.
    arrays["player_id"] = 1 - arrays["player_id"]
    np.testing.assert_array_equal(game_shortterm_value_targets(arrays, meta)["shortterm_value_target"],
                                  result["shortterm_value_target"])


def test_missing_roots_gaps_old_data_and_draws_are_explicit():
    arrays, meta = timeline()
    arrays["per_world_root_visit_counts"][:, :2] = 0
    labels = game_shortterm_value_targets(arrays, meta)
    assert labels["shortterm_value_mask"].tolist() == [False, False, True, False]
    arrays, meta = timeline()
    arrays["completed_player_turns"][2:] = 2
    meta["terminal_result"]["completed_turns"] = [2, 1]
    assert game_shortterm_value_targets(arrays, meta)["shortterm_value_mask"].tolist() == [False, False, True, False]
    meta["terminal_result"]["completed_turns"] = [2, 2]
    assert not game_shortterm_value_targets(arrays, meta)["shortterm_value_mask"].any()
    arrays.pop("per_world_root_value_sums")
    with pytest.raises(ValueError, match="requires v3"):
        game_shortterm_value_targets(arrays, meta)
    arrays, meta = timeline()
    arrays["value_target"][:] = 0
    lam = LAMBDAS
    np.testing.assert_allclose(game_shortterm_value_targets(arrays, meta)["shortterm_value_target"][2], (1-lam)*.6)


@pytest.mark.parametrize("opponent", [False, True])
def test_heads_preserve_inference_strict_reload_and_auxiliary_gradients(opponent):
    base = build_actor_critic({"architecture": "entity_attention_v1", "hidden": 16, "heads": 2,
        "layers": 1, "ffn_hidden": 32, "market_relations": True, "value_readout": "global_players_market"})
    model = add_continuation_heads(base, predict_opponent_policy=opponent)
    batch = tensorize_batch(collate_encoded_decisions([encode_decision(GameEngine.new(42).decision())]))
    for expected, actual in zip(base(batch), model(batch)):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    state = TorchStateBatch(**{k: getattr(batch, k) for k in TorchStateBatch.__dataclass_fields__})
    torch.testing.assert_close(model.value_from_state(state), model(batch)[1])
    _, predictions = model.value_with_continuation(batch)
    cfg = ContinuationAuxiliaryConfig()
    labels = {"shortterm_value_target": np.ones((1, 3), np.float32), "shortterm_value_mask": np.ones(1, bool),
              "shortterm_main_mask": np.ones(1, bool)}
    continuation_loss(continuation_loss_sums(predictions, labels, torch.ones(1), cfg), cfg).backward()
    assert model.context_encoder.weight.grad.abs().sum() > 0
    assert model.shortterm_value_head.network[1].weight.grad.abs().sum() > 0
    assert model.critic.network[1].weight.grad is None and model.actor[1].weight.grad is None
    restored = build_actor_critic(asdict(model.config))
    restored.load_state_dict(model.state_dict(), strict=True)
    for expected, actual in zip(model(batch), restored(batch)):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    prediction = torch.zeros(2, 3, requires_grad=True)
    sums = continuation_loss_sums({"shortterm_value": prediction}, {
        "shortterm_value_target": np.ones((2, 3)), "shortterm_value_mask": np.zeros(2, bool),
        "shortterm_main_mask": np.ones(2, bool)}, torch.ones(2), cfg)
    loss = continuation_loss(sums, cfg)
    loss.backward()
    assert loss == 0 and prediction.grad.count_nonzero() == 0


@pytest.fixture(params=[0., .15], ids=["shortterm", "shortterm_opponent"])
def continuation_pool(tmp_path, monkeypatch, request):
    from test_random_budget_iteration import archive
    import splendor.offline_objectives as entry
    monkeypatch.setattr(entry, "source_identity", lambda *_: {"test": True})
    rows = [archive(tmp_path / f"round-{r}", r, g, mixed=True) for r in range(2) for g in range(2)]
    for row in rows:
        path = row["path"]
        with np.load(path) as raw:
            arrays = dict(raw)
        meta = json.loads(arrays["metadata_json"].tobytes())
        meta.update(status="natural_terminal", terminal_result={"winner_ids": [0], "completed_turns": [2, 2]})
        width = arrays["candidate_offsets"][1]
        visits = np.zeros((2, width*4), dtype=np.int32)
        visits[0, ::width] = 1
        visits[1, ::width] = 3
        arrays.update(player_id=np.array([0, 1, 0, 1]), value_target=np.array([1, -1, 1, -1], np.int8),
            policy_weight=np.array([0, 0, 1, 0], np.float32), low_budget=np.array([1, 1, 0, 1], bool),
            valid_world_count=np.full(4, 2), per_world_root_visit_counts=visits,
            per_world_root_value_sums=visits.astype(np.float64)*.3,
            metadata_json=np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8))
        np.savez_compressed(path, **arrays)
        from pathlib import Path
        stat = Path(path).stat()
        row.update(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    manifest = tmp_path / "pool.jsonl"
    manifest.write_text("".join(json.dumps(row)+"\n" for row in rows))
    return ContinuationOfflineConfig(manifest=str(manifest), manifest_sha256=sha256(manifest),
        first_round=1, last_round=2, expected_games_per_round=2, draw_exposures=2.5,
        cache_mib=8, training_workers=0, validation_positions=8, validation_percent_step=25,
        encoder_grad_norm_every_steps=2,
        continuation_auxiliary={"shortterm_value_weight": .25, "opponent_policy_weight": request.param},
        model_config={"architecture": "entity_attention_v1", "hidden": 16, "heads": 2,
            "layers": 1, "ffn_hidden": 32, "market_relations": True, "value_readout": "global_players_market"},
        learner=LearnerConfig(device="cpu", batch_size=4, history_batch_size=2,
            learning_rate=1e-4, validation_batch_size=8, checkpoint_every_steps=2))


def test_offline_training_exact_resume_validation_and_selection(continuation_pool, tmp_path):
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    run(continuation_pool, full)
    run(continuation_pool, resumed, max_steps_this_session=1)
    run(continuation_pool, resumed)
    a = load_alphazero_checkpoint(full / "checkpoints/final.pt")
    b = load_alphazero_checkpoint(resumed / "checkpoints/final.pt")
    assert a.total_train_step == b.total_train_step == 5
    assert a.sampling_rng_state == b.sampling_rng_state
    for name, value in a.model_state_dict.items():
        torch.testing.assert_close(value, b.model_state_dict[name], rtol=0, atol=0)
    for pid, state in a.optimizer_state_dict["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(value, b.optimizer_state_dict["state"][pid][name], rtol=0, atol=0)
    metrics = read_json(resumed / "full_validation/step-000000005.json")["metrics"]
    assert metrics["shortterm_positions"] == 8
    assert all(f"shortterm_h{h}_mse" in metrics for h in HORIZONS)
    if continuation_pool.continuation_auxiliary["opponent_policy_weight"]:
        assert metrics["opponent_positions"] == 2
        assert metrics["opponent_action_ce"] > 0 and metrics["opponent_buy_mass"] == 0
        assert metrics["opponent_label_coverage"] == .25
        assert metrics["opponent_buy_conditional_ce"] is None
        assert metrics["opponent_reserve_unknown_ratio"] is None
    for root in (full, resumed):
        rows = [json.loads(line) for line in (root / "training_metrics.jsonl").read_text().splitlines()]
        recorded = [row["total_train_step"] for row in rows if "encoder_grad_norm_total" in row]
        assert recorded == [1, 2, 4]
    selected = select_checkpoint(continuation_pool, resumed)
    assert selected["selected_step"] in [2, 3, 4, 5]
    assert selected["selection_objective"] == pytest.approx(min(
        row["metrics"]["policy_kl"] + row["metrics"]["value_loss"] for row in selected["candidates"]))
    with pytest.raises(ValueError, match="identical frozen"):
        run(replace(continuation_pool, continuation_auxiliary={"shortterm_value_weight": .3}), resumed)


def opponent_timeline():
    from splendor.features import ACTION_FEATURE_NAMES as A, CARD_FEATURE_NAMES as C
    from splendor.selfplay_data import STATE_FIELDS
    encoded = encode_decision(GameEngine.new(42).decision())
    arrays = {name: np.stack([getattr(encoded, name)]*4) for name in STATE_FIELDS}
    arrays["global_features"][:, G.index("phase_main_action")] = [1, 0, 1, 0]
    arrays["global_features"][:, G.index("phase_payment")] = [0, 1, 0, 1]
    arrays["pending_card"][:] = arrays["market_t1"][:, 0]
    arrays["pending_card_mask"][:] = True  # duplicate pending/market identity
    # The future player sees a private reserved card; current observer sees only an unknown slot.
    arrays["opponent_reserved_mask"][:, 0] = True
    arrays["opponent_reserved"][:, 0, :2] = [1, 0]
    candidates = np.zeros((8, len(A)), np.float32)
    candidates[:, A.index("type_take_pair")] = 1
    card_start = A.index("target_card.valid")
    for row, kind, card in ((2, "buy_reserved", arrays["market_t1"][0, 0]),
                            (3, "buy_market", arrays["market_t1"][0, 2]),
                            (4, "reserve_market", arrays["market_t1"][0, 1]),
                            (5, "reserve_deck", np.zeros(len(C)))):
        candidates[row, A.index("type_take_pair")] = 0
        candidates[row, A.index("type_" + kind)] = 1
        candidates[row, card_start:card_start+len(C)] = card
    candidates[3, card_start + C.index("points_scaled")] = .987  # unseen future refill
    # Same intrinsic card but changed source/actor-relative deficits must still match.
    candidates[2, card_start + C.index("source_market")] = 0
    candidates[2, card_start + C.index("source_deck")] = 1
    candidates[2, card_start + C.index("ordinary_deficit_white_scaled")] = .987
    arrays.update(player_id=np.array([0, 0, 1, 1]), candidate_offsets=np.array([0, 1, 2, 7, 8]),
        candidate_features=candidates, policy_target=np.array([1, 1, .2, .1, .3, .1, .3, 1], np.float32),
        policy_weight=np.ones(4, np.float32))
    return arrays, {"status": "natural_terminal"}


def test_opponent_next_main_soft_aggregation_visibility_other_and_terminal_masks():
    from splendor.opponent_policy import game_opponent_policy_targets, UNKNOWN_SLOT, visible_card_arrays
    arrays, meta = opponent_timeline()
    original = {k: v.copy() for k, v in arrays.items()}
    labels = game_opponent_policy_targets(arrays, meta)
    np.testing.assert_array_equal(labels["opponent_policy_weight"], [1, 1, 0, 0])
    np.testing.assert_allclose(labels["opponent_action_target"][:2], [[.3, .3, .4, 0]]*2)
    np.testing.assert_allclose(labels["opponent_buy_target"][0, [0, UNKNOWN_SLOT]], [.2, .1])
    np.testing.assert_allclose(labels["opponent_reserve_target"][0, [1, UNKNOWN_SLOT]], [.3, .1])
    np.testing.assert_allclose(labels["opponent_buy_target"].sum(1), labels["opponent_action_target"][:, 0])
    np.testing.assert_allclose(labels["opponent_reserve_target"].sum(1), labels["opponent_action_target"][:, 2])
    assert not visible_card_arrays(arrays)[1][:, 18].any()  # pending duplicates excluded
    for key, value in original.items():
        np.testing.assert_array_equal(arrays[key], value)
    arrays["policy_weight"][2] = 0
    assert not game_opponent_policy_targets(arrays, meta)["opponent_policy_weight"].any()


def test_opponent_hierarchical_ce_weighting_independent_card_softmax_and_mask_gradients():
    from splendor.opponent_policy import CARD_CLASSES, game_opponent_policy_targets, opponent_loss_sums
    arrays, meta = opponent_timeline()
    labels = game_opponent_policy_targets(arrays, meta)
    predictions = {"opponent_action": torch.zeros(4, 4, requires_grad=True),
        "opponent_buy": torch.zeros(4, CARD_CLASSES, requires_grad=True),
        "opponent_reserve": torch.zeros(4, CARD_CLASSES, requires_grad=True)}
    sums = opponent_loss_sums(predictions, labels, torch.ones(4))
    assert sums["opponent_action_ce_sum"].detach().item() == pytest.approx(2*np.log(4))
    assert sums["opponent_buy_ce_sum"].detach().item() == pytest.approx(2*.3*np.log(CARD_CLASSES))
    assert sums["opponent_reserve_ce_sum"].detach().item() == pytest.approx(2*.4*np.log(CARD_CLASSES))
    sum(sums[f"opponent_{name}_ce_sum"] for name in ("action", "buy", "reserve")).backward()
    for prediction in predictions.values():
        assert prediction.grad[:2].abs().sum() > 0 and prediction.grad[2:].abs().sum() == 0


def test_opponent_heads_mask_unknown_padding_duplicates_and_backpropagate_shared_encoder():
    from splendor.opponent_policy import CARD_CLASSES, UNKNOWN_SLOT, game_opponent_policy_targets, opponent_loss_sums
    from splendor.selfplay_data import SelfPlaySample, collate_selfplay_samples
    arrays, meta = opponent_timeline()
    arrays.update(value_target=np.zeros(4, np.int8), value_weight=np.ones(4, np.float32),
                  behavior_policy=arrays["policy_target"], decision_index=np.arange(4))
    meta["game_id"] = 0
    samples = [SelfPlaySample(arrays, meta, i) for i in range(4)]
    batch = collate_selfplay_samples(samples)
    base = build_actor_critic({"architecture": "entity_attention_v1", "hidden": 16, "heads": 2, "layers": 1, "ffn_hidden": 32})
    model = add_continuation_heads(base, predict_opponent_policy=True)
    _, _, predictions = model.forward_with_continuation(batch.decisions)
    for name in ("buy", "reserve"):
        logits = predictions["opponent_" + name]
        assert logits.shape == (4, CARD_CLASSES)
        assert (logits[:, 18] < -1e8).all() and (logits[:, 15] < -1e8).all()
        assert (logits[:, UNKNOWN_SLOT].abs() < 1e8).all()
    labels = game_opponent_policy_targets(arrays, meta)
    sums = opponent_loss_sums(predictions, labels, torch.ones(4))
    sum(sums[f"opponent_{name}_ce_sum"] for name in ("action", "buy", "reserve")).backward()
    assert model.opponent_policy_head.card[1].weight.grad.abs().sum() > 0
    assert model.opponent_policy_head.action.network[1].weight.grad.abs().sum() > 0
    assert model.context_encoder.weight.grad.abs().sum() > 0
    assert model.actor[1].weight.grad is None and model.critic.network[1].weight.grad is None


def test_cache_accounts_for_labels_without_mutating_source_and_catalog_identity_is_unique(continuation_pool, tmp_path):
    from splendor.catalog import load_default_catalog
    from splendor.compact_game_cache import compact
    from splendor.continuation_auxiliary import game_continuation_targets
    from splendor.offline_continuation import prepare_inputs
    from splendor.selfplay_data import load_game
    catalog = load_default_catalog()
    assert len({(c.tier, c.bonus, c.points, c.cost) for c in catalog.cards}) == len(catalog.cards)
    index, *_ = prepare_inputs(continuation_pool, tmp_path / "prepare")
    game = index.games[0]
    before = sha256(game.path)
    arrays, meta = load_game(game.path)
    cfg = ContinuationAuxiliaryConfig(**continuation_pool.continuation_auxiliary)
    expected = game_continuation_targets(arrays, meta, cfg)
    expected_bytes = compact(arrays, meta)[2] + sum(a.nbytes for a in expected.values())
    sample = index.cache.sample(game, 0)
    for key, value in expected.items():
        np.testing.assert_array_equal(sample.arrays[key], value)
    assert index.cache._bytes == expected_bytes
    assert index.cache.sample(game, 1).arrays is sample.arrays
    assert index.cache.stats()["hits"] == 1
    assert sha256(game.path) == before


def test_resume_allows_diagnostic_cadence_changes_and_initial_config_without_new_field(continuation_pool, tmp_path):
    from splendor.alphazero_recording import write_json
    from splendor.offline_continuation import prepare_inputs
    output = tmp_path / "logging_change"
    initial = replace(continuation_pool, encoder_grad_norm_every_steps=0)
    run(initial, output, max_steps_this_session=1)
    identity_before = read_json(output / "plan.json")["replay_identity"]
    # Initial releases had no diagnostic-cadence key in the frozen config.
    frozen = read_json(output / "config.json")
    frozen.pop("encoder_grad_norm_every_steps")
    write_json(output / "config.json", frozen)
    run(continuation_pool, output, max_steps_this_session=1)
    assert read_json(output / "plan.json")["replay_identity"] == identity_before
    rows = [json.loads(line) for line in (output / "training_metrics.jsonl").read_text().splitlines()]
    assert [row["total_train_step"] for row in rows if "encoder_grad_norm_total" in row] == [2]
    assert prepare_inputs(replace(continuation_pool, encoder_grad_norm_every_steps=1), output)[2] == identity_before
    explicitly_disabled = replace(continuation_pool, continuation_auxiliary=dict(
        continuation_pool.continuation_auxiliary, score_ownership_weight=0.))
    assert prepare_inputs(explicitly_disabled, output)[2] == identity_before
