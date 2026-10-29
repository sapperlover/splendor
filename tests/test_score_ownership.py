from dataclasses import asdict, replace
from collections import Counter
import json

import numpy as np
import pytest
import torch

from splendor.actor_critic import build_actor_critic
from splendor.alphazero_checkpoint import load_alphazero_checkpoint
from splendor.alphazero_recording import read_json, sha256
from splendor.alphazero_train import LearnerConfig
from splendor.continuation_auxiliary import (ContinuationAuxiliaryConfig, PROTOCOL, add_continuation_heads,
    continuation_config_identity, continuation_gradient_tasks, continuation_loss, continuation_loss_sums, game_continuation_targets)
from splendor.engine import GameEngine
from splendor.features import CARD_FEATURE_NAMES as C, GLOBAL_FEATURE_NAMES as G, SCALES, encode_decision
from splendor.offline_continuation import ContinuationOfflineConfig, run, select_checkpoint
from splendor.score_ownership import (OUTPUT_NAMES, game_score_ownership_targets, score_ownership_metrics,
    score_ownership_diagnostic_sums, score_ownership_loss_sums, score_ownership_loss)
from splendor.selfplay_data import STATE_FIELDS, SelfPlaySample, collate_selfplay_samples


@pytest.fixture(autouse=True, scope="module")
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def score_timeline():
    # Includes an already earned score, a zero-point blind-reserved purchase,
    # both nobles, all card tiers, and a final payment with no later snapshot.
    actors = np.array([0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0], np.int8)
    phases = ["main_action", "payment", "choose_noble", "main_action", "payment", "main_action",
              "payment", "main_action", "payment", "choose_noble", "main_action", "payment"]
    gains = [0, 1, 3, 0, 0, 0, 2, 0, 3, 3, 0, 5]
    g = np.zeros((len(actors), len(G)), np.float32)
    pending = np.zeros((len(actors), len(C)), np.float32)
    mask = np.zeros(len(actors), bool)
    scores = [4, 2]
    nobles = [0, 0]
    for i, (p, phase, gain) in enumerate(zip(actors, phases, gains)):
        g[i, G.index("phase_" + phase)] = 1
        g[i, G.index("self_score_scaled")] = scores[p] / SCALES.player_score
        g[i, G.index("opponent_score_scaled")] = scores[1-p] / SCALES.player_score
        g[i, G.index("self_nobles_scaled")] = nobles[p] / SCALES.noble_count
        g[i, G.index("opponent_nobles_scaled")] = nobles[1-p] / SCALES.noble_count
        if phase == "payment":
            tier = {1: 1, 4: 1, 6: 2, 8: 2, 11: 3}[i]
            pending[i, :2] = 1
            pending[i, C.index(f"tier_{tier}")] = 1
            pending[i, C.index("points_scaled")] = gain / SCALES.card_points
            mask[i] = True
        scores[p] += gain
        if phase == "choose_noble":
            nobles[p] += 1
    return {"global_features": g, "player_id": actors, "pending_card": pending, "pending_card_mask": mask}, {
        "status": "natural_terminal", "terminal_result": {"scores": scores, "completed_turns": [3, 2], "winner_ids": [0]}}


def test_exact_source_timing_relative_player_increment_only_and_final_payment():
    arrays, meta = score_timeline()
    y = game_score_ownership_targets(arrays, meta)["score_ownership_target"]
    assert y.shape == (12, 8) and y.dtype == np.int16
    np.testing.assert_array_equal(y[0], [1, 2, 5, 3, 0, 3, 0, 3])
    np.testing.assert_array_equal(y[1], y[0])  # purchase is credited only by Pay
    np.testing.assert_array_equal(y[2], [0, 2, 5, 3, 0, 3, 0, 3])
    np.testing.assert_array_equal(y[3], [0, 3, 0, 3, 0, 2, 5, 0])
    np.testing.assert_array_equal(y[8], [0, 3, 0, 3, 0, 0, 5, 0])
    np.testing.assert_array_equal(y[9], [0, 0, 0, 3, 0, 0, 5, 0])
    np.testing.assert_array_equal(y[-1], [0, 0, 5, 0, 0, 0, 0, 0])
    own = np.rint(arrays["global_features"][:, G.index("self_score_scaled")]*15)
    opp = np.rint(arrays["global_features"][:, G.index("opponent_score_scaled")]*15)
    final = np.asarray(meta["terminal_result"]["scores"])
    np.testing.assert_array_equal(y[:, :4].sum(1), final[arrays["player_id"]] - own)
    np.testing.assert_array_equal(y[:, 4:].sum(1), final[1-arrays["player_id"]] - opp)
    arrays["player_id"] = 1-arrays["player_id"]
    meta["terminal_result"]["scores"] = final[::-1].tolist()
    np.testing.assert_array_equal(game_score_ownership_targets(arrays, meta)["score_ownership_target"], y)


def test_final_noble_and_forced_skip_terminal_without_search_or_trace():
    arrays, meta = score_timeline()
    short = {key: value[:10] for key, value in arrays.items()}
    meta["terminal_result"]["scores"] = [10, 8]
    y = game_score_ownership_targets(short, meta)["score_ownership_target"]
    np.testing.assert_array_equal(y[-1], [0, 0, 0, 3, 0, 0, 0, 0])
    # After this noble has been resolved, an additional main-action snapshot
    # followed only by automatic skips earns nothing further, even in a draw.
    after = {key: np.concatenate((value, value[-1:]), axis=0) for key, value in short.items()}
    after["global_features"][-1] = 0
    after["global_features"][-1, G.index("phase_main_action")] = 1
    after["global_features"][-1, G.index("self_score_scaled")] = 8/15
    after["global_features"][-1, G.index("opponent_score_scaled")] = 10/15
    after["global_features"][-1, G.index("self_nobles_scaled")] = 1/3
    after["global_features"][-1, G.index("opponent_nobles_scaled")] = 1/3
    after["pending_card_mask"][-1] = False
    meta["terminal_result"].update(reason="consecutive_forced_skips", winner_ids=[0, 1], completed_turns=[7, 7])
    np.testing.assert_array_equal(game_score_ownership_targets(after, meta)["score_ownership_target"][-1], np.zeros(8))


@pytest.mark.parametrize("phase", ["payment", "main_action", "return_tokens"])
def test_automatic_single_noble_is_split_from_card_points_even_at_terminal(phase):
    arrays, meta = score_timeline()
    # A card and an automatic noble are resolved together by the last Pay.
    meta["terminal_result"]["scores"][0] += 3
    y = game_score_ownership_targets(arrays, meta)["score_ownership_target"]
    np.testing.assert_array_equal(y[-1], [0, 0, 5, 3, 0, 0, 0, 0])
    # Automatic noble acquisition on a take/reserve/return needs no Claim row.
    if phase != "payment":
        arrays["global_features"][-1, G.index("phase_payment")] = 0
        arrays["global_features"][-1, G.index("phase_"+phase)] = 1
        meta["terminal_result"]["scores"][0] -= 5
        y = game_score_ownership_targets(arrays, meta)["score_ownership_target"]
        np.testing.assert_array_equal(y[-1], [0, 0, 0, 3, 0, 0, 0, 0])


@pytest.mark.parametrize("removed", [(2,), (9,), (2, 9)])
def test_folded_automatic_nobles_between_snapshots_keep_the_same_targets(removed):
    arrays, meta = score_timeline()
    expected = game_score_ownership_targets(arrays, meta)["score_ownership_target"]
    folded = {key:np.delete(value, removed, axis=0) for key,value in arrays.items()}
    np.testing.assert_array_equal(game_score_ownership_targets(folded, meta)["score_ownership_target"],
                                  np.delete(expected, removed, axis=0))


@pytest.mark.parametrize("bad", ["negative", "phase", "unknown", "tier", "points", "count", "not_terminal"])
def test_inconsistent_archives_fail_instead_of_fabricating_targets(bad):
    arrays, meta = score_timeline()
    if bad == "negative":
        meta["terminal_result"]["scores"][0] = 0
    elif bad == "phase":
        arrays["global_features"][1, G.index("phase_payment")] = 0
    elif bad == "unknown":
        arrays["pending_card"][1, C.index("known")] = 0
    elif bad == "tier":
        arrays["pending_card"][1, C.index("tier_1")] = 0
    elif bad == "points":
        arrays["pending_card"][1, C.index("points_scaled")] = 0
    elif bad == "count":
        arrays["global_features"][3, G.index("opponent_nobles_scaled")] = 0
    else:
        meta["status"] = "engineering_truncation"
    with pytest.raises(ValueError):
        game_score_ownership_targets(arrays, meta)


def model_and_batch():
    arrays, meta = score_timeline()
    encoded = encode_decision(GameEngine.new(42).decision())
    n = len(arrays["player_id"])
    all_arrays = {name: np.stack([getattr(encoded, name)]*n) for name in STATE_FIELDS} | arrays
    width = len(encoded.candidate_features)
    all_arrays.update(candidate_features=np.tile(encoded.candidate_features, (n, 1)),
        candidate_offsets=np.arange(n+1)*width, policy_target=np.full(n*width, 1/width, np.float32),
        behavior_policy=np.full(n*width, 1/width, np.float32), root_network_prior=np.full(n*width, 1/width, np.float32),
        value_target=(1-2*arrays["player_id"]).astype(np.int8), policy_weight=np.ones(n, np.float32),
        value_weight=np.ones(n, np.float32), decision_index=np.arange(n, dtype=np.int32),
        completed_player_turns=np.array([0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4], np.int32))
    meta.update(game_id=0, run_id="ownership-test", round_index=0, split="train")
    return all_arrays, meta


def test_eight_output_loss_scaling_metrics_gradients_inference_and_strict_loading():
    arrays, meta = model_and_batch()
    labels = game_score_ownership_targets(arrays, meta)
    batch = collate_selfplay_samples([SelfPlaySample(arrays, meta, i) for i in range(12)])
    base = build_actor_critic({"architecture": "entity_attention_v1", "hidden": 16, "heads": 2,
        "layers": 1, "ffn_hidden": 32, "market_relations": True, "value_readout": "global_players_market"})
    model = add_continuation_heads(base, predict_shortterm_value=False, predict_score_ownership=True)
    assert not hasattr(model, "shortterm_value_head") and not hasattr(model, "opponent_policy_head")
    for a, b in zip(base(batch.decisions), model(batch.decisions)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    _, _, pred = model.forward_with_continuation(batch.decisions)
    cfg = ContinuationAuxiliaryConfig(shortterm_value_weight=0, score_ownership_weight=.25)
    assert pred["score_ownership"].shape == (12, 8)
    sums = continuation_loss_sums(pred, labels, torch.ones(12), cfg)
    continuation_loss(sums, cfg).backward()
    assert model.context_encoder.weight.grad.abs().sum() > 0
    assert model.score_ownership_head.network[1].weight.grad.abs().sum() > 0
    assert model.critic.network[1].weight.grad is None and model.actor[1].weight.grad is None
    assert len(continuation_gradient_tasks(sums, cfg)) == 9
    restored = build_actor_critic(asdict(model.config))
    restored.load_state_dict(model.state_dict(), strict=True)
    for a, b in zip(model(batch.decisions), restored(batch.decisions)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    zeros = torch.zeros(2, 8, requires_grad=True)
    sums = continuation_loss_sums({"score_ownership": zeros}, {"score_ownership_target": np.full((2, 8), 3, np.int16)},
                                   torch.tensor([1., 0.]), cfg)
    assert continuation_loss(sums, cfg).detach().item() == pytest.approx(.25*(3/15)**2)
    stats = score_ownership_metrics({key:value.detach() for key,value in sums.items()})
    assert stats["score_ownership_mse_points"] == pytest.approx(9)
    assert stats["score_ownership_mae_points"] == pytest.approx(3)
    continuation_loss(sums, cfg).backward()
    assert zeros.grad[0].abs().sum() > 0 and zeros.grad[1].abs().sum() == 0


def test_legacy_identity_and_three_way_model_composition():
    cfg = ContinuationAuxiliaryConfig()
    assert continuation_config_identity(cfg) == {"shortterm_value_weight": .25, "opponent_policy_weight": 0., "protocol": PROTOCOL}
    for bad in (-1, True, float("nan")):
        with pytest.raises(ValueError):
            ContinuationAuxiliaryConfig(score_ownership_weight=bad)
    base = build_actor_critic({"architecture": "entity_attention_v1", "hidden": 16, "heads": 2, "layers": 1, "ffn_hidden": 32})
    torch.manual_seed(26)
    both = add_continuation_heads(base, predict_opponent_policy=True)
    # Old model configs omit the new false flag and still load strictly.
    legacy = asdict(both.config)
    legacy.pop("predict_score_ownership")
    build_actor_critic(legacy).load_state_dict(both.state_dict(), strict=True)
    torch.manual_seed(26)
    all_heads = add_continuation_heads(base, predict_opponent_policy=True, predict_score_ownership=True)
    for name, value in both.state_dict().items():
        torch.testing.assert_close(value, all_heads.state_dict()[name], rtol=0, atol=0)
    arrays, meta = model_and_batch()
    batch = collate_selfplay_samples([SelfPlaySample(arrays, meta, 0)])
    _, _, predictions = all_heads.forward_with_continuation(batch.decisions)
    assert set(predictions) == {"shortterm_value", "opponent_action", "opponent_buy", "opponent_reserve", "score_ownership"}
    width = int(arrays["candidate_offsets"][1])
    arrays.update(valid_world_count=np.ones(12, np.int8),
        per_world_root_visit_counts=np.ones((1, 12*width), np.int32),
        per_world_root_value_sums=np.full((1, 12*width), .25))
    cfg = ContinuationAuxiliaryConfig(opponent_policy_weight=.15, score_ownership_weight=.25)
    labels = {key:value[:1] for key,value in game_continuation_targets(arrays, meta, cfg).items()}
    continuation_loss(continuation_loss_sums(predictions, labels, torch.ones(1), cfg), cfg).backward()
    assert all_heads.shortterm_value_head.network[1].weight.grad.abs().sum() > 0
    assert all_heads.opponent_policy_head.action.network[1].weight.grad.abs().sum() > 0
    assert all_heads.score_ownership_head.network[1].weight.grad.abs().sum() > 0


def test_four_worker_offline_updates_resume_state_transport_and_selection(tmp_path, monkeypatch):
    import splendor.offline_objectives as entry
    import splendor.score_ownership as ownership
    monkeypatch.setattr(entry, "source_identity", lambda *_: {"test": True})
    arrays, meta = model_and_batch()
    manifest_rows = []
    for round_index in range(2):
        for game_id in range(2):
            path = tmp_path/f"r{round_index}-g{game_id}.npz"
            metadata = meta | {"round_index":round_index, "game_id":game_id, "split":"train" if game_id == 0 else "validation"}
            np.savez_compressed(path, **arrays, metadata_json=np.frombuffer(json.dumps(metadata).encode(), dtype=np.uint8))
            stat = path.stat()
            manifest_rows.append(metadata | {"path":str(path), "size":stat.st_size, "mtime_ns":stat.st_mtime_ns,
                "decision_count":12, "policy_samples":12})
    manifest = tmp_path/"pool.jsonl"
    manifest.write_text("".join(json.dumps(row)+"\n" for row in manifest_rows))
    cfg = ContinuationOfflineConfig(manifest=str(manifest), manifest_sha256=sha256(manifest), first_round=1, last_round=2,
        expected_games_per_round=2, cache_mib=8, training_workers=0, draw_exposures=1, validation_positions=8,
        validation_percent_step=50, encoder_grad_norm_every_steps=2,
        continuation_auxiliary={"shortterm_value_weight":0, "score_ownership_weight":.25},
        model_config={"architecture":"entity_attention_v1", "hidden":16, "heads":2, "layers":1, "ffn_hidden":32},
        learner=LearnerConfig(device="cpu", batch_size=4, history_batch_size=2, validation_batch_size=8, checkpoint_every_steps=2))
    serial, parallel = tmp_path/"serial", tmp_path/"parallel"
    # Compare the old validation path to diagnostics plus four-worker/resume.
    with monkeypatch.context() as baseline:
        baseline.setattr(ownership, "score_ownership_diagnostic_sums", lambda *args: {})
        run(cfg, serial)
    run(replace(cfg, training_workers=4), parallel, max_steps_this_session=2)
    run(replace(cfg, training_workers=4), parallel)
    a, b = [load_alphazero_checkpoint(path/"checkpoints/final.pt") for path in (serial, parallel)]
    assert a.replay_identity == b.replay_identity and a.sampling_rng_state == b.sampling_rng_state
    torch.testing.assert_close(a.torch_cpu_rng_state, b.torch_cpu_rng_state, rtol=0, atol=0)
    for name, value in a.model_state_dict.items():
        torch.testing.assert_close(value, b.model_state_dict[name], rtol=0, atol=0)
    for pid, state in a.optimizer_state_dict["state"].items():
        for key, value in state.items():
            torch.testing.assert_close(value, b.optimizer_state_dict["state"][pid][key], rtol=0, atol=0)
    metrics = read_json(parallel/"full_validation/step-000000006.json")["metrics"]
    baseline_metrics = read_json(serial/"full_validation/step-000000006.json")["metrics"]
    for key, value in baseline_metrics.items():
        assert metrics[key] == value, key
    assert metrics["score_ownership_positions"] == 24
    assert metrics["value_weight_sum"] == 24
    assert metrics["score_ownership_phase_payment_positions"] == 10
    assert metrics["score_ownership_non_payment_positions"] == 14
    assert "score_ownership_constant_mse_points" in metrics
    fixed_curve = [json.loads(line) for line in (parallel/"validation_curve.jsonl").read_text().splitlines()]
    assert all("score_ownership_constant_mse_points" in point["metrics"] for point in fixed_curve)
    assert all(f"score_ownership_{name}_mse_points" in metrics for name in OUTPUT_NAMES)
    selected = select_checkpoint(replace(cfg, training_workers=4), parallel)
    assert selected["selection_objective"] == pytest.approx(min(
        row["metrics"]["policy_kl"] + row["metrics"]["value_loss"] for row in selected["candidates"]))
    logged = [json.loads(line) for line in (parallel/"training_metrics.jsonl").read_text().splitlines()]
    assert [row["total_train_step"] for row in logged if "encoder_grad_norm_score_ownership" in row] == [1, 2, 4, 6]
    assert logged and all("score_ownership_constant_mse_points" not in row for row in logged)
    prefetch = read_json(parallel/"prefetch_progress.json")
    assert prefetch["workers"] == 4 and 0 < prefetch["arena_bytes"] < 8*1024**2


def diagnostic_inputs():
    # Unequal weights, all decision phases, exact score-bin boundaries and a
    # zero-weight outlier make per-batch averaging visibly incorrect.
    target = np.arange(8)[None, :] + np.array([0, 2, 4, 6, 100])[:, None]
    labels = {"score_ownership_target": target.astype(np.int16)}
    weights = torch.tensor([1., 2., .5, 3., 0.])
    prediction = torch.tensor((target + np.array([1, -2, 3, -4, 500])[:, None]) / 15,
                              dtype=torch.float32, requires_grad=True)
    g = np.zeros((5, len(G)), np.float32)
    for i, phase in enumerate(("main_action", "payment", "return_tokens", "choose_noble", "payment")):
        g[i, G.index("phase_" + phase)] = 1
    g[:, G.index("self_score_scaled")] = np.array([4, 5, 10, 0, 15]) / 15
    g[:, G.index("opponent_score_scaled")] = np.array([3, 0, 0, 9, 0]) / 15
    return {"score_ownership": prediction}, labels, weights, g


def test_diagnostic_baselines_weighted_groups_and_batch_partition_independence():
    predictions, labels, weights, g = diagnostic_inputs()

    def summarize(chunks):
        total = Counter()
        for indices in chunks:
            p = {key: value[indices] for key, value in predictions.items()}
            y = {key: value[indices] for key, value in labels.items()}
            sums = score_ownership_loss_sums(p, y, weights[indices])
            sums.update(score_ownership_diagnostic_sums(p, y, weights[indices], g[indices]))
            total.update({key: float(value.detach()) for key, value in sums.items()})
        return score_ownership_metrics(total)

    full = summarize([np.arange(5)])
    split = summarize([np.arange(1), np.arange(1, 3), np.arange(3, 5)])
    assert full.keys() == split.keys()
    for key in full:
        assert split[key] == pytest.approx(full[key], abs=2e-5), key
    w = weights.numpy().astype(float)
    target = labels["score_ownership_target"].astype(float)
    pred = predictions["score_ownership"].detach().numpy().astype(float) * 15
    mean = np.average(target, axis=0, weights=w)
    zero = np.average(target**2, axis=0, weights=w)
    constant = np.average((target-mean)**2, axis=0, weights=w)
    for j, name in enumerate(OUTPUT_NAMES):
        prefix = "score_ownership_" + name
        assert full[prefix+"_target_mean_points"] == pytest.approx(mean[j])
        assert full[prefix+"_prediction_mean_points"] == pytest.approx(np.average(pred[:, j], weights=w))
        assert full[prefix+"_target_zero_fraction"] == pytest.approx(np.average(target[:, j] == 0, weights=w))
        assert full[prefix+"_zero_mse_points"] == pytest.approx(zero[j])
        assert full[prefix+"_constant_mse_points"] == pytest.approx(constant[j])
    assert full["score_ownership_constant_mse_points"] == pytest.approx(constant.mean())
    assert full["score_ownership_zero_mse_points"] == pytest.approx(zero.mean())
    model_mse = np.average((pred-target)**2, axis=0, weights=w).mean()
    assert full["score_ownership_constant_skill"] == pytest.approx(1-model_mse/constant.mean())
    expected = {"phase_main_action": 1, "phase_payment": 2, "phase_return_tokens": .5,
                "phase_choose_noble": 3, "non_payment": 4.5,
                "stage_early": 1, "stage_middle": 5, "stage_late": .5}
    for group, n in expected.items():
        assert full[f"score_ownership_{group}_positions"] == n
    # The actual payment remains in the training loss and its gradient.
    grad, = torch.autograd.grad(score_ownership_loss(score_ownership_loss_sums(predictions, labels, weights)),
                                predictions["score_ownership"])
    assert grad[1].abs().sum() > 0 and grad[-1].abs().sum() == 0


def test_diagnostics_are_detached_leave_loss_rng_and_samples_unchanged_and_empty_groups_null():
    predictions, labels, weights, g = diagnostic_inputs()
    before = {key: value.copy() for key, value in labels.items()}
    rng = torch.get_rng_state().clone()
    sums = score_ownership_loss_sums(predictions, labels, weights)
    loss_before = score_ownership_loss(sums)
    grad_before, = torch.autograd.grad(loss_before, predictions["score_ownership"], retain_graph=True)
    diagnostics = score_ownership_diagnostic_sums(predictions, labels, weights, g)
    assert all(not value.requires_grad for value in diagnostics.values())
    loss_after = score_ownership_loss(sums | diagnostics)
    grad_after, = torch.autograd.grad(loss_after, predictions["score_ownership"])
    torch.testing.assert_close(loss_before, loss_after, rtol=0, atol=0)
    torch.testing.assert_close(grad_before, grad_after, rtol=0, atol=0)
    assert torch.equal(rng, torch.get_rng_state())
    for key in before:
        np.testing.assert_array_equal(labels[key], before[key])
    p = {key: value[:1] for key, value in predictions.items()}
    y = {key: value[:1] for key, value in labels.items()}
    small = score_ownership_loss_sums(p, y, weights[:1])
    small.update(score_ownership_diagnostic_sums(p, y, weights[:1], g[:1]))
    metrics = score_ownership_metrics({key: float(value.detach()) for key, value in small.items()})
    assert metrics["score_ownership_phase_payment_positions"] == 0
    assert metrics["score_ownership_phase_payment_mse_points"] is None
    assert metrics["score_ownership_phase_payment_constant_mse_points"] is None
    assert metrics["score_ownership_constant_skill"] is None  # one sample: zero constant variance
    assert metrics["score_ownership_stage_late_positions"] == 0
