"""Finite CPU engineering check: archived labels, one update, strict load, 3 S8N1 roots."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import torch

from splendor.alphazero_recording import read_json, sha256, write_json
from splendor.alphazero_train import AlphazeroLearner
from splendor.continuation_auxiliary import (ContinuationAuxiliaryConfig, add_continuation_heads,
    continuation_targets, game_continuation_targets)
from splendor.engine import GameEngine
from splendor.features import GLOBAL_FEATURE_NAMES as G
from splendor.mcts_policy import MCTSConfig
from splendor.native_policy import NativePUCTSearchPolicy
from splendor.offline_continuation import ContinuationOfflineConfig
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.rng import DeterministicRng
from splendor.selfplay_data import SelfPlaySample, collate_selfplay_samples, load_game


def checked_score_ownership_oracle(arrays, metadata):
    """Checked inventory oracle, including automatically acquired nobles."""
    from splendor.catalog import load_default_catalog
    from splendor.compact_trace import replay_trace
    from splendor.replay import _decode_action
    catalog = load_default_catalog()
    trace = metadata["trace"]
    replay_trace(trace)
    increments = np.zeros((len(arrays["player_id"]), 2, 4), dtype=np.int64)
    engine = GameEngine.new(trace["game_seed"], first_player=trace["first_player"], catalog=catalog, mode="checked")
    for i, item in enumerate(trace["actions"]):
        before = engine._debug_state().players
        engine.apply(_decode_action(item["action"]))
        after = engine._debug_state().players
        for actor in (0, 1):
            for card_id in set(after[actor].purchased_card_ids)-set(before[actor].purchased_card_ids):
                card = catalog.card(card_id)
                increments[i, actor, card.tier.value-1] += card.points
            for noble_id in set(after[actor].noble_ids)-set(before[actor].noble_ids):
                increments[i, actor, 3] += catalog.noble(noble_id).points
    future = np.cumsum(increments[::-1], axis=0)[::-1]
    rows, players = np.arange(len(future)), arrays["player_id"]
    return np.concatenate((future[rows, players], future[rows, 1-players]), axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="base entity_attention_v1 weights")
    parser.add_argument("--output", type=Path, required=True, help="runtime directory for the smoke weights")
    parser.add_argument("--artifacts", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    cfg = ContinuationOfflineConfig.from_dict(read_json(args.config))
    auxiliary = ContinuationAuxiliaryConfig(**cfg.continuation_auxiliary)
    if sha256(Path(cfg.manifest)) != cfg.manifest_sha256:
        raise ValueError("historical manifest changed")
    rows = [json.loads(line) for line in Path(cfg.manifest).read_text().splitlines()]
    rows = [r for r in rows if cfg.first_round-1 <= r["round_index"] < cfg.last_round and r["split"] == "train"][-3:]
    samples, games = [], []
    for row in rows:
        path = Path(row["path"])
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (row["size"], row["mtime_ns"]):
            raise ValueError("source archive changed")
        arrays, meta = load_game(path)
        extra = game_continuation_targets(arrays, meta, auxiliary)
        if auxiliary.score_ownership_weight > 0:
            np.testing.assert_array_equal(extra["score_ownership_target"], checked_score_ownership_oracle(arrays, meta))
        main = arrays["global_features"][:, G.index("phase_main_action")] > .5
        games.append({"path": str(path), "sha256": sha256(path), "decisions": len(main),
            "main_actions": int(main.sum()),
            "shortterm_valid": int(extra["shortterm_value_mask"].sum()) if auxiliary.shortterm_value_weight else None,
            "opponent_weight": float(extra["opponent_policy_weight"].sum()) if auxiliary.opponent_policy_weight else None,
            "score_ownership_trace_oracle": bool(auxiliary.score_ownership_weight)})
        enriched = arrays | extra
        samples.extend(SelfPlaySample(enriched, meta, int(i)) for i in np.linspace(0, len(main)-1, 16, dtype=int))
    policies = [s for s in samples if s.arrays["policy_weight"][s.index] > 0][:8]
    values = samples[::3][:16]
    if not policies or not values:
        raise ValueError("smoke requires eligible archived samples")
    base = load_frozen_actor_critic_checkpoint(args.checkpoint, device="cpu")
    torch.manual_seed(cfg.auxiliary_initialization_seed)
    model = add_continuation_heads(base, predict_opponent_policy=auxiliary.opponent_policy_weight > 0,
        predict_shortterm_value=auxiliary.shortterm_value_weight > 0, predict_score_ownership=auxiliary.score_ownership_weight > 0)
    policy_batch = collate_selfplay_samples(policies)
    value_batch = collate_selfplay_samples(values)
    with torch.inference_mode():
        for a, b in zip(base(policy_batch.decisions), model(policy_batch.decisions)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    learner = AlphazeroLearner(model, replace(cfg.learner, device="cpu"), sampling_rng=DeterministicRng(cfg.sampling_seed))
    learner.round_steps_target = 1
    metrics = learner.train_step(policy_batch, references=[], value_batch=value_batch,
        continuation_auxiliary=auxiliary, continuation_labels=continuation_targets(values, auxiliary),
        record_encoder_grad_norms=True)
    if ((auxiliary.shortterm_value_weight > 0 and metrics["shortterm_positions"] <= 0)
            or (auxiliary.opponent_policy_weight > 0 and metrics["opponent_positions"] <= 0)
            or (auxiliary.score_ownership_weight > 0 and metrics["score_ownership_positions"] <= 0)):
        raise ValueError("smoke did not exercise the configured auxiliary objectives")
    gradient_tasks = ["policy", "value"]
    if auxiliary.shortterm_value_weight > 0:
        gradient_tasks.extend(("shortterm_h2", "shortterm_h6", "shortterm_h14", "shortterm_value"))
    if auxiliary.opponent_policy_weight > 0:
        gradient_tasks.extend(("opponent_action", "opponent_buy", "opponent_reserve", "opponent_policy"))
    if auxiliary.score_ownership_weight > 0:
        from splendor.score_ownership import OUTPUT_NAMES
        gradient_tasks.extend(("score_ownership", *("score_ownership_"+name for name in OUTPUT_NAMES)))
    if not all(np.isfinite(metrics["encoder_grad_norm_" + name]) for name in gradient_tasks):
        raise ValueError("smoke did not record finite encoder task gradients")
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "smoke_checkpoint.pt"
    learner.save(checkpoint)
    restored = load_frozen_actor_critic_checkpoint(checkpoint, device="cpu")
    with torch.inference_mode():
        for a, b in zip(model(policy_batch.decisions), restored(policy_batch.decisions)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    def unexpected_auxiliary(*_):
        raise AssertionError("search/inference executed a training-only head")
    hooks = []
    if restored.config.predict_shortterm_value:
        hooks.append(restored.shortterm_value_head.register_forward_pre_hook(unexpected_auxiliary))
    if restored.config.predict_opponent_policy:
        hooks.append(restored.opponent_policy_head.register_forward_pre_hook(unexpected_auxiliary))
    if restored.config.predict_score_ownership:
        hooks.append(restored.score_ownership_head.register_forward_pre_hook(unexpected_auxiliary))
    searches = []
    for seed in (902301, 902302, 902303):
        request = GameEngine.new(seed, mode="fast").decision()
        result = NativePUCTSearchPolicy(restored, config=MCTSConfig(simulations=8, determinizations=1,
            seed=seed, root_noise_fraction=0)).search(request)
        if result.simulations != 8:
            raise ValueError("native smoke did not finish S8N1 (total 8)")
        searches.append({"seed": seed, "simulations": result.simulations})
    for hook in hooks:
        hook.remove()
    files = ["src/splendor/continuation_auxiliary.py", "src/splendor/opponent_policy.py", "src/splendor/score_ownership.py",
        "src/splendor/offline_continuation.py", "src/splendor/actor_critic.py", "src/splendor/entity_attention.py", "src/splendor/compact_game_cache.py",
        "src/splendor/alphazero_train.py", "src/splendor/offline_objectives.py", __file__]
    write_json(args.artifacts / "smoke.json", {"scope": "engineering verification only; no strength conclusion",
        "command_config": str(args.config), "config_sha256": sha256(args.config), "config": asdict(cfg),
        "base_checkpoint": str(args.checkpoint), "base_checkpoint_sha256": sha256(args.checkpoint),
        "base_parameters": base.trainable_parameter_count(), "parameters": model.trainable_parameter_count(),
        "games": games, "one_cpu_update": metrics, "strict_reload": True, "base_outputs_bitwise_preserved": True,
        "native_searches": searches, "inference_auxiliary_calls": 0,
        "source_sha256": {str(path): sha256(Path(path)) for path in files}, "torch": torch.__version__})
    print({"games": games, "total_loss": metrics["total_loss"], "parameters": model.trainable_parameter_count()}, flush=True)


if __name__ == "__main__":
    main()
