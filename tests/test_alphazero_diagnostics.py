from __future__ import annotations
import copy
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.nn.utils import clip_grad_norm_

from splendor.alphazero_analysis import analyze_run
from splendor.alphazero_diagnostics import evaluate_set, freeze_set
from splendor.alphazero_loop import LoopConfig, run_loop
from splendor.alphazero_recording import DiagnosticsConfig, ReferenceWriter, read_json, summarize_references
from splendor.alphazero_recording import write_json, append_json
from splendor.alphazero_replay import ReplayReference, build_replay_index
from splendor.alphazero_train import AlphazeroLearner, LearnerConfig, _torch_batch, alphazero_loss_sums, draw_training_batch, label_statistics, metric_ratios, run_training_round
from splendor.rng import DeterministicRng
from splendor.selfplay_data import collate_selfplay_samples
from test_alphazero_loop import _loop_config, _weights_checkpoint
from test_alphazero_training import az_runs


def test_label_entropy_mask_weights_and_same_denominator():
    batch = SimpleNamespace(decisions=SimpleNamespace(candidate_mask=np.array([[1, 1, 0], [1, 0, 0], [1, 1, 0]], bool)),
        policy_targets=np.array([[.25, .75, 100], [1, 100, 100], [.5, .5, 100]], np.float32),
        policy_weights=np.array([2, 0, 0], np.float32), value_targets=np.array([1, -1, 0], np.float32), value_weights=np.array([1, 2, 0], np.float32))
    stats = label_statistics(batch)
    h = -(.25 * np.log(.25) + .75 * np.log(.75))
    assert stats["target_entropy_sum"] == pytest.approx(2 * h)
    assert stats["policy_positions"] == 1
    assert stats["value_positions"] == 2
    ratios = metric_ratios({**stats, "policy_weight_sum": 2, "policy_ce_sum": 2 * (h + .3), "entropy_sum": 2 * .4,
                            "value_weight_sum": 3, "value_sq_sum": .75})
    assert ratios["policy_kl"] == pytest.approx(.3)
    assert ratios["target_entropy"] == pytest.approx(h)
    assert ratios["value_zero_mse"] == 1
    assert metric_ratios({**stats, "policy_weight_sum": 0, "policy_ce_sum": 0, "entropy_sum": 0, "value_weight_sum": 3, "value_sq_sum": .75})["policy_kl"] is None


def test_reference_coverage_resume_deduplicates_and_detects_changed_draws(tmp_path):
    refs = [ReplayReference(0, "run-0", 0, 0), ReplayReference(1, "run-1", 0, 0)]
    for session, steps in (("a", [1, 2]), ("b", [2, 3])):
        writer = ReferenceWriter(tmp_path, session, 2, {"run_id": "run"})
        for step in steps:
            writer.append(step, refs)
        writer.close()
    with (tmp_path / "b.bin").open("ab") as handle:
        handle.write(b"torn")
    result = summarize_references(tmp_path, 3, 1, {"current": 3, "history": 4})
    assert result["committed_draws"] == 6
    assert result["repeated_or_uncommitted_draws"] == 2
    assert result["torn_bytes"] == 4
    assert result["current"]["covered_positions"] == 1
    assert result["current"]["unseen_fraction"] == pytest.approx(2 / 3)
    assert result["history"]["draws"] == 3
    writer = ReferenceWriter(tmp_path, "c", 2, {})
    writer.append(2, refs[::-1]); writer.close()
    with pytest.raises(ValueError, match="sequence differs"):
        summarize_references(tmp_path, 3, 1, {"current": 3, "history": 4})


def test_fixed_set_no_leak_identity_stage_and_rng_restore(az_runs, tmp_path):
    checkpoint, first, _ = az_runs
    index = build_replay_index([first], max_games=3, cache_bytes=1024**2)
    games = index.split_games("validation")
    samples = [index.cache.sample(game, i) for game in games for i in range(game.decision_count)]
    paths = {game.key: game.path for game in games}
    manifest = freeze_set(samples, paths, limit=16, seed=77)
    assert manifest == freeze_set(samples, paths, limit=16, seed=77)
    assert len(manifest["references"]) == 16
    assert sum(manifest["composition"].values()) == 16
    assert all(ref["game_id"] in {g.game_id for g in games} for ref in manifest["references"])
    train = index.position_pool(split="train", round_index=0)
    sample, _ = train.sample(DeterministicRng(1))
    with pytest.raises(ValueError, match="validation games"):
        freeze_set([sample], {g.key: g.path for g in index.games})
    config = LearnerConfig(device="cpu", batch_size=4, history_batch_size=2, validation_batch_size=4)
    rng_before = torch.get_rng_state().clone()
    metrics = evaluate_set(checkpoint, manifest, config, {"available": True, "value": .2, "source": "test training-only constant"})
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert metrics["set_sha256"] == manifest["set_sha256"]
    overall = metrics["metrics"]["overall"]
    assert overall["positions"] == 16
    assert overall["policy_loss"] - overall["target_entropy"] == pytest.approx(overall["policy_kl"])
    assert overall["value_train_constant_mse"] == pytest.approx((overall["value_target_sq_sum"] - .4 * overall["value_target_sum"] + .04 * overall["value_weight_sum"]) / overall["value_weight_sum"])


def test_new_logging_matches_original_update_and_sampling(az_runs):
    checkpoint, first, _ = az_runs
    index = build_replay_index([first], max_games=3, cache_bytes=16 * 1024**2)
    pool = index.position_pool(split="train", round_index=0)
    config = LearnerConfig(device="cpu", batch_size=4, history_batch_size=2)
    learner = AlphazeroLearner.warm_start(checkpoint, config, sampling_seed=11, replay_identity=index.identity(), parent_checkpoint={}, round_index=0, round_steps_target=2)
    baseline = copy.deepcopy(learner.model)
    optimizer = torch.optim.AdamW(baseline.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    rng = DeterministicRng(11)
    for _ in range(2):
        drawn = draw_training_batch(pool, None, learner.sampling_rng, config)
        old_drawn = draw_training_batch(pool, None, rng, config)
        assert drawn.references == old_drawn.references
        batch = collate_selfplay_samples(drawn.samples)
        tensors = _torch_batch(batch, torch.device("cpu"))
        baseline.train(); logits, values = baseline(batch.decisions)
        terms = alphazero_loss_sums(logits, values, candidate_mask=tensors["candidate_mask"], policy_targets=tensors["policy_targets"],
            value_targets=tensors["value_targets"], policy_weights=tensors["policy_weights"], value_weights=tensors["value_weights"])
        optimizer.zero_grad(set_to_none=True); terms.total(config).backward(); clip_grad_norm_(baseline.parameters(), config.max_grad_norm); optimizer.step()
        learner.train_step(batch, references=drawn.references)
        assert all(torch.equal(a, b) for a, b in zip(baseline.parameters(), learner.model.parameters()))
        assert rng.snapshot() == learner.sampling_rng.snapshot()


def test_prepared_config_and_historical_fingerprint_compatible(tmp_path):
    root = Path(__file__).resolve().parents[1]
    old_path = root / "runs/az_loop_seed1/config.json"
    if old_path.exists():
        raw = read_json(old_path)
        assert LoopConfig.from_dict(raw["config"]).to_dict() == raw["config"]
    formal_raw = read_json(root / "configs/az_theta5_10x1000/formal.json")
    formal_raw["initial_checkpoint"] = str(_weights_checkpoint(tmp_path / "initial.pt"))
    formal = LoopConfig.from_dict(formal_raw)
    assert (formal.num_rounds, formal.new_games_per_round, formal.replay_max_games, formal.replay_cache_bytes) == (10, 1000, 6000, 4294967296)
    assert formal.evaluation.greedy_vs_initial.pairs == 50
    assert formal.evaluation.mcts_vs_parent.bootstrap_resamples == 10000
    inventory = read_json(root / "artifacts/az_theta5_10x1000_seed1/seeds.json")
    assert len(inventory["training"]) == 10
    assert inventory["audit"]["collisions"] == []
    worlds = {name: {panel["search_seed_base"] ^ p for p in range(panel["policy_seed_start"], panel["policy_seed_end_inclusive"] + 1)}
              for name, panel in inventory["evaluation"].items() if panel["search_seed_base"] is not None}
    assert not worlds["round-002-vs-theta5.json"] & worlds["round-002-vs-ppo.json"]
    assert worlds["round-002-vs-theta5.json"] == worlds["round-004-vs-theta5.json"]
    assert not worlds["round-009-vs-theta5.json"] & worlds["round-002-vs-theta5.json"]


def test_interrupted_training_reference_ledger_and_updates_resume_exactly(az_runs, tmp_path):
    checkpoint, first, _ = az_runs
    index = build_replay_index([first], max_games=3, cache_bytes=16 * 1024**2)
    pool = index.position_pool(split="train", round_index=0)
    config = LearnerConfig(device="cpu", batch_size=4, history_batch_size=2, checkpoint_every_steps=2)
    def fresh():
        return AlphazeroLearner.warm_start(checkpoint, config, sampling_seed=31, replay_identity=index.identity(), parent_checkpoint={}, round_index=0, round_steps_target=4)
    expected = fresh()
    run_training_round(expected, steps_target=4, current_pool=pool, history_pool=None, validation_samples=[], validation_games=0,
                       checkpoint_dir=tmp_path / "baseline", metrics_path=tmp_path / "baseline.jsonl")
    actual = fresh()
    writer = ReferenceWriter(tmp_path / "usage", "interrupted", 4, {})
    def stop(record):
        if record["round_train_step"] == 3:
            raise KeyboardInterrupt("unit test interruption after uncommitted step")
    with pytest.raises(KeyboardInterrupt):
        try:
            run_training_round(actual, steps_target=4, current_pool=pool, history_pool=None, validation_samples=[], validation_games=0,
                               checkpoint_dir=tmp_path / "resume", metrics_path=tmp_path / "metrics.jsonl", progress_callback=stop, reference_writer=writer)
        finally:
            writer.close()
    actual = AlphazeroLearner.resume(tmp_path / "resume/step-000000002.pt", config, replay_identity=index.identity(), round_index=0, round_steps_target=4)
    writer = ReferenceWriter(tmp_path / "usage", "resumed", 4, {})
    try:
        run_training_round(actual, steps_target=4, current_pool=pool, history_pool=None, validation_samples=[], validation_games=0,
                           checkpoint_dir=tmp_path / "resume", metrics_path=tmp_path / "metrics.jsonl", reference_writer=writer)
    finally:
        writer.close()
    usage = summarize_references(tmp_path / "usage", 4, 0, {"current": len(pool), "history": 0})
    assert usage["committed_draws"] == 16
    assert usage["repeated_or_uncommitted_draws"] == 4
    assert all(torch.equal(a, b) for a, b in zip(expected.model.parameters(), actual.model.parameters()))
    assert expected.sampling_rng.snapshot() == actual.sampling_rng.snapshot()
    records = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [r["round_train_step"] for r in records if r["event"] == "train_step"] == [1, 2, 3, 4]


def test_recorded_loop_calls_diagnostics_and_noop_resume(tmp_path, monkeypatch):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        checkpoint = _weights_checkpoint(tmp_path / "initial.pt")
        config = replace(_loop_config(checkpoint), num_rounds=1, new_games_per_round=4,
                         selfplay=replace(_loop_config(checkpoint).selfplay, validation_fraction=.5),
                         evaluation=replace(_loop_config(checkpoint).evaluation,
                             mcts_vs_parent=replace(_loop_config(checkpoint).evaluation.mcts_vs_parent, bootstrap_resamples=32, bootstrap_seed=777),
                             greedy_vs_initial=replace(_loop_config(checkpoint).evaluation.greedy_vs_initial, bootstrap_resamples=32, bootstrap_seed=778)),
                         diagnostics=DiagnosticsConfig(str(tmp_path / "artifacts"), fixed_positions=16, resource_interval_seconds=5))
        output = tmp_path / "run"
        result = run_loop(config, output, progress_interval_seconds=0, progress_format="human")
        assert result["loop_complete"]
        comparison = read_json(tmp_path / "artifacts/round-000/current_before_after.json")
        assert comparison["before"]["set_sha256"] == comparison["after"]["set_sha256"]
        summary = read_json(tmp_path / "artifacts/summary.json")
        assert summary["rounds"][0]["committed_steps"] == 2
        assert all(row["paired_bootstrap"]["resamples"] == 32 for row in summary["evaluation"])
        usage = read_json(tmp_path / "artifacts/round-000/usage.json")
        assert usage["committed_draws"] == 8
        run_loop(config, output, resume=True, progress_interval_seconds=0)
        summary = read_json(tmp_path / "artifacts/summary.json")
        assert len(summary["sessions"]) == 2
        assert summary["rounds"][0]["committed_steps"] == 2
        assert usage == read_json(tmp_path / "artifacts/round-000/usage.json")
    finally:
        torch.set_num_threads(previous_threads)


def test_hard_kill_reports_lower_bound_instead_of_false_throughput(tmp_path):
    output = tmp_path / "run"
    (output / "rounds/round-000").mkdir(parents=True)
    write_json(output / "recording/sessions/ended.json", {"session_id": "ended", "start_unix": 100, "end_unix": 250,
        "wall_seconds": 150, "status": "interrupted", "phases": [{"phase": "collect", "round_index": 0, "start_unix": 100, "end_unix": 200, "seconds": 100}]})
    write_json(output / "recording/sessions/killed.json", {"session_id": "killed", "start_unix": 500, "status": "running",
        "phases": [{"phase": "collect", "round_index": 0, "start_unix": 500, "status": "running"}]})
    append_json(output / "recording/resources.jsonl", {"session_id": "killed", "unix_time": 580, "round_index": 0, "phase": "collect", "tree_rss_bytes": 10, "tree_pss_bytes": 5})
    result = analyze_run(output, tmp_path / "artifact")
    assert result["recorded_actual_wall_seconds"] == 150
    assert result["pause_seconds_known"] == 250
    assert result["hard_kill_wall_lower_bound_seconds"] == 80
    assert result["phase_seconds_mutually_exclusive"]["invocation_remainder"] == 50
    assert not result["rounds"][0]["cumulative_rates"]["available"]


def test_external_panel_wrapper_preserves_config_and_resume_ledger(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("recorded_eval_test", root / "scripts/run_recorded_evaluation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    raw = read_json(root / "configs/az_theta5_10x1000/round-002-vs-theta5.json")
    cfg = tmp_path / "panel.json"
    write_json(cfg, raw)
    output = tmp_path / "anchors/panel.json"
    seen = []
    def fixed(config, path, *, resume, progress_callback):
        seen.append((config, resume))
        progress_callback(200 if resume else 0, 200)
        progress_callback(200, 200)
        summary = {"completed_pairs": 200, "total_pairs": 200, "passed": True}
        write_json(path, summary)
        return SimpleNamespace(summary=summary)
    monkeypatch.setattr(module, "run_fixed_evaluation", fixed)
    args = ["run_recorded_evaluation.py", "--config", str(cfg), "--output", str(output)]
    monkeypatch.setattr(sys, "argv", args)
    assert module.main() == 0
    monkeypatch.setattr(sys, "argv", args + ["--resume"])
    assert module.main() == 0
    assert seen[0][0].bootstrap_seed == raw["bootstrap_seed"]
    assert seen[0][0].execution == raw["execution"]
    assert seen[0][0].policy_a.checkpoint == seen[1][0].policy_a.checkpoint
    assert [value[1] for value in seen] == [False, True]
    sessions = sorted((output.parent / "recording/panel/recording/sessions").glob("*.json"))
    assert len(sessions) == 2
    assert sorted(read_json(p)["end_state"]["new_completed_pairs"] for p in sessions) == [0, 200]
