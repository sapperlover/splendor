from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest
import torch

from splendor import decision_probe as probe
from splendor.actor_critic import ActorCritic
from splendor.features import FEATURE_SCHEMA_VERSION
from splendor.ppo import TRAINING_CHECKPOINT_FORMAT_VERSION
from splendor.replay import RULESET_ID


@pytest.fixture(scope="module")
def bank(tmp_path_factory):
    path = tmp_path_factory.mktemp("decision-bank") / "bank.json"
    probe.build_bank(path, positions=8)
    return path


@pytest.fixture
def checkpoint(tmp_path):
    torch.set_num_threads(1)
    model = ActorCritic().eval()
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    path = tmp_path / "weights.pt"
    torch.save({"format_version": TRAINING_CHECKPOINT_FORMAT_VERSION,
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "metadata": {"rules_version": RULESET_ID},
                "model_config": asdict(model.config), "model_state_dict": model.state_dict()}, path)
    return path


def search(*, worlds=1, backend="native"):
    return {"backend": backend, "simulations": 9, "determinizations": worlds, "c_puct": 1.5,
            "fold_forced_phases": True, "max_simulation_transitions": 512, "root_noise_fraction": 0.}


def test_bank_real_replay_stage_balance_and_immutable_resume(bank):
    report = probe.verify_bank(bank)
    assert report["positions"] == report["independent_games"] == 8
    assert report["by_stage"] == {"early_mid": 4, "late": 4}
    identity = probe.digest(bank)
    probe.build_bank(bank, positions=8, resume=True)
    assert probe.digest(bank) == identity
    with pytest.raises(ValueError, match="identity changed"):
        probe.build_bank(bank, positions=8, seed_start=9, resume=True)
    with pytest.raises(FileExistsError):
        probe.build_bank(bank, positions=8)


def test_jsd_and_baseline_high_visit_ranks():
    assert probe.jsd(np.array([1., 0.]), np.array([1., 0.])) == 0
    assert probe.jsd(np.array([1., 0.]), np.array([0., 1.])) == pytest.approx(np.log(2))
    rank = probe.rank_changes(np.array([.5, .3, .2]), np.array([.3, .5, .2]))
    assert rank["high_visit_actions"] == 2
    assert rank["high_visit_rank_inversion"] == 1
    assert rank["high_visit_mean_rank_displacement"] == 1
    assert probe.rank_changes(np.array([.9, .1]), np.array([.1, .9]))["high_visit_rank_inversion"] is None
    assert probe.rank_changes(np.array([.5, .3, .2]), np.array([.4, .4, .2]))["high_visit_rank_inversion"] == .5


def test_stage_ineligible_natural_game_is_preserved_and_replaced(tmp_path):
    path = tmp_path / "bank.json"
    report = probe.build_bank(path, positions=8, seed_start=2026100932)
    assert report["excluded_games"] >= 1
    assert report["source_games_played"] == 8 + report["excluded_games"]
    excluded = [__import__("json").loads(line) for line in (tmp_path / "excluded_source_games.jsonl").read_text().splitlines()]
    assert any(r["game_seed"] == 2026100939 for r in excluded)
    assert 2026100939 not in {r["game_seed"] for r in probe.read(path)["positions"]}


def test_native_saved_scores_use_world_targets_and_pair_exactly(bank, checkpoint, tmp_path):
    output = tmp_path / "score"
    summary = probe.evaluate(bank, checkpoint, output, search=search(worlds=2), active_roots=4)
    raw = probe.read(output / "searches.json")
    assert summary["status"] == "complete"
    assert summary["by_stage"]["all"]["positions"] == 8
    for run in raw:
        for row in run:
            assert row["simulation_budgets"] == [5, 4]
            target = np.mean([np.asarray(v) / sum(v) for v in row["per_world_root_visit_counts"]], axis=0)
            np.testing.assert_allclose(row["policy_target"], target)
            assert sum(row["root_visit_counts"]) == 9
    report = probe.compare_scores(output, output, tmp_path / "comparison.json")
    for group in report["by_stage"].values():
        for metric in ("action_changed", "visit_jsd", "prior_jsd", "root_value_absolute_change"):
            assert group["metrics"][metric]["mean"] == 0
    assert probe.evaluate(bank, checkpoint, output, search=search(worlds=2), active_roots=4,
                          resume=True) == summary
    with pytest.raises(ValueError, match="identity changed"):
        probe.evaluate(bank, checkpoint, output, search=search(), active_roots=4, resume=True)

    altered = tmp_path / "altered"
    altered.mkdir()
    probe.write(altered / "searches.json", raw)
    changed = {**summary, "repeat_seeds": [1, 2]}
    probe.write(altered / "summary.json", changed)
    with pytest.raises(ValueError, match="repeat_seeds"):
        probe.compare_scores(output, altered, tmp_path / "bad.json", allow_search_change=True)
    changed = {**summary, "search": search()}
    probe.write(altered / "summary.json", changed)
    with pytest.raises(ValueError, match="search configurations"):
        probe.compare_scores(output, altered, tmp_path / "bad.json")
    # Git provenance may change without any execution file changing.
    changed = {**summary, "code": {**summary["code"], "git_head": "documentation-commit"}}
    probe.write(altered / "summary.json", changed)
    assert not probe.compare_scores(output, altered, tmp_path / "same.json")["code_changed"]
    files = {**summary["code"]["files"], "native_policy.py": "different"}
    changed["code"] = {**summary["code"], "files": files}
    probe.write(altered / "summary.json", changed)
    with pytest.raises(ValueError, match="code changed"):
        probe.compare_scores(output, altered, tmp_path / "bad.json")


def test_interruption_resumes_only_missing_roots(bank, checkpoint, tmp_path, monkeypatch):
    output = tmp_path / "interrupted"
    original_write = probe.write

    def interrupt(path, value):
        original_write(path, value)
        if path.name == "seed-2026101500-root-000000.json":
            raise KeyboardInterrupt

    monkeypatch.setattr(probe, "write", interrupt)
    with pytest.raises(KeyboardInterrupt):
        probe.evaluate(bank, checkpoint, output, search=search(), active_roots=4)
    assert not (output / "summary.json").exists()
    monkeypatch.setattr(probe, "write", original_write)
    jobs = []
    original_start = probe.NativePUCTSearchPolicy.start_job

    def observe(self, request, **kwargs):
        jobs.append(request)
        return original_start(self, request, **kwargs)

    monkeypatch.setattr(probe.NativePUCTSearchPolicy, "start_job", observe)
    summary = probe.evaluate(bank, checkpoint, output, search=search(), active_roots=4, resume=True)
    assert summary["status"] == "complete" and len(jobs) == 15
    assert probe.compare_scores(output, output, tmp_path / "same.json")["by_stage"]["all"]["metrics"]["visit_jsd"]["mean"] == 0


def test_native_chance_uses_same_bank_protocol(bank, checkpoint, tmp_path):
    config = {**search(backend="native_chance"), "chance_widening": {"k": .25, "alpha": .5}}
    summary = probe.evaluate(bank, checkpoint, tmp_path / "chance", search=config,
                             seeds=(9,), active_roots=4)
    assert summary["status"] == "complete"
    assert summary["within_checkpoint_seed_variation"]["all"]["positions"] == 0
