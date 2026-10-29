import importlib.util
from pathlib import Path
import sys

import pytest

from splendor.alphazero_recording import read_json, write_json

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/r37_lr2e5_step140221_vs_r37_10k_20261029"


def wrapper():
    spec = importlib.util.spec_from_file_location("fixed140221_test", ART / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_formal_fixed140221_two_modes_10k_pairs_same_old_seed_bank_and_search_a():
    plan = read_json(ART / "plan.json")
    assert plan["models"][0]["selected_step"] == 140221
    assert plan["models"][0]["sha256"] == "42fbd07c413190389ac5f2cab73ef41ff4a10f5402a13e099a37d49aa3cb6caa"
    assert plan["total_pairs"] == 20000 and plan["total_games"] == 40000
    assert [p["kind"] for p in plan["panels"]] == ["greedy", "search"]
    old_plan = read_json(ROOT / "artifacts/offline5_vs_r37_10k_20261027/plan.json")
    assert plan["seeds"] == old_plan["seeds"]
    for p in plan["panels"]:
        config = read_json(ROOT / p["config"]["path"])
        assert len(config["game_seeds"]) == 10000 and p["pairs"] == 10000
        assert config["policy_a"]["checkpoint"].endswith("step-000140221.pt")
        assert config["metadata"]["candidate_sha256"] == plan["models"][0]["sha256"]
        assert config["policy_b"]["checkpoint"] == plan["opponent"]["checkpoint"]
        if p["kind"] == "greedy":
            assert config["device"] == "cpu" and config["workers"] == 8
        else:
            assert config["execution"]["search_workers"] == 12
            assert config["execution"]["total_active_games"] == 1000
            assert config["execution"]["pipeline_slots"] == 2
            for side in ["policy_a", "policy_b"]:
                search = config[side]["search_config"]
                assert search["simulations"] == 128 and search["determinizations"] == 1
                assert search["root_noise_fraction"] == 0 and search["forced_playouts"] == 0
                assert not search["policy_target_pruning"]


@pytest.mark.parametrize("fail_first", [False, True])
def test_search_only_starts_after_successful_greedy(monkeypatch, fail_first):
    e = wrapper().configured_entry()
    plan = read_json(ART / "plan.json")
    done, started = set(), []
    monkeypatch.setattr(e, "panel_status", lambda p, _: {"status":"complete" if p["id"] in done else "pending"})
    monkeypatch.setattr(e, "write_report", lambda *_: None)
    def run_panel(p, *_):
        started.append(p["kind"])
        if fail_first:
            raise RuntimeError("injected greedy failure")
        done.add(p["id"])
    monkeypatch.setattr(e, "run_panel", run_panel)
    if fail_first:
        with pytest.raises(RuntimeError, match="greedy failure"):
            e.run_sequence(plan, Path("source"))
        assert started == ["greedy"]
    else:
        e.run_sequence(plan, Path("source"))
        assert started == ["greedy", "search"]


def test_status_does_not_prepare_or_modify_reports(monkeypatch, tmp_path, capsys):
    w = wrapper()
    write_json(tmp_path / "plan.json", {"panels":[{"id":"fixture"}]})
    fake = w.configured_entry()
    monkeypatch.setattr(fake, "panel_status", lambda *_: {"status":"pending", "completed_pairs":0})
    monkeypatch.setattr(w, "ART", tmp_path)
    monkeypatch.setattr(w, "configured_entry", lambda: fake)
    monkeypatch.setattr(sys, "argv", ["run.py", "status"])
    before = {p:p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert w.main() == 0
    assert before == {p:p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert "20000" in capsys.readouterr().out
