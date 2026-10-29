import ast
import copy
import fcntl
import importlib.util
import sys
from pathlib import Path

import pytest

from splendor.alphazero_recording import read_json, write_json

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/offline5_vs_r37_10k_20261027"


def entry():
    spec = importlib.util.spec_from_file_location("offline5_evaluation", ART / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("existing_reports", [False, True])
def test_status_is_read_only_while_run_holds_chain_lock(monkeypatch, tmp_path, capsys, existing_reports):
    e = entry()
    art, runtime = tmp_path / "artifacts", tmp_path / "runtime"
    art.mkdir()
    runtime.mkdir()
    plan = {"panels": [{"id": "fixture"}], "total_pairs": 10000}
    row = {"id": "fixture", "model": "noaux", "kind": "greedy", "status": "incomplete", "completed_pairs": 37}
    write_json(art / "plan.json", plan)
    if existing_reports:
        write_json(art / "summary.json", {"previous_report": True})
        (art / "results.md").write_text("Previous report\n")
    monkeypatch.setattr(e, "ART", art)
    monkeypatch.setattr(e, "RUNTIME", runtime)
    monkeypatch.setattr(e, "panel_status", lambda *_: row)
    monkeypatch.setattr(sys, "argv", ["run.py", "status"])
    with (runtime / "chain.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()}
        assert e.main() == 0
        after = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()}
    assert after == before
    assert ast.literal_eval(capsys.readouterr().out) == {
        "status": "incomplete", "panels": [row], "planned_pairs": 10000, "completed_pairs": 37}


def test_formal_panels_have_requested_order_best_models_common_seeds_and_execution_a():
    e = entry()
    plan = read_json(ART / "plan.json")
    keys = ["value", "value_policy", "policy", "noaux", "policy_score"]
    assert [p["model"] for p in plan["panels"]] == keys*2
    assert [p["kind"] for p in plan["panels"]] == ["greedy"]*5+["search"]*5
    assert plan["total_pairs"] == 100000 and plan["total_games"] == 200000
    assert [m["selected_step"] for m in plan["models"]] == [140221, 140221, 140221, 122694, 140221]
    seeds = None
    for panel in plan["panels"]:
        config = read_json(ROOT / panel["config"]["path"])
        assert len(config["game_seeds"]) == 10000 and panel["pairs"] == 10000
        if seeds is None:
            seeds = config["game_seeds"]
        assert config["game_seeds"] == seeds
        assert config["policy_b"]["checkpoint"] == plan["opponent"]["checkpoint"]
        if panel["kind"] == "greedy":
            assert config["device"] == "cpu" and config["workers"] == 8
            assert all(config[s]["kind"] == "greedy_checkpoint" for s in ("policy_a", "policy_b"))
        else:
            x = config["execution"]
            assert config["device"] == "cuda" and x["search_workers"] == 12 and x["total_active_games"] == 1000
            assert x["max_batch_size"] == 500 and x["local_microbatch"] == 128 and x["pipeline_slots"] == 2
            assert x["batch_buckets"] == [64,128,256,500] and x["candidate_buckets"] == [32,48,64]
            for side in ("policy_a", "policy_b"):
                search = config[side]["search_config"]
                assert search["simulations"] == 128 and search["determinizations"] == 1
                assert search["root_noise_fraction"] == 0 and search["forced_playouts"] == 0
                assert not search["policy_target_pruning"]
    assert e.SUMMARY_EVERY == 1000


@pytest.mark.parametrize("fail_at", [None, 4, 6])
def test_strict_serial_order_and_failure_stop(monkeypatch, fail_at):
    e = entry()
    plan = read_json(ART / "plan.json")
    done, started = set(), []
    monkeypatch.setattr(e, "panel_status", lambda p, _: {"id": p["id"], "status": "complete" if p["id"] in done else "pending"})
    monkeypatch.setattr(e, "write_report", lambda *_: None)
    def run(panel, *_):
        started.append(panel["id"])
        if len(started) == fail_at:
            raise RuntimeError("injected panel failure")
        done.add(panel["id"])
    monkeypatch.setattr(e, "run_panel", run)
    if fail_at:
        with pytest.raises(RuntimeError, match="injected"):
            e.run_sequence(plan, Path("source"))
        assert len(started) == fail_at
    else:
        e.run_sequence(plan, Path("source"))
        assert started == [p["id"] for p in plan["panels"]]


def test_resume_skips_completed_panels_and_passes_resume_for_partial_sidecar(monkeypatch, tmp_path):
    e = entry()
    plan = read_json(ART / "plan.json")
    done = {p["id"] for p in plan["panels"][:3]}
    started = []
    monkeypatch.setattr(e, "panel_status", lambda p, _: {"id": p["id"], "status": "complete" if p["id"] in done else "pending"})
    monkeypatch.setattr(e, "write_report", lambda *_: None)
    monkeypatch.setattr(e, "run_panel", lambda p, *_: (started.append(p["id"]), done.add(p["id"])))
    e.run_sequence(plan, Path("source"))
    assert started == [p["id"] for p in plan["panels"][3:]]
    monkeypatch.setattr(e, "ROOT", tmp_path)
    panel = plan["panels"][3]
    command = e.panel_command(panel, tmp_path / "frozen")
    assert "--resume" not in command and command[3] == "splendor.evaluation"
    Path(str(tmp_path / panel["output"])+".games").mkdir(parents=True)
    assert "--resume" in e.panel_command(panel, tmp_path / "frozen")
    assert command[command.index("--summary-every")+1] == "1000"


@pytest.mark.parametrize("kind,simulations,accepted", [("search",128,True), ("search",127,False),
                                                       ("greedy",0,True), ("greedy",128,False)])
def test_result_checks_pair_seats_and_actual_search_budget(tmp_path, monkeypatch, kind, simulations, accepted):
    e = entry()
    monkeypatch.setattr(e, "ROOT", tmp_path)
    config = {"game_seeds": [7], "metadata": {"candidate_sha256":"candidate", "opponent_sha256":"r37"},
              "bootstrap_resamples":10000, "bootstrap_seed":3}
    panel = {"id":"fixture", "model":"model", "kind":kind, "pairs":1,
             "config":{"path":"config.json"}, "output":"result.json"}
    stats = {"simulations":simulations, "searchable_decisions":1 if kind == "search" else 0,
             "determinizations":1 if kind == "search" else 0}
    games = [{"game_in_pair":i, "first_player":i, "game_seed":7, "status":"natural_terminal",
              "outcome_for_a":"a_win" if i == 0 else "a_loss", "policy_a_stats":stats, "policy_b_stats":stats} for i in range(2)]
    result = {"config":config, "catalog_fingerprint":"catalog", "passed":True, "valid_games":2,
        "completed_pairs":1, "win_rate_score":.5, "pairs":[{"pair_id":0,"game_seed":7,"games":games}],
        "checkpoint_identity":{"policy_a":{"sha256":"candidate"},"policy_b":{"sha256":"r37"}},
        "paired_bootstrap":{"pair_count":1,"resamples":10000,"seed":3,"estimate":.5},
        **{key:0 for key in ("a_wins","draws","a_losses","exceptions","illegal_actions",
                            "engineering_truncations","invariant_failures","policy_cycle_draws")}}
    write_json(tmp_path / "config.json", config)
    write_json(tmp_path / "result.json", result)
    if accepted:
        assert e.panel_status(panel, {"catalog_fingerprint":"catalog"})["status"] == "complete"
        bad = copy.deepcopy(result)
        bad["pairs"][0]["games"][1]["first_player"] = 0
        write_json(tmp_path / "result.json", bad)
        with pytest.raises(ValueError, match="seats"):
            e.panel_status(panel, {"catalog_fingerprint":"catalog"})
    else:
        with pytest.raises(ValueError, match="budget|Greedy"):
            e.panel_status(panel, {"catalog_fingerprint":"catalog"})
