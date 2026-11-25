import importlib.util
from pathlib import Path
import sys

import pytest

from splendor.alphazero_recording import read_json, write_json

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/lambda161924_search3000_20261123"


def wrapper():
    spec = importlib.util.spec_from_file_location("lambda3000_test", ART / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_user_fixed_models_same_pool_and_two_independent_3000pair_search_panels():
    plan = read_json(ART / "plan.json")
    models = plan["models"]
    assert models["lambda"]["selected_step"] == 161924 and models["mc"]["selected_step"] == 70842
    assert models["lambda"]["sha256"] == "561f055af56719ad7fbfa9779e93955ce139d5651b6a2b9b9dbc7e10b7182b3b"
    assert models["mc"]["sha256"] == "37cbd0bf0082e530bf303fe56e027b99af63b9f9a7afea15ccf4d236a9c69cec"
    assert models["lambda"]["dataset_digest"] == models["mc"]["dataset_digest"]
    assert models["lambda"]["manifest_sha256"] == models["mc"]["manifest_sha256"]
    common_seeds = None
    for key in ("mc", "r37"):
        panel = plan["panels"][key]
        config = read_json(ROOT / panel["config"]["path"])
        assert panel["pairs"] == 3000 and len(config["game_seeds"]) == 3000
        assert config["policy_a"]["checkpoint"] == models["lambda"]["checkpoint"]
        assert config["policy_b"]["checkpoint"] == models[key]["checkpoint"]
        assert config["metadata"]["opponent_sha256"] == models[key]["sha256"]
        if common_seeds is None:
            common_seeds = config["game_seeds"]
        assert config["game_seeds"] == common_seeds
        x = config["execution"]
        assert config["device"] == "cuda" and x["backend"] == "native_cuda_pool"
        assert x["search_workers"] == 12 and x["total_active_games"] == 1000
        assert x["max_batch_size"] == 500 and x["local_microbatch"] == 128 and x["pipeline_slots"] == 2
        assert x["batch_buckets"] == [64,128,256,500] and x["candidate_buckets"] == [32,48,64]
        for side in ("policy_a", "policy_b"):
            assert config[side]["kind"] == "mcts_checkpoint"
            search = config[side]["search_config"]
            assert search["simulations"] == 128 and search["determinizations"] == 1
            assert search["root_noise_fraction"] == 0 and search["forced_playouts"] == 0
            assert not search["policy_target_pruning"]


@pytest.mark.parametrize("key", ["mc", "r37"])
def test_each_command_resumes_its_own_output_and_has_no_training_stage(key, tmp_path, monkeypatch):
    e = wrapper().configured_entry()
    panel = read_json(ART / "plan.json")["panels"][key]
    monkeypatch.setattr(e, "ROOT", tmp_path)
    command = e.panel_command(panel, tmp_path / "source")
    assert command[3] == "splendor.evaluation" and "--resume" not in command
    Path(str(tmp_path / panel["output"])+".games").mkdir(parents=True)
    assert "--resume" in e.panel_command(panel, tmp_path / "source")
    assert command[command.index("--summary-every")+1] == "500"


@pytest.mark.parametrize("key", ["mc", "r37"])
def test_status_is_read_only_and_does_not_prepare_or_start_eval(key, tmp_path, monkeypatch, capsys):
    w = wrapper()
    plan = {"panels":{key:{"id":"fixture"}}}
    write_json(tmp_path / "plan.json", plan)
    e = w.configured_entry()
    monkeypatch.setattr(e, "panel_status", lambda *_:{"status":"pending", "completed_pairs":0})
    monkeypatch.setattr(w, "ART", tmp_path)
    monkeypatch.setattr(w, "configured_entry", lambda:e)
    monkeypatch.setattr(sys, "argv", ["run.py",key,"status"])
    before = {p:p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert w.main() == 0
    assert before == {p:p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert "pending" in capsys.readouterr().out
