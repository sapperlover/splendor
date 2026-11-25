import copy
import importlib.util
from pathlib import Path
import sys

import pytest

from splendor.alphazero_recording import read_json, write_json

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT / "artifacts/lambda161924_vs_l12_fpu2000_20261123"


def wrapper():
    spec=importlib.util.spec_from_file_location("lambda_l12_test",ART / "run.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_uses_prior_highest_point_recipe_and_fixed_lambda_l12_identities():
    plan=read_json(ART / "plan.json")
    history=read_json(ROOT / "artifacts/search_fpu_cpuct_l12_20261104/summary.json")
    assert plan["prior_best_fpu"]["panel"]==max(history["panels"],key=lambda p:p["score"])["id"]
    assert plan["prior_best_fpu"]["score"]==.5575
    assert plan["models"]["lambda"]["sha256"]=="561f055af56719ad7fbfa9779e93955ce139d5651b6a2b9b9dbc7e10b7182b3b"
    assert plan["models"]["lambda"]["total_train_step"]==161924
    assert plan["models"]["l12"]["sha256"]=="2a6304e3470cab3dad78ec0a2789ee892892e741f3731cafc2197cde9bd2d45d"
    assert plan["models"]["l12"]["checkpoint"].endswith("round-011/checkpoints/final.pt")


def test_two_panels_only_change_the_requested_candidate_search_recipe():
    plan=read_json(ART / "plan.json")
    configs={k:read_json(ROOT / p["config"]["path"]) for k,p in plan["panels"].items()}
    fpu,plain=configs["fpu"],configs["nofpu"]
    assert fpu["policy_a"]["search_config"]["fpu_reduction"]==.3
    assert fpu["policy_a"]["search_config"]["c_puct"]==1.
    assert plain["policy_a"]["search_config"]["fpu_reduction"] is None
    assert plain["policy_a"]["search_config"]["c_puct"]==1.5
    for key,config in configs.items():
        assert len(config["game_seeds"])==2000 and plan["panels"][key]["pairs"]==2000
        assert config["policy_b"]["search_config"]["fpu_reduction"] is None
        assert config["policy_b"]["search_config"]["c_puct"]==1.5
        assert config["execution"]["search_workers"]==12
        assert config["execution"]["total_active_games"]==1000 and config["execution"]["pipeline_slots"]==2
        for side in ("policy_a","policy_b"):
            s=config[side]["search_config"]
            assert s["simulations"]==128 and s["determinizations"]==1
            assert s["root_noise_fraction"]==0 and s["forced_playouts"]==0 and not s["policy_target_pruning"]
    a,b=copy.deepcopy(fpu),copy.deepcopy(plain)
    for config in (a,b):
        config.pop("metadata")
        config["policy_a"].pop("name")
        config["policy_a"]["search_config"].pop("fpu_reduction")
        config["policy_a"]["search_config"].pop("c_puct")
    assert a==b


@pytest.mark.parametrize("key",["fpu","nofpu"])
def test_independent_output_resume_and_readonly_status(key,tmp_path,monkeypatch,capsys):
    w=wrapper();e=w.configured_entry()
    panel=read_json(ART / "plan.json")["panels"][key]
    monkeypatch.setattr(e,"ROOT",tmp_path)
    command=e.panel_command(panel,tmp_path / "source")
    assert "--resume" not in command and command[3]=="splendor.evaluation"
    Path(str(tmp_path / panel["output"])+".games").mkdir(parents=True)
    assert "--resume" in e.panel_command(panel,tmp_path / "source")
    write_json(tmp_path / "plan.json",{"panels":{key:panel}})
    monkeypatch.setattr(w,"ART",tmp_path)
    monkeypatch.setattr(w,"configured_entry",lambda:e)
    monkeypatch.setattr(e,"panel_status",lambda *_:{"status":"pending","completed_pairs":0})
    monkeypatch.setattr(sys,"argv",["run.py",key,"status"])
    before={p:p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert w.main()==0
    assert before=={p:p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert "pending" in capsys.readouterr().out
