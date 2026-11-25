import copy
import importlib.util
from pathlib import Path
import sys

import pytest

from splendor.alphazero_recording import read_json,write_json

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT / "artifacts/lambda_online_r3_comparisons_20261125"


def wrapper():
    spec=importlib.util.spec_from_file_location("lambda_r3_test",ART / "run.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_fixed_actual_used_models_and_round3_3000_step_are_distinguished():
    p=read_json(ART / "plan.json");m=p["models"]
    assert (m["r3_used"]["local_train_step"],m["r3_used"]["total_train_step"])==(10000,191924)
    assert m["r3_used"]["sha256"]=="8b6798463c23feb1745faaff0364b9de7a92e27ba825d507063355309e6c1667"
    assert (m["r1"]["local_train_step"],m["r1"]["total_train_step"])==(10000,171924)
    assert m["lambda"]["total_train_step"]==161924
    assert (m["r3_step3000"]["local_train_step"],m["r3_step3000"]["total_train_step"])==(3000,184924)
    assert m["r3_step3000"]["replay_digest"]==m["r3_used"]["replay_digest"]
    assert m["r3_step3000"]["sha256"]!=m["r3_used"]["sha256"]
    for key in ("r1","r3_used"):
        evidence=p["actual_use_evidence"][key]
        assert evidence["committed_checkpoint"]==evidence["next_collection_teacher"]
        assert evidence["committed_checkpoint"]["sha256"]==m[key]["sha256"]


def test_all_three_panels_share_seed_and_fpu_recipe_on_both_sides():
    plan=read_json(ART / "plan.json");normalized=[]
    assert set(plan["panels"])=={"lambda","r1","r3_step3000"}
    for key,panel in plan["panels"].items():
        config=read_json(ROOT / panel["config"]["path"])
        assert panel["pairs"]==2000 and len(config["game_seeds"])==2000
        assert config["policy_a"]["checkpoint"]==plan["models"]["r3_used"]["checkpoint"]
        assert config["policy_b"]["checkpoint"]==plan["models"][key]["checkpoint"]
        assert config["policy_a"]["search_config"]==config["policy_b"]["search_config"]
        for side in ("policy_a","policy_b"):
            s=config[side]["search_config"]
            assert s["simulations"]==128 and s["determinizations"]==1
            assert s["fpu_reduction"]==.3 and s["c_puct"]==1.
            assert s["root_noise_fraction"]==0 and s["forced_playouts"]==0 and not s["policy_target_pruning"]
        x=config["execution"]
        assert x["search_workers"]==12 and x["total_active_games"]==1000
        assert x["max_batch_size"]==500 and x["local_microbatch"]==128 and x["pipeline_slots"]==2
        assert x["batch_buckets"]==[64,128,256,500] and x["candidate_buckets"]==[32,48,64]
        q=copy.deepcopy(config);q.pop("metadata");q["policy_b"].pop("checkpoint");q["policy_b"].pop("name")
        normalized.append(q)
    assert normalized[0]==normalized[1]==normalized[2]


@pytest.mark.parametrize("key",["lambda","r1","r3_step3000"])
def test_each_script_resumes_only_its_panel_and_status_is_read_only(key,tmp_path,monkeypatch,capsys):
    w=wrapper();entry=w.configured_entry();panel=read_json(ART / "plan.json")["panels"][key]
    monkeypatch.setattr(entry,"ROOT",tmp_path)
    cmd=entry.panel_command(panel,tmp_path / "source")
    assert cmd[3]=="splendor.evaluation" and "--resume" not in cmd
    Path(str(tmp_path / panel["output"])+".games").mkdir(parents=True)
    assert "--resume" in entry.panel_command(panel,tmp_path / "source")
    write_json(tmp_path / "plan.json",{"panels":{key:panel}})
    monkeypatch.setattr(w,"ART",tmp_path);monkeypatch.setattr(w,"configured_entry",lambda:entry)
    monkeypatch.setattr(entry,"panel_status",lambda *_:{"status":"pending","completed_pairs":0})
    monkeypatch.setattr(sys,"argv",["run.py",key,"status"])
    before={p:p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert w.main()==0
    assert before=={p:p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert "pending" in capsys.readouterr().out
