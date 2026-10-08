"""Audit the two recorded initialize-only calls; no collection or training."""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from splendor.alphazero_checkpoint import load_alphazero_checkpoint
from splendor.alphazero_recording import read_json, sha256, source_identity, write_json
from splendor.hybrid_train import HybridConfig

ROOT = Path(__file__).resolve().parents[2]
ART = Path(__file__).resolve().parent
PARENT = ROOT / "runs/entity_attention_hybrid_static_graph_seed1"
RUNTIME = ROOT / "runtime/entity_attention_hybrid_fork_validation_20261005"


def record(path):
    return {"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size}


def main():
    source = source_identity(ROOT, RUNTIME / "delivery_source")
    parent_state = read_json(PARENT / "hybrid_state.json")
    parent_cp = PARENT / "rounds/round-008/checkpoints/final.pt"
    parent_hash = sha256(parent_cp)
    assert parent_hash == "af3b2b64007705cc9adcbb3e8a6ad68239f4dc967c31c515e857b1240479e44a"
    assert parent_state["completed_rounds"] == list(range(9))
    assert parent_state["latest_checkpoint"]["sha256"] == parent_hash
    rows, sessions, configurations = [], [], []
    for simulations in (512, 1024):
        cfg_path = ROOT / f"configs/entity_attention_v1/hybrid_fork_s{simulations}_n2.json"
        cfg = HybridConfig.from_dict(read_json(cfg_path))
        configurations.append(record(cfg_path))
        output = RUNTIME / f"final_s{simulations}_n2"
        state, fork = read_json(output / "hybrid_state.json"), read_json(output / "fork_manifest.json")
        assert read_json(output / "config.json") == cfg.to_dict()
        assert state["completed_rounds"] == [] and state["online_round_offset"] == 9
        assert state["latest_checkpoint"]["total_train_step"] == 9500
        assert not (output / "rounds").exists() and not (output / "offline/checkpoints").exists()
        assert state["fork_manifest_sha256"] == sha256(output / "fork_manifest.json")
        assert sha256(output / "checkpoints/fork_initial.pt") == parent_hash
        for relative in ("checkpoints/theta_5000.pt", "offline/dataset.json", "offline/fixed_validation.json",
                         "fixed_online_dataset.json", "fixed_online_validation.json"):
            assert sha256(output / relative) == sha256(PARENT / relative)
        assert len(fork["games"]) == 2000 and {g["round_index"] for g in fork["games"]} == {7, 8}
        for game in fork["games"]:
            assert sha256(Path(game["path"])) == game["archive_sha256"]
        payload = load_alphazero_checkpoint(output / "checkpoints/fork_initial.pt")
        steps = {int(p["step"]) for p in payload.optimizer_state_dict["state"].values()}
        assert steps == {9500} and payload.total_train_step == 9500 and payload.round_index == 9
        identity_paths = list((output / "source").glob("*/identity.json"))
        assert len(identity_paths) == 1
        identity = read_json(identity_paths[0])
        assert identity["source_sha256"] == source["source_sha256"]
        target = ART / "recording" / f"s{simulations}_n2_identity.json"
        write_json(target, identity)
        sessions.append({"command": identity["command"], "identity": record(target),
                         "runtime_state": record(output / "hybrid_state.json"),
                         "runtime_fork_manifest": record(output / "fork_manifest.json")})
        rows.append({"simulations": simulations, "determinizations": 2, "global_step": 9500,
                     "optimizer_parameter_states": len(payload.optimizer_state_dict["state"]),
                     "optimizer_steps": sorted(steps), "inherited_games": len(fork["games"]),
                     "inherited_split_counts": dict(Counter(g["split"] for g in fork["games"])),
                     "inherited_train_positions": sum(g["decision_count"] for g in fork["games"] if g["split"] == "train"),
                     "collected_games": 0, "optimizer_updates": 0})
    suites = ET.parse(ART / "cpu_tests.xml").getroot().findall("testsuite")
    tests = {key: sum(int(s.attrib[key]) for s in suites) for key in ("tests", "failures", "errors", "skipped")}
    assert tests == {"tests": 37, "failures": 0, "errors": 0, "skipped": 0}
    result = {"passed": True, "scope": "37 CPU regressions plus two final-source initialize-only CLI calls with the real theta_9500; no formal branch training",
              "cpu_tests": tests, "branches": rows, "parent_checkpoint": record(parent_cp),
              "parent_state": record(PARENT / "hybrid_state.json"), "configurations": configurations,
              "sessions": sessions, "source": source,
              "tests_command": "conda run --no-capture-output -n splendor env PYTHONPATH=src python -m pytest -q tests/test_hybrid_training.py tests/test_alphazero_training.py tests/test_alphazero_loop.py tests/test_alphazero_replay.py --tb=short --junitxml=artifacts/entity_attention_hybrid_fork_20261005/cpu_tests.xml"}
    write_json(ART / "verification.json", result)
    print(json.dumps({"passed": True, "tests": tests, "branches": rows}, ensure_ascii=False))


if __name__ == "__main__":
    main()
