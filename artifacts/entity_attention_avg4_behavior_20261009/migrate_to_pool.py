"""One-time, audited switch of the stopped first collection to root-pool labels.

No games/checkpoints/state are rewritten. Refuse runs that reached labeling or
training; acquire the same supervisor and hybrid locks before updating configs.
"""
from __future__ import annotations

import fcntl
import json
import shutil
from contextlib import ExitStack
from pathlib import Path

from splendor import iteration_suite as suite
from splendor.alphazero_recording import read_json, sha256, write_json
from splendor.hybrid_train import HybridConfig


def main():
    plan = read_json(suite.ROOT / "configs/entity_attention_v1/iteration_avg4_behavior_20261009.json")
    suite.validate_plan(plan)
    artifacts = suite.ROOT / plan["artifact_root"]
    runtime = suite.ROOT / plan["runtime_root"]
    run = suite.ROOT / plan["run_root"] / "seed1_avg4"
    expected = {"protocol": suite.PROTOCOL, "plan": plan, "configs": suite.arm_configs(plan),
                "evaluation_template": suite.evaluation_template(plan), "jobs": list(suite.evaluation_jobs(plan))}
    with ExitStack() as stack:
        for path in (runtime / ".suite.lock", run / ".hybrid.lock"):
            handle = stack.enter_context(path.open("a"))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        old = read_json(artifacts / "manifest.json")
        new_cfg = HybridConfig.from_dict(expected["configs"]["seed1_avg4"]).to_dict()
        assert new_cfg == expected["configs"]["seed1_avg4"]
        receipt = artifacts / "pool_migration.json"
        if old == expected:
            if not receipt.exists() or read_json(run / "config.json") != new_cfg:
                raise ValueError("pool suite/config migration is incomplete")
            print(json.dumps({"status": "already_migrated", "receipt": str(receipt)}))
            return
        state = read_json(run / "hybrid_state.json")
        if state["completed_rounds"] or state["latest_checkpoint"]["sha256"] != plan["start_sha256"]:
            raise ValueError("migration requires the original untrained first round")
        rounds = run / "rounds"
        if any(rounds.glob("round-*/policy_labels")) or any(rounds.glob("round-*/checkpoints/*.pt")):
            raise ValueError("labeling/training already started; use a new branch")
        old_cfg = read_json(run / "config.json")
        if old_cfg != old["configs"]["seed1_avg4"]:
            raise ValueError("old frozen suite/hybrid configs disagree")
        old_without = {k: v for k, v in old_cfg.items() if k != "policy_averaging"}
        new_without = {k: v for k, v in new_cfg.items() if k != "policy_averaging"}
        if old_without != new_without:
            raise ValueError("migration may only change label execution")
        for key in ("samples", "world_seed", "noise_seed", "games_per_block"):
            if old_cfg["policy_averaging"][key] != new_cfg["policy_averaging"][key]:
                raise ValueError(f"label semantics changed: {key}")
        collection = rounds / "round-000/selfplay"
        games = {str(p.relative_to(run)): sha256(p) for p in sorted((collection / "games").glob("game-*.npz"))}
        protected = [collection / "manifest.json", run / "hybrid_state.json", run / "fork_manifest.json",
                     run / "checkpoints/fork_initial.pt"]
        unchanged = {str(p.relative_to(suite.ROOT)): sha256(p) for p in protected}
        snapshot = runtime / "before_pool_switch"
        paths = [artifacts / "manifest.json", artifacts / "training_configs/seed1_avg4.json",
                 artifacts / "verification.json", artifacts / "README.md", run / "config.json"]
        for path in paths:
            target = snapshot / path.relative_to(suite.ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise ValueError(f"migration snapshot already exists: {target}")
            shutil.copy2(path, target)
        record = {"status": "migrated", "scope": "stopped first collection; only unstarted label execution changed",
                  "before_manifest_sha256": sha256(artifacts / "manifest.json"),
                  "old_label_config": old_cfg["policy_averaging"], "new_label_config": new_cfg["policy_averaging"],
                  "snapshot": str(snapshot), "preserved_games": len(games), "archive_sha256": games,
                  "unchanged_sha256": unchanged, "training_steps_added": 0}
        write_json(run / "config.json", new_cfg)
        write_json(artifacts / "training_configs/seed1_avg4.json", new_cfg)
        write_json(artifacts / "manifest.json", expected)
        for path, checksum in unchanged.items():
            assert sha256(suite.ROOT / path) == checksum, path
        assert all(sha256(run / path) == checksum for path, checksum in games.items())
        record["after_manifest_sha256"] = sha256(artifacts / "manifest.json")
        write_json(receipt, record)
        print(json.dumps({"status": "migrated", "preserved_games": len(games), "label_workers": 8,
                          "training_steps_added": 0, "receipt": str(receipt)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
