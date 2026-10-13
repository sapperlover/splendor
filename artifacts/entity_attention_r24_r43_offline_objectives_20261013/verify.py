"""Bounded CPU verification; no formal training, collection or arena."""
from dataclasses import replace
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
import torch

from splendor.alphazero_recording import read_json, sha256, write_json
from splendor.alphazero_replay import PositionPool
from splendor.alphazero_train import draw_training_batch
from splendor.offline_objectives import OfflineObjectiveConfig, digest, initialize_learner
from splendor.offline_train import make_index
from splendor.compact_game_cache import CompactGameCache
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.policy_surprise import PolicySurpriseConfig, PolicySurpriseWeighting, game_policy_surprise
from splendor.selfplay_data import SelfPlaySample, collate_selfplay_samples, load_game
from splendor.value_auxiliary import ValueAuxiliaryConfig, auxiliary_targets

HERE = Path(__file__).resolve().parent


def main():
    torch.set_num_threads(1)
    branches = ("policy_surprise", "value_auxiliary")
    configs = {b: OfflineObjectiveConfig.from_dict(read_json(Path(
        f"configs/entity_attention_v1/offline_r24_r43_{b}_20261013.json"))) for b in branches}
    outputs = {b: Path(f"runs/entity_attention_r24_r43_{b}_offline_20261013") for b in branches}
    datasets = {b: read_json(o / "dataset.json") for b, o in outputs.items()}
    fixed = {b: read_json(o / "fixed_validation.json") for b, o in outputs.items()}
    plans = {b: read_json(o / "plan.json") for b, o in outputs.items()}
    assert datasets[branches[0]] == datasets[branches[1]]
    assert fixed[branches[0]] == fixed[branches[1]]
    assert {p["steps_target"] for p in plans.values()} == {22681}
    records = datasets[branches[0]]["games"]
    selected = {r["round_index"]: r for r in records}
    assert set(selected) == set(range(23, 43))
    checked = []
    for generation, record in sorted(selected.items()):
        path = Path(record["path"])
        assert sha256(path) == record["archive_sha256"]
        arrays, metadata = load_game(path)
        assert metadata["status"] == "natural_terminal"
        samples = [SelfPlaySample(arrays, metadata, i) for i in range(record["decision_count"])]
        score, turns = auxiliary_targets(samples, ValueAuxiliaryConfig())
        assert np.isfinite(score).all() and np.isfinite(turns).all() and np.all(turns >= 0)
        assert np.all(np.diff(turns) <= 1e-7)
        _, weights = game_policy_surprise(arrays, PolicySurpriseConfig())
        np.testing.assert_allclose(weights.sum(), arrays["policy_weight"].sum(), atol=1e-4, rtol=1e-6)
        checked.append({"absolute_round": generation + 1, "game_id": record["game_id"],
                        "archive_sha256": record["archive_sha256"], "positions": len(samples)})
    index = make_index(datasets[branches[0]], 64 * 1024**2)
    index.cache = CompactGameCache(64 * 1024**2)
    pool = PositionPool(index.split_games("train"), index.cache, "verification")
    smoke, references = {}, {}
    runtime = Path("runtime") / HERE.name
    runtime.mkdir(parents=True, exist_ok=True)
    for branch in branches:
        config = configs[branch]
        config = replace(config, learner=replace(config.learner, device="cpu", batch_size=8, history_batch_size=4))
        identity = {"digest": digest({"smoke": branch, "source": config.manifest_sha256}),
                    "policy_surprise_weighting": config.policy_surprise_weighting, "value_auxiliary": config.value_auxiliary}
        learner = initialize_learner(config, identity, 1)
        drawn = draw_training_batch(pool, None, learner.sampling_rng, learner.config)
        references[branch] = [r.to_dict() for r in drawn.references]
        batch = collate_selfplay_samples(drawn.samples)
        kwargs = {}
        if branch == "policy_surprise":
            batch, _ = PolicySurpriseWeighting(PolicySurpriseConfig()).apply(batch, drawn.samples, drawn.references)
        else:
            cfg = ValueAuxiliaryConfig(**config.value_auxiliary)
            kwargs = {"value_auxiliary": cfg, "auxiliary_labels": auxiliary_targets(drawn.samples, cfg)}
        metrics = learner.train_step(batch, references=drawn.references, **kwargs)
        checkpoint = runtime / f"{branch}_cpu_smoke.pt"
        learner.save(checkpoint)
        frozen = load_frozen_actor_critic_checkpoint(checkpoint)
        for a, b in zip(learner.model(batch.decisions), frozen(batch.decisions)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        smoke[branch] = {"updates": 1, "device": "cpu", "batch_size": 8,
                         "metrics": metrics, "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint)}
    assert references[branches[0]] == references[branches[1]]
    suites = list(ET.parse(HERE / "tests.xml").getroot().iter("testsuite"))
    tests = {key: sum(int(s.attrib.get(key, 0)) for s in suites) for key in ("tests", "errors", "failures", "skipped")}
    assert tests["errors"] == tests["failures"] == 0
    result = {"status": "prepared; formal training and strength not run", "passed": True,
              "same_dataset": True, "same_fixed_validation": True, "same_smoke_sampling": True,
              "git_head": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip(),
              "smoke_references": references[branches[0]],
              "plans": plans, "representative_archive_checks": checked, "actual_r43_cpu_smoke": smoke,
              "tests": tests, "verification_command": "conda run --no-capture-output -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python artifacts/entity_attention_r24_r43_offline_objectives_20261013/verify.py",
              "code_hashes": {str(p): sha256(p) for p in map(Path, (
                  "src/splendor/offline_objectives.py", "src/splendor/value_auxiliary.py", "src/splendor/entity_attention.py",
                  "src/splendor/actor_critic.py", "src/splendor/alphazero_train.py", "src/splendor/compact_game_cache.py"))}}
    write_json(HERE / "verification.json", result)
    print({"passed": True, "tests": tests, "games": len(records), "steps_per_branch": 22681})


if __name__ == "__main__":
    main()
