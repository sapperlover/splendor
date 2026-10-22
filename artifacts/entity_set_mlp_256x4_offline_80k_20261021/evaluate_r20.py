"""Evaluate raw R20 with the stopped MLP run's exact frozen holdout protocol."""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import torch

from splendor.alphazero_recording import read_json, sha256, write_json
from splendor.alphazero_train import LearnerConfig
from splendor.compact_game_cache import CompactGameCache
from splendor.offline_objectives import digest, evaluate
from splendor.offline_train import make_index
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint


def main():
    root = Path(__file__).resolve().parents[2]
    art = Path(__file__).resolve().parent
    output = root / "runs" / art.name
    plan = read_json(art / "plan.json")
    config = read_json(output / "config.json")
    dataset = read_json(output / "dataset.json")
    fixed = read_json(output / "fixed_validation.json")
    if dataset["digest"] != digest({k: v for k, v in dataset.items() if k != "digest"}):
        raise ValueError("frozen dataset identity differs")
    if fixed["digest"] != digest(fixed["references"]):
        raise ValueError("fixed validation identity differs")
    checkpoint = root / plan["opponent"]["path"]
    if sha256(checkpoint) != plan["opponent"]["sha256"]:
        raise ValueError("raw R20 checkpoint differs")
    for game in dataset["games"]:
        if game["split"] == "validation":
            stat = Path(game["path"]).stat()
            if (stat.st_size, stat.st_mtime_ns) != (game["size"], game["mtime_ns"]):
                raise ValueError(f"validation archive changed: {game['path']}")
    torch.set_num_threads(1)
    learner_config = LearnerConfig(**config["learner"])
    model = load_frozen_actor_critic_checkpoint(checkpoint, device=learner_config.device)
    learner = SimpleNamespace(model=model, device=torch.device(learner_config.device), config=learner_config)
    index = make_index(dataset, config["cache_mib"] * 1024**2)
    index.cache = CompactGameCache(config["cache_mib"] * 1024**2)
    record = {
        "status": "running", "model": "raw R20", "checkpoint": plan["opponent"],
        "dataset_digest": dataset["digest"], "fixed_validation_digest": fixed["digest"],
        "model_config": asdict(model.config), "command": sys.argv,
        "executable": sys.executable, "torch": torch.__version__,
        "device": str(learner.device), "script_sha256": sha256(Path(__file__)),
        "evaluator": {"path": evaluate.__code__.co_filename,
                      "sha256": sha256(Path(evaluate.__code__.co_filename))},
        "source_record_sha256": sha256(art / "source.json"),
        "scope": "No training or matches; identical archived targets, policy masks, batch size and metric formulas as the MLP run.",
    }
    result_path = art / "r20_same_holdout.json"
    tick = time.perf_counter()
    for name, refs in (("fixed_validation", fixed), ("full_validation", None)):
        start = time.perf_counter()
        metrics = evaluate(learner, index, refs, None)
        expected_positions = fixed["positions"] if refs else plan["splits"]["validation"]["positions"]
        if metrics["positions"] != expected_positions:
            raise ValueError("validation population differs")
        if refs is None and metrics["policy_positions"] != plan["splits"]["validation"]["policy_positions"]:
            raise ValueError("positive policy population differs")
        record[name] = {"metrics": metrics, "seconds": time.perf_counter() - start}
        write_json(result_path, record)
        print(json.dumps({"scope": name, "KL": metrics["policy_kl"], "MSE": metrics["value_loss"],
                          "positions": metrics["positions"]}), flush=True)
    record.update(status="complete", seconds=time.perf_counter() - tick)
    write_json(result_path, record)


if __name__ == "__main__":
    main()
