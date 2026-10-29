"""Six online rounds share the complete validation cohort of the previous six."""
from __future__ import annotations

import argparse
import fcntl
from pathlib import Path

import torch

from .alphazero_recording import read_json, sha256, write_json
from .fixed_holdout import FixedHoldout, PROTOCOL, check_disjoint, read_manifest
from .offline_train import _digest
from .opponent_pool import load_frozen_actor_critic_checkpoint


def check_settings(settings):
    if settings.get("block_rounds") != 6:
        raise ValueError("six-round holdout requires block_rounds=6")
    return read_manifest(settings)


class HoldoutSchedule:
    """Keep only the current cohort loaded; freeze each block independently."""

    def __init__(self, settings, output, device, *, six_rounds=False):
        if six_rounds:
            check_settings(settings)
        self.settings, self.output, self.device = settings, Path(output), torch.device(device)
        self.six_rounds = six_rounds
        self.start = 1
        self.current = FixedHoldout(settings, self.output, device)
        if six_rounds:
            path = self.output / "fixed_holdout/schedule.json"
            identity = {"protocol": "previous-six-round-holdout-v1", "block_rounds": 6,
                "initial_manifest": settings["manifest"],
                "rule": "local1-6 use initial manifest; local7-12 use local1-6 validation; repeat every6"}
            if path.exists() and read_json(path) != identity:
                raise ValueError("six-round holdout schedule changed on resume")
            if not path.exists():
                write_json(path, identity)

    def for_iteration(self, iteration):
        if iteration < 1:
            raise ValueError("holdout iteration must be positive")
        start = (iteration-1)//6*6+1 if self.six_rounds else 1
        if start == self.start:
            return self.current
        if start == 1:
            settings, destination = self.settings, self.output
        else:
            destination = self.output / "fixed_holdout/blocks" / f"iterations-{start:03d}-{start+5:03d}"
            manifest_path = destination / "manifest.json"
            if manifest_path.exists():
                frozen = read_json(manifest_path)
                if (frozen["source_iterations"] != list(range(start-6, start))
                        or frozen["evaluated_iterations"] != list(range(start, start+6))):
                    raise ValueError("six-round holdout block boundary changed")
                for source in frozen["sources"]:
                    if sha256(Path(source["path"])) != source["sha256"]:
                        raise ValueError("six-round holdout collection index changed")
            else:
                records, sources = [], []
                # Reading original collection indices preserves games retired from replay.
                for source_iteration in range(start-6, start):
                    path = self.output / "rounds" / f"round-{source_iteration-1:03d}" / "collection_index.json"
                    sources.append({"iteration": source_iteration, "path": str(path.resolve()), "sha256": sha256(path)})
                    games = read_json(path)["games"]
                    records.extend({**g, "source_iteration": source_iteration,
                        "archive_sha256": sha256(Path(g["path"]))} for g in games if g["split"] == "validation")
                frozen = {"protocol": PROTOCOL, "source_iterations": list(range(start-6, start)),
                    "evaluated_iterations": list(range(start, start+6)), "sources": sources, "games": records,
                    "selection": "all original validation from previous six online rounds, fixed within this six-round block",
                    "games_count": len(records), "value_positions": sum(g["decision_count"] for g in records),
                    "policy_positions": sum(g["policy_samples"] for g in records)}
                frozen["digest"] = _digest(frozen)
                write_json(manifest_path, frozen)
            settings = {"manifest": {"path": str(manifest_path.resolve()), "sha256": sha256(manifest_path)},
                        "batch_size": self.settings["batch_size"]}
        # Release the previous CPU panel before the next panel is decoded.
        self.current = FixedHoldout(settings, destination, self.device)
        self.start = start
        return self.current

    def backfill(self, completed, steps):
        devices = [self.device.index or 0] if self.device.type == "cuda" else []
        for row in completed:
            observer = self.for_iteration(row["iteration"])
            snapshot = Path(row["snapshot"]["path"])
            if sha256(snapshot) != row["snapshot"]["sha256"]:
                raise ValueError("backfill replay snapshot changed")
            check_disjoint(observer.manifest, read_json(snapshot)["games"])
            stage = self.output / "rounds" / f"round-{row['iteration']-1:03d}"
            for step in steps:
                checkpoint = stage / "checkpoints" / ("initial.pt" if step == 0 else f"step-{step:09d}.pt")
                with torch.random.fork_rng(devices=devices):
                    model = load_frozen_actor_critic_checkpoint(checkpoint, device=self.device)
                    observer.observe(model, checkpoint, row["iteration"], step)
                del model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from .random_budget_iteration import Config
    config = Config.from_dict(read_json(args.config))
    if config.six_round_holdout is None:
        parser.error("config must enable six_round_holdout")
    torch.set_num_threads(1)
    output = args.output.resolve()
    with (output / ".iteration.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        schedule = HoldoutSchedule(config.six_round_holdout, output, config.learner.device, six_rounds=True)
        schedule.backfill(read_json(output / "state.json")["completed_rounds"], config.validation_steps)


if __name__ == "__main__":
    main()
