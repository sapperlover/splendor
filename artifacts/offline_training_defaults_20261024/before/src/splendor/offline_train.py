"""Bounded offline AZ fitting over existing, frozen whole-game splits."""
from __future__ import annotations

import argparse
import bisect
import fcntl
import hashlib
import json
import math
import shutil
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from .actor_critic import build_actor_critic, model_config_from_dict
from .alphazero_checkpoint import load_alphazero_checkpoint
from .alphazero_recording import (ReferenceWriter, append_json, read_json, sha256,
                                  source_identity, summarize_references, write_json)
from .alphazero_replay import PositionPool, ReplayGame, ReplayIndex
from .alphazero_train import AlphazeroLearner, LearnerConfig, metric_ratios, run_training_round
from .rng import DeterministicRng


@dataclass(frozen=True)
class OfflineConfig:
    source_run: str
    source_artifacts: str
    model_config: dict[str, Any]
    max_steps: int
    rounds: tuple[int, ...] = tuple(range(10))
    initialization_seed: int = 13100000
    sampling_seed: int = 13110000
    validation_seed: int = 13120000
    validation_positions: int = 4096
    replay_cache_bytes: int = 4 * 1024**3
    games_per_round: int | None = None
    full_validation_at_end: bool = True
    learner: LearnerConfig = field(default_factory=lambda: LearnerConfig(checkpoint_every_steps=1000))

    def __post_init__(self):
        if not self.rounds or tuple(sorted(set(self.rounds))) != self.rounds or any(r < 0 for r in self.rounds):
            raise ValueError("rounds must be sorted, unique, nonnegative and nonempty")
        for name in ("max_steps", "validation_positions", "replay_cache_bytes"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.validation_positions > 4096:
            raise ValueError("fixed validation subset is limited to 4096 positions")
        if self.games_per_round is not None and (type(self.games_per_round) is not int or self.games_per_round <= 0):
            raise ValueError("games_per_round must be positive or null")
        if self.learner.max_steps_per_round is not None:
            raise ValueError("offline budget is max_steps, not online max_steps_per_round")
        if type(self.full_validation_at_end) is not bool:
            raise ValueError("full_validation_at_end must be bool")

    @classmethod
    def from_dict(cls, raw):
        values = dict(raw)
        values["rounds"] = tuple(values.get("rounds", range(10)))
        values["learner"] = LearnerConfig(**values.get("learner", {"checkpoint_every_steps": 1000}))
        return cls(**values)


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def freeze_dataset(config: OfflineConfig) -> dict[str, Any]:
    """Reuse completed-run metadata/hashes; neither decode nor replay every archive."""
    source = Path(config.source_run)
    completed = read_json(source / "loop_state.json")["completed_rounds"]
    missing = sorted(set(config.rounds) - set(completed))
    if missing:
        raise ValueError(f"source rounds are incomplete: {missing}")
    records, sources = [], []
    for round_index in config.rounds:
        round_root = source / "rounds" / f"round-{round_index:03d}"
        summary = read_json(round_root / "selfplay/summary.json")
        if summary["completed_games"] != summary["expected_games"]:
            raise ValueError(f"source collection incomplete: {round_root}")
        snapshot_path = round_root / "replay_snapshot.json"
        dataset_path = Path(config.source_artifacts) / round_root.name / "dataset.json"
        snapshot = read_json(snapshot_path)
        existing = {g["game_id"]: g for g in read_json(dataset_path)["games"]}
        games = [g for g in snapshot["games"] if g["round_index"] == round_index]
        if len(games) != summary["completed_games"]:
            raise ValueError(f"source snapshot lacks current-round games: {round_root}")
        if config.games_per_round is not None:
            games = games[:config.games_per_round]
        for game in games:
            original = existing[game["game_id"]]
            if any(original[k] != game[k] for k in ("round_index", "split", "decision_count", "run_id")):
                raise ValueError("snapshot and existing dataset metadata disagree")
            if original["status"] != "natural_terminal":
                raise ValueError("offline source must be a complete natural-terminal game")
            stat = Path(game["path"]).stat()
            records.append({**game, "archive_sha256": original["archive_sha256"],
                            "checkpoint_sha256": original["checkpoint_sha256"],
                            "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
        for path in (snapshot_path, dataset_path, round_root / "selfplay/manifest.json"):
            sources.append({"path": str(path.resolve()), "sha256": sha256(path)})
    result = {"protocol": "az-offline-data-v1", "rounds": list(config.rounds), "games": records, "sources": sources,
              "scope": "original whole-game splits; all selected rounds, no online replay eviction"}
    result["digest"] = _digest(result)
    return result


def make_index(dataset, cache_bytes) -> ReplayIndex:
    fields = ("round_index", "run_id", "game_id", "split", "decision_count", "policy_samples")
    return ReplayIndex([ReplayGame(**{k: g[k] for k in fields}, path=Path(g["path"])) for g in dataset["games"]], cache_bytes=cache_bytes)


def freeze_validation(index: ReplayIndex, config: OfflineConfig) -> dict[str, Any]:
    grouped = {r: [g for g in index.split_games("validation") if g.round_index == r] for r in config.rounds}
    rng = DeterministicRng(config.validation_seed)
    selections = {r: set() for r in grouped}
    sizes = {r: sum(g.decision_count for g in games) for r, games in grouped.items()}
    target = min(config.validation_positions, sum(sizes.values()))
    if not target:
        raise ValueError("selected source contains no validation positions")
    # Balance across source rounds; exhausted rounds donate their remaining quota.
    while sum(map(len, selections.values())) < target:
        for r, selected in selections.items():
            if len(selected) == sizes[r] or sum(map(len, selections.values())) == target:
                continue
            position = rng.randbelow(sizes[r])
            while position in selected:
                position = rng.randbelow(sizes[r])
            selected.add(position)
    references = []
    for r, games in grouped.items():
        starts = [0]
        for game in games:
            starts.append(starts[-1] + game.decision_count)
        for position in sorted(selections[r]):
            i = bisect.bisect_right(starts, position) - 1
            references.append({"round_index": r, "game_id": games[i].game_id, "decision_index": position - starts[i]})
    return {"references": references, "positions": len(references), "seed": config.validation_seed,
            "composition_by_round": dict(Counter(str(ref["round_index"]) for ref in references)), "digest": _digest(references)}


def evaluate_validation(learner: AlphazeroLearner, index: ReplayIndex, fixed=None, *, policy_transform=None):
    games = {g.key: g for g in index.split_games("validation")}
    if fixed is not None:
        samples = [index.cache.sample(games[(ref["round_index"], ref["game_id"])], ref["decision_index"])
                   for ref in fixed["references"]]
        return learner.evaluate_samples([policy_transform(s) for s in samples] if policy_transform else samples)
    sums = Counter()
    # Stream full holdout; never retain every decoded game in a Python list.
    fields = ("policy_ce_sum", "policy_weight_sum", "value_sq_sum", "value_weight_sum", "entropy_sum", "positions",
              "target_entropy_sum", "policy_positions", "value_positions", "value_target_sum", "value_target_sq_sum")
    for game in games.values():
        for start in range(0, game.decision_count, learner.config.validation_batch_size):
            samples = [index.cache.sample(game, i) for i in range(start, min(
                game.decision_count, start + learner.config.validation_batch_size))]
            metrics = learner.evaluate_samples([policy_transform(s) for s in samples] if policy_transform else samples)
            sums.update({key: metrics[key] for key in fields})
    return {"available": bool(sums["positions"]), **sums, **metric_ratios(sums)}


def _copy_checkpoint(source: Path, destination: Path):
    temporary = destination.with_suffix(".pt.tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)


def run_offline(config: OfflineConfig, output: Path, *, resume=False, max_steps_this_session=None,
                replay_cache_bytes_override: int | None = None):
    """Resume the frozen experiment with an optional execution-only LRU budget."""
    if replay_cache_bytes_override is not None and (type(replay_cache_bytes_override) is not int
                                                   or replay_cache_bytes_override <= 0):
        raise ValueError("replay cache override must be a positive integer byte count")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run_offline(config, output, resume=resume, max_steps_this_session=max_steps_this_session,
                            replay_cache_bytes_override=replay_cache_bytes_override)


def _run_offline(config, output, *, resume, max_steps_this_session, replay_cache_bytes_override):
    started = time.perf_counter()
    torch.set_num_threads(1)
    frozen_config = json.loads(json.dumps(asdict(config)))
    config_path, data_path = output / "config.json", output / "dataset.json"
    if config_path.exists():
        if not resume:
            raise ValueError("offline run exists; use --resume or a new output")
        if read_json(config_path) != frozen_config:
            raise ValueError("offline resume requires the identical frozen configuration")
        dataset = read_json(data_path)
        for source in dataset["sources"]:
            if sha256(Path(source["path"])) != source["sha256"]:
                raise ValueError(f"frozen source metadata changed: {source['path']}")
        for game in dataset["games"]:
            stat = Path(game["path"]).stat()
            if stat.st_size != game["size"] or stat.st_mtime_ns != game["mtime_ns"]:
                raise ValueError(f"frozen source archive changed: {game['path']}")
    else:
        if resume:
            raise ValueError("cannot resume an uninitialized offline run")
        dataset = freeze_dataset(config)
        write_json(data_path, dataset)
        write_json(config_path, frozen_config)
    if dataset["digest"] != _digest({key: value for key, value in dataset.items() if key != "digest"}):
        raise ValueError("frozen dataset identity mismatch")
    cache_bytes = config.replay_cache_bytes if replay_cache_bytes_override is None else replay_cache_bytes_override
    index = make_index(dataset, cache_bytes)
    print(f"离线 LRU 容量 {cache_bytes / 1024**2:g} MiB（冻结配置 {config.replay_cache_bytes / 1024**2:g} MiB）", flush=True)
    pool = PositionPool(index.split_games("train"), index.cache, "offline-all-train")
    if not len(pool):
        raise ValueError("selected data has no training positions")
    identity = {**index.identity(), "digest": dataset["digest"], "sampling": "all train positions uniformly with replacement"}
    fixed_path = output / "fixed_validation.json"
    if not fixed_path.exists():
        write_json(fixed_path, freeze_validation(index, config))
    fixed = read_json(fixed_path)
    if fixed["digest"] != _digest(fixed["references"]):
        raise ValueError("frozen validation references identity mismatch")
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    steps = sorted(checkpoints.glob("step-*.pt"))
    if resume and steps:
        learner = AlphazeroLearner.resume(steps[-1], config.learner, replay_identity=identity,
                                          round_index=0, round_steps_target=config.max_steps)
        payload = load_alphazero_checkpoint(steps[-1])
        if payload.model_config != asdict(model_config_from_dict(config.model_config)):
            raise ValueError("offline checkpoint architecture disagrees with configuration")
    elif resume and (checkpoints / "initial.pt").exists():
        learner = AlphazeroLearner.resume(checkpoints / "initial.pt", config.learner, replay_identity=identity,
                                          round_index=0, round_steps_target=config.max_steps)
    else:
        torch.manual_seed(config.initialization_seed)
        learner = AlphazeroLearner(build_actor_critic(config.model_config), config.learner,
                                  sampling_rng=DeterministicRng(config.sampling_seed))
        learner.replay_identity = identity
        learner.round_steps_target = config.max_steps
        learner.parent_checkpoint = {"kind": "random_initialization", "seed": config.initialization_seed,
                                     "model_config": asdict(learner.model.config)}
        learner.save(checkpoints / "initial.pt")
    # Truncate uncommitted curve points using the learner's checkpoint boundary.
    curve_path = output / "validation_curve.jsonl"
    curve = [json.loads(line) for line in curve_path.read_text().splitlines()] if curve_path.exists() else []
    curve = [point for point in curve if point["step"] <= learner.round_train_step]
    curve_path.write_text("".join(json.dumps(point, sort_keys=True) + "\n" for point in curve))

    def validate_and_publish(learner, checkpoint):
        tick = time.perf_counter()
        metrics = evaluate_validation(learner, index, fixed)
        point = {"step": learner.round_train_step, "checkpoint": str(checkpoint.resolve()),
                 "set_digest": fixed["digest"], "metrics": metrics, "seconds": time.perf_counter() - tick}
        curve.append(point)
        append_json(curve_path, point)
        write_json(output / "best_validation_kl.json", min(curve, key=lambda item: item["metrics"]["policy_kl"]))
        _copy_checkpoint(checkpoint, checkpoints / "latest.pt")
        print(f"离线验证 step={point['step']}/{config.max_steps} KL={metrics['policy_kl']:.5f} value MSE={metrics['value_loss']:.5f}", flush=True)

    if not curve or curve[-1]["step"] != learner.round_train_step:
        validate_and_publish(learner, steps[-1] if steps else checkpoints / "initial.pt")
    else:
        write_json(output / "best_validation_kl.json", min(curve, key=lambda item: item["metrics"]["policy_kl"]))
        _copy_checkpoint(steps[-1] if steps else checkpoints / "initial.pt", checkpoints / "latest.pt")
    session_id = str(time.time_ns())
    provenance = source_identity(Path.cwd(), output / "source" / session_id)
    write_json(output / "source" / session_id / "identity.json", {**provenance, "command": sys.argv,
               "torch": torch.__version__, "device": config.learner.device, "model_config": asdict(learner.model.config),
               "cuda_device": torch.cuda.get_device_name(learner.device) if learner.device.type == "cuda" else None,
               "replay_cache_capacity_bytes": cache_bytes,
               "parameter_count": learner.model.trainable_parameter_count()})
    writer = ReferenceWriter(output / "usage", session_id, config.learner.batch_size, identity)
    try:
        result = run_training_round(learner, steps_target=config.max_steps, current_pool=pool, history_pool=None,
            validation_samples=(), validation_games=0, checkpoint_dir=checkpoints,
            metrics_path=output / "training_metrics.jsonl", reference_writer=writer,
            checkpoint_callback=validate_and_publish, max_steps_this_session=max_steps_this_session,
            progress_callback=lambda row: print(f"离线训练 {row['round_train_step']}/{config.max_steps} loss={row['total_loss']:.5f}", flush=True)
                if row['round_train_step'] % min(100, config.learner.checkpoint_every_steps) == 0 else None)
    finally:
        writer.close()
        append_json(output / "sessions.jsonl", {"session_id": session_id, "seconds": time.perf_counter() - started,
                    "scope": "setup, training and fixed-subset validation; excludes final holdout and usage summary",
                    "replay_cache_capacity_bytes": cache_bytes,
                    "step": learner.round_train_step, "peak_allocated_bytes": torch.cuda.max_memory_allocated()
                    if config.learner.device == "cuda" else None,
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved() if config.learner.device == "cuda" else None})
    complete = learner.round_train_step == config.max_steps
    if complete:
        _copy_checkpoint(checkpoints / f"step-{config.max_steps:09d}.pt", checkpoints / "final.pt")
        if config.full_validation_at_end and not (output / "full_validation.json").exists():
            tick = time.perf_counter()
            metrics = evaluate_validation(learner, index)
            write_json(output / "full_validation.json", {**metrics, "seconds": time.perf_counter() - tick,
                       "dataset_digest": dataset["digest"], "step": learner.round_train_step})
    usage = summarize_references(output / "usage", learner.round_train_step, max(config.rounds) + 1,
                                 {"current": 0, "history": len(pool)})
    if usage.get("history"):
        usage["all_train"] = usage.pop("history")
        usage.pop("current")
    write_json(output / "usage.json", usage)
    result.update(complete=complete, train_positions=len(pool), average_draw_exposure=learner.round_train_step * config.learner.batch_size / len(pool),
                  session_id=session_id, session_total_seconds=time.perf_counter() - started,
                  model_config=asdict(learner.model.config), parameter_count=learner.model.trainable_parameter_count(),
                  validation=curve[-1]["metrics"], replay_cache=index.cache.stats(), dataset_digest=dataset["digest"])
    write_json(output / "training.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-steps-this-session", type=int)
    parser.add_argument("--replay-cache-mib", type=int,
                        help="execution-only LRU capacity in MiB; may change on resume without changing frozen training settings")
    args = parser.parse_args()
    result = run_offline(OfflineConfig.from_dict(read_json(args.config)), args.output,
                         resume=args.resume, max_steps_this_session=args.max_steps_this_session,
                         replay_cache_bytes_override=None if args.replay_cache_mib is None else args.replay_cache_mib * 1024**2)
    print(json.dumps({"complete": result["complete"], "steps": result["steps_completed"], "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
