"""Warm-start fixed-history offline policy surprise or value auxiliary fitting."""
from __future__ import annotations

import argparse
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

import torch

from .alphazero_checkpoint import load_alphazero_checkpoint, move_optimizer_state
from .alphazero_recording import ReferenceWriter, append_json, read_json, sha256, source_identity, write_json
from .alphazero_train import (AlphazeroLearner, LearnerConfig, _torch_batch, alphazero_loss_sums,
                             label_statistics, metric_ratios, run_training_round)
from .alphazero_replay import PositionPool
from .compact_game_cache import CompactGameCache
from .offline_train import OfflineConfig, freeze_validation, make_index
from .policy_surprise import PolicySurpriseConfig
from .rng import DeterministicRng
from .selfplay_data import collate_selfplay_samples
from .value_auxiliary import ValueAuxiliaryConfig, add_auxiliary_heads, auxiliary_targets, extend_optimizer_state


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class OfflineObjectiveConfig:
    manifest: str
    manifest_sha256: str
    initial_checkpoint: str
    initial_checkpoint_sha256: str
    first_round: int = 24
    last_round: int = 43
    expected_games_per_round: int = 1000
    draw_exposures: float = 4.0
    cache_mib: int = 8192
    sampling_seed: int = 34410000
    validation_seed: int = 34420000
    auxiliary_initialization_seed: int = 36100000
    validation_positions: int = 4096
    policy_surprise_weighting: dict | None = None
    value_auxiliary: dict | None = None
    learner: LearnerConfig = field(default_factory=lambda: LearnerConfig(
        learning_rate=2e-5, checkpoint_every_steps=1000))

    def __post_init__(self):
        if not 1 <= self.first_round <= self.last_round:
            raise ValueError("absolute source rounds must be ordered and positive")
        if type(self.expected_games_per_round) is not int or self.expected_games_per_round <= 0:
            raise ValueError("expected_games_per_round must be a positive integer")
        if not math.isfinite(self.draw_exposures) or self.draw_exposures <= 0 or self.cache_mib <= 0:
            raise ValueError("exposures and cache must be positive")
        if self.learner.max_steps_per_round is not None:
            raise ValueError("offline budget is determined by draw_exposures")
        if (self.policy_surprise_weighting is None) == (self.value_auxiliary is None):
            raise ValueError("select exactly one of policy surprise or value auxiliary fitting")
        if not 0 < self.validation_positions <= 4096:
            raise ValueError("validation_positions must be in [1, 4096]")
        if self.policy_surprise_weighting is not None:
            PolicySurpriseConfig(**self.policy_surprise_weighting)
        if self.value_auxiliary is not None:
            ValueAuxiliaryConfig(**self.value_auxiliary)

    @classmethod
    def from_dict(cls, raw):
        raw = dict(raw)
        raw["learner"] = LearnerConfig(**raw.get("learner", {}))
        return cls(**raw)


def freeze_inputs(config, output):
    """Filter absolute generation IDs, retaining original whole-game splits."""
    frozen = asdict(config)
    output.mkdir(parents=True, exist_ok=True)
    if sha256(Path(config.manifest)) != config.manifest_sha256:
        raise ValueError("source historical manifest changed")
    if sha256(Path(config.initial_checkpoint)) != config.initial_checkpoint_sha256:
        raise ValueError("initial checkpoint changed")
    if (output / "config.json").exists():
        if read_json(output / "config.json") != frozen:
            raise ValueError("offline resume requires the identical frozen configuration")
        dataset = read_json(output / "dataset.json")
        if dataset["digest"] != digest({k: v for k, v in dataset.items() if k != "digest"}):
            raise ValueError("frozen dataset identity mismatch")
    else:
        records = [json.loads(line) for line in Path(config.manifest).read_text().splitlines()]
        records = [r for r in records if config.first_round - 1 <= r["round_index"] < config.last_round]
        counts = Counter(r["round_index"] for r in records)
        if set(counts) != set(range(config.first_round - 1, config.last_round)) or set(counts.values()) != {config.expected_games_per_round}:
            raise ValueError("unexpected archived game count in selected absolute generations")
        if len({(r["round_index"], r["game_id"]) for r in records}) != len(records):
            raise ValueError("duplicate historical game identity")
        dataset = {"protocol": "az-offline-objectives-data-v1", "games": records,
                   "manifest_sha256": config.manifest_sha256,
                   "absolute_rounds": [config.first_round, config.last_round],
                   "scope": "original archived search targets and whole-game splits; no relabeling"}
        dataset["digest"] = digest(dataset)
        write_json(output / "dataset.json", dataset)
        write_json(output / "config.json", frozen)
    for game in dataset["games"]:
        stat = Path(game["path"]).stat()
        if (stat.st_size, stat.st_mtime_ns) != (game["size"], game["mtime_ns"]):
            raise ValueError(f"frozen source archive changed: {game['path']}")
    index = make_index(dataset, config.cache_mib * 1024**2)
    index.cache = CompactGameCache(config.cache_mib * 1024**2)
    pool = PositionPool(index.split_games("train"), index.cache, "all-frozen-train")
    steps = math.ceil(config.draw_exposures * len(pool) / config.learner.batch_size)
    identity = {**index.identity(), "digest": digest({"data": dataset["digest"], "config": frozen}),
                "policy_surprise_weighting": config.policy_surprise_weighting,
                "value_auxiliary": config.value_auxiliary}
    fixed_path = output / "fixed_validation.json"
    if not fixed_path.exists():
        subset_config = OfflineConfig(source_run="", source_artifacts="", model_config={}, max_steps=steps,
                                     rounds=tuple(range(config.first_round - 1, config.last_round)),
                                     validation_positions=config.validation_positions,
                                     validation_seed=config.validation_seed)
        write_json(fixed_path, freeze_validation(index, subset_config))
    fixed = read_json(fixed_path)
    if fixed["digest"] != digest(fixed["references"]):
        raise ValueError("frozen validation references changed")
    train_counts = {split: {"games": len(index.split_games(split)),
                           "positions": sum(g.decision_count for g in index.split_games(split))}
                    for split in ("train", "validation")}
    plan = {"status": "prepared", "absolute_rounds": dataset["absolute_rounds"],
            "games": len(index.games), "splits": train_counts, "steps_target": steps,
            "draw_exposures": config.draw_exposures, "replay_identity": identity,
            "fixed_validation_digest": fixed["digest"],
            "initial_checkpoint": config.initial_checkpoint,
            "initial_checkpoint_sha256": config.initial_checkpoint_sha256}
    write_json(output / "plan.json", plan)
    return index, pool, identity, fixed, plan


def initialize_learner(config, identity, steps):
    payload = load_alphazero_checkpoint(config.initial_checkpoint)
    base = payload.build_model()
    if config.value_auxiliary is not None:
        torch.manual_seed(config.auxiliary_initialization_seed)
        model = add_auxiliary_heads(base)
    else:
        model = base
    learner = AlphazeroLearner(model, config.learner, sampling_rng=DeterministicRng(config.sampling_seed))
    if config.value_auxiliary is not None:
        extend_optimizer_state(payload.optimizer_state_dict, base, model, learner.optimizer)
    else:
        learner.optimizer.load_state_dict(payload.optimizer_state_dict)
    move_optimizer_state(learner.optimizer, learner.device)
    for group in learner.optimizer.param_groups:
        group.update(lr=config.learner.learning_rate, weight_decay=config.learner.weight_decay)
    learner.total_train_step = payload.total_train_step
    learner.replay_identity = identity
    learner.round_steps_target = steps
    learner.parent_checkpoint = {"path": config.initial_checkpoint, "sha256": config.initial_checkpoint_sha256,
                                 "preserve_optimizer": True, "new_heads_optimizer": "empty state"}
    torch.set_rng_state(payload.torch_cpu_rng_state.cpu())
    if payload.torch_cuda_rng_states and config.learner.device == "cuda":
        torch.cuda.set_rng_state_all([state.cpu() for state in payload.torch_cuda_rng_states])
    return learner


def evaluate(learner, index, fixed, auxiliary):
    """Unweighted primary validation, plus scaled and natural-unit auxiliary errors."""
    games = {g.key: g for g in index.games}
    refs = (fixed["references"] if fixed is not None else
            ({"round_index": g.round_index, "game_id": g.game_id, "decision_index": i}
             for g in index.split_games("validation") for i in range(g.decision_count)))
    totals = Counter()
    was_training = learner.model.training
    learner.model.eval()

    def flush(samples):
        batch = collate_selfplay_samples(samples)
        tensors = _torch_batch(batch, learner.device)
        with torch.inference_mode():
            if auxiliary is None:
                logits, values = learner.model(batch.decisions)
            else:
                logits, values, score, turns = learner.model.forward_with_auxiliary(batch.decisions)
                ys, yt = auxiliary_targets(samples, auxiliary)
                es = score - torch.as_tensor(ys, device=learner.device)
                et = turns - torch.as_tensor(yt, device=learner.device)
                weight = tensors["value_weights"]
                totals.update(score_sq_sum=float((es.square() * weight).sum()),
                              turns_sq_sum=float((et.square() * weight).sum()),
                              score_abs_sum=float((es.abs() * weight).sum()),
                              turns_abs_sum=float((et.abs() * weight).sum()))
            terms = alphazero_loss_sums(logits, values, **tensors)
            totals.update({k: float(getattr(terms, k)) for k in (
                "policy_ce_sum", "policy_weight_sum", "value_sq_sum", "value_weight_sum", "entropy_sum")})
            totals.update(label_statistics(batch))
            totals["positions"] += len(samples)

    chunk = []
    for ref in refs:
        chunk.append(index.cache.sample(games[(ref["round_index"], ref["game_id"])], ref["decision_index"]))
        if len(chunk) == learner.config.validation_batch_size:
            flush(chunk)
            chunk = []
    if chunk:
        flush(chunk)
    learner.model.train(was_training)
    result = {**totals, **metric_ratios(totals)}
    result["total_loss"] = learner.config.policy_loss_weight * result["policy_loss"] + learner.config.value_loss_weight * result["value_loss"]
    if auxiliary is not None:
        n = totals["value_weight_sum"]
        result.update(score_loss=totals["score_sq_sum"] / n, remaining_turns_loss=totals["turns_sq_sum"] / n,
                      score_mae_points=totals["score_abs_sum"] / n * auxiliary.score_scale,
                      remaining_turns_mae=totals["turns_abs_sum"] / n * auxiliary.turns_scale)
        result["total_loss"] += auxiliary.score_loss_weight * result["score_loss"] + auxiliary.turns_loss_weight * result["remaining_turns_loss"]
    return result


def run(config, output, *, prepare_only=False, max_steps_this_session=None):
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        index, pool, identity, fixed, plan = freeze_inputs(config, output)
        print(json.dumps({k: plan[k] for k in ("absolute_rounds", "games", "splits", "steps_target")}, ensure_ascii=False), flush=True)
        if prepare_only:
            return plan
        torch.set_num_threads(1)
        checkpoints = output / "checkpoints"
        checkpoints.mkdir(exist_ok=True)
        saved = sorted(checkpoints.glob("step-*.pt"))
        latest = saved[-1] if saved else checkpoints / "initial.pt"
        if latest.exists():
            learner = AlphazeroLearner.resume(latest, config.learner, replay_identity=identity,
                                              round_index=0, round_steps_target=plan["steps_target"])
        else:
            learner = initialize_learner(config, identity, plan["steps_target"])
            learner.save(latest)
        auxiliary = ValueAuxiliaryConfig(**config.value_auxiliary) if config.value_auxiliary is not None else None
        session = str(time.time_ns())
        write_json(output / "source" / session / "identity.json", {
            **source_identity(Path.cwd(), output / "source" / session), "command": sys.argv,
            "config": asdict(config), "torch": torch.__version__, "device": str(learner.device)})
        curve_path = output / "validation_curve.jsonl"
        curve = [json.loads(line) for line in curve_path.read_text().splitlines()] if curve_path.exists() else []
        curve = [p for p in curve if p["step"] <= learner.round_train_step]
        curve_path.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in curve))

        def validate(current, checkpoint):
            step = current.round_train_step
            if any(point["step"] == step for point in curve):
                return
            metrics = evaluate(current, index, fixed, auxiliary)
            point = {"step": step, "metrics": metrics, "checkpoint": str(checkpoint),
                     "draw_exposure": step * config.learner.batch_size / len(pool)}
            curve.append(point)
            append_json(curve_path, point)
            print(f"验证 step={step}/{plan['steps_target']} KL={metrics['policy_kl']:.6f} MSE={metrics['value_loss']:.6f}", flush=True)

        validate(learner, latest)
        writer = ReferenceWriter(output / "references", session, config.learner.batch_size, identity)
        def progress(record):
            if record["round_train_step"] % 100 == 0:
                print(f"训练 step={record['round_train_step']}/{plan['steps_target']} loss={record['total_loss']:.6f}", flush=True)
        try:
            result = run_training_round(
                learner, steps_target=plan["steps_target"], current_pool=pool, history_pool=None,
                validation_samples=[], validation_games=len(index.split_games("validation")),
                checkpoint_dir=checkpoints, metrics_path=output / "training_metrics.jsonl",
                checkpoint_callback=validate, reference_writer=writer, progress_callback=progress,
                max_steps_this_session=max_steps_this_session,
                policy_surprise_weighting=(PolicySurpriseConfig(**config.policy_surprise_weighting)
                                          if config.policy_surprise_weighting is not None else None),
                value_auxiliary=auxiliary)
        finally:
            writer.close()
        completed = learner.round_train_step == plan["steps_target"]
        if completed:
            shutil.copyfile(checkpoints / f"step-{learner.round_train_step:09d}.pt", checkpoints / "final.pt")
            if not (output / "full_validation.json").exists():
                write_json(output / "full_validation.json", evaluate(learner, index, None, auxiliary))
        write_json(output / "progress.json", {"status": "complete" if completed else "incomplete",
                                              "step": learner.round_train_step, "steps_target": plan["steps_target"],
                                              "replay_identity": identity, "training": result})
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--max-steps-this-session", type=int)
    args = parser.parse_args()
    run(OfflineObjectiveConfig.from_dict(read_json(args.config)), args.output,
        prepare_only=args.prepare_only, max_steps_this_session=args.max_steps_this_session)


if __name__ == "__main__":
    main()
