"""Continuous offline-to-online AZ fitting with a prescribed two-pool schedule."""
from __future__ import annotations

import argparse
import fcntl
import gc
import json
import math
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch

from .alphazero_checkpoint import load_alphazero_checkpoint
from .actor_critic import model_config_from_dict
from .alphazero_loop import LoopSeeds, SelfPlaySection
from .alphazero_recording import ReferenceWriter, append_json, read_json, sha256, source_identity, write_json
from .alphazero_replay import PositionPool, ReplayRun, build_replay_index
from .alphazero_train import AlphazeroLearner, DrawnBatch, run_training_round
from .evaluation import FixedEvaluationConfig, run_fixed_evaluation
from .offline_train import (OfflineConfig, _copy_checkpoint, _digest, evaluate_validation,
                            freeze_dataset, freeze_validation, make_index, run_offline)
from .policy_averaging import (AveragedTargetStore, PolicyAveragingConfig, generate_labels, label_root)
from .policy_surprise import PolicySurpriseConfig
from .selfplay_fast import ExecutionConfig, FastSelfPlayConfig, generate

PROTOCOL = "az-continuous-hybrid-v1"
DEFAULT_SCHEDULE = ((1, 224), (2, 192), (3, 160), (5, 128), (7, 96), (10, 64), (15, 32))
COLLECTION_BATCHING_FIELDS = ("max_batch_size", "batch_buckets")


@dataclass(frozen=True)
class SplitTrainingConfig:
    recent_rounds: int = 4
    value_recent_positions: int = 128
    policy_max_grad_norm: float = 1.0
    value_max_grad_norm: float = 1.0
    # None keeps the existing shared recent window. Only policy may narrow it.
    policy_recent_rounds: int | None = None

    def __post_init__(self):
        for name in ("recent_rounds", "value_recent_positions"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("policy_max_grad_norm", "value_max_grad_norm"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.policy_recent_rounds is not None and (type(self.policy_recent_rounds) is not int
                or not 0 < self.policy_recent_rounds <= self.recent_rounds):
            raise ValueError("policy_recent_rounds must be in 1..recent_rounds")

    def to_dict(self):
        values = asdict(self)
        if self.policy_recent_rounds is None:
            del values["policy_recent_rounds"]
        return values


@dataclass(frozen=True)
class HybridConfig:
    run_id: str
    offline: OfflineConfig
    selfplay: SelfPlaySection
    seeds: LoopSeeds
    new_games_per_round: int = 1000
    updates_per_round: int = 500
    replay_max_games: int = 6000
    online_validation_positions: int = 2048
    online_validation_seed: int = 14640000
    # Last human-numbered round at a tier, and exact offline batch quota.
    mix_schedule: tuple[tuple[int, int], ...] = DEFAULT_SCHEDULE
    evaluation_every_rounds: int = 3
    evaluation: dict[str, Any] | None = None
    split_training: SplitTrainingConfig | None = None
    # Optional external fixed opponent. Path and hash are frozen in the config.
    evaluation_anchor: dict[str, str] | None = None
    # A frozen experimental treatment, applied once after restoring a fork.
    # Unlike execution cache sizing this changes the learning algorithm.
    learning_rate_override: float | None = None
    policy_averaging: PolicyAveragingConfig | None = None
    # Warm start on a separately frozen, multi-source offline pool.
    initialization: dict[str, Any] | None = None
    fitting_diagnostics: dict[str, Any] | None = None
    # Fork an existing online window without loading or evaluating an offline pool.
    online_only: bool = False
    # Joint updates with a narrower policy window; value keeps replay_max_games.
    policy_replay_max_games: int | None = None
    policy_surprise_weighting: PolicySurpriseConfig | None = None

    def __post_init__(self):
        if not self.run_id:
            raise ValueError("run_id must be nonempty")
        if type(self.online_only) is not bool:
            raise ValueError("online_only must be bool")
        if self.online_only and (self.initialization is not None or self.mix_schedule):
            raise ValueError("online_only requires a replay fork and an empty mixing schedule")
        if self.policy_surprise_weighting is not None and (not self.online_only
                or self.split_training is not None or self.policy_averaging is not None):
            raise ValueError("policy surprise weighting requires joint online-only archived targets")
        if self.policy_replay_max_games is not None:
            if (type(self.policy_replay_max_games) is not int
                    or not 0 < self.policy_replay_max_games <= self.replay_max_games):
                raise ValueError("policy_replay_max_games must be in 1..replay_max_games")
            if not self.online_only or self.split_training is not None:
                raise ValueError("policy replay window requires joint online-only training")
            if self.offline.model_config.get("architecture") != "entity_attention_v1":
                raise ValueError("joint head pools require entity_attention_v1")
        if self.initialization is not None:
            if set(self.initialization) != {"checkpoint", "manifest"}:
                raise ValueError("initialization requires checkpoint and manifest identities")
            for identity in self.initialization.values():
                if set(identity) != {"path", "sha256"}:
                    raise ValueError("initialization identities require path and sha256")
        if self.fitting_diagnostics is not None:
            if (self.initialization is None and not self.online_only) or self.split_training is not None:
                raise ValueError("pool fitting diagnostics require joint initialized or online-only training")
            steps = self.fitting_diagnostics["steps"]
            if (steps != sorted(set(steps)) or not steps or steps[0] != 0
                    or steps[-1] != self.updates_per_round
                    or any(type(s) is not int or s < 0 for s in steps)
                    or any(s % self.offline.learner.checkpoint_every_steps for s in steps)):
                raise ValueError("fitting steps must include 0/final and align with checkpoints")
            names = ("train_positions_per_generation",) if self.online_only else (
                "train_positions_per_generation", "offline_positions_per_source_split")
            for name in names:
                if type(self.fitting_diagnostics[name]) is not int or self.fitting_diagnostics[name] <= 0:
                    raise ValueError(f"{name} must be positive")
        if self.policy_averaging is not None and (self.split_training is None
                or self.selfplay.search_config["determinizations"] != 1):
            raise ValueError("policy averaging requires split training and native N1 behavior")
        if self.learning_rate_override is not None and (isinstance(self.learning_rate_override, bool)
                or not isinstance(self.learning_rate_override, (int, float))
                or not math.isfinite(self.learning_rate_override) or self.learning_rate_override <= 0):
            raise ValueError("learning_rate_override must be finite and positive")
        for name in ("new_games_per_round", "updates_per_round", "replay_max_games"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("online_validation_positions", "evaluation_every_rounds"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.online_validation_positions > 4096:
            raise ValueError("online fixed validation is limited to 4096 positions")
        previous_round, previous_quota = 0, self.offline.learner.batch_size
        for end, quota in self.mix_schedule:
            if type(end) is not int or type(quota) is not int or end <= previous_round:
                raise ValueError("mix tiers require increasing positive end rounds")
            if not 0 <= quota < previous_quota:
                raise ValueError("offline quotas must decrease and fit the batch")
            previous_round, previous_quota = end, quota
        if self.offline.replay_cache_bytes < 2:
            raise ValueError("two caches require at least two bytes")
        if self.evaluation_every_rounds and self.evaluation is None:
            raise ValueError("periodic evaluation requires an evaluation template")
        if self.seeds.sampling != self.offline.sampling_seed:
            raise ValueError("hybrid sampling uses the continuous offline sampling seed")
        if self.split_training is not None:
            if self.mix_schedule:
                raise ValueError("split training requires pure online sampling")
            if self.offline.model_config.get("architecture") != "entity_attention_v1":
                raise ValueError("split training requires entity_attention_v1")
            if self.split_training.value_recent_positions >= self.offline.learner.batch_size:
                raise ValueError("value recent quota must leave a nonempty history quota")
            if self.replay_max_games <= self.new_games_per_round * self.split_training.recent_rounds:
                raise ValueError("split replay must retain older games beyond the recent rounds")
        if self.evaluation_anchor is not None:
            if set(self.evaluation_anchor) != {"path", "sha256"} or self.evaluation is None:
                raise ValueError("external evaluation anchor requires path, sha256 and evaluation template")
            if Path(self.evaluation["policy_b"]["checkpoint"]).resolve() != Path(self.evaluation_anchor["path"]).resolve():
                raise ValueError("external evaluation anchor differs from policy_b checkpoint")

    @classmethod
    def from_dict(cls, raw):
        values = dict(raw)
        if values.pop("protocol", None) != PROTOCOL:
            raise ValueError("hybrid config protocol mismatch")
        values["offline"] = OfflineConfig.from_dict(values["offline"])
        values["selfplay"] = SelfPlaySection(**values["selfplay"])
        values["seeds"] = LoopSeeds(**values["seeds"])
        values["mix_schedule"] = tuple(tuple(row) for row in values.get("mix_schedule", DEFAULT_SCHEDULE))
        if values.get("split_training") is not None:
            values["split_training"] = SplitTrainingConfig(**values["split_training"])
        if values.get("policy_averaging") is not None:
            values["policy_averaging"] = PolicyAveragingConfig(**values["policy_averaging"])
        if values.get("policy_surprise_weighting") is not None:
            values["policy_surprise_weighting"] = PolicySurpriseConfig(**values["policy_surprise_weighting"])
        return cls(**values)

    def to_dict(self):
        values = {"protocol": PROTOCOL, **asdict(self)}
        values["selfplay"] = self.selfplay.to_dict()
        if not self.online_only:
            del values["online_only"]
        # Preserve the frozen representation of existing joint runs.
        for name in ("split_training", "evaluation_anchor", "learning_rate_override", "policy_averaging",
                     "initialization", "fitting_diagnostics", "policy_replay_max_games", "policy_surprise_weighting"):
            if values[name] is None:
                del values[name]
        if self.split_training is not None:
            values["split_training"] = self.split_training.to_dict()
        return json.loads(json.dumps(values))

    def quotas(self, round_index):
        if type(round_index) is not int or round_index < 0:
            raise ValueError("online round index must be nonnegative")
        offline = next((quota for end, quota in self.mix_schedule if round_index + 1 <= end), 0)
        return offline, self.offline.learner.batch_size - offline


def draw_mixed_batch(offline_pool, online_pool, rng, offline_count, online_count,
                     *, inherited_run_ids=None, record_rounds=False):
    samples, references = [], []
    for pool, count in ((offline_pool, offline_count), (online_pool, online_count)):
        for _ in range(count):
            sample, reference = pool.sample(rng)
            samples.append(sample)
            references.append(reference)
    sources = {"offline": offline_count, "online": online_count}
    if record_rounds:
        for ref in references[offline_count:]:
            key = f"online_round_{ref.round_index}"
            sources[key] = sources.get(key, 0) + 1
    if inherited_run_ids is not None:
        inherited = sum(ref.run_id in inherited_run_ids for ref in references[offline_count:])
        sources.update(inherited_online=inherited, branch_online=online_count - inherited)
    return DrawnBatch(tuple(samples), tuple(references), online_count, offline_count, sources)


def joint_policy_pool(index, max_games):
    """Count complete games before excluding whole-game validation, as replay does."""
    games = index.games[-max_games:]
    pool = PositionPool([g for g in games if g.split == "train"], index.cache, "policy")
    if not len(pool):
        raise ValueError("joint policy training pool is empty")
    return pool


def joint_replay_description(index, policy_pool, value_pool, config):
    policy_keys = {g.key for g in policy_pool.games}
    generations = []
    for r in sorted({g.round_index for g in index.games}):
        games = [g for g in index.games if g.round_index == r]
        policy_positions = sum(g.decision_count for g in games if g.key in policy_keys)
        value_positions = sum(g.decision_count for g in games if g.split == "train")
        generations.append({"round_index": r, "run_ids": sorted({g.run_id for g in games}),
            "games": len(games), "policy_train_positions": policy_positions,
            "value_train_positions": value_positions,
            "policy_expected_fraction": policy_positions / len(policy_pool),
            "value_expected_fraction": value_positions / len(value_pool)})
    return {"mode": "joint_separate_batches",
        "policy_replay_max_games": config.policy_replay_max_games,
        "value_replay_max_games": config.replay_max_games,
        "policy_batch_size": config.offline.learner.batch_size,
        "value_batch_size": config.offline.learner.batch_size,
        "pool_positions": {"policy": len(policy_pool), "value": len(value_pool)},
        "sampling": "uniform_train_positions_with_replacement; coupled head RNGs; value advances canonical stream",
        "update": "shared encoder receives both losses; one weighted sum, global clip and AdamW step",
        "generations": generations}


def draw_joint_head_batch(policy_pool, value_pool, rng, batch_size, *, inherited_run_ids=None):
    """Match R30 head-pool sampling and preserve the unchanged value RNG stream."""
    from .rng import DeterministicRng
    policy_rng = DeterministicRng.restore(rng.snapshot())
    policy_samples, value_samples, references = [], [], []
    sources = {"policy": batch_size, "value": batch_size}
    for head, pool, head_rng, samples in (("policy", policy_pool, policy_rng, policy_samples),
                                         ("value", value_pool, rng, value_samples)):
        for _ in range(batch_size):
            sample, ref = pool.sample(head_rng)
            samples.append(sample)
            references.append(ref)
            key = f"{head}_round_{ref.round_index}"
            sources[key] = sources.get(key, 0) + 1
    if inherited_run_ids is not None:
        inherited = sum(ref.run_id in inherited_run_ids for ref in references)
        sources.update(inherited_online=inherited, branch_online=len(references) - inherited)
    return DrawnBatch(tuple(policy_samples), tuple(references), len(references), 0, sources,
                     tuple(value_samples))


def split_replay_pools(index, config):
    """Select recent collection teachers, and a disjoint older value bucket."""
    rounds = sorted({game.round_index for game in index.games})
    n = config.split_training.recent_rounds
    if len(rounds) <= n:
        raise ValueError("split training needs recent rounds and older inherited replay")
    recent = set(rounds[-n:])
    train = index.split_games("train")
    recent_pool = PositionPool([g for g in train if g.round_index in recent], index.cache, "recent")
    history_pool = PositionPool([g for g in train if g.round_index not in recent], index.cache, "history")
    if not len(recent_pool) or not len(history_pool):
        raise ValueError("split training requires nonempty recent and older training pools")
    return recent_pool, history_pool, sorted(recent)


def split_policy_pool(index, config, recent_pool):
    n = config.split_training.policy_recent_rounds
    if n is None:
        return recent_pool
    rounds = set(sorted({g.round_index for g in index.games})[-n:])
    pool = PositionPool([g for g in index.split_games("train") if g.round_index in rounds],
                        index.cache, "policy_recent")
    if not len(pool):
        raise ValueError("policy recent training pool is empty")
    return pool


def split_replay_description(index, config, pools, recent_rounds):
    batch = config.offline.learner.batch_size
    recent_count = config.split_training.value_recent_positions
    policy_pool = split_policy_pool(index, config, pools[0])
    policy_rounds = {g.round_index for g in policy_pool.games}
    teachers = []
    for r in sorted({g.round_index for g in index.games}):
        games = [g for g in index.games if g.round_index == r]
        manifest_path = games[0].path.parent.parent / "manifest.json"
        manifest = read_json(manifest_path)
        positions = sum(g.decision_count for g in games if g.split == "train")
        is_recent = r in recent_rounds
        teachers.append({"round_index": r, "run_id": games[0].run_id,
            "checkpoint_identity": manifest["checkpoint_identity"], "manifest_sha256": sha256(manifest_path),
            "games": len(games), "train_games": sum(g.split == "train" for g in games),
            "train_positions": positions, "bucket": "recent" if is_recent else "history",
            "policy_expected_fraction": positions / len(policy_pool) if r in policy_rounds else 0.0,
            "value_expected_fraction": (recent_count if is_recent else batch - recent_count)
                / batch * positions / len(pools[0 if is_recent else 1])})
    description = {"mode": "split_policy_detached_value", "config": config.split_training.to_dict(),
            "recent_round_indices": recent_rounds, "sampling": "uniform_positions_with_replacement",
            "policy_batch_size": batch, "value_batch_size": batch,
            "quotas": {"policy_recent": batch, "value_recent": recent_count,
                       "value_history": batch - recent_count},
            "pool_positions": {"recent": len(pools[0]), "history": len(pools[1])}, "teachers": teachers}
    if config.split_training.policy_recent_rounds is not None:
        description["policy_recent_round_indices"] = sorted(policy_rounds)
        description["pool_positions"]["policy_recent"] = len(policy_pool)
    return description


def draw_split_batch(recent_pool, history_pool, rng, batch_size, value_recent_positions,
                     *, inherited_run_ids=None, policy_pool=None):
    policy_samples, value_samples, references = [], [], []
    sources = {"policy_recent": batch_size, "value_recent": value_recent_positions,
               "value_history": batch_size - value_recent_positions}
    for head, pool, count in (("policy", recent_pool if policy_pool is None else policy_pool, batch_size),
                              ("value", recent_pool, value_recent_positions),
                              ("value", history_pool, batch_size - value_recent_positions)):
        for _ in range(count):
            sample, ref = pool.sample(rng)
            (policy_samples if head == "policy" else value_samples).append(sample)
            references.append(ref)
            key = f"{head}_round_{ref.round_index}"
            sources[key] = sources.get(key, 0) + 1
    if inherited_run_ids is not None:
        inherited = sum(ref.run_id in inherited_run_ids for ref in references)
        sources.update(inherited_online=inherited, branch_online=len(references) - inherited)
    return DrawnBatch(tuple(policy_samples), tuple(references), len(references), 0, sources,
                     tuple(value_samples))


class HybridReferenceWriter:
    """Keep per-source identities and batch hashes; full audit records are opt-in."""

    def __init__(self, root, session, identity, quotas, sources=("offline", "online"),
                 *, persist_records=False):
        self.counts = quotas
        self.writers = [ReferenceWriter(root / source, session, count,
                        {"source": source, **identity[source]}, persist_records=persist_records)
                        for source, count in zip(sources, quotas)]
        self.path = root / f"{session}.json"
        write_json(self.path, {"protocol": "az-hybrid-reference-sources-v1",
                              "persist_records": persist_records,
                              "sources": {name: str(writer.path.resolve()) for name, writer in
                                          zip(sources, self.writers)}})

    def append(self, step, references):
        if len(references) != sum(self.counts):
            raise ValueError("reference count differs from source quotas")
        digests, start = [], 0
        for writer, count in zip(self.writers, self.counts):
            digests.append(writer.append(step, references[start:start + count]))
            start += count
        return _digest(digests)

    def flush(self):
        for writer in self.writers:
            writer.flush()

    def record_step(self, record):
        for writer in self.writers:
            writer.record_step(record)

    def archive_metrics(self, path):
        self.writers[0].archive_metrics(path)

    def close(self):
        for writer in self.writers:
            writer.close()


def _identity(path):
    payload = load_alphazero_checkpoint(path)
    return {"path": str(path.resolve()), "sha256": sha256(path),
            "total_train_step": payload.total_train_step}


def _emit(event, **values):
    print(json.dumps({"event": event, **values}, ensure_ascii=False), flush=True)


def _release_inference_memory():
    """No inference is live at this boundary; release process-local graphs."""
    torch.compiler.reset()
    gc.collect()
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()


def _fixed_online(config, index, output, first_online_round=0):
    manifest_path = output / "fixed_online_dataset.json"
    refs_path = output / "fixed_online_validation.json"
    if not config.online_validation_positions:
        return None
    if not manifest_path.exists():
        games = [g.to_dict() for g in index.split_games("validation") if g.round_index == first_online_round]
        if not games:
            raise ValueError("first online round has no validation games; cannot freeze diagnostic set")
        write_json(manifest_path, {"games": games})
    fixed_index = make_index(read_json(manifest_path), index.cache.max_bytes)
    fixed_index.cache = index.cache
    if not refs_path.exists():
        selection = OfflineConfig(source_run="", source_artifacts="", model_config={}, max_steps=1,
                                  rounds=(first_online_round,), validation_positions=config.online_validation_positions,
                                  validation_seed=config.online_validation_seed)
        write_json(refs_path, freeze_validation(fixed_index, selection))
    return fixed_index, read_json(refs_path)


def _evaluation(config, checkpoint, anchor, stage):
    raw = json.loads(json.dumps(config.evaluation))
    raw["policy_a"]["checkpoint"] = str(checkpoint.resolve())
    raw["policy_b"]["checkpoint"] = str(anchor.resolve())
    if config.evaluation_anchor is None:
        raw["policy_b"]["name"] = "theta_5000"
        raw["policy_b"]["inference"] = dict(raw["policy_a"].get("inference", {}))
    elif sha256(anchor) != config.evaluation_anchor["sha256"]:
        raise ValueError("external fixed evaluation checkpoint changed")
    path = stage / ("evaluation/mcts_vs_theta5000.json" if config.evaluation_anchor is None else
                    "evaluation/mcts_vs_fixed_anchor.json")
    write_json(stage / "evaluation/config.json", raw)
    result = run_fixed_evaluation(FixedEvaluationConfig.from_dict(raw), path,
                                  resume=path.exists() or Path(str(path) + ".games").exists(),
                                  require_scored_pairs=True)
    if not result.summary["passed"]:
        raise RuntimeError(f"evaluation contains engineering failures: {path}")
    _emit("evaluation_complete", score=result.summary["win_rate_score"], path=str(path))


def prepare_fork(config, parent, round_number=None):
    """Read and freeze a completed hybrid boundary; do not modify the parent."""
    parent = Path(parent).resolve()
    state = read_json(parent / "hybrid_state.json")
    completed = state["completed_rounds"]
    if state["protocol"] != PROTOCOL or completed != list(range(len(completed))) or not completed:
        raise ValueError("fork source must have completed consecutive hybrid rounds")
    chosen = len(completed) if round_number is None else round_number
    if type(chosen) is not int or not 1 <= chosen <= len(completed):
        raise ValueError("fork round must be a completed human-numbered parent round")
    stage = parent / "rounds" / f"round-{chosen - 1:03d}"
    checkpoint = stage / "checkpoints/final.pt"
    payload = load_alphazero_checkpoint(checkpoint)
    if payload.learner_config != asdict(config.offline.learner):
        raise ValueError("fork must preserve the checkpoint learner configuration and optimizer")
    if payload.round_train_step != payload.round_steps_target:
        raise ValueError("fork checkpoint is not a completed training round")
    checkpoint_identity = _identity(checkpoint)
    if chosen == len(completed) and checkpoint_identity != state["latest_checkpoint"]:
        raise ValueError("fork checkpoint differs from the parent's committed latest checkpoint")
    dataset_path = parent / "offline/dataset.json"
    dataset = None
    if not config.online_only:
        dataset = read_json(dataset_path)
        if freeze_dataset(config.offline)["digest"] != dataset["digest"]:
            raise ValueError("fork must preserve the original frozen offline dataset")
    anchor = Path(state["anchor_checkpoint"]["path"])
    if _identity(anchor) != state["anchor_checkpoint"]:
        raise ValueError("parent fixed evaluation anchor changed")
    snapshot_path = stage / "replay_snapshot.json"
    games = sorted(read_json(snapshot_path)["games"], key=lambda g: (g["round_index"], g["game_id"]))
    games = games[-config.replay_max_games:]
    if not games or max(g["round_index"] for g in games) != payload.round_index - 1:
        raise ValueError("fork replay does not end at the selected checkpoint round")
    records, runs = [], {}
    for game in games:
        path = Path(game["path"]).resolve()
        records.append({**game, "path": str(path), "archive_sha256": sha256(path)})
        labels = label_root(path) / "manifest.json"
        if labels.exists():
            records[-1]["policy_labels_manifest"] = {"path": str(labels), "sha256": sha256(labels)}
        key = (game["round_index"], game["run_id"], str(path.parent.parent))
        runs[key] = {"round_index": key[0], "run_id": key[1], "root": key[2]}
    if config.split_training is not None:
        split_replay_pools(make_index({"games": records}, config.offline.replay_cache_bytes), config)
    inputs = [parent / "config.json", snapshot_path]
    if not config.online_only:
        inputs.extend([dataset_path, parent / "offline/fixed_validation.json"])
    train_refs = stage / "fitting/train_references.json"
    if config.online_only and config.fitting_diagnostics is not None:
        references = read_json(train_refs)
        # Freeze exactly the parent's fixed subsets for still-active generations.
        active = {(g["round_index"], g["game_id"]) for g in games}
        inherited_refs = [ref for ref in references["references"]
                          if (ref["round_index"], ref["game_id"]) in active]
        inputs.append(train_refs)
    fixed_online = [parent / name for name in ("fixed_online_dataset.json", "fixed_online_validation.json")]
    if any(path.exists() for path in fixed_online):
        if not all(path.exists() for path in fixed_online):
            raise ValueError("parent fixed online holdout is incomplete")
        inputs.extend(fixed_online)
    result = {"protocol": "az-hybrid-fork-v1", "parent_run": str(parent), "parent_round": chosen,
              "parent_checkpoint": checkpoint_identity, "anchor_checkpoint": state["anchor_checkpoint"],
              "online_round_offset": payload.round_index, "games": records,
              "inherited_runs": list(runs.values()), "offline_dataset_digest": dataset["digest"] if dataset else None,
              "inputs": [{"path": str(p.resolve()), "sha256": sha256(p)} for p in inputs]}
    if config.online_only:
        result["online_only"] = True
        if config.fitting_diagnostics is not None:
            result["train_references"] = inherited_refs
    result["digest"] = _digest(result)
    return result


def _initialize_fork(output, manifest):
    parent = Path(manifest["parent_run"])
    offline = output / "offline"
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    initial, anchor = checkpoints / "fork_initial.pt", checkpoints / "theta_5000.pt"
    _copy_checkpoint(Path(manifest["parent_checkpoint"]["path"]), initial)
    _copy_checkpoint(Path(manifest["anchor_checkpoint"]["path"]), anchor)
    if not manifest.get("online_only", False):
        offline.mkdir(parents=True, exist_ok=True)
        for filename in ("dataset.json", "fixed_validation.json"):
            shutil.copyfile(parent / "offline" / filename, offline / filename)
    # Preserve the shared historical holdout even when its games leave replay.
    for filename in ("fixed_online_dataset.json", "fixed_online_validation.json"):
        if (parent / filename).exists():
            shutil.copyfile(parent / filename, output / filename)
    write_json(output / "fork_manifest.json", manifest)
    return {"protocol": PROTOCOL, "completed_rounds": [], "latest_checkpoint": _identity(initial),
            "anchor_checkpoint": _identity(anchor), "online_round_offset": manifest["online_round_offset"],
            "fork_manifest_sha256": sha256(output / "fork_manifest.json")}


def _load_fork(output, state, config):
    if "fork_manifest_sha256" not in state:
        return None
    path = output / "fork_manifest.json"
    if sha256(path) != state["fork_manifest_sha256"]:
        raise ValueError("frozen fork manifest changed")
    manifest = read_json(path)
    if manifest.get("online_only", False) != config.online_only:
        raise ValueError("fork online-only mode changed")
    if manifest["online_round_offset"] != state["online_round_offset"]:
        raise ValueError("fork round offset changed")
    if sha256(output / "checkpoints/fork_initial.pt") != manifest["parent_checkpoint"]["sha256"]:
        raise ValueError("fork initial checkpoint changed")
    if sha256(output / "checkpoints/theta_5000.pt") != manifest["anchor_checkpoint"]["sha256"]:
        raise ValueError("fork evaluation anchor changed")
    for record in manifest["inputs"]:
        source = Path(record["path"])
        if source.parent == Path(manifest["parent_run"]) / "offline":
            if sha256(output / "offline" / source.name) != record["sha256"]:
                raise ValueError("copied fork offline metadata changed")
        elif source.name in ("fixed_online_dataset.json", "fixed_online_validation.json"):
            if sha256(output / source.name) != record["sha256"]:
                raise ValueError("copied fork online holdout changed")
    # Only still-active inherited archives are required after rolling eviction.
    remaining = min(len(manifest["games"]), max(0, config.replay_max_games
                    - len(state["completed_rounds"]) * config.new_games_per_round))
    for game in manifest["games"][-remaining:] if remaining else ():
        if sha256(Path(game["path"])) != game["archive_sha256"]:
            raise ValueError("inherited replay archive changed")
        if "policy_labels_manifest" in game:
            label_manifest = game["policy_labels_manifest"]
            if sha256(Path(label_manifest["path"])) != label_manifest["sha256"]:
                raise ValueError("inherited policy labels changed")
    return manifest


def _replay_runs(config, output, r, offset, fork):
    earliest = max(0, r - (config.replay_max_games + config.new_games_per_round - 1)
                   // config.new_games_per_round + 1)
    runs = []
    if fork and (r + 1) * config.new_games_per_round < config.replay_max_games:
        runs = [ReplayRun(g["round_index"], g["run_id"], Path(g["root"]))
                for g in fork["inherited_runs"]]
    runs.extend(ReplayRun(offset + i, f"{config.run_id}-round-{i:03d}",
                          output / "rounds" / f"round-{i:03d}" / "selfplay")
                for i in range(earliest, r + 1))
    return runs


def frozen_offline_dataset(config):
    if config.initialization is None:
        return freeze_dataset(config.offline)
    identity = config.initialization["manifest"]
    path = Path(identity["path"])
    if sha256(path) != identity["sha256"]:
        raise ValueError("frozen offline pool manifest changed")
    games = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    keys = {(g["round_index"], g["game_id"]) for g in games}
    if len(keys) != len(games) or {g["split"] for g in games} != {"train", "validation"}:
        raise ValueError("offline pool requires unique games and preserved train/validation splits")
    if set(config.offline.rounds) != {g["round_index"] for g in games}:
        raise ValueError("offline source rounds differ from the frozen manifest")
    for game in games:
        stat = Path(game["path"]).stat()
        if stat.st_size != game["size"] or stat.st_mtime_ns != game["mtime_ns"]:
            raise ValueError(f"offline source archive changed: {game['path']}")
    dataset = {"protocol": "az-offline-data-v1", "games": games,
               "sources": [identity], "scope": "frozen multi-source whole-game splits; original behavior labels"}
    dataset["digest"] = _digest(dataset)
    return dataset


def prepare_initialization(config):
    identity = config.initialization["checkpoint"]
    if sha256(Path(identity["path"])) != identity["sha256"]:
        raise ValueError("initial checkpoint changed")
    payload = load_alphazero_checkpoint(identity["path"])
    original, target = dict(payload.learner_config), asdict(config.offline.learner)
    # Checkpoint cadence is execution/recording only; AdamW settings stay intact.
    original.pop("checkpoint_every_steps")
    target.pop("checkpoint_every_steps")
    if original != target:
        raise ValueError("warm start must preserve the learner/optimizer settings")
    if payload.model_config != asdict(model_config_from_dict(config.offline.model_config)):
        raise ValueError("warm start architecture differs from configuration")
    return frozen_offline_dataset(config)


def _initialize_checkpoint(output, config):
    dataset = prepare_initialization(config)
    offline = output / "offline"
    write_json(offline / "dataset.json", dataset)
    index = make_index(dataset, config.offline.replay_cache_bytes // 2)
    write_json(offline / "fixed_validation.json", freeze_validation(index, config.offline))
    checkpoint = output / "checkpoints/starting_model.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    _copy_checkpoint(Path(config.initialization["checkpoint"]["path"]), checkpoint)
    return {"protocol": PROTOCOL, "completed_rounds": [], "latest_checkpoint": _identity(checkpoint),
            "anchor_checkpoint": _identity(checkpoint)}


def run_hybrid(config, output, *, resume=False, online_rounds=None, forever=False,
               max_steps_this_session=None, fork_from=None, fork_round=None,
               replay_cache_bytes_override=None, round_callback=None, round_commit_callback=None):
    if config.online_only and not resume and fork_from is None:
        raise ValueError("online-only training must inherit a completed replay fork")
    if config.initialization is not None and fork_from is not None:
        raise ValueError("checkpoint initialization and inherited replay fork are mutually exclusive")
    if config.split_training is not None and not resume and fork_from is None:
        raise ValueError("split training must fork from a completed hybrid replay window")
    if config.learning_rate_override is not None and not resume and fork_from is None:
        raise ValueError("learning rate override must initialize from a completed fork")
    if resume and fork_from is not None:
        raise ValueError("use fork-from only to initialize a new run; use resume afterwards")
    if fork_round is not None and fork_from is None:
        raise ValueError("fork-round requires fork-from")
    if (online_rounds is None) == (not forever):
        raise ValueError("choose online_rounds or forever")
    if online_rounds is not None and (type(online_rounds) is not int or online_rounds < 0):
        raise ValueError("online_rounds is an absolute nonnegative target")
    if max_steps_this_session is not None and max_steps_this_session <= 0:
        raise ValueError("max_steps_this_session must be positive")
    if replay_cache_bytes_override is not None and (type(replay_cache_bytes_override) is not int
                                                    or replay_cache_bytes_override < 2):
        raise ValueError("two replay caches require at least two bytes")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".hybrid.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run_hybrid(config, output, resume, online_rounds, forever, max_steps_this_session,
                           fork_from, fork_round, replay_cache_bytes_override, round_callback, round_commit_callback)


def _algorithm_config(raw):
    """Keep resume strict except for the two collection batching controls."""
    values = json.loads(json.dumps(raw))
    for name in COLLECTION_BATCHING_FIELDS:
        values["selfplay"]["execution"].pop(name, None)
    return values


def _run_hybrid(config, output, resume, online_rounds, forever, budget, fork_from, fork_round,
                replay_cache_bytes_override, round_callback=None, round_commit_callback=None):
    torch.set_num_threads(1)
    execution = ExecutionConfig(**config.selfplay.execution)
    if config.evaluation_anchor is not None:
        if sha256(Path(config.evaluation_anchor["path"])) != config.evaluation_anchor["sha256"]:
            raise ValueError("external fixed evaluation checkpoint changed")
        FixedEvaluationConfig.from_dict(config.evaluation)
    cfg_path, state_path = output / "config.json", output / "hybrid_state.json"
    if cfg_path.exists():
        if not resume:
            raise ValueError("run exists; use --resume or a new output")
        if _algorithm_config(read_json(cfg_path)) != _algorithm_config(config.to_dict()):
            raise ValueError("hybrid resume requires identical frozen algorithm configuration")
        state = read_json(state_path)
    else:
        if resume:
            raise ValueError("cannot resume an uninitialized hybrid run")
        state = (_initialize_checkpoint(output, config) if config.initialization is not None else
                 _initialize_fork(output, prepare_fork(config, fork_from, fork_round))
                 if fork_from is not None else
                 {"protocol": PROTOCOL, "completed_rounds": [], "latest_checkpoint": None})
        write_json(state_path, state)
        # config.json is the initialization commit marker used by resumable runners.
        write_json(cfg_path, config.to_dict())
    fork = _load_fork(output, state, config)
    learner_config = (replace(config.offline.learner, learning_rate=config.learning_rate_override)
                      if config.learning_rate_override is not None else config.offline.learner)
    offset = state.get("online_round_offset", 0)
    inherited_ids = {g["run_id"] for g in fork["games"]} if fork else None
    inherited_keys = {(g["round_index"], g["game_id"]) for g in fork["games"]} if fork else None
    cache_bytes = (config.offline.replay_cache_bytes if replay_cache_bytes_override is None
                   else replay_cache_bytes_override)
    session = str(time.time_ns())
    write_json(output / "source" / session / "identity.json",
               {**source_identity(Path.cwd(), output / "source" / session), "command": sys.argv,
                "torch": torch.__version__, "device": config.offline.learner.device,
                "online_rounds_target": online_rounds, "forever": forever,
                "frozen_config_sha256": sha256(cfg_path),
                "replay_cache_capacity_bytes": cache_bytes,
                "effective_learner_config": asdict(learner_config),
                "inductor_compile_threads_env": os.environ.get("TORCHINDUCTOR_COMPILE_THREADS"),
                "collection_execution": asdict(execution)})
    offline_root = output / "offline"
    anchor = output / ("checkpoints/starting_model.pt" if config.initialization is not None
                       else "checkpoints/theta_5000.pt")
    if state["latest_checkpoint"] is None:
        previous = sorted((offline_root / "checkpoints").glob("step-*.pt"))
        before = load_alphazero_checkpoint(previous[-1]).total_train_step if previous else 0
        result = run_offline(config.offline, offline_root,
                             resume=(offline_root / "config.json").exists(), max_steps_this_session=budget,
                             replay_cache_bytes_override=cache_bytes)
        completed = result["steps_completed"]
        if budget is not None:
            budget -= completed - before
        if completed < config.offline.max_steps:
            return state
        anchor.parent.mkdir(parents=True, exist_ok=True)
        _copy_checkpoint(offline_root / "checkpoints/final.pt", anchor)
        state["latest_checkpoint"] = _identity(anchor)
        state["anchor_checkpoint"] = state["latest_checkpoint"]
        write_json(state_path, state)
    if sha256(anchor) != state["anchor_checkpoint"]["sha256"]:
        raise ValueError("fixed theta_5000 checkpoint changed")
    evaluation_anchor = (Path(config.evaluation_anchor["path"]).resolve()
                         if config.evaluation_anchor is not None else anchor)
    dataset = off_index = off_pool = old_fixed = None
    cache_half = 0 if config.online_only else cache_bytes // 2
    if not config.online_only:
        # Legacy runs continue to verify and observe their frozen offline pool.
        dataset = read_json(offline_root / "dataset.json")
        if frozen_offline_dataset(config)["digest"] != dataset["digest"]:
            raise ValueError("original offline dataset changed")
        off_index = make_index(dataset, cache_half)
        if config.initialization is not None:
            from .compact_game_cache import CompactGameCache
            off_index.cache = CompactGameCache(cache_half)
        off_pool = PositionPool(off_index.split_games("train"), off_index.cache, "offline")
        old_fixed = read_json(offline_root / "fixed_validation.json")
        offline_holdout_keys = {(ref["round_index"], ref["game_id"]) for ref in old_fixed["references"]}
    _emit("replay_cache_capacity", offline_bytes=cache_half, online_bytes=cache_bytes - cache_half,
          frozen_total_bytes=config.offline.replay_cache_bytes)
    online_cache = None
    while forever or len(state["completed_rounds"]) < online_rounds:
        if budget is not None and budget <= 0:
            break
        r = len(state["completed_rounds"])
        display_round = offset + r + 1 if config.online_only else r + 1
        quotas = config.quotas(r)
        if quotas[0] == 0 and off_index is not None:
            removed = off_index.cache.retain(offline_holdout_keys)
            if removed:
                _emit("replay_cache_pruned", round=display_round, source="offline", removed_games=removed,
                      **off_index.cache.stats())
        data_round = offset + r
        stage = output / "rounds" / f"round-{r:03d}"
        teacher = state["latest_checkpoint"]
        if sha256(Path(teacher["path"])) != teacher["sha256"]:
            raise ValueError("collection teacher checkpoint changed")
        raw = asdict(config.selfplay)
        raw.update(run_id=f"{config.run_id}-round-{r:03d}", round_index=data_round, checkpoint=teacher["path"],
                   num_games=config.new_games_per_round, seed=config.seeds.collection_for(data_round),
                   split_seed=config.seeds.split_for(data_round))
        root = stage / "selfplay"
        manifest_path = root / "manifest.json"
        if manifest_path.exists():
            # Execution is part of the data fingerprint. Finish an existing
            # collection with its original sizes; tune only unstarted rounds.
            frozen_execution = asdict(ExecutionConfig(**read_json(manifest_path)["config"]["execution"]))
            if any(raw["execution"].get(name) != frozen_execution[name]
                   for name in COLLECTION_BATCHING_FIELDS):
                _emit("collection_batching_preserved", round=display_round,
                      **{name: frozen_execution[name] for name in COLLECTION_BATCHING_FIELDS})
            for name in COLLECTION_BATCHING_FIELDS:
                raw["execution"][name] = frozen_execution[name]
        collection = FastSelfPlayConfig.from_dict(raw)
        _emit("collect_start", round=display_round, teacher=teacher["path"], games=config.new_games_per_round)
        devices = list(range(torch.cuda.device_count())) if config.offline.learner.device == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            try:
                summary = generate(collection, root, resume=(root / "manifest.json").exists(),
                                   progress_callback=lambda done, total: _emit("collect_progress", round=display_round,
                                                                               completed=done, total=total))
            finally:
                _release_inference_memory()
        if summary["completed_games"] != config.new_games_per_round or summary["missing_games"]:
            raise RuntimeError("online collection is incomplete")
        if config.policy_averaging is not None:
            _emit("policy_averaging_start", round=display_round, samples=4, behavior_search_reused=True, extra_searches=3)
            with torch.random.fork_rng(devices=devices):
                try:
                    generate_labels(root, stage / "policy_labels", config.policy_averaging,
                                    device=config.offline.learner.device,
                                    progress_callback=lambda block, repeat, done, total:
                                    _emit("policy_averaging_search" if block < 0 else "policy_averaging_chunk",
                                          round=display_round, block=block, repeat=repeat,
                                          completed_roots=done, total_roots=total))
                finally:
                    _release_inference_memory()
        runs = _replay_runs(config, output, r, offset, fork)
        allowed_keys = (inherited_keys | {(run.round_index, gid) for run in runs if run.round_index >= offset
                                         for gid in range(config.new_games_per_round)} if fork else None)
        index = build_replay_index(runs, max_games=config.replay_max_games,
                                   cache_bytes=cache_bytes - cache_half,
                                   state_path=output / "online_replay_index.json", allowed_game_keys=allowed_keys)
        if online_cache is not None:
            index.cache = online_cache
        elif config.initialization is not None or config.online_only:
            from .compact_game_cache import CompactGameCache
            index.cache = CompactGameCache(cache_bytes - cache_half)
        online_cache = index.cache
        fixed_online = _fixed_online(config, index, output, first_online_round=offset)
        online_keep_keys = {game.key for game in index.games}
        if fixed_online is not None:
            online_keep_keys.update(game.key for game in fixed_online[0].games)
        removed = online_cache.retain(online_keep_keys)
        if removed:
            _emit("replay_cache_pruned", round=display_round, source="online", removed_games=removed,
                  **online_cache.stats())
        online_pool = PositionPool(index.split_games("train"), index.cache, "online")
        if not len(online_pool):
            raise ValueError("online training pool is empty")
        snapshot = {"games": [g.to_dict() for g in index.games], "teacher": teacher,
                    "quotas": list(quotas), "round_index": data_round}
        if config.policy_surprise_weighting is not None:
            snapshot["policy_surprise_weighting"] = asdict(config.policy_surprise_weighting)
        target_store = None
        if config.policy_averaging is not None:
            target_store = AveragedTargetStore(index.games, required_run_ids={run.run_id for run in runs
                                              if run.round_index >= offset})
            snapshot["policy_labels"] = target_store.manifests
        split_pools = None
        policy_pool = None
        joint_pool = None
        if config.policy_replay_max_games is not None:
            joint_pool = joint_policy_pool(index, config.policy_replay_max_games)
            snapshot["joint_head_pools"] = joint_replay_description(index, joint_pool, online_pool, config)
        if config.split_training is not None:
            recent_pool, history_pool, recent_rounds = split_replay_pools(index, config)
            split_pools = (recent_pool, history_pool)
            policy_pool = split_policy_pool(index, config, recent_pool)
            snapshot["split_training"] = split_replay_description(index, config, split_pools, recent_rounds)
            if target_store is not None:
                teachers = snapshot["split_training"]["teachers"]
                for info in teachers:
                    info["policy_target_samples"] = 4 if info["run_id"] in target_store.manifests else 1
                snapshot["split_training"]["policy_expected_m4_fraction"] = sum(
                    info["policy_expected_fraction"] for info in teachers if info["policy_target_samples"] == 4)
        if fork:
            inherited_games = [g for g in index.games if g.run_id in inherited_ids]
            inherited_positions = sum(g.decision_count for g in inherited_games if g.split == "train")
            snapshot["replacement"] = {"inherited_games": len(inherited_games),
                "branch_games": index.total_games - len(inherited_games),
                "inherited_train_positions": inherited_positions,
                "branch_train_positions": len(online_pool) - inherited_positions,
                "branch_online_position_fraction": 1 - inherited_positions / len(online_pool)}
        snapshot["digest"] = _digest(snapshot)
        snap_path = stage / "replay_snapshot.json"
        if snap_path.exists() and read_json(snap_path) != snapshot:
            raise ValueError("online training snapshot changed on resume")
        write_json(snap_path, snapshot)
        identity = {"protocol": PROTOCOL,
                    "digest": _digest([dataset["digest"], snapshot["digest"]]) if dataset else snapshot["digest"],
                    "online": {"digest": snapshot["digest"], "dataset_path": str(snap_path),
                               "run_ids": sorted({g.run_id for g in index.games})}}
        if dataset is not None:
            identity["offline"] = {"digest": dataset["digest"], "dataset_path": str(offline_root / "dataset.json"),
                                   "run_ids": sorted({g["run_id"] for g in dataset["games"]})}
        if split_pools is not None:
            identity["split_training"] = snapshot["split_training"]
            for name in ("policy_recent", "value_recent", "value_history"):
                identity[name] = {**identity["online"], "head_bucket": name,
                                  "split_training": snapshot["split_training"]}
        if joint_pool is not None:
            identity["joint_head_pools"] = snapshot["joint_head_pools"]
            for name in ("policy", "value"):
                identity[name] = {**identity["online"], "head_bucket": name,
                                  "joint_head_pools": snapshot["joint_head_pools"]}
        if config.policy_surprise_weighting is not None:
            identity["policy_surprise_weighting"] = asdict(config.policy_surprise_weighting)
        cps = stage / "checkpoints"
        steps = sorted(cps.glob("step-*.pt"))
        if steps or (cps / "initial.pt").exists():
            learner = AlphazeroLearner.resume(steps[-1] if steps else cps / "initial.pt", learner_config,
                      replay_identity=identity, round_index=data_round + 1, round_steps_target=config.updates_per_round)
        else:
            payload = load_alphazero_checkpoint(teacher["path"])
            first_override = config.learning_rate_override is not None and r == 0 and fork is not None
            warm_start = config.initialization is not None and r == 0
            learner = AlphazeroLearner.resume(teacher["path"],
                      type(learner_config)(**payload.learner_config) if warm_start else
                      config.offline.learner if first_override else learner_config,
                      replay_identity=payload.replay_identity, round_index=payload.round_index,
                      round_steps_target=payload.round_steps_target)
            if first_override:
                learner.config = learner_config
                for group in learner.optimizer.param_groups:
                    group["lr"] = learner_config.learning_rate
            if warm_start:
                learner.config = learner_config
            learner.round_index, learner.round_train_step = data_round + 1, 0
            learner.round_steps_target = config.updates_per_round
            learner.replay_identity, learner.parent_checkpoint = identity, teacher
            learner.save(cps / "initial.pt")
        if any(group["lr"] != learner_config.learning_rate for group in learner.optimizer.param_groups):
            raise ValueError("optimizer learning rate differs from effective learner configuration")
        current = make_index({"games": [g.to_dict() for g in index.games if g.round_index == data_round]}, index.cache.max_bytes)
        current.cache = index.cache
        recent_validation = None
        if split_pools is not None:
            recent_validation = make_index({"games": [g.to_dict() for g in index.games
                                            if g.round_index in recent_rounds]}, index.cache.max_bytes)
            recent_validation.cache = index.cache
        before_path = stage / "validation_before.json"
        validation_kwargs = {"policy_transform": target_store.transform} if target_store is not None else {}
        if not before_path.exists():
            before = {"total_train_step": learner.total_train_step,
                      "current": evaluate_validation(learner, current, **validation_kwargs)}
            if recent_validation is not None:
                before["recent"] = evaluate_validation(learner, recent_validation, **validation_kwargs)
            if target_store is not None:
                before["policy_target_scope"] = "current=M4; recent=mixed legacy single/M4 by teacher; value unchanged"
            write_json(before_path, before)
        split_grad_norms = None
        if joint_pool is not None:
            batch_size = config.offline.learner.batch_size
            writer = HybridReferenceWriter(stage / "usage", session, identity,
                                           (batch_size, batch_size), ("policy", "value"))
            drawer = lambda rng: draw_joint_head_batch(joint_pool, online_pool, rng, batch_size,
                                                       inherited_run_ids=inherited_ids)
        elif split_pools is None:
            writer = (HybridReferenceWriter(stage / "usage", session, identity, (quotas[1],), ("online",))
                      if config.online_only else HybridReferenceWriter(stage / "usage", session, identity, quotas))
            drawer = lambda rng: draw_mixed_batch(off_pool, online_pool, rng, *quotas,
                inherited_run_ids=inherited_ids, record_rounds=config.fitting_diagnostics is not None)
        else:
            split = config.split_training
            names = ("policy_recent", "value_recent", "value_history")
            writer = HybridReferenceWriter(stage / "usage", session, identity,
                tuple(snapshot["split_training"]["quotas"][name] for name in names), names)
            drawer = lambda rng: draw_split_batch(*split_pools, rng, config.offline.learner.batch_size,
                split.value_recent_positions, inherited_run_ids=inherited_ids, policy_pool=policy_pool)
            split_grad_norms = (split.policy_max_grad_norm, split.value_max_grad_norm)
        if target_store is not None:
            original_drawer = drawer
            def drawer(rng):
                batch = original_drawer(rng)
                sources = {**batch.source_positions, "policy_m4": sum(target_store.is_averaged(s) for s in batch.samples)}
                return replace(batch, samples=tuple(target_store.transform(s) for s in batch.samples),
                               source_positions=sources)
        fitting_observer = None
        if config.fitting_diagnostics is not None:
            from .pool_fitting import RoundFitting
            fitting_observer = RoundFitting(config.fitting_diagnostics, output, stage, off_index, index, dataset,
                inherited_references=fork.get("train_references") if fork else None)
            fitting_observer(learner, cps / (f"step-{learner.round_train_step:09d}.pt"
                                           if learner.round_train_step else "initial.pt"))
        start_step = learner.round_train_step
        _emit("train_start", round=display_round, global_step=learner.total_train_step,
              offline_positions=quotas[0], online_positions=quotas[1], target=config.updates_per_round)
        try:
            run_training_round(learner, steps_target=config.updates_per_round,
                               current_pool=online_pool, history_pool=None,
                               validation_samples=(), validation_games=0, checkpoint_dir=cps,
                               metrics_path=stage / "training_metrics.jsonl", reference_writer=writer,
                               batch_drawer=drawer, split_grad_norms=split_grad_norms,
                               joint_separate_batches=joint_pool is not None,
                               policy_surprise_weighting=config.policy_surprise_weighting,
                               checkpoint_callback=fitting_observer,
                               max_steps_this_session=budget,
                               progress_callback=lambda row: _emit("train_progress", round=display_round,
                                   global_step=row["total_train_step"], step=row["round_train_step"],
                                   loss=row["total_loss"]) if row["round_train_step"] % 100 == 0 else None)
        finally:
            writer.close()
        if budget is not None:
            budget -= learner.round_train_step - start_step
        if learner.round_train_step < config.updates_per_round:
            break
        final = cps / "final.pt"
        _copy_checkpoint(cps / f"step-{learner.round_train_step:09d}.pt", final)
        validation = {"total_train_step": learner.total_train_step,
                      "offline_fixed": evaluate_validation(learner, off_index, old_fixed) if off_index else None,
                      "current": evaluate_validation(learner, current, **validation_kwargs)}
        if recent_validation is not None:
            validation["recent"] = evaluate_validation(learner, recent_validation, **validation_kwargs)
        if target_store is not None:
            validation["policy_target_scope"] = "current=M4; recent=mixed legacy single/M4 by teacher; value unchanged"
        if fixed_online is not None:
            validation["online_fixed"] = evaluate_validation(learner, *fixed_online)
        write_json(stage / "validation_after.json", validation)
        usage = {
            name: {"draws": config.updates_per_round * count, "pool_positions": len(pool) if pool else 0,
                   "average_exposure": config.updates_per_round * count / len(pool) if pool else None}
            for name, count, pool in zip(("offline", "online"), quotas, (off_pool, online_pool))}
        if config.fitting_diagnostics is not None and joint_pool is None:
            by_round = {}
            for line in (stage / "training_metrics.jsonl").read_text().splitlines():
                record = json.loads(line)
                if record["event"] == "train_step":
                    for key, count in record["source_positions"].items():
                        if key.startswith("online_round_"):
                            by_round[key] = by_round.get(key, 0) + count
            usage["online_by_generation"] = [{"round_index": gen,
                "train_positions": sum(g.decision_count for g in index.split_games("train") if g.round_index == gen),
                "actual_draws": by_round.get(f"online_round_{gen}", 0)}
                for gen in sorted({g.round_index for g in index.games})]
            for entry in usage["online_by_generation"]:
                entry["average_exposure"] = entry["actual_draws"] / entry["train_positions"]
        if joint_pool is not None:
            count = config.updates_per_round * config.offline.learner.batch_size
            usage = {name: {"draws": count, "pool_positions": len(pool),
                            "average_exposure": count / len(pool)}
                     for name, pool in (("policy", joint_pool), ("value", online_pool))}
            by_round = {}
            for line in (stage / "training_metrics.jsonl").read_text().splitlines():
                record = json.loads(line)
                if record["event"] == "train_step":
                    for key, draws in record["source_positions"].items():
                        if key.startswith(("policy_round_", "value_round_")):
                            by_round[key] = by_round.get(key, 0) + draws
            usage["joint_head_pools"] = {**snapshot["joint_head_pools"],
                                         "actual_draws_by_head_round": by_round}
        if split_pools is not None:
            counts = snapshot["split_training"]["quotas"]
            usage = {name: {"draws": config.updates_per_round * counts[name], "pool_positions": len(pool),
                           "average_exposure": config.updates_per_round * counts[name] / len(pool)}
                     for name, pool in zip(("policy_recent", "value_recent", "value_history"),
                                           (policy_pool, split_pools[0], split_pools[1]))}
            by_round = {}
            m4_draws = 0
            for line in (stage / "training_metrics.jsonl").read_text().splitlines():
                record = json.loads(line)
                if record["event"] == "train_step":
                    if target_store is not None:
                        m4_draws += record["source_positions"]["policy_m4"]
                    for key, count in record["source_positions"].items():
                        if "_round_" in key:
                            by_round[key] = by_round.get(key, 0) + count
            usage["split_training"] = {**snapshot["split_training"], "actual_draws_by_head_round": by_round}
            if target_store is not None:
                usage["policy_averaging"] = {"samples": 4, "behavior_search_reused": True,
                    "policy_m4_draws": m4_draws, "policy_total_draws": config.updates_per_round * counts["policy_recent"],
                    "actual_policy_m4_fraction": m4_draws / (config.updates_per_round * counts["policy_recent"]),
                    "supervision_manifests": target_store.manifests}
        if fork:
            draws = {"inherited_online": 0, "branch_online": 0}
            for line in (stage / "training_metrics.jsonl").read_text().splitlines():
                record = json.loads(line)
                if record["event"] == "train_step":
                    for source in draws:
                        draws[source] += record["source_positions"][source]
            usage["online_replacement"] = {**snapshot["replacement"], "actual_draws": draws}
        write_json(stage / "source_usage.json", usage)
        evaluation_round = data_round + 1 if config.online_only else r + 1
        if config.evaluation_every_rounds and evaluation_round % config.evaluation_every_rounds == 0:
            with torch.random.fork_rng(devices=devices):
                try:
                    _evaluation(config, final, evaluation_anchor, stage)
                finally:
                    _release_inference_memory()
        if round_callback is not None:
            with torch.random.fork_rng(devices=devices):
                try:
                    round_callback(display_round, final, stage)
                finally:
                    _release_inference_memory()
        state["latest_checkpoint"] = _identity(final)
        state["completed_rounds"].append(r)
        write_json(state_path, state)
        round_record = {"round_index": r, "teacher": teacher,
                       "checkpoint": state["latest_checkpoint"], "quotas": list(quotas), "validation": validation}
        if config.online_only:
            round_record.update(absolute_round=data_round + 1, local_round=r + 1)
        append_json(output / "round_metrics.jsonl", round_record)
        if round_commit_callback is not None:
            round_commit_callback(data_round + 1, final, stage, state)
        _emit("round_complete", round=display_round, global_step=learner.total_train_step,
              checkpoint=str(final), online_fraction=quotas[1] / config.offline.learner.batch_size)
        del learner
        gc.collect()
        if config.offline.learner.device == "cuda":
            torch.cuda.empty_cache()
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--resume", action="store_true")
    initialization.add_argument("--fork-from", type=Path, help="initialize from a completed hybrid run; skip offline fitting")
    parser.add_argument("--fork-round", type=int, help="human-numbered parent round; default latest completed")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--online-rounds", type=int, help="absolute online round target, not additional rounds")
    group.add_argument("--forever", action="store_true", help="continue latest-model self-iteration until interrupted")
    parser.add_argument("--max-steps-this-session", type=int)
    parser.add_argument("--replay-cache-mib", type=int,
                        help="execution-only total cache budget; does not change the frozen training plan")
    parser.add_argument("--check-config", action="store_true", help="validate config/source metadata without training")
    args = parser.parse_args(argv)
    config = HybridConfig.from_dict(read_json(args.config))
    if args.fork_round is not None and args.fork_from is None:
        parser.error("--fork-round requires --fork-from")
    if args.check_config:
        data = (prepare_initialization(config) if config.initialization is not None
                else freeze_dataset(config.offline))
        fork = prepare_fork(config, args.fork_from, args.fork_round) if args.fork_from else None
        if config.evaluation_anchor is not None:
            if sha256(Path(config.evaluation_anchor["path"])) != config.evaluation_anchor["sha256"]:
                raise ValueError("external fixed evaluation checkpoint changed")
            FixedEvaluationConfig.from_dict(config.evaluation)
        _emit("config_valid", source_games=len(data["games"]), offline_steps=config.offline.max_steps,
              fork=({"parent_round": fork["parent_round"], "global_step": fork["parent_checkpoint"]["total_train_step"],
                     "inherited_games": len(fork["games"]), "online_round_offset": fork["online_round_offset"]} if fork else None),
              schedule=[{"round": r + 1, "new_games": (r + 1) * config.new_games_per_round,
                         "quotas": config.quotas(r)} for r in range(max(16, config.mix_schedule[-1][0] + 1)
                                                                    if config.mix_schedule else 16)],
              execution=config.selfplay.execution)
        return 0
    if args.output is None or (args.online_rounds is None and not args.forever):
        parser.error("training requires --output and either --online-rounds or --forever")
    run_hybrid(config, args.output, resume=args.resume, online_rounds=args.online_rounds,
               forever=args.forever, max_steps_this_session=args.max_steps_this_session,
               fork_from=args.fork_from, fork_round=args.fork_round,
               replay_cache_bytes_override=None if args.replay_cache_mib is None
                                           else args.replay_cache_mib * 1024**2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
