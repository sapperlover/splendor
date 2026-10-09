"""Resumable M4 targets: archived behavior search plus three independent searches.

Behavior archives remain immutable. Separate numeric supervision is applied to
policy training/validation only; original visits, actions and returns are kept.
"""
from __future__ import annotations

import fcntl
import json
import os
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch

from .alphazero_recording import environment_identity, read_json, sha256, write_json
from .catalog import load_default_catalog
from .contracts import DecisionRequest
from .engine import GameEngine
from .mcts_policy import MCTSConfig
from .native_policy import NativeBatchRunner, NativePUCTSearchPolicy
from .opponent_pool import load_frozen_actor_critic_checkpoint
from .replay import to_primitive
from .result import TerminalResult
from .search_inference import FrozenSearchInference, InferenceConfig
from .selfplay import derive_seed
from .selfplay_data import load_game

PROTOCOL = "policy-target-behavior-plus3-v1"


@dataclass(frozen=True)
class PolicyAveragingConfig:
    samples: int = 4
    world_seed: int = 26300000
    noise_seed: int = 26400000
    active_roots: int = 64
    games_per_block: int = 8
    search_workers: int = 1
    max_batch_size: int = 200
    local_microbatch: int = 64
    shared_slots: int = 2

    def __post_init__(self):
        if self.samples != 4:
            raise ValueError("this protocol requires behavior + 3 searches, samples=4")
        for name in ("world_seed", "noise_seed", "active_roots", "games_per_block", "search_workers",
                     "max_batch_size", "local_microbatch", "shared_slots"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.shared_slots not in (1, 2):
            raise ValueError("shared_slots must be 1 or 2")


def label_root(archive: Path) -> Path:
    return archive.parent.parent.parent / "policy_labels"


def _npz(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".npz.tmp")
    with temp.open("wb") as f:
        np.savez_compressed(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def _code_identity():
    root = Path(__file__).parent
    names = ("policy_averaging.py", "policy_averaging_pool.py", "multiworker_gpu.py",
             "selfplay_fast.py", "native_policy.py", "native_search.cpp", "mcts_policy.py",
             "search_policy.py", "features.py", "actor_critic.py", "entity_attention.py",
             "search_inference.py", "engine.py", "rules.py", "compact_trace.py", "selfplay.py")
    return {"files": {n: sha256(root / n) for n in names},
            "native_extensions": {p.name: sha256(p) for p in root.glob("_native_search*.so")}}


def replay_roots(arrays, metadata):
    """Reconstruct only real archived decisions, preserving candidate ordering."""
    engine = GameEngine.new(metadata["game_seed"], first_player=metadata["first_player"], mode="checked")
    offsets = arrays["candidate_offsets"]
    roots = []
    for i, action in enumerate(metadata["trace"]["actions"]):
        request = engine.decision()
        chosen = int(action["candidate_index"])
        if (not isinstance(request, DecisionRequest) or action["decision_index"] != i
                or action["player_id"] != engine.current_player
                or chosen != int(arrays["chosen_candidate_index"][i])
                or to_primitive(request.legal_actions[chosen]) != action["action"]
                or len(request.legal_actions) != int(offsets[i + 1] - offsets[i])):
            raise ValueError("archive trace/candidate identity mismatch")
        if arrays["searchable"][i]:
            roots.append((i, request))
        engine.apply(request.legal_actions[chosen])
    terminal = engine.decision()
    if not isinstance(terminal, TerminalResult) or to_primitive(terminal) != metadata["terminal_result"]:
        raise ValueError("archive did not reproduce its natural terminal")
    return roots


def generate_labels(collection_root: Path, output: Path, config: PolicyAveragingConfig,
                    *, device: str, progress_callback=None):
    """Save progress after each repeat/chunk; completion commits per-game means."""
    collection_root, output = Path(collection_root).resolve(), Path(output).resolve()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source_manifest = read_json(collection_root / "manifest.json")
    raw = source_manifest["config"]
    if raw.get("search_backend", "native") != "native":
        raise ValueError("behavior-plus3 currently supports native determinization only")
    search = raw["search_config"]
    if search["determinizations"] != 1:
        raise ValueError("behavior-plus3 uses complete N1 searches")
    checkpoint = Path(raw["checkpoint"])
    files = sorted((collection_root / "games").glob("game-*.npz"))
    if len(files) != raw["num_games"]:
        raise ValueError("averaging requires all source games to be committed")
    identity = {"protocol": PROTOCOL, "config": asdict(config), "search": search,
                "add_root_noise": raw["add_root_noise"], "teacher_sha256": sha256(checkpoint),
                "source_manifest_sha256": sha256(collection_root / "manifest.json"),
                "sources": {p.name: sha256(p) for p in files}, "code": _code_identity(),
                "device": device}
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        frozen = output / "identity.json"
        if frozen.exists():
            if read_json(frozen) != identity:
                raise ValueError("averaging identity changed; cannot resume")
        else:
            write_json(frozen, identity)
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            if manifest["identity_sha256"] != sha256(frozen):
                raise ValueError("averaged supervision identity changed")
            for record in manifest["games"]:
                if sha256(output / record["file"]) != record["sha256"]:
                    raise ValueError("averaged supervision changed")
            return manifest
        started = time.perf_counter()
        model = load_frozen_actor_critic_checkpoint(checkpoint, device=device)
        # Eager FP32 uses the existing contiguous native transport, bounded by active_roots.
        evaluator = FrozenSearchInference(model, device=device,
                                          config=InferenceConfig(mode="eager", compile_failure_policy="fail_fast"))
        pooled_source = lookup = None
        if config.search_workers > 1:
            from .policy_averaging_pool import run_pool, read_task
            pooled_source, lookup = run_pool(files, raw, output, config, evaluator, sha256(frozen),
                                             identity["teacher_sha256"], progress_callback)
        entries = []
        extra_simulations = 0
        for block_start in range(0, len(files), config.games_per_block):
            block_files = files[block_start:block_start + config.games_per_block]
            block = [load_game(p) for p in block_files]
            if any(meta["checkpoint_sha256"] != identity["teacher_sha256"] for _, meta in block):
                raise ValueError("source archive teacher differs from averaging teacher")
            roots = [(g, i, request) for g, (arrays, metadata) in enumerate(block)
                     for i, request in replay_roots(arrays, metadata)]
            width = max((len(request.legal_actions) for _, _, request in roots), default=1)
            extra_targets = np.zeros((3, len(roots), width), dtype=np.float64)
            extra_counts = np.zeros((3, len(roots), width), dtype=np.int32)
            world_seeds = np.zeros((3, len(roots)), dtype=np.uint64)
            noise_seeds = np.zeros((3, len(roots)), dtype=np.uint64)
            for rep in range(3):
                runner = (NativeBatchRunner(evaluator, load_default_catalog(), capacity=config.active_roots,
                                            buckets=(16, 32, 48, 64)) if pooled_source is None else None)
                for start in range(0, len(roots), config.active_roots):
                    part = output / "work.games" / f"block-{block_start:04d}-rep-{rep}-chunk-{start:05d}.npz"
                    chunk = roots[start:start + config.active_roots]
                    seed_pairs = []
                    for g, i, _ in chunk:
                        meta = block[g][1]
                        seed_pairs.append((derive_seed(config.world_seed, raw["round_index"], meta["game_id"],
                                                       "m4_extra_world", rep, i),
                                           derive_seed(config.noise_seed, raw["round_index"], meta["game_id"],
                                                       "m4_extra_noise", rep, i)))
                    if not part.exists():
                        targets = np.zeros((len(chunk), width), dtype=np.float64)
                        counts = np.zeros((len(chunk), width), dtype=np.int32)
                        if pooled_source is not None:
                            for j, (g, i, request) in enumerate(chunk):
                                target, count = read_task(output, pooled_source, lookup[block_start + g, i, rep],
                                                          sha256(frozen))
                                targets[j, :len(request.legal_actions)] = target
                                counts[j, :len(request.legal_actions)] = count
                        else:
                            jobs = []
                            for (_, _, request), (world, noise) in zip(chunk, seed_pairs, strict=True):
                                policy = NativePUCTSearchPolicy(model, evaluator=evaluator,
                                                               config=MCTSConfig(**{**search, "seed": world}))
                                jobs.append(policy.start_job(request, add_root_noise=raw["add_root_noise"],
                                                             temperature=0., sampling_seed=noise))
                            while any(not job.done for job in jobs):
                                if runner.run([j for j in jobs if not j.done]) <= 0:
                                    raise RuntimeError("averaging search made no progress")
                            for j, job in enumerate(jobs):
                                result = job.result
                                if not result.searchable or sum(result.root_visit_counts) != search["simulations"]:
                                    raise ValueError("averaging search budget mismatch")
                                targets[j, :len(result.policy_target)] = result.policy_target
                                counts[j, :len(result.root_visit_counts)] = result.root_visit_counts
                            runner.batch_records.clear()
                            del jobs, policy, job
                        _npz(part, targets=targets, counts=counts,
                             seed_pairs=np.asarray(seed_pairs, dtype=np.uint64))
                    with np.load(part, allow_pickle=False) as z:
                        if z["targets"].shape != (len(chunk), width) or not np.array_equal(
                                z["seed_pairs"], np.asarray(seed_pairs, dtype=np.uint64)) or (
                                z["counts"].shape != z["targets"].shape or
                                not np.all(z["counts"].sum(axis=1) == search["simulations"])):
                            raise ValueError("saved averaging chunk identity mismatch")
                        extra_targets[rep, start:start + len(chunk)] = z["targets"]
                        extra_counts[rep, start:start + len(chunk)] = z["counts"]
                        world_seeds[rep, start:start + len(chunk)] = z["seed_pairs"][:, 0]
                        noise_seeds[rep, start:start + len(chunk)] = z["seed_pairs"][:, 1]
                    if progress_callback:
                        progress_callback(block_start, rep + 1, start + len(chunk), len(roots))
            for g, (arrays, meta) in enumerate(block):
                original = arrays["policy_target"]
                targets = np.tile(original.astype(np.float64), (4, 1))
                counts = np.zeros((3, len(original)), dtype=np.int32)
                worlds = np.zeros((3, len(arrays["searchable"])), dtype=np.uint64)
                noises = np.zeros_like(worlds)
                for j, (source, i, request) in enumerate(roots):
                    if source != g:
                        continue
                    a, b = arrays["candidate_offsets"][i:i + 2]
                    targets[1:, a:b] = extra_targets[:, j, :len(request.legal_actions)]
                    counts[:, a:b] = extra_counts[:, j, :len(request.legal_actions)]
                    worlds[:, i], noises[:, i] = world_seeds[:, j], noise_seeds[:, j]
                mean = targets.mean(axis=0).astype(np.float32)
                for a, b in zip(arrays["candidate_offsets"][:-1], arrays["candidate_offsets"][1:]):
                    if not np.isfinite(mean[a:b]).all() or (mean[a:b] < 0).any() or abs(mean[a:b].sum() - 1) > 1e-6:
                        raise ValueError("invalid averaged policy distribution")
                destination = output / "games" / block_files[g].name
                _npz(destination, target_mean=mean, targets=targets, extra_root_visit_counts=counts,
                     candidate_offsets=arrays["candidate_offsets"], searchable=arrays["searchable"],
                     extra_world_seeds=worlds, extra_noise_seeds=noises)
                entries.append({"game_id": meta["game_id"], "run_id": meta["run_id"], "split": meta["split"],
                                "file": str(destination.relative_to(output)), "sha256": sha256(destination),
                                "archive_sha256": identity["sources"][block_files[g].name],
                                "searchable_positions": int(arrays["searchable"].sum())})
                extra_simulations += int(arrays["searchable"].sum()) * 3 * search["simulations"]
            print(json.dumps({"event": "policy_averaging_progress", "games": len(entries), "total": len(files),
                              "seconds": time.perf_counter() - started}), flush=True)
        manifest = {"protocol": PROTOCOL, "status": "complete", "identity_sha256": sha256(frozen),
                    "samples": 4, "behavior_search_reused": True, "extra_searches": 3,
                    "teacher_sha256": identity["teacher_sha256"], "source_manifest_sha256": identity["source_manifest_sha256"],
                    "games": entries, "extra_simulations": extra_simulations,
                    "seconds_this_invocation": time.perf_counter() - started, "environment": environment_identity()}
        if pooled_source is not None:
            manifest["execution"] = {"backend": "shared_memory_root_pool", "search_workers": config.search_workers,
                "active_roots": config.active_roots, "max_batch_size": config.max_batch_size,
                "local_microbatch": config.local_microbatch, "shared_slots": config.shared_slots,
                "inference_mode": "eager", "device": device,
                "statistics": [str(p.relative_to(output)) for p in sorted((output / "pool_statistics").glob("*.json"))]}
        write_json(manifest_path, manifest)
        return manifest


class AveragedTargetStore:
    """Small bounded cache of policy-only means, including mixed legacy/M4 history."""

    def __init__(self, games, *, required_run_ids=(), cache_games=1024):
        self.records, self.manifests = {}, {}
        self.cache = OrderedDict()
        self.cache_games = cache_games
        for game in games:
            root = label_root(game.path)
            path = root / "manifest.json"
            if not path.exists():
                if game.run_id in required_run_ids:
                    raise ValueError("new M4 teacher has no completed supervision")
                continue
            if game.run_id not in self.manifests:
                manifest = read_json(path)
                if manifest["protocol"] != PROTOCOL or manifest["status"] != "complete" or manifest["samples"] != 4:
                    raise ValueError("invalid M4 supervision manifest")
                self.manifests[game.run_id] = {"path": str(path), "sha256": sha256(path)}
                for entry in manifest["games"]:
                    self.records[(game.run_id, entry["game_id"])] = (root, entry)
            if (game.run_id, game.game_id) not in self.records:
                raise ValueError("M4 manifest is missing an active game")

    def is_averaged(self, sample):
        return (sample.metadata["run_id"], sample.metadata["game_id"]) in self.records

    def transform(self, sample):
        key = (sample.metadata["run_id"], sample.metadata["game_id"])
        if key not in self.records:
            return sample
        mean = self.cache.get(key)
        if mean is None:
            root, entry = self.records[key]
            path = root / entry["file"]
            if sha256(path) != entry["sha256"]:
                raise ValueError("M4 supervision file changed")
            with np.load(path, allow_pickle=False) as z:
                if not np.array_equal(z["candidate_offsets"], sample.arrays["candidate_offsets"]):
                    raise ValueError("M4 candidate ordering changed")
                if not np.array_equal(z["targets"][0].astype(np.float32), sample.arrays["policy_target"]):
                    raise ValueError("M4 behavior target differs from original archive")
                mean = z["target_mean"].copy()
            self.cache[key] = mean
            if len(self.cache) > self.cache_games:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return replace(sample, arrays={**sample.arrays, "policy_target": mean})
