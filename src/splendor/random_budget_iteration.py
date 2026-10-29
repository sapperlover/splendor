"""Continuous mixed-budget self-play with a frozen historical policy subset."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import uuid
from collections import Counter
import ctypes
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .alphazero_checkpoint import load_alphazero_checkpoint
from .alphazero_loop import LoopSeeds, SelfPlaySection
from .alphazero_recording import read_json, sha256, source_identity, write_json
from .alphazero_replay import PositionPool, ReplayRun, build_replay_index
from .alphazero_train import AlphazeroLearner, DrawnBatch, LearnerConfig, run_training_round
from .compact_game_cache import CompactGameCache
from .continuation_auxiliary import ContinuationAuxiliaryConfig, continuation_config_identity
from .fixed_holdout import check_disjoint, read_manifest
from .six_round_holdout import HoldoutSchedule, check_settings
from .hybrid_train import HybridReferenceWriter, _release_inference_memory
from .offline_train import _digest, make_index
from .pool_iteration import panel
from .rng import DeterministicRng
from .selfplay import derive_seed
from .selfplay_fast import ExecutionConfig, FastSelfPlayConfig, generate

PROTOCOL = "random-budget-iteration-v1"
COLLECTION_TUNING_FIELDS = ("max_batch_size", "batch_buckets", "pipeline_slots")


@dataclass(frozen=True)
class Config:
    run_id: str
    initial_checkpoint: dict
    history_manifest: dict
    selfplay: SelfPlaySection
    learner: LearnerConfig
    seeds: LoopSeeds
    retention_seed: int
    historical_policy_seed: int
    first_round: int = 24
    last_round: int = 43
    expected_games_per_round: int | dict[str, int] = 1000
    new_games_per_round: int = 4000
    replay_max_games: int = 24000
    updates_per_round: int = 1000
    historical_policy_fraction: float = 0.25
    cache_bytes: int = 8 * 1024**3
    validation_positions: int = 2048
    validation_seed: int = 36840000
    validation_steps: tuple[int, ...] = (0, 100, 250, 500, 1000)
    greedy_evaluation: dict | None = None
    search_evaluation: dict | None = None
    evaluation_anchor: dict | None = None
    search_evaluation_every_rounds: int = 3
    continuation_state: dict | None = None
    fixed_holdout: dict | None = None
    six_round_holdout: dict | None = None
    initial_history_eviction: bool = False
    continuation_auxiliary: dict | None = None
    encoder_grad_norm_every_steps: int = 0

    def __post_init__(self):
        if self.fixed_holdout is not None and self.six_round_holdout is not None:
            raise ValueError("choose fixed_holdout or six_round_holdout")
        if not self.run_id or not 1 <= self.first_round <= self.last_round:
            raise ValueError("invalid run/source rounds")
        from .offline_objectives import expected_round_game_counts
        expected_round_game_counts(self.first_round, self.last_round, self.expected_games_per_round)
        if type(self.initial_history_eviction) is not bool:
            raise ValueError("initial_history_eviction must be boolean")
        if self.continuation_auxiliary is not None:
            ContinuationAuxiliaryConfig(**self.continuation_auxiliary)
        if type(self.encoder_grad_norm_every_steps) is not int or self.encoder_grad_norm_every_steps < 0:
            raise ValueError("encoder_grad_norm_every_steps must be a nonnegative integer")
        if self.encoder_grad_norm_every_steps and self.continuation_auxiliary is None:
            raise ValueError("encoder diagnostics require continuation auxiliary")
        for name in ("new_games_per_round", "replay_max_games",
                     "updates_per_round", "cache_bytes", "validation_positions", "search_evaluation_every_rounds"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (not 0 < self.historical_policy_fraction <= 1 or self.validation_positions > 4096
                or self.new_games_per_round > self.replay_max_games):
            raise ValueError("invalid historical policy fraction, validation size or replay capacity")
        if self.validation_steps != tuple(sorted(set(self.validation_steps))) or self.validation_steps[0] != 0 or self.validation_steps[-1] != self.updates_per_round:
            raise ValueError("validation steps must include zero/final in increasing order")
        search = self.selfplay.search_config
        if search.get("forced_playouts", 0) != 0 or search.get("policy_target_pruning", False):
            raise ValueError("this experiment requires forced playouts and pruning disabled")
        if self.selfplay.budget_randomization is None:
            raise ValueError("budget randomization is required")

    @classmethod
    def from_dict(cls, raw):
        values = dict(raw)
        if values.pop("protocol", None) != PROTOCOL:
            raise ValueError("random budget iteration protocol mismatch")
        values["selfplay"] = SelfPlaySection(**values["selfplay"])
        values["learner"] = LearnerConfig(**values["learner"])
        values["seeds"] = LoopSeeds(**values["seeds"])
        values["validation_steps"] = tuple(values.get("validation_steps", (0, 100, 250, 500, 1000)))
        return cls(**values)

    def to_dict(self):
        raw = json.loads(json.dumps({"protocol": PROTOCOL, **asdict(self)}))
        if self.continuation_state is None:
            raw.pop("continuation_state")
        if self.fixed_holdout is None:
            raw.pop("fixed_holdout")
        if self.six_round_holdout is None:
            raw.pop("six_round_holdout")
        if not self.initial_history_eviction:
            raw.pop("initial_history_eviction")
        if self.continuation_auxiliary is None:
            raw.pop("continuation_auxiliary")
        if not self.encoder_grad_norm_every_steps:
            raw.pop("encoder_grad_norm_every_steps")
        return raw


def check(config):
    """Read-only preflight; no archive decoding, collection or training."""
    ExecutionConfig(**config.selfplay.execution)
    if config.fixed_holdout is not None:
        read_manifest(config.fixed_holdout)
    if config.six_round_holdout is not None:
        check_settings(config.six_round_holdout)
    for identity in (config.initial_checkpoint, config.history_manifest):
        if sha256(Path(identity["path"])) != identity["sha256"]:
            raise ValueError(f"frozen input changed: {identity['path']}")
    records = [json.loads(line) for line in Path(config.history_manifest["path"]).read_text().splitlines()]
    records = [g for g in records if config.first_round - 1 <= g["round_index"] < config.last_round]
    counts = Counter(g["round_index"] for g in records)
    from .offline_objectives import expected_round_game_counts
    if dict(counts) != expected_round_game_counts(config.first_round, config.last_round, config.expected_games_per_round):
        raise ValueError("historical round/game count mismatch")
    if len({_key(g) for g in records}) != len(records):
        raise ValueError("duplicate historical game key")
    first_pool_games = len(records) + config.new_games_per_round
    first_evicted_history_games = max(0, first_pool_games-config.replay_max_games)
    if first_evicted_history_games and not config.initial_history_eviction:
        raise ValueError("first round exceeds replay capacity and would evict historical games")
    first_pool_games = min(first_pool_games, config.replay_max_games)
    for game in records:
        stat = Path(game["path"]).stat()
        if (stat.st_size, stat.st_mtime_ns) != (game["size"], game["mtime_ns"]):
            raise ValueError(f"historical archive changed: {game['path']}")
    payload = load_alphazero_checkpoint(config.initial_checkpoint["path"])
    if config.continuation_auxiliary is None:
        if payload.model_config.get("architecture") != "entity_attention_v1":
            raise ValueError("this branch requires the model without auxiliary heads")
    else:
        auxiliary = ContinuationAuxiliaryConfig(**config.continuation_auxiliary)
        if (payload.model_config.get("architecture") != "entity_attention_continuation_v1"
                or payload.model_config.get("predict_opponent_policy", False) != (auxiliary.opponent_policy_weight > 0)
                or payload.model_config.get("predict_shortterm_value", True) != (auxiliary.shortterm_value_weight > 0)
                or payload.model_config.get("predict_score_ownership", False) != (auxiliary.score_ownership_weight > 0)):
            raise ValueError("initial checkpoint heads differ from continuation auxiliary")
    for template in (config.greedy_evaluation, config.search_evaluation):
        if template is not None:
            from .evaluation import FixedEvaluationConfig
            FixedEvaluationConfig.from_dict(template)
            if config.evaluation_anchor is None or Path(template["policy_b"]["checkpoint"]).resolve() != Path(config.evaluation_anchor["path"]).resolve():
                raise ValueError("evaluation opponent differs from the fixed anchor")
    if config.evaluation_anchor is not None and sha256(Path(config.evaluation_anchor["path"])) != config.evaluation_anchor["sha256"]:
        raise ValueError("evaluation anchor changed")
    # Preserve AdamW settings; lr, checkpoint cadence and batch schedule are explicit.
    if payload.learner_config["weight_decay"] != config.learner.weight_decay:
        raise ValueError("weight decay differs from source optimizer")
    inherited_rounds = 0
    first_new_policy_fraction = config.new_games_per_round / first_pool_games
    if config.continuation_state is not None:
        parent = Path(config.continuation_state["path"])
        if sha256(parent) != config.continuation_state["sha256"]:
            raise ValueError("continuation state changed")
        state = read_json(parent)
        completed = state["completed_rounds"]
        inherited_rounds = len(completed)
        if (not completed or state["latest_checkpoint"] != config.initial_checkpoint
                or completed[-1]["checkpoint"] != state["latest_checkpoint"]
                or [row["iteration"] for row in completed] != list(range(1, inherited_rounds+1))
                or payload.round_index != config.last_round + inherited_rounds):
            raise ValueError("continuation checkpoint/round boundary mismatch")
        prior = completed[-1]["snapshot"]
        if sha256(Path(prior["path"])) != prior["sha256"]:
            raise ValueError("continuation replay snapshot changed")
        games = read_json(Path(prior["path"]))["games"]
        if not 0 < len(games) <= config.replay_max_games or len({_key(g) for g in games}) != len(games):
            raise ValueError("continuation replay capacity/keys mismatch")
        for game in games:
            stat = Path(game["path"]).stat()
            if (stat.st_size, stat.st_mtime_ns) != (game["size"], game["mtime_ns"]):
                raise ValueError(f"continuation archive changed: {game['path']}")
        original_keys = {_key(g) for g in records}
        original_count = sum(_key(g) in original_keys for g in games)
        overflow = max(0, len(games) + config.new_games_per_round - config.replay_max_games)
        first_evicted_history_games = min(original_count, overflow)
        first_pool_games = min(config.replay_max_games, len(games) + config.new_games_per_round)
        first_new_policy_fraction = 1 - max(0, original_count-overflow) / first_pool_games
    return sorted(records, key=_key), {"initial_step": payload.total_train_step,
        "inherited_completed_rounds": inherited_rounds,
        "historical_games": len(records), "historical_train_games": sum(g["split"] == "train" for g in records),
        "source_rounds": [config.first_round, config.last_round], "games_per_iteration": config.new_games_per_round,
        "first_pool_games": first_pool_games,
        "pool_capacity": config.replay_max_games, "policy_batch": config.learner.batch_size,
        "value_batch": config.learner.batch_size, "updates": config.updates_per_round,
        "learning_rate": config.learner.learning_rate,
        "historical_policy_fraction": config.historical_policy_fraction,
        "first_new_policy_fraction": first_new_policy_fraction,
        **({"first_evicted_history_games": first_evicted_history_games} if config.initial_history_eviction else {}),
        **({"continuation_auxiliary": continuation_config_identity(ContinuationAuxiliaryConfig(**config.continuation_auxiliary))}
           if config.continuation_auxiliary is not None else {})}


def _key(game):
    return game["round_index"], game["game_id"]


def retain_games(previous, incoming, original_keys, config, data_round):
    combined = sorted([*previous, *incoming], key=_key)
    if len({_key(g) for g in combined}) != len(combined):
        raise ValueError("replay selection contains duplicate games")
    overflow = max(0, len(combined) - config.replay_max_games)
    historical = [g for g in previous if _key(g) in original_keys]
    historical.sort(key=_key)
    rng = DeterministicRng(derive_seed(config.retention_seed, data_round, 0, "historical_eviction"))
    rng.shuffle(historical)
    removed = historical[:overflow]
    if len(removed) < overflow:
        # Original games leave randomly first; afterwards retire online games FIFO.
        online = sorted((g for g in previous if _key(g) not in original_keys), key=_key)
        removed.extend(online[:overflow-len(removed)])
    removed_keys = {_key(g) for g in removed}
    active = [g for g in combined if _key(g) not in removed_keys]
    if len(active) != min(config.replay_max_games, len(combined)):
        raise ValueError("replay eviction did not reach its target size")
    return active, sorted(removed, key=_key)


class PolicyPool(PositionPool):
    """Uniform positive-weight decisions, optionally a fixed historical subset."""
    def __init__(self, games, cache, *, fraction=1.0, seed=0, eligibility_cache=None):
        self.label = "eligible-policy"
        self._games, self._cache = tuple(games), cache
        rows = []
        for game in games:
            key = (str(game.path), fraction, seed)
            indices = eligibility_cache.get(key) if eligibility_cache is not None else None
            if indices is None:
                with np.load(game.path, allow_pickle=False) as archive:
                    indices = np.flatnonzero(archive["policy_weight"] > 0).astype(np.int32)
                    if "low_budget" in archive.files and np.any(archive["low_budget"][indices]):
                        raise ValueError("policy candidates include low-budget steps")
                if len(indices) != game.policy_samples:
                    raise ValueError("eligible policy count differs from archive metadata")
                if fraction < 1:
                    # Fixed per-game shuffle: no re-draw when games survive into another round.
                    chosen = indices.tolist()
                    DeterministicRng(derive_seed(seed, game.round_index, game.game_id, "historical_policy_subset")).shuffle(chosen)
                    indices = np.asarray(sorted(chosen[:int(len(chosen) * fraction)]), dtype=np.int32)
                if eligibility_cache is not None:
                    eligibility_cache[key] = indices
            rows.append(indices)
        self._ends = np.cumsum([len(row) for row in rows], dtype=np.int64)
        self._indices = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int32)
        self._size = len(self._indices)
        self.subset_digest = _digest([[g.round_index, g.game_id, row.tolist()] for g, row in zip(games, rows)])

    def sample(self, rng):
        if not self._size:
            raise ValueError("eligible policy pool is empty")
        from .alphazero_replay import ReplayReference
        position = rng.randbelow(self._size)
        game = self._games[int(np.searchsorted(self._ends, position, side="right"))]
        decision = int(self._indices[position])
        return self._cache.sample(game, decision), ReplayReference(game.round_index, game.run_id, game.game_id, decision)


def new_policy_quota(step, batch_size, new_games, total_games):
    return step * batch_size * new_games // total_games - (step-1) * batch_size * new_games // total_games


def draw_batch(old_policy, new_policy, value, rng, batch_size, step, new_games, total_games, original_keys):
    policy_rng = DeterministicRng.restore(rng.snapshot())
    new_count = new_policy_quota(step, batch_size, new_games, total_games)
    samples, refs, values = [], [], []
    for pool, count in ((old_policy, batch_size-new_count), (new_policy, new_count)):
        for _ in range(count):
            sample, ref = pool.sample(policy_rng)
            samples.append(sample); refs.append(ref)
    value_old = 0
    for _ in range(batch_size):
        sample, ref = value.sample(rng)
        values.append(sample); refs.append(ref)
        value_old += (ref.round_index, ref.game_id) in original_keys
    return DrawnBatch(tuple(samples), tuple(refs), batch_size*2, 0,
        {"policy_historical": batch_size-new_count, "policy_new": new_count,
         "value_historical": value_old, "value_new": batch_size-value_old}, tuple(values))


def start_learner(teacher, config, identity, data_round):
    payload = load_alphazero_checkpoint(teacher["path"])
    learner = AlphazeroLearner.resume(teacher["path"], LearnerConfig(**payload.learner_config),
        replay_identity=payload.replay_identity, round_index=payload.round_index,
        round_steps_target=payload.round_steps_target)
    learner.config = config.learner
    for group in learner.optimizer.param_groups:
        group["lr"] = config.learner.learning_rate
    learner.round_index, learner.round_train_step = data_round + 1, 0
    learner.round_steps_target = config.updates_per_round
    learner.replay_identity, learner.parent_checkpoint = identity, teacher
    return learner


def _emit(event, **kwargs):
    print(json.dumps({"event": event, **kwargs}, ensure_ascii=False), flush=True)


def _trim_cpu_heap():
    """Return freed NumPy cache pages on this Linux/glibc training entry."""
    trim = ctypes.CDLL(None).malloc_trim
    trim.argtypes, trim.restype = (ctypes.c_size_t,), ctypes.c_int
    return bool(trim(0))


def _fixed_collection_config(raw):
    values = json.loads(json.dumps(raw))
    for name in COLLECTION_TUNING_FIELDS:
        values["execution"].pop(name, None)
    return values


def _algorithm_config(raw):
    values = json.loads(json.dumps(raw))
    # Scheduled diagnostic nodes are saved separately from periodic recovery.
    values["learner"].pop("checkpoint_every_steps")
    values.pop("encoder_grad_norm_every_steps", None)
    # A fixed diagnostic panel changes neither replay nor learner updates.
    # Its immutable identity is checked separately in FixedHoldout.
    values.pop("fixed_holdout", None)
    values.pop("six_round_holdout", None)
    values["selfplay"] = _fixed_collection_config(raw["selfplay"])
    # Evaluation size and inference execution do not alter the learner/replay.
    # Each prepared panel keeps its own immutable configuration when resumed.
    search = values.get("search_evaluation")
    if search is not None:
        search.pop("pairs", None)
        search.pop("execution", None)
        for name in ("policy_a", "policy_b"):
            search[name].pop("inference", None)
    return values


def _collection_config(raw, stage):
    """Freeze execution per round, including rounds started by the old driver."""
    requested = FastSelfPlayConfig.from_dict(raw)
    frozen_path = stage / "collection_config.json"
    manifest_path = stage / "selfplay" / "manifest.json"
    if frozen_path.exists():
        actual = FastSelfPlayConfig.from_dict(read_json(frozen_path)["config"])
    elif manifest_path.exists():
        actual = FastSelfPlayConfig.from_dict(read_json(manifest_path)["config"])
    else:
        actual = requested
    if _fixed_collection_config(actual.to_dict()) != _fixed_collection_config(requested.to_dict()):
        raise ValueError("collection resume requires identical frozen algorithm configuration")
    if not frozen_path.exists():
        write_json(frozen_path, {"config": actual.to_dict(),
            "requested_execution": requested.to_dict()["execution"]})
    return actual


def run(config, output, *, rounds=None, max_steps_this_session=None, config_path=None):
    original, preflight = check(config)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".iteration.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run(config, output, original, preflight, rounds, max_steps_this_session, config_path)


def _run(config, output, original, preflight, rounds, budget, config_path):
    torch.set_num_threads(1)
    frozen, state_path = output / "config.json", output / "state.json"
    if frozen.exists():
        if _algorithm_config(read_json(frozen)) != _algorithm_config(config.to_dict()):
            raise ValueError("resume requires identical frozen algorithm configuration")
        state = read_json(state_path)
    else:
        write_json(frozen, config.to_dict())
        write_json(output / "historical_games.json", {"games": original, "preflight": preflight})
        state = (read_json(Path(config.continuation_state["path"])) if config.continuation_state is not None
                 else {"completed_rounds": [], "latest_checkpoint": config.initial_checkpoint})
        write_json(state_path, state)
    session = uuid.uuid4().hex
    write_json(output / "source" / session / "identity.json", source_identity(Path.cwd(), output / "source" / session))
    original_keys = {_key(g) for g in original}
    eligibility_cache = {}
    auxiliary = (ContinuationAuxiliaryConfig(**config.continuation_auxiliary)
                 if config.continuation_auxiliary is not None else None)
    cache = CompactGameCache(config.cache_bytes, continuation_auxiliary=auxiliary)
    holdout_settings = config.six_round_holdout if config.six_round_holdout is not None else config.fixed_holdout
    holdout_schedule = (HoldoutSchedule(holdout_settings, output, config.learner.device,
                        six_rounds=config.six_round_holdout is not None) if holdout_settings is not None else None)
    if holdout_schedule is not None:
        holdout_schedule.backfill(state["completed_rounds"], config.validation_steps)
    while rounds is None or len(state["completed_rounds"]) < rounds:
        if budget is not None and budget <= 0:
            break
        r = len(state["completed_rounds"])
        if config_path is not None:
            updated = Config.from_dict(read_json(Path(config_path)))
            if _algorithm_config(updated.to_dict()) != _algorithm_config(config.to_dict()):
                raise ValueError("runtime update requires identical frozen algorithm configuration")
            if (updated.fixed_holdout, updated.six_round_holdout) != (config.fixed_holdout, config.six_round_holdout):
                raise ValueError("changing fixed holdout requires restarting at a checkpoint boundary")
            ExecutionConfig(**updated.selfplay.execution)
            config = updated
        fixed_holdout = holdout_schedule.for_iteration(r+1) if holdout_schedule is not None else None
        write_json(output / "source" / session / f"config-iteration-{r+1:03d}.json", config.to_dict())
        data_round = config.last_round + r
        stage = output / "rounds" / f"round-{r:03d}"
        teacher = state["latest_checkpoint"]
        if sha256(Path(teacher["path"])) != teacher["sha256"]:
            raise ValueError("collection teacher changed")
        raw = config.selfplay.to_dict() | {"run_id": f"{config.run_id}-round-{r:03d}",
            "round_index": data_round, "checkpoint": teacher["path"], "num_games": config.new_games_per_round,
            "seed": config.seeds.collection_for(data_round), "split_seed": config.seeds.split_for(data_round)}
        collection_config = _collection_config(raw, stage)
        _emit("collect_start", iteration=r+1, games=config.new_games_per_round,
              execution={name: getattr(collection_config.execution, name) for name in COLLECTION_TUNING_FIELDS})
        devices = list(range(torch.cuda.device_count())) if config.learner.device == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            try:
                collection = generate(collection_config, stage / "selfplay",
                    resume=(stage / "selfplay/manifest.json").exists(),
                    progress_callback=lambda done, total: _emit("collect_progress", iteration=r+1, completed=done, total=total))
            finally:
                _release_inference_memory()
        if collection["completed_games"] != config.new_games_per_round or collection["missing_games"]:
            raise ValueError("collection is incomplete")
        incoming = build_replay_index([ReplayRun(data_round, raw["run_id"], stage / "selfplay")],
            max_games=config.new_games_per_round, state_path=stage / "collection_index.json")
        previous = original
        if r:
            prior = state["completed_rounds"][-1]["snapshot"]
            if sha256(Path(prior["path"])) != prior["sha256"]:
                raise ValueError("previous replay snapshot changed")
            previous = read_json(Path(prior["path"]))["games"]
        new_records = []
        for game in incoming.games:
            stat = game.path.stat()
            new_records.append(game.to_dict() | {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
        active, removed = retain_games(previous, new_records, original_keys, config, data_round)
        if fixed_holdout is not None:
            check_disjoint(fixed_holdout.manifest, active)
        for game in active:
            stat = Path(game["path"]).stat()
            if (stat.st_size, stat.st_mtime_ns) != (game["size"], game["mtime_ns"]):
                raise ValueError(f"active replay archive changed: {game['path']}")
        index = make_index({"games": active}, config.cache_bytes)
        index.cache = cache
        cache.retain(g.key for g in index.games)
        active_paths = {str(g.path) for g in index.games}
        eligibility_cache = {k: v for k, v in eligibility_cache.items() if k[0] in active_paths}
        train_games = index.split_games("train")
        old = [g for g in train_games if g.key in original_keys]
        new = [g for g in train_games if g.key not in original_keys]
        old_policy = PolicyPool(old, cache, fraction=config.historical_policy_fraction,
            seed=config.historical_policy_seed, eligibility_cache=eligibility_cache)
        new_policy = PolicyPool(new, cache, eligibility_cache=eligibility_cache)
        value = PositionPool(train_games, cache, "value-all-budgets")
        new_games = sum(_key(g) not in original_keys for g in active)
        snapshot = {"games": active, "evicted_games": removed, "teacher": teacher,
            "historical_games": len(active)-new_games, "new_games": new_games,
            "policy_new_fraction": new_games/len(active),
            "policy_positions": {"historical": len(old_policy), "new": len(new_policy)},
            "policy_subsets": {"historical": old_policy.subset_digest, "new": new_policy.subset_digest},
            "value_positions": len(value)}
        snapshot["digest"] = _digest(snapshot)
        snapshot_path = stage / "replay_snapshot.json"
        if snapshot_path.exists() and read_json(snapshot_path) != snapshot:
            raise ValueError("active pool/subset changed on resume")
        write_json(snapshot_path, snapshot)
        _emit("pool_ready", iteration=r+1, historical_games=snapshot["historical_games"],
              new_games=new_games, policy_new_fraction=snapshot["policy_new_fraction"],
              policy_positions=snapshot["policy_positions"], value_positions=len(value))
        identity = {"protocol": PROTOCOL, "digest": snapshot["digest"]}
        if auxiliary is not None:
            identity.update(continuation_auxiliary=continuation_config_identity(auxiliary),
                digest=_digest({"snapshot": snapshot["digest"], "continuation_auxiliary": continuation_config_identity(auxiliary)}))
        checkpoints = stage / "checkpoints"
        saved = sorted(checkpoints.glob("step-*.pt"))
        if saved or (checkpoints / "initial.pt").exists():
            learner = AlphazeroLearner.resume(saved[-1] if saved else checkpoints / "initial.pt", config.learner,
                replay_identity=identity, round_index=data_round+1, round_steps_target=config.updates_per_round)
        else:
            learner = start_learner(teacher, config, identity, data_round)
            learner.save(checkpoints / "initial.pt")
        validation = PositionPool(index.split_games("validation"), cache, "validation")
        validation_rng = DeterministicRng(derive_seed(config.validation_seed, data_round, 0, "validation_subset"))
        validation_draws = [validation.sample(validation_rng) for _ in range(min(config.validation_positions, len(validation)))]
        validation_samples = [sample for sample, _ in validation_draws]
        references = [ref.to_dict() for _, ref in validation_draws]
        write_json(stage / "validation_references.json", {"references": references, "digest": _digest(references)})
        def observe(current, checkpoint):
            if fixed_holdout is not None:
                fixed_holdout.observe(current.model, checkpoint, r+1, current.round_train_step)
            path = stage / "validation" / f"step-{current.round_train_step:09d}.json"
            if path.exists():
                if read_json(path)["checkpoint_sha256"] != sha256(checkpoint):
                    raise ValueError("validation checkpoint changed")
                return
            write_json(path, {"step": current.round_train_step, "checkpoint_sha256": sha256(checkpoint),
                "metrics": current.evaluate_samples(validation_samples), "snapshot_digest": snapshot["digest"],
                "validation_references_digest": _digest(references),
                "scope": "fixed pooled validation positions; policy metrics use positive archive weights, value uses all"})
        if learner.round_train_step == 0:
            observe(learner, checkpoints / "initial.pt")
        elif fixed_holdout is not None:
            # Recover diagnostic nodes saved before a session was interrupted.
            holdout_schedule.backfill([{"iteration": r+1,
                "snapshot": {"path": str(snapshot_path), "sha256": sha256(snapshot_path)}}],
                [s for s in config.validation_steps if s <= learner.round_train_step])
        writer = HybridReferenceWriter(stage / "usage", session, {"policy": identity, "value": identity},
            (config.learner.batch_size, config.learner.batch_size), ("policy", "value"))
        before = learner.round_train_step
        try:
            training = run_training_round(learner, steps_target=config.updates_per_round, current_pool=value,
                history_pool=None, validation_samples=validation_samples, validation_games=len(validation.games),
                checkpoint_dir=checkpoints, metrics_path=stage / "training_metrics.jsonl", reference_writer=writer,
                batch_drawer=lambda rng: draw_batch(old_policy, new_policy, value, rng, config.learner.batch_size,
                    learner.round_train_step+1, new_games, len(active), original_keys),
                joint_separate_batches=True, checkpoint_steps=config.validation_steps,
                continuation_auxiliary=auxiliary,
                encoder_grad_norm_every_steps=config.encoder_grad_norm_every_steps,
                checkpoint_callback=lambda current, path: observe(current, path) if current.round_train_step in config.validation_steps else None,
                max_steps_this_session=budget,
                progress_callback=lambda row: _emit("train_progress", iteration=r+1, step=row["round_train_step"],
                    policy_loss=row["policy_loss"], value_loss=row["value_loss"]) if row["round_train_step"] % 50 == 0 else None)
        finally:
            writer.close()
        write_json(stage / "training.json", training)
        if budget is not None:
            budget -= learner.round_train_step-before
        if learner.round_train_step < config.updates_per_round:
            return state
        final = checkpoints / "final.pt"
        if not final.exists():
            learner.save(final)
        # Evaluation and the next collection do not consume training arrays.
        # Drop sampled views as well as the LRU's references before collecting.
        released_bytes = cache.bytes
        released_games = cache.retain(())
        del learner, index, old_policy, new_policy, value, validation
        del validation_draws, validation_samples, observe
        _release_inference_memory()
        heap_trimmed = _trim_cpu_heap()
        _emit("training_cache_released", iteration=r+1, released_games=released_games,
              released_bytes=released_bytes, cache_bytes=cache.bytes, heap_trimmed=heap_trimmed)
        evaluations = {}
        if fixed_holdout is not None:
            result_path = fixed_holdout.output / "rounds" / f"round-{r:03d}" / "fixed_holdout" / f"step-{config.updates_per_round:09d}.json"
            evaluations["fixed_holdout"] = {"path": str(result_path), "sha256": sha256(result_path),
                "panel_digest": fixed_holdout.manifest["digest"], "metrics": read_json(result_path)["metrics"]}
        for name, template, due in (("greedy", config.greedy_evaluation, True),
                ("search", config.search_evaluation, (r+1) % config.search_evaluation_every_rounds == 0)):
            if template is not None and due:
                evaluations[name] = panel(template, final, template["policy_b"]["checkpoint"], stage / f"{name}.json",
                    opponent_name=template["policy_b"]["name"]).summary
        completed = {"iteration": r+1, "checkpoint": {"path": str(final), "sha256": sha256(final)},
            "snapshot": {"path": str(snapshot_path), "sha256": sha256(snapshot_path)}, "evaluations": evaluations}
        state["completed_rounds"].append(completed)
        state["latest_checkpoint"] = completed["checkpoint"]
        write_json(state_path, state)
        _emit("round_complete", iteration=r+1, checkpoint=str(final))
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--rounds", type=int, help="absolute completed-round target; omitted means continuous")
    parser.add_argument("--max-steps-this-session", type=int)
    args = parser.parse_args()
    config = Config.from_dict(read_json(args.config))
    if args.check:
        print(json.dumps(check(config)[1], ensure_ascii=False, indent=2))
        return
    if args.output is None or (args.rounds is not None and args.rounds <= 0) or (args.max_steps_this_session is not None and args.max_steps_this_session <= 0):
        parser.error("provide --output and positive optional round/step limits")
    run(config, args.output, rounds=args.rounds, max_steps_this_session=args.max_steps_this_session,
        config_path=args.config.resolve())


if __name__ == "__main__":
    main()
