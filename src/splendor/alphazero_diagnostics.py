"""Post-training diagnostics over explicit archived validation references."""
from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .alphazero_recording import DiagnosticsConfig, distribution, read_json, sha256, unavailable, write_json
from .alphazero_train import AlphazeroLearner, LearnerConfig, metric_ratios
from .opponent_pool import load_frozen_actor_critic_checkpoint
from .rng import DeterministicRng
from .selfplay_data import SelfPlaySample, load_game
from .compact_trace import replay_trace
from .replay import to_primitive

STAGES = ("early", "middle", "late")
SUM_FIELDS = ("policy_ce_sum", "policy_weight_sum", "value_sq_sum", "value_weight_sum", "entropy_sum",
              "target_entropy_sum", "value_target_sum", "value_target_sq_sum", "policy_positions", "value_positions", "positions")


def stage_for(turns: int) -> str:
    return "early" if turns < 10 else "middle" if turns < 50 else "late"


def freeze_set(samples: Sequence[SelfPlaySample], paths: Mapping[tuple[int, int], Path], *, limit: int | None = None, seed: int = 0) -> dict[str, Any]:
    hashes = {key: sha256(path) for key, path in paths.items()}
    references = []
    for sample in samples:
        if sample.metadata["split"] != "validation":
            raise ValueError("diagnostic set must contain validation games only")
        key = (sample.metadata["round_index"], sample.metadata["game_id"])
        references.append({"round_index": key[0], "run_id": sample.metadata["run_id"], "game_id": key[1],
                           "decision_index": sample.index, "path": str(paths[key].resolve()), "archive_sha256": hashes[key],
                           "stage": stage_for(int(sample.arrays["completed_player_turns"][sample.index]))})
    if limit is not None and len(references) > limit:
        groups = {stage: [] for stage in STAGES}
        for ref in references:
            groups[ref["stage"]].append(ref)
        for refs in groups.values():
            refs.sort(key=lambda ref: hashlib.sha256(json.dumps([seed, ref], sort_keys=True).encode()).digest())
        selected = []
        # Round-robin strata; empty/exhausted groups transfer their quota.
        for offset in range(max(map(len, groups.values()))):
            for stage in STAGES:
                if offset < len(groups[stage]) and len(selected) < limit:
                    selected.append(groups[stage][offset])
            if len(selected) == limit:
                break
        references = selected
    references.sort(key=lambda r: (r["round_index"], r["game_id"], r["decision_index"]))
    digest = hashlib.sha256(json.dumps(references, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"protocol": "az-validation-references-v1", "set_sha256": digest, "positions": len(references),
            "scope": "root decisions; combined completed player turns <10 / 10..49 / >=50",
            "selection_seed": seed, "limit": limit, "composition": dict(Counter(r["stage"] for r in references)), "references": references}


def evaluate_set(checkpoint: Path, manifest: Mapping[str, Any], config: LearnerConfig, constant: Mapping[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    totals = {stage: {key: 0.0 for key in SUM_FIELDS} for stage in ("overall", *STAGES)}
    device_ids = [torch.cuda.current_device()] if config.device == "cuda" else []
    # Frozen model construction initializes tensors; restore global RNG as well as never drawing learner RNG.
    with torch.random.fork_rng(devices=device_ids):
        model = load_frozen_actor_critic_checkpoint(checkpoint, device=config.device)
        evaluator = AlphazeroLearner(model, config, sampling_rng=DeterministicRng(0))
        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for ref in manifest["references"]:
            grouped[ref["path"]].append(ref)
        for path, refs in grouped.items():
            if sha256(Path(path)) != refs[0]["archive_sha256"]:
                raise ValueError(f"diagnostic archive changed: {path}")
            arrays, metadata = load_game(Path(path))
            if metadata["split"] != "validation":
                raise ValueError("diagnostic archive is no longer validation")
            for stage in STAGES:
                samples = [SelfPlaySample(arrays, metadata, int(ref["decision_index"])) for ref in refs if ref["stage"] == stage]
                if not samples:
                    continue
                metrics = evaluator.evaluate_samples(samples)
                for key in SUM_FIELDS:
                    totals[stage][key] += metrics[key]
                    totals["overall"][key] += metrics[key]
    results = {}
    for stage, sums in totals.items():
        if not sums["positions"]:
            results[stage] = unavailable("empty validation stratum")
            continue
        values = {"available": True, **sums, **metric_ratios(sums)}
        if constant.get("available") and sums["value_weight_sum"]:
            c = constant["value"]
            values["value_train_constant_mse"] = (sums["value_target_sq_sum"] - 2 * c * sums["value_target_sum"] + c * c * sums["value_weight_sum"]) / sums["value_weight_sum"]
        else:
            values["value_train_constant_mse"] = unavailable("no training labels to fit constant")
        results[stage] = values
    return {"checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256(checkpoint), "set_sha256": manifest["set_sha256"],
            "constant_baseline": dict(constant), "metrics": results, "seconds": time.perf_counter() - started}


def training_constant(games: Sequence[Mapping[str, Any]], digest: str) -> dict[str, Any]:
    weight = total = 0.0
    for game in games:
        if game["split"] != "train":
            continue
        arrays, _ = load_game(Path(game["path"]))
        weights = arrays["value_weight"].astype(np.float64)
        total += float(np.sum(arrays["value_target"] * weights))
        weight += float(weights.sum())
    return {"available": True, "value": total / weight, "value_weight_sum": weight,
            "source": "retained train games only; no validation labels", "replay_digest": digest} if weight else unavailable("empty retained training pool")


def dataset_summary(games: Sequence[Mapping[str, Any]], round_index: int, max_games: int) -> dict[str, Any]:
    records = []
    pools: dict[str, dict[str, int]] = {}
    rounds: dict[str, Counter] = defaultdict(Counter)
    target_sums = Counter()
    candidate_counts = Counter()
    root_stages = Counter()
    wins = {"first": Counter(), "second": Counter()}
    checked_count = 0
    checked_seconds = 0.0
    for game in games:
        scope = ("current" if game["round_index"] == round_index else "history") + "_" + game["split"]
        pool = pools.setdefault(scope, {"games": 0, "positions": 0, "policy_positions": 0, "compressed_bytes": 0})
        pool["games"] += 1
        pool["positions"] += game["decision_count"]
        pool["policy_positions"] += game["policy_samples"]
        pool["compressed_bytes"] += Path(game["path"]).stat().st_size
        rounds[str(game["round_index"])][game["split"]] += 1
        if game["round_index"] != round_index:
            continue
        arrays, meta = load_game(Path(game["path"]))
        terminal = meta["terminal_result"]
        tick = time.perf_counter()
        if to_primitive(replay_trace(meta["trace"])) != terminal:
            raise ValueError(f"checked replay terminal differs for {game['path']}")
        checked_seconds += time.perf_counter() - tick
        checked_count += 1
        scores = terminal.get("scores")
        record = {key: meta.get(key) for key in ("round_index", "run_id", "game_id", "game_seed", "first_player", "checkpoint_sha256", "completed_player_turns", "decision_count", "policy_samples", "player_decisions", "split", "status")}
        record.update(path=str(Path(game["path"]).resolve()), archive_sha256=sha256(Path(game["path"])), terminal_result=terminal,
                      value_samples=int(np.count_nonzero(arrays["value_weight"])),
                      score_margin=scores[meta["first_player"]] - scores[1 - meta["first_player"]] if scores else None)
        records.append(record)
        widths = np.diff(arrays["candidate_offsets"])
        candidate_counts.update(map(int, widths))
        for i, width in enumerate(widths):
            stage = stage_for(int(arrays["completed_player_turns"][i]))
            root_stages[stage] += 1
            target = arrays["policy_target"][arrays["candidate_offsets"][i]:arrays["candidate_offsets"][i + 1]].astype(np.float64)
            positive = target > 0
            weight = float(arrays["policy_weight"][i])
            target_sums[stage + "_entropy_sum"] += float(-np.sum(target[positive] * np.log(target[positive]))) * weight
            target_sums[stage + "_weight_sum"] += weight
        winners = terminal["winner_ids"]
        outcome = "draw" if len(winners) == 2 else "win" if meta["first_player"] in winners else "loss"
        wins["first"][outcome] += 1
        wins["second"]["draw" if outcome == "draw" else "loss" if outcome == "win" else "win"] += 1
    total_games = sum(p["games"] for p in pools.values())
    return {"scope": "new round root decisions; retained replay composition", "round_index": round_index,
            "games": records, "pools": pools, "retained_by_round": {k: dict(v) for k, v in rounds.items()},
            "retained_round_game_shares": {k: sum(v.values()) / total_games for k, v in rounds.items()},
            "replay_max_games": max_games, "validation_game_fraction": sum(p["games"] for k, p in pools.items() if k.endswith("validation")) / total_games,
            "combined_turns": distribution(r["completed_player_turns"] for r in records), "decisions": distribution(r["decision_count"] for r in records),
            "first_second_wdl": {k: dict(v) for k, v in wins.items()},
            "terminal_scores": distribution(s for r in records for s in r["terminal_result"].get("scores", [])),
            "first_player_score_margin": distribution(r["score_margin"] for r in records if r["score_margin"] is not None),
            "terminal_reasons": dict(Counter(r["terminal_result"].get("reason") for r in records)),
            "root_stage_positions": dict(root_stages), "root_candidate_count_histogram": dict(candidate_counts), "target_entropy_sums": dict(target_sums),
            "root_target_entropy": {stage: target_sums[stage + "_entropy_sum"] / target_sums[stage + "_weight_sum"] if target_sums[stage + "_weight_sum"] else unavailable("no effective policy roots") for stage in STAGES},
            "extra_checked_replay": {"scope": "all new round games, Python checked trace replay, post-training diagnostics",
                                     "games_passed": checked_count, "seconds": checked_seconds, "passed": True}}


def diagnose_round(output: Path, round_index: int, parent: Path, initial: Path, config: LearnerConfig, diagnostics: DiagnosticsConfig, max_games: int) -> dict[str, Any]:
    started = time.perf_counter()
    round_root = output / "rounds" / f"round-{round_index:03d}"
    artifact = Path(diagnostics.artifact_dir) / f"round-{round_index:03d}"
    snapshot = read_json(round_root / "replay_snapshot.json")
    games = snapshot["games"]
    paths = {(g["round_index"], g["game_id"]): Path(g["path"]) for g in games if g["round_index"] == round_index and g["split"] == "validation"}
    current_path = round_root / "diagnostics" / "current_validation.json"
    if not current_path.exists():
        samples = []
        for path in paths.values():
            arrays, metadata = load_game(path)
            samples.extend(SelfPlaySample(arrays, metadata, i) for i in range(metadata["decision_count"]))
        write_json(current_path, freeze_set(samples, paths))
    current = read_json(current_path)
    fixed_path = output / "diagnostics" / "fixed_validation.json"
    if not fixed_path.exists():
        if round_index != 0:
            raise ValueError("fixed diagnosis must be frozen from round-000 validation")
        samples = []
        for path in paths.values():
            arrays, metadata = load_game(path)
            samples.extend(SelfPlaySample(arrays, metadata, i) for i in range(metadata["decision_count"]))
        write_json(fixed_path, freeze_set(samples, paths, limit=diagnostics.fixed_positions, seed=diagnostics.fixed_seed))
    fixed = read_json(fixed_path)
    write_json(Path(diagnostics.artifact_dir) / "fixed_validation.json", fixed)
    constant_path = round_root / "diagnostics" / "training_constant.json"
    if not constant_path.exists():
        write_json(constant_path, training_constant(games, snapshot["identity"]["digest"]))
    constant = read_json(constant_path)
    data_path = artifact / "dataset.json"
    if not data_path.exists():
        data = dataset_summary(games, round_index, max_games)
        data["evicted_games"] = snapshot["identity"].get("evicted_games")
        data["evicted_by_round"] = {r: count - sum(data["retained_by_round"].get(r, {}).values()) for r, count in snapshot["collected_games_by_round"].items()}
        data["replay_identity"] = snapshot["identity"]
        write_json(data_path, data)
    curve_path = artifact / "fixed_curve.json"
    curve = read_json(curve_path) if curve_path.exists() else []
    known = {(row["checkpoint_sha256"], row["set_sha256"]) for row in curve}
    checkpoints = ([initial] if round_index == 0 else []) + sorted((round_root / "checkpoints").glob("step-*.pt")) + [round_root / "checkpoints" / "final.pt"]
    for checkpoint in checkpoints:
        key = (sha256(checkpoint), fixed["set_sha256"])
        if key in known:
            continue
        row = evaluate_set(checkpoint, fixed, config, constant)
        row.update(round_index=round_index, node="initial" if checkpoint == initial else checkpoint.stem)
        curve.append(row)
        known.add(key)
        write_json(curve_path, curve)
    comparison_path = artifact / "current_before_after.json"
    if not comparison_path.exists():
        write_json(comparison_path, {"set": {k: v for k, v in current.items() if k != "references"},
                   "before": evaluate_set(parent, current, config, constant),
                   "after": evaluate_set(round_root / "checkpoints" / "final.pt", current, config, constant),
                   "before_scope": "parent weights evaluated after training; identical pre-training frozen checkpoint"})
    return {"artifact_dir": str(artifact.resolve()), "current_set_sha256": current["set_sha256"], "fixed_set_sha256": fixed["set_sha256"],
            "seconds_this_session": time.perf_counter() - started, "fixed_positions": fixed["positions"], "current_positions": current["positions"]}
