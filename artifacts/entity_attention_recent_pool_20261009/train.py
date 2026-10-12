"""Resumable fixed-pool offline experiment using the production AZ learner."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import math
import os
import resource
import shutil
import signal
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from pool import HERE, digest, prepare
from compact_cache import CompactGameCache
from splendor.actor_critic import build_actor_critic
from splendor.alphazero_recording import append_json, environment_identity, read_json, sha256, source_identity, write_json
from splendor.alphazero_replay import PositionPool
from splendor.alphazero_train import AlphazeroLearner, LearnerConfig, draw_training_batch
from splendor.features import GLOBAL_FEATURE_NAMES, SCALES
from splendor.offline_train import make_index
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.rng import DeterministicRng
from splendor.selfplay_data import collate_selfplay_samples


def load_records(path):
    import json
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def freeze_subsets(records, seed, per_source):
    rng = DeterministicRng(seed)
    buckets = defaultdict(list)
    for game in records:
        buckets[(game["source_id"], game["split"])].append(game)
    references = []
    for (source, split), games in sorted(buckets.items()):
        starts = np.cumsum([0] + [g["decision_count"] for g in games])
        selected = set()
        while len(selected) < min(per_source, int(starts[-1])):
            selected.add(rng.randbelow(int(starts[-1])))
        for position in sorted(selected):
            i = int(np.searchsorted(starts, position, side="right") - 1)
            references.append({"round_index": games[i]["round_index"], "game_id": games[i]["game_id"],
                "decision_index": int(position - starts[i]), "source_id": source, "split": split})
    return {"references": references, "seed": seed, "per_source_split": per_source, "digest": digest(references)}


def ratios(sums):
    p, v = sums["policy_weight"], sums["positions"]
    result = {**sums, "policy_kl": sums["kl"] / p if p else None,
        "policy_ce": sums["ce"] / p if p else None,
        "target_entropy": sums["target_entropy"] / p if p else None,
        "policy_entropy": sums["entropy"] / p if p else None,
        "teacher_top1_agreement": sums["top1"] / p if p else None,
        "value_mse": sums["sq_error"] / v if v else None,
        "value_zero_mse": sums["target_sq"] / v if v else None,
        "value_bias": (sums["pred"] - sums["target"]) / v if v else None}
    if v:
        x = max(0., sums["pred_sq"] - sums["pred"] ** 2 / v)
        y = max(0., sums["target_sq"] - sums["target"] ** 2 / v)
        result["value_correlation"] = (sums["product"] - sums["pred"] * sums["target"] / v) / math.sqrt(x*y) if x*y > 0 else None
    return result


def evaluate(learner, index, records, *, fixed=None):
    """Stream <= one minibatch of sample references, including full holdout."""
    by_key = {(r["round_index"], r["game_id"]): r for r in records}
    games = {g.key: g for g in index.games}
    if fixed is None:
        refs = ({"round_index": g.round_index, "game_id": g.game_id, "decision_index": i}
                for g in index.split_games("validation") for i in range(g.decision_count))
    else:
        refs = iter(fixed["references"])
    totals = defaultdict(Counter)
    game_totals = defaultdict(Counter)
    score_columns = [GLOBAL_FEATURE_NAMES.index(n) for n in ("self_score_scaled", "opponent_score_scaled")]
    was_training = learner.model.training
    learner.model.eval()

    def flush(samples, identities):
        batch = collate_selfplay_samples(samples)
        with torch.inference_mode():
            logits, values = learner.model(batch.decisions)
            mask = torch.as_tensor(batch.decisions.candidate_mask, device=learner.device)
            lp = torch.log_softmax(logits.masked_fill(~mask, -torch.inf), -1)
            lp = torch.where(mask, lp, torch.zeros_like(lp)).cpu().numpy().astype(np.float64)
            pred = values.cpu().numpy().astype(np.float64)
        target = batch.policy_targets.astype(np.float64)
        logs = np.zeros_like(target)
        np.log(target, out=logs, where=target > 0)
        pw = batch.policy_weights.astype(np.float64)
        y = batch.value_targets.astype(np.float64)
        probs = np.where(batch.decisions.candidate_mask, np.exp(lp), 0)
        ce = -(target * lp).sum(-1)
        th = -(target * logs).sum(-1)
        best = np.argmax(np.where(batch.decisions.candidate_mask, lp, -np.inf), -1)
        correct = target[np.arange(len(samples)), best] >= target.max(-1) - 1e-7
        late = batch.decisions.global_features[:, score_columns].max(-1) * SCALES.player_score >= 10 - 1e-5
        for i, identity in enumerate(identities):
            row = {"positions": 1, "policy_weight": float(pw[i]), "ce": ce[i]*pw[i],
                "kl": (ce[i]-th[i])*pw[i], "target_entropy": th[i]*pw[i],
                "entropy": float(-(probs[i]*lp[i]).sum())*pw[i], "top1": float(correct[i])*pw[i],
                "sq_error": (pred[i]-y[i])**2, "pred": pred[i], "pred_sq": pred[i]**2,
                "target": y[i], "target_sq": y[i]**2, "product": pred[i]*y[i]}
            for source in (identity["source_id"], "__all_observed__"):
                for stage in ("all", "late" if late[i] else "early_mid"):
                    totals[f"{source}/{identity['split']}/{stage}"].update(row)
            if fixed is None:
                key = f"{identity['round_index']}:{identity['game_id']}"
                for stage in ("all", "late" if late[i] else "early_mid"):
                    game_totals[f"{key}/{stage}"].update(row)

    samples, identities = [], []
    for ref in refs:
        key = (ref["round_index"], ref["game_id"])
        samples.append(index.cache.sample(games[key], ref["decision_index"]))
        identities.append(by_key[key])
        if len(samples) == learner.config.validation_batch_size:
            flush(samples, identities)
            samples, identities = [], []
    if samples:
        flush(samples, identities)
    learner.model.train(was_training)
    metrics = {k: ratios(v) for k, v in sorted(totals.items())}
    populations = defaultdict(Counter)
    for r in records:
        populations[r["split"]][r["source_id"]] += r["policy_samples"]
    # The fixed subsets balance sources. Rank by original pool policy-position
    # weights, never by the unweighted/balanced subset mean.
    weighted = {}
    for split, weights in populations.items():
        if not any(f"{source}/{split}/all" in metrics for source in weights):
            continue
        weighted[split] = {"policy_kl": sum(metrics[f"{source}/{split}/all"]["policy_kl"] * n for source,n in weights.items()) / sum(weights.values())}
        counts = Counter()
        for r in records:
            if r["split"] == split:
                counts[r["source_id"]] += r["decision_count"]
        weighted[split]["value_mse"] = sum(metrics[f"{source}/{split}/all"]["value_mse"] * n for source,n in counts.items()) / sum(counts.values())
    result = {"scope": "complete original validation games" if fixed is None else "fixed per-source train/validation subsets; observed aggregate is source-balanced",
        "set_digest": None if fixed is None else fixed["digest"], "metrics": metrics, "pool_weighted": weighted}
    if fixed is None:
        result["per_game"] = {k: ratios(v) for k,v in game_totals.items()}
    return result


def reconcile(path, step):
    import json
    if not path.exists():
        return []
    rows = []
    temporary = path.with_suffix(".tmp")
    with path.open() as src, temporary.open("w") as dst:
        for line in src:
            row = json.loads(line)
            if row["step"] <= step:
                dst.write(line)
                rows.append(row)
    temporary.replace(path)
    return rows


def summarize_usage(ledger, records, step, batch_size):
    """One linear pass; avoid the legacy O(steps * ledger size) summarizer."""
    games = {(r["round_index"], r["game_id"]): r for r in records if r["split"] == "train"}
    starts, cursor = {}, 0
    for key, r in games.items():
        starts[key] = cursor
        cursor += r["decision_count"]
    counts = np.zeros(cursor, dtype=np.int32)
    shape = (1 + max(k[0] for k in games), 1 + max(k[1] for k in games))
    offsets = np.full(shape, -1, dtype=np.int64)
    source_ids = sorted({r["source_id"] for r in records})
    source_codes = np.zeros(shape, dtype=np.int64)
    source_draws = np.zeros(len(source_ids), dtype=np.int64)
    for key, r in games.items():
        offsets[key] = starts[key]
        source_codes[key] = source_ids.index(r["source_id"])
    with ledger.open("rb") as handle:
        while raw := handle.read(32 * batch_size * 1000):
            rows = np.frombuffer(raw, dtype="<i8").reshape(-1, 4)
            base = offsets[rows[:, 1], rows[:, 2]]
            if np.any(base < 0):
                raise ValueError("ledger contains a game outside the train pool")
            np.add.at(counts, base + rows[:, 3], 1)
            source_draws += np.bincount(source_codes[rows[:, 1], rows[:, 2]], minlength=len(source_ids))
    result = {"steps": step, "draws": int(counts.sum()), "train_positions": cursor,
        "covered_positions": int(np.count_nonzero(counts)), "unseen_fraction": float(np.mean(counts == 0)),
        "average_draw_exposure": float(counts.mean()), "source_draws": dict(zip(source_ids, map(int, source_draws))),
        "use_count_histogram": dict(zip(*[list(map(int, x)) for x in np.unique(counts, return_counts=True)]))}
    if result["draws"] != step * batch_size:
        raise ValueError("ledger draw total mismatch")
    return result


def run(config, *, max_steps_this_session=None):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    output = Path(config["output"])
    artifacts = Path(config["artifacts"])
    output.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    with (output / "train.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run(config, output, artifacts, max_steps_this_session)


def _run(config, output, artifacts, max_steps_this_session):
    started = time.perf_counter()
    config_path = output / "config.json"
    if config_path.exists() and read_json(config_path) != config:
        raise ValueError("frozen training configuration changed")
    write_json(config_path, config)
    if sha256(Path(config["manifest"])) != config["manifest_sha256"]:
        raise ValueError("pool manifest changed")
    records = load_records(config["manifest"])
    for r in records:
        stat = Path(r["path"]).stat()
        if stat.st_size != r["size"] or stat.st_mtime_ns != r["mtime_ns"]:
            raise ValueError(f"source archive changed: {r['path']}")
    index = make_index({"games": records}, config["cache_mib"] * 1024**2)
    index.cache = CompactGameCache(config["cache_mib"] * 1024**2)
    pool = PositionPool(index.split_games("train"), index.cache, "all-frozen-train")
    cfg = LearnerConfig(**config["learner"])
    milestones = {math.ceil(e * len(pool) / cfg.batch_size): e for e in config["full_holdout_exposures"]}
    steps_target = math.ceil(config["draw_exposures"] * len(pool) / cfg.batch_size)
    identity = {**index.identity(), "digest": digest({"manifest": config["manifest_sha256"], "config": config})}
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    steps = sorted(checkpoints.glob("step-*.pt"))
    if steps:
        last_checkpoint = steps[-1]
        learner = AlphazeroLearner.resume(last_checkpoint, cfg, replay_identity=identity, round_index=0, round_steps_target=steps_target)
    elif (checkpoints / "initial.pt").exists():
        last_checkpoint = checkpoints / "initial.pt"
        learner = AlphazeroLearner.resume(last_checkpoint, cfg, replay_identity=identity, round_index=0, round_steps_target=steps_target)
    else:
        torch.manual_seed(config["initialization_seed"])
        learner = AlphazeroLearner(build_actor_critic(config["model_config"]), cfg, sampling_rng=DeterministicRng(config["sampling_seed"]))
        learner.round_steps_target = steps_target
        learner.replay_identity = identity
        learner.parent_checkpoint = {"kind": "random_initialization", "seed": config["initialization_seed"], "reason": "replicate the previously successful fresh attention offline fit"}
        last_checkpoint = checkpoints / "initial.pt"
        learner.save(last_checkpoint)
    fixed_path = output / "fixed_subsets.json"
    if not fixed_path.exists():
        write_json(fixed_path, freeze_subsets(records, config["validation_seed"], config["positions_per_source_split"]))
    fixed = read_json(fixed_path)
    baseline_path = artifacts / "baselines.json"
    if not baseline_path.exists() and config.get("baseline_checkpoints"):
        baselines = {}
        with torch.random.fork_rng(devices=[0] if cfg.device == "cuda" else []):
            for name, path in config["baseline_checkpoints"].items():
                model = load_frozen_actor_critic_checkpoint(Path(path), device="cpu")
                baseline = AlphazeroLearner(model, cfg, sampling_rng=DeterministicRng(config["sampling_seed"]))
                if asdict(model.config) != asdict(learner.model.config):
                    raise ValueError("baseline architecture mismatch")
                baselines[name] = {"checkpoint": path, "sha256": sha256(Path(path)),
                    "evaluation": evaluate(baseline, index, records, fixed=fixed)}
                print(f"基线 {name}: {baselines[name]['evaluation']['pool_weighted']['validation']}", flush=True)
                del baseline, model
        write_json(baseline_path, baselines)
    curve_path = artifacts / "curve.jsonl"
    curve = reconcile(curve_path, learner.round_train_step)
    metrics_path = output / "training_metrics.jsonl"
    reconcile(metrics_path, learner.round_train_step)
    ledger_path = output / "references.i64"
    ledger = ledger_path.open("r+b" if ledger_path.exists() else "w+b")
    expected_bytes = learner.round_train_step * cfg.batch_size * 32
    if ledger.seek(0, 2) < expected_bytes:
        raise ValueError("reference ledger shorter than checkpoint progress")
    ledger.truncate(expected_bytes)
    ledger.seek(expected_bytes)
    stop = {"requested": False}
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.update(requested=True))
    session = str(time.time_ns())
    source_dir = output / "source" / session
    provenance = source_identity(Path.cwd(), source_dir)
    for path in HERE.glob("*.py"):
        shutil.copyfile(path, source_dir / path.name)
    write_json(source_dir / "identity.json", {**provenance, "command": sys.argv,
        "experiment_files": {p.name: sha256(p) for p in HERE.glob("*.py")}, "environment": environment_identity(), "config": config})

    def validate(checkpoint):
        tick = time.perf_counter()
        result = evaluate(learner, index, records, fixed=fixed)
        point = {"step": learner.round_train_step, "draw_exposure": learner.round_train_step * cfg.batch_size / len(pool),
            "checkpoint": str(checkpoint.resolve()), "learning_rate": learner.optimizer.param_groups[0]["lr"],
            "evaluation": result, "seconds": time.perf_counter()-tick,
            "cache": index.cache.stats(), "rss_peak_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if cfg.device == "cuda" else None,
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved() if cfg.device == "cuda" else None}
        curve.append(point)
        append_json(curve_path, point)
        best = min(curve, key=lambda p: p["evaluation"]["pool_weighted"]["validation"]["policy_kl"])
        write_json(artifacts / "best_validation_kl.json", best)
        if best["step"] == point["step"]:
            shutil.copyfile(checkpoint, checkpoints / "best_validation_kl.pt")
        write_json(artifacts / "progress.json", {"status": "training", "step": learner.round_train_step,
            "steps_target": steps_target, "train_positions": len(pool), "latest": point,
            "best_validation_kl_step": best["step"], "dataset_manifest_sha256": config["manifest_sha256"]})
        val = result["pool_weighted"]["validation"]
        print(f"验证 step={point['step']}/{steps_target} exposure={point['draw_exposure']:.3f} KL={val['policy_kl']:.6f} MSE={val['value_mse']:.6f}", flush=True)

    def full_holdout():
        path = artifacts / f"full_holdout_step_{learner.round_train_step:09d}.json"
        if not path.exists():
            tick = time.perf_counter()
            result = evaluate(learner, index, records)
            write_json(path, {"step": learner.round_train_step, "manifest_sha256": config["manifest_sha256"],
                "evaluation": result, "seconds": time.perf_counter()-tick})
            print(f"完整留出 step={learner.round_train_step}: {result['pool_weighted']['validation']}", flush=True)

    status = "failed"
    try:
        if (artifacts / "progress.json").exists():
            progress = read_json(artifacts / "progress.json")
            progress["status"] = "training"
            write_json(artifacts / "progress.json", progress)
        if not curve or curve[-1]["step"] != learner.round_train_step:
            validate(last_checkpoint)
        if learner.round_train_step in milestones:
            full_holdout()
        session_target = steps_target if max_steps_this_session is None else min(steps_target, learner.round_train_step + max_steps_this_session)
        source_by_round = {r["round_index"]: r["source_id"] for r in records}
        while learner.round_train_step < session_target and not stop["requested"]:
            tick = time.perf_counter()
            exposure = learner.round_train_step * cfg.batch_size / len(pool)
            lr = cfg.learning_rate * (config["late_lr_multiplier"] if exposure >= config["lr_drop_exposure"] else 1.)
            for group in learner.optimizer.param_groups:
                group["lr"] = lr
            drawn = draw_training_batch(pool, None, learner.sampling_rng, cfg)
            draw_seconds = time.perf_counter()-tick
            tick = time.perf_counter()
            batch = collate_selfplay_samples(drawn.samples)
            collate_seconds = time.perf_counter()-tick
            tick = time.perf_counter()
            row = learner.train_step(batch, references=drawn.references)
            update_seconds = time.perf_counter()-tick
            step = learner.round_train_step
            references = np.asarray([(step, r.round_index, r.game_id, r.decision_index) for r in drawn.references], dtype="<i8")
            raw = references.tobytes()
            ledger.write(raw)
            row.update(step=step, draw_exposure=step*cfg.batch_size/len(pool), seconds={"draw_load": draw_seconds,
                "collate": collate_seconds, "update": update_seconds}, reference_sha256=hashlib.sha256(raw).hexdigest())
            row["source_draws"] = dict(Counter(source_by_round[r.round_index] for r in drawn.references))
            append_json(metrics_path, row)
            if step % 100 == 0:
                print(f"训练 step={step}/{steps_target} loss={row['total_loss']:.5f} lr={lr:g} cache={index.cache.bytes/1024**3:.2f}GiB", flush=True)
                write_json(output / "live.json", {"status":"training", "step":step, "steps_target":steps_target,
                    "draw_exposure":row["draw_exposure"],"time_unix":time.time(),"last_step_seconds":row["seconds"],
                    "cache":index.cache.stats(),"rss_peak_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
            if step % cfg.checkpoint_every_steps == 0 or step in milestones or step == session_target or stop["requested"]:
                ledger.flush()
                os.fsync(ledger.fileno())
                last_checkpoint = checkpoints / f"step-{step:09d}.pt"
                learner.save(last_checkpoint, metadata={"ledger_bytes": step*cfg.batch_size*32, "last_reference_sha256": row["reference_sha256"]})
                validate(last_checkpoint)
                if step in milestones:
                    full_holdout()
        if last_checkpoint.name != f"step-{learner.round_train_step:09d}.pt" and learner.round_train_step:
            ledger.flush()
            os.fsync(ledger.fileno())
            last_checkpoint = checkpoints / f"step-{learner.round_train_step:09d}.pt"
            learner.save(last_checkpoint, metadata={"ledger_bytes": learner.round_train_step*cfg.batch_size*32})
            validate(last_checkpoint)
        if learner.round_train_step == steps_target:
            full_holdout()
            shutil.copyfile(last_checkpoint, checkpoints / "final.pt")
            best = read_json(artifacts / "best_validation_kl.json")
            best_full = artifacts / f"full_holdout_step_{best['step']:09d}.json"
            if not best_full.exists():
                best_learner = AlphazeroLearner.resume(Path(best["checkpoint"]), cfg,
                    replay_identity=identity, round_index=0, round_steps_target=steps_target)
                write_json(best_full, {"step": best["step"], "manifest_sha256": config["manifest_sha256"],
                    "evaluation": evaluate(best_learner, index, records)})
                del best_learner
            status = "complete"
        else:
            status = "interrupted" if stop["requested"] else "session_limit"
        ledger.flush()
        write_json(artifacts / "usage.json", summarize_usage(ledger_path, records, learner.round_train_step, cfg.batch_size))
        progress = read_json(artifacts / "progress.json")
        progress.update(status=status, latest_checkpoint=str(last_checkpoint.resolve()))
        write_json(artifacts / "progress.json", progress)
        write_json(output / "live.json", {"status":status,"step":learner.round_train_step,
            "steps_target":steps_target,"time_unix":time.time()})
        return progress
    finally:
        ledger.close()
        append_json(output / "sessions.jsonl", {"session_id": session, "status": status,
            "step": learner.round_train_step, "seconds": time.perf_counter()-started,
            "cache": index.cache.stats(), "rss_peak_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if cfg.device == "cuda" else None,
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved() if cfg.device == "cuda" else None})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--max-steps-this-session", type=int)
    args = parser.parse_args()
    if args.max_steps_this_session is not None and args.max_steps_this_session <= 0:
        parser.error("--max-steps-this-session must be positive")
    run(read_json(args.config), max_steps_this_session=args.max_steps_this_session)


if __name__ == "__main__":
    main()
