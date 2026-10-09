"""Independent-M4 paired Delta-KL probe on the behavior+M4 self-iteration branch.

For human rounds 4 / 8 / 12 (round-003 / round-007 / round-011):

1. sample 256 searchable roots from the round's train split and 256 from its
   validation split;
2. run four *fresh* independent S1024/N1 searches per root with the round's
   teacher checkpoint and new world/noise seed bases, averaging them into a
   fresh M4 target;
3. evaluate the round's initial (before) and final (after) student on the same
   roots and compute KL(fresh M4 || student) per root, then the paired
   Delta-KL = KL_after - KL_before on the same label;
4. repeat the paired computation with the stored behavior+3 M4 supervision as
   a secondary check (descriptive; the primary result is the fresh label).

Interpreting Delta-KL on one fixed label avoids comparing absolute KL across
labels with different entropy.  Cluster bootstrap resamples source games.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from splendor.alphazero_checkpoint import load_alphazero_checkpoint, model_state_sha256
from splendor.alphazero_recording import environment_identity, read_json, sha256, write_json
from splendor.catalog import load_default_catalog
from splendor.mcts_policy import MCTSConfig
from splendor.native_policy import NativeBatchRunner, NativePUCTSearchPolicy
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.policy_averaging import replay_roots
from splendor.search_inference import FrozenSearchInference, InferenceConfig
from splendor.selfplay import derive_seed
from splendor.selfplay_data import load_game

ROOT = Path(__file__).resolve().parents[2]
ART = Path(__file__).resolve().parent
RUNTIME = ROOT / "runtime/entity_attention_avg4_delta_kl_20261009"
BRANCH = ROOT / "runs/entity_attention_avg4_behavior_20261009/seed1_avg4/rounds"

ROUNDS = {"r004": "round-003", "r008": "round-007", "r012": "round-011"}
SAMPLES_PER_SET = 256
FRESH_SAMPLES = 4
FRESH_WORLD_SEED = 26700000
FRESH_NOISE_SEED = 26800000
SAMPLE_SEED_BASE = 20261009
BOOTSTRAP_SEED = 20261010
BOOTSTRAP_RESAMPLES = 10000
ACTIVE_ROOTS = 64
BUCKETS = (16, 32, 48, 64)
SETS = ("train", "validation", "train_ledger", "train_unseen")


def emit(**kwargs):
    print(json.dumps(kwargs, ensure_ascii=False), flush=True)


def record(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def git_identity():
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                          capture_output=True, check=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, text=True,
                            capture_output=True, check=True).stdout
    return {"git_head": head,
            "git_status_sha256": hashlib.sha256(status.encode()).hexdigest(),
            "script": record(Path(__file__))}


def round_stage(name):
    return BRANCH / ROUNDS[name]


def teacher_identity(name):
    stage = round_stage(name)
    manifest = read_json(stage / "selfplay/manifest.json")
    config = manifest["config"]
    teacher = Path(config["checkpoint"])
    labels = read_json(stage / "policy_labels/manifest.json")
    if labels["teacher_sha256"] != sha256(teacher):
        raise ValueError(f"{name}: policy-label teacher differs from the collection teacher")
    before, after = stage / "checkpoints/initial.pt", stage / "checkpoints/final.pt"
    before_payload = load_alphazero_checkpoint(before, device="cpu")
    teacher_payload = load_alphazero_checkpoint(teacher, device="cpu")
    before_state = model_state_sha256(before_payload.build_model().state_dict())
    teacher_state = model_state_sha256(teacher_payload.build_model().state_dict())
    if before_state != teacher_state:
        raise ValueError(f"{name}: initial checkpoint differs from the collection teacher")
    after_payload = load_alphazero_checkpoint(after, device="cpu")
    if after_payload.total_train_step != before_payload.total_train_step + 100:
        raise ValueError(f"{name}: before/after step gap is not one 100-step round")
    return {
        "name": name, "stage": str(stage), "round_index": int(config["round_index"]),
        "teacher": {**record(teacher), "model_state_sha256": teacher_state},
        "before": {**record(before), "model_state_sha256": before_state,
                   "total_train_step": before_payload.total_train_step},
        "after": {**record(after), "total_train_step": after_payload.total_train_step},
        "search_config": dict(config["search_config"]),
        "add_root_noise": bool(config["add_root_noise"]),
    }


def load_round_roots(name):
    """Return per-game roots in a deterministic order."""
    games = []
    for path in sorted((round_stage(name) / "selfplay/games").glob("game-*.npz")):
        arrays, meta = load_game(path)
        roots = replay_roots(arrays, meta)
        games.append({"path": path, "arrays": arrays, "meta": meta,
                      "roots": {int(i): request for i, request in roots}})
    if len(games) != 200:
        raise ValueError(f"{name}: expected 200 games, found {len(games)}")
    return games


def prepare():
    spec = {"protocol": "avg4-independent-m4-delta-kl-v1", "client_date": "2026-10-09",
            "rounds": {}, "samples_per_set": SAMPLES_PER_SET, "fresh_samples": FRESH_SAMPLES,
            "fresh_world_seed": FRESH_WORLD_SEED, "fresh_noise_seed": FRESH_NOISE_SEED,
            "sample_seed_base": SAMPLE_SEED_BASE, "bootstrap": {
                "resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED,
                "cluster": "source game"},
            "source": git_identity(), "environment": environment_identity()}
    samples = {"protocol": "avg4-independent-m4-delta-kl-samples-v1", "rounds": {}}
    for name in ROUNDS:
        identity = teacher_identity(name)
        games = load_round_roots(name)
        pools = {"train": [], "validation": []}
        for game in games:
            split = str(game["meta"]["split"])
            for i in game["roots"]:
                pools[split].append({"game_id": int(game["meta"]["game_id"]),
                                     "decision_index": int(i),
                                     "file": game["path"].name})
        entry = {"identity": identity, "available": {k: len(v) for k, v in pools.items()},
                 "train": [], "validation": []}
        for set_index, split in enumerate(("train", "validation")):
            rng = np.random.default_rng(derive_seed(SAMPLE_SEED_BASE, identity["round_index"],
                                                    0, "sample", set_index))
            count = min(SAMPLES_PER_SET, len(pools[split]))
            chosen = rng.choice(len(pools[split]), size=count, replace=False)
            entry[split] = [{**pools[split][int(i)], "split": split}
                            for i in sorted(int(j) for j in chosen)]
        samples["rounds"][name] = entry
        spec["rounds"][name] = {k: identity[k] for k in
                                ("stage", "round_index", "teacher", "before", "after",
                                 "search_config", "add_root_noise")}
        emit(event="prepared", round=name, available=entry["available"],
             sampled={k: len(entry[k]) for k in ("train", "validation")})
    RUNTIME.mkdir(parents=True, exist_ok=True)
    write_json(RUNTIME / "samples.json", samples)
    write_json(ART / "specification.json", spec)
    return samples


def load_samples():
    path = RUNTIME / "samples.json"
    if not path.exists():
        raise ValueError("run prepare first")
    return read_json(path)


def request_map(name, refs):
    games = load_round_roots(name)
    by_game = {int(game["meta"]["game_id"]): game for game in games}
    out = []
    for ref in refs:
        game = by_game[int(ref["game_id"])]
        out.append(game["roots"][int(ref["decision_index"])])
    return out


def search_round(name, identity, refs, device, simulations_override=None, tag=""):
    teacher = identity["teacher"]
    model = load_frozen_actor_critic_checkpoint(teacher["path"], device=device)
    evaluator = FrozenSearchInference(
        model, device=device,
        config=InferenceConfig(mode="eager", compile_failure_policy="fail_fast"))
    requests = request_map(name, refs)
    search = identity["search_config"]
    if simulations_override is not None:
        search = {**search, "simulations": int(simulations_override)}
    output = RUNTIME / name / (f"search_{tag}" if tag else "search")
    output.mkdir(parents=True, exist_ok=True)
    combined = {}
    runner = NativeBatchRunner(evaluator, load_default_catalog(),
                               capacity=ACTIVE_ROOTS, buckets=BUCKETS)
    for rep in range(FRESH_SAMPLES):
        targets = []
        counts = []
        worlds = []
        noises = []
        for start in range(0, len(requests), ACTIVE_ROOTS):
            part = output / f"rep-{rep}-chunk-{start:05d}-sim{search['simulations']}.npz"
            chunk = requests[start:start + ACTIVE_ROOTS]
            seed_pairs = []
            for ref in refs[start:start + ACTIVE_ROOTS]:
                seed_pairs.append((
                    derive_seed(FRESH_WORLD_SEED, identity["round_index"],
                                int(ref["game_id"]), "fresh_m4_world", rep,
                                int(ref["decision_index"])),
                    derive_seed(FRESH_NOISE_SEED, identity["round_index"],
                                int(ref["game_id"]), "fresh_m4_noise", rep,
                                int(ref["decision_index"]))))
            if not part.exists():
                width = max(len(request.legal_actions) for request in chunk)
                block_targets = np.zeros((len(chunk), width), dtype=np.float64)
                block_counts = np.zeros((len(chunk), width), dtype=np.int32)
                jobs = []
                policies = []
                for request, (world, noise) in zip(chunk, seed_pairs, strict=True):
                    policy = NativePUCTSearchPolicy(
                        model, evaluator=evaluator,
                        config=MCTSConfig(**{**search, "seed": int(world)}))
                    policies.append(policy)
                    jobs.append(policy.start_job(request, add_root_noise=identity["add_root_noise"],
                                                 temperature=0.0, sampling_seed=int(noise)))
                while any(not job.done for job in jobs):
                    if runner.run([job for job in jobs if not job.done]) <= 0:
                        raise RuntimeError(f"{name}: search made no progress")
                for j, job in enumerate(jobs):
                    result = job.result
                    if not result.searchable or sum(result.root_visit_counts) != search["simulations"]:
                        raise ValueError(f"{name}: fresh search budget mismatch")
                    width_j = len(result.policy_target)
                    block_targets[j, :width_j] = result.policy_target
                    block_counts[j, :width_j] = result.root_visit_counts
                temporary = part.with_suffix(".npz.tmp")
                with temporary.open("wb") as handle:
                    np.savez_compressed(handle, targets=block_targets, counts=block_counts,
                                        seed_pairs=np.asarray(seed_pairs, dtype=np.uint64))
                os.replace(temporary, part)
                runner.batch_records.clear()
                del jobs, policies
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            with np.load(part, allow_pickle=False) as archive:
                if not np.array_equal(archive["seed_pairs"],
                                      np.asarray(seed_pairs, dtype=np.uint64)) or (
                        not np.all(archive["counts"].sum(axis=1) == search["simulations"])):
                    raise ValueError(f"{name}: saved fresh chunk identity mismatch")
                targets.append(archive["targets"])
                counts.append(archive["counts"])
                worlds.append(archive["seed_pairs"][:, 0])
                noises.append(archive["seed_pairs"][:, 1])
        width = max(array.shape[1] for array in targets)
        padded_targets = [np.pad(array, ((0, 0), (0, width - array.shape[1])))
                          for array in targets]
        padded_counts = [np.pad(array, ((0, 0), (0, width - array.shape[1])))
                         for array in counts]
        combined[rep] = (np.concatenate(padded_targets), np.concatenate(padded_counts),
                         np.concatenate(worlds), np.concatenate(noises))
        emit(event="search_rep_done", round=name, rep=rep + 1, roots=len(requests))
    width = max(array[0].shape[1] for array in combined.values())
    fresh_targets = np.zeros((FRESH_SAMPLES, len(requests), width), dtype=np.float64)
    fresh_counts = np.zeros((FRESH_SAMPLES, len(requests), width), dtype=np.int32)
    fresh_worlds = np.zeros((FRESH_SAMPLES, len(requests)), dtype=np.uint64)
    fresh_noises = np.zeros((FRESH_SAMPLES, len(requests)), dtype=np.uint64)
    for rep, (targets, counts, worlds, noises) in combined.items():
        fresh_targets[rep, :, :targets.shape[1]] = targets
        fresh_counts[rep, :, :counts.shape[1]] = counts
        fresh_worlds[rep] = worlds
        fresh_noises[rep] = noises
    mean = fresh_targets.mean(axis=0)
    mean /= np.maximum(mean.sum(axis=1, keepdims=True), 1e-12)
    payload = {
        "round": name, "teacher": teacher, "round_index": identity["round_index"],
        "fresh_samples": FRESH_SAMPLES, "world_seed": FRESH_WORLD_SEED,
        "noise_seed": FRESH_NOISE_SEED, "search_config": search,
        "add_root_noise": identity["add_root_noise"],
        "game_id": np.asarray([r["game_id"] for r in refs], dtype=np.int64),
        "decision_index": np.asarray([r["decision_index"] for r in refs], dtype=np.int64),
        "split": np.asarray([r["file"] for r in refs]),
        "targets": fresh_targets, "counts": fresh_counts, "target_mean": mean,
        "world_seeds": fresh_worlds, "noise_seeds": fresh_noises,
    }
    filename = f"fresh_targets_{tag}.npz" if tag else "fresh_targets.npz"
    np.savez_compressed(RUNTIME / name / filename, **payload)
    emit(event="search_done", round=name, roots=len(requests),
         simulations=int(fresh_counts.sum()))


def softmax(logits):
    values = np.asarray(logits, dtype=np.float64)
    values = values - values.max()
    exp = np.exp(values)
    return exp / exp.sum()


def kl(pi, p):
    pi = np.asarray(pi, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    mask = pi > 0
    return float(np.sum(pi[mask] * (np.log(pi[mask]) - np.log(p[mask]))))


def entropy(pi):
    pi = np.asarray(pi, dtype=np.float64)
    mask = pi > 0
    return float(-np.sum(pi[mask] * np.log(pi[mask])))


def js(p, q):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    m = 0.5 * (p + q)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def evaluate_round(name, identity, refs, device, tag=""):
    requests = request_map(name, refs)
    filename = f"fresh_targets_{tag}.npz" if tag else "fresh_targets.npz"
    with np.load(RUNTIME / name / filename, allow_pickle=False) as fresh:
        if (not np.array_equal(fresh["game_id"], [r["game_id"] for r in refs])
                or not np.array_equal(fresh["decision_index"], [r["decision_index"] for r in refs])):
            raise ValueError(f"{name}: fresh targets and evaluation roots differ")
        fresh_mean = fresh["target_mean"]
        fresh_targets = fresh["targets"]
    before = FrozenSearchInference(
        load_frozen_actor_critic_checkpoint(identity["before"]["path"], device=device),
        device=device, config=InferenceConfig(mode="eager", compile_failure_policy="fail_fast"))
    after = FrozenSearchInference(
        load_frozen_actor_critic_checkpoint(identity["after"]["path"], device=device),
        device=device, config=InferenceConfig(mode="eager", compile_failure_policy="fail_fast"))
    stored = {}
    offsets = {}
    for ref in refs:
        path = round_stage(name) / "policy_labels/games" / ref["file"]
        if path not in stored:
            with np.load(path, allow_pickle=False) as archive:
                stored[path] = archive["target_mean"]
        if ref["file"] not in offsets:
            game_path = round_stage(name) / "selfplay/games" / ref["file"]
            arrays, _ = load_game(game_path)
            offsets[ref["file"]] = arrays["candidate_offsets"]
    rows = []
    for start in range(0, len(requests), 128):
        chunk = requests[start:start + 128]
        before_values = before.policy_value(chunk)
        after_values = after.policy_value(chunk)
        for offset, request in enumerate(chunk):
            index = start + offset
            ref = refs[index]
            pi_fresh = fresh_mean[index, :len(request.legal_actions)]
            pi_stored = np.asarray(
                stored[(round_stage(name) / "policy_labels/games" / ref["file"])][
                    :], dtype=np.float64)
            candidate_offsets = offsets[ref["file"]]
            a, b = (candidate_offsets[ref["decision_index"]:
                                      ref["decision_index"] + 2])
            pi_stored = pi_stored[a:b]
            p_before = softmax(before_values[offset][0])
            p_after = softmax(after_values[offset][0])
            pair_js = [js(fresh_targets[i, index, :len(request.legal_actions)],
                          fresh_targets[j, index, :len(request.legal_actions)])
                       for i in range(FRESH_SAMPLES) for j in range(i + 1, FRESH_SAMPLES)]
            rows.append({
                "round": name, "set": ref["split"],
                "game_id": int(ref["game_id"]), "decision_index": int(ref["decision_index"]),
                "width": int(len(request.legal_actions)),
                "kl_fresh_before": kl(pi_fresh, p_before),
                "kl_fresh_after": kl(pi_fresh, p_after),
                "kl_stored_before": kl(pi_stored, p_before),
                "kl_stored_after": kl(pi_stored, p_after),
                "top1_fresh_before": int(np.argmax(pi_fresh) == np.argmax(p_before)),
                "top1_fresh_after": int(np.argmax(pi_fresh) == np.argmax(p_after)),
                "top1_stored_before": int(np.argmax(pi_stored) == np.argmax(p_before)),
                "top1_stored_after": int(np.argmax(pi_stored) == np.argmax(p_after)),
                "entropy_fresh": entropy(pi_fresh),
                "entropy_stored": entropy(pi_stored),
                "entropy_before": entropy(p_before),
                "entropy_after": entropy(p_after),
                "fresh_pair_js_mean": float(np.mean(pair_js)),
                "js_fresh_stored": js(pi_fresh, pi_stored),
            })
        emit(event="evaluate_progress", round=name, done=start + len(chunk), total=len(requests))
    path = RUNTIME / name / (f"positions_{tag}.jsonl" if tag else "positions.jsonl")
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    del before, after
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    emit(event="evaluate_done", round=name, positions=len(rows))
    return rows


def bootstrap_mean(values, clusters, resamples, seed, *, strata=None):
    """Resample whole games with multiplicity, keeping round weights fixed."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"estimate": None, "lower": None, "upper": None}
    clusters = np.asarray(clusters)
    strata = np.zeros(len(values), dtype=np.int64) if strata is None else np.asarray(strata)
    if values.shape != clusters.shape or values.shape != strata.shape:
        raise ValueError("bootstrap values, clusters and strata must have matching shapes")
    rng = np.random.default_rng(seed)
    means = np.zeros(resamples, dtype=np.float64)
    games = 0
    for stratum in np.unique(strata):
        mask = strata == stratum
        unique, inverse = np.unique(clusters[mask], return_inverse=True)
        sums = np.bincount(inverse, weights=values[mask])
        counts = np.bincount(inverse)
        chosen = rng.integers(0, len(unique), size=(resamples, len(unique)))
        means += (mask.sum() / len(values)) * (
            sums[chosen].sum(axis=1) / counts[chosen].sum(axis=1))
        games += len(unique)
    return {"estimate": float(values.mean()),
            "lower": float(np.percentile(means, 2.5)),
            "upper": float(np.percentile(means, 97.5)), "games": games}


def summarize(rows):
    summary = {}
    for key in SETS:
        set_rows = [r for r in rows if r["set"] == key]
        entry = {}
        for metric in ("kl_fresh", "kl_stored"):
            if not set_rows:
                entry[metric] = {"positions": 0}
                continue
            delta = [r[f"{metric}_after"] - r[f"{metric}_before"] for r in set_rows]
            clusters = [r["game_id"] for r in set_rows]
            branch = metric.split("_")[1]
            entry[metric] = {
                **bootstrap_mean(delta, clusters, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED),
                "positions": len(set_rows),
                "improved_fraction": float(np.mean([d < 0 for d in delta])),
                "median_delta": float(np.median(delta)),
                "mean_before": float(np.mean([r[f"{metric}_before"] for r in set_rows])),
                "mean_after": float(np.mean([r[f"{metric}_after"] for r in set_rows])),
                "top1_before": float(np.mean([r[f"top1_{branch}_before"] for r in set_rows])),
                "top1_after": float(np.mean([r[f"top1_{branch}_after"] for r in set_rows])),
            }
        if set_rows:
            entry["entropy"] = {
                "fresh": float(np.mean([r["entropy_fresh"] for r in set_rows])),
                "stored": float(np.mean([r["entropy_stored"] for r in set_rows])),
                "student_before": float(np.mean([r["entropy_before"] for r in set_rows])),
                "student_after": float(np.mean([r["entropy_after"] for r in set_rows])),
            }
        summary[key] = entry
    return summary


def load_result_rows(round_names):
    """Read every completed set; analysis must not silently drop supplements."""
    output = {}
    for name in round_names:
        paths = [RUNTIME / name / "positions.jsonl"]
        paths += [RUNTIME / name / f"positions_{tag}.jsonl"
                  for tag in ("ledger", "unseen")
                  if (RUNTIME / name / f"positions_{tag}.jsonl").exists()]
        output[name] = [json.loads(line) for path in paths
                        for line in path.read_text().splitlines()]
    return output


def analyze(rows_by_round, *, output_path=None):
    output = {"protocol": "avg4-independent-m4-delta-kl-summary-v2",
              "client_date": "2026-10-09",
              "bootstrap": {"cluster": "source game", "strata": "round",
                            "multiplicity": "retained", "round_weights": "position counts fixed",
                            "resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED},
              "rounds": {}, "pooled": {}}
    for name, rows in rows_by_round.items():
        output["rounds"][name] = summarize(rows)
    for key in SETS:
        subset = [r for rows in rows_by_round.values() for r in rows if r["set"] == key]
        if not subset:
            continue
        delta = [r["kl_fresh_after"] - r["kl_fresh_before"] for r in subset]
        clusters = [f"{r['round']}:{r['game_id']}" for r in subset]
        strata = [r["round"] for r in subset]
        output["pooled"][key] = {
            **bootstrap_mean(delta, clusters, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, strata=strata),
            "positions": len(subset),
            "improved_fraction": float(np.mean([d < 0 for d in delta])),
        }
        stored_delta = [r["kl_stored_after"] - r["kl_stored_before"] for r in subset]
        output["pooled"][key]["stored_delta"] = bootstrap_mean(
            stored_delta, clusters, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, strata=strata)
        output["pooled"][key]["fresh_minus_stored_delta"] = bootstrap_mean(
            np.asarray(delta) - np.asarray(stored_delta), clusters,
            BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, strata=strata)
    output["fresh_label_internal_js"] = float(np.mean(
        [r["fresh_pair_js_mean"] for rows in rows_by_round.values() for r in rows]))
    output["fresh_vs_stored_js"] = float(np.mean(
        [r["js_fresh_stored"] for rows in rows_by_round.values() for r in rows]))
    if (RUNTIME / "ledger_samples.json").exists():
        output["ledger_samples"] = read_json(RUNTIME / "ledger_samples.json")
    if (ART / "unseen_specification.json").exists():
        output["unseen_specification"] = record(ART / "unseen_specification.json")
    output["source"] = git_identity()
    write_json(output_path or ART / "summary.json", output)
    return output


def run_all(device, limit=None, simulations=None):
    samples = load_samples() if (RUNTIME / "samples.json").exists() else prepare()
    spec = read_json(ART / "specification.json")
    for name, identity in spec["rounds"].items():
        refs = samples["rounds"][name]["train"] + samples["rounds"][name]["validation"]
        if limit is not None:
            half = max(1, limit // 2)
            refs = (samples["rounds"][name]["train"][:half]
                    + samples["rounds"][name]["validation"][:half])
        search_round(name, identity, refs, device, simulations_override=simulations)
    rows_by_round = {}
    for name, identity in spec["rounds"].items():
        refs = samples["rounds"][name]["train"] + samples["rounds"][name]["validation"]
        if limit is not None:
            half = max(1, limit // 2)
            refs = (samples["rounds"][name]["train"][:half]
                    + samples["rounds"][name]["validation"][:half])
        rows_by_round[name] = evaluate_round(name, identity, refs, device)
    analyze(load_result_rows(spec["rounds"]))
    emit(event="all_done", rounds=list(rows_by_round))


def current_policy_draws(name, identity):
    usage = round_stage(name) / "usage/policy_recent"
    bins = sorted(usage.glob("*.bin"))
    if len(bins) != 1:
        raise ValueError(f"{name}: expected one policy ledger, found {bins}")
    rows = np.frombuffer(bins[0].read_bytes(), dtype="<i8").reshape(-1, 4)
    current = rows[rows[:, 1] == identity["round_index"]]
    return current, bins[0]


def ledger_refs(name, identity, count=SAMPLES_PER_SET):
    """Sample unique current-round positions actually drawn by the policy ledger."""
    current, _ = current_policy_draws(name, identity)
    unique = sorted({(int(round_index), int(game_id), int(decision))
                     for _, round_index, game_id, decision in current})
    games = load_round_roots(name)
    by_game = {int(game["meta"]["game_id"]): game for game in games}
    searchable = []
    dropped = 0
    for _, game_id, decision in unique:
        game = by_game.get(game_id)
        if (game is None or str(game["meta"]["split"]) != "train"
                or decision not in game["roots"]):
            dropped += 1
            continue
        searchable.append((game_id, decision))
    rng = np.random.default_rng(derive_seed(SAMPLE_SEED_BASE, identity["round_index"],
                                            0, "ledger_sample"))
    take = min(count, len(searchable))
    chosen = sorted(int(i) for i in rng.choice(len(searchable), size=take, replace=False))
    refs = []
    for i in chosen:
        game_id, decision = searchable[i]
        game = by_game[game_id]
        refs.append({"game_id": game_id, "decision_index": decision,
                     "file": game["path"].name, "split": "train_ledger"})
    return {"draws_current_round": int(len(current)),
            "unique_current_round": int(len(unique)),
            "searchable_unique_current_round": len(searchable),
            "dropped_non_searchable": dropped, "sampled": len(refs)}, refs


def run_ledger(device, simulations=None):
    samples = load_samples()
    spec = read_json(ART / "specification.json")
    stats_out = {}
    for name, identity in spec["rounds"].items():
        stats, refs = ledger_refs(name, identity)
        stats_out[name] = stats
        emit(event="ledger_sample", round=name, **stats)
        search_round(name, identity, refs, device, simulations_override=simulations,
                     tag="ledger")
        evaluate_round(name, identity, refs, device, tag="ledger")
    write_json(RUNTIME / "ledger_samples.json", stats_out)
    analyze(load_result_rows(spec["rounds"]))
    emit(event="ledger_done", rounds=list(spec["rounds"]))


def unseen_refs(name, identity, count=SAMPLES_PER_SET):
    """Uniform searchable Train roots absent from this round's policy draws."""
    current, ledger = current_policy_draws(name, identity)
    seen = {(int(game), int(decision)) for _, _, game, decision in current}
    games = load_round_roots(name)
    train = [{"game_id": int(game["meta"]["game_id"]), "decision_index": int(i),
              "file": game["path"].name, "split": "train_unseen"}
             for game in games if game["meta"]["split"] == "train"
             for i in game["roots"]]
    pool = [ref for ref in train if (ref["game_id"], ref["decision_index"]) not in seen]
    if len(pool) < count:
        raise ValueError(f"{name}: not enough never-drawn Train roots")
    rng = np.random.default_rng(derive_seed(SAMPLE_SEED_BASE, identity["round_index"],
                                            0, "unseen_sample"))
    chosen = sorted(int(i) for i in rng.choice(len(pool), size=count, replace=False))
    refs = [pool[i] for i in chosen]
    return {"draws_current_round": len(current), "train_searchable": len(train),
            "seen_searchable": len(train) - len(pool), "unseen_searchable": len(pool),
            "sampled": len(refs), "sampled_games": len({r["game_id"] for r in refs}),
            "ledger": record(ledger),
            "game_archives_sha256": hashlib.sha256(json.dumps(
                [(g["path"].name, sha256(g["path"])) for g in games],
                separators=(",", ":")).encode()).hexdigest()}, refs


def run_unseen(device):
    spec = read_json(ART / "specification.json")
    path = ART / "unseen_specification.json"
    saved = read_json(path) if path.exists() else None
    rounds = {}
    for name, identity in spec["rounds"].items():
        for who in ("teacher", "before", "after"):
            if sha256(Path(identity[who]["path"])) != identity[who]["sha256"]:
                raise ValueError(f"{name}: {who} checkpoint changed")
        stats, refs = unseen_refs(name, identity)
        rounds[name] = {"identity": identity, "statistics": stats, "refs": refs}
        emit(event="unseen_sample", round=name, **stats)
    if saved is not None and saved["rounds"] != rounds:
        raise ValueError("never-drawn samples or their source identity changed")
    if saved is None:
        write_json(path, {"protocol": "avg4-independent-m4-train-unseen-v1",
                         "selection": "uniform without replacement among Train searchable roots absent from current-round policy ledger",
                         "samples_per_round": SAMPLES_PER_SET, "fresh_samples": FRESH_SAMPLES,
                         "sample_seed_base": SAMPLE_SEED_BASE, "sample_stream": "unseen_sample",
                         "fresh_world_seed": FRESH_WORLD_SEED, "fresh_noise_seed": FRESH_NOISE_SEED,
                         "source": git_identity(), "environment": environment_identity(),
                         "execution": {"device": device, "inference_mode": "eager", "active_roots": ACTIVE_ROOTS,
                                       "buckets": BUCKETS, "tf32": False}, "rounds": rounds})
    for name, entry in rounds.items():
        search_round(name, entry["identity"], entry["refs"], device, tag="unseen")
        evaluate_round(name, entry["identity"], entry["refs"], device, tag="unseen")
    analyze(load_result_rows(spec["rounds"]))
    emit(event="unseen_done", rounds=list(rounds))


def run_smoke(limit=8, simulations=16):
    """Use a separate directory; a small-budget test cannot overwrite results."""
    global RUNTIME
    samples = load_samples()
    spec = read_json(ART / "specification.json")
    RUNTIME = RUNTIME / f"smoke-limit{limit}-sim{simulations}"
    name = next(iter(spec["rounds"]))
    identity = spec["rounds"][name]
    half = max(1, limit // 2)
    refs = samples["rounds"][name]["train"][:half] + samples["rounds"][name]["validation"][:half]
    search_round(name, identity, refs, "cpu", simulations_override=simulations)
    rows = evaluate_round(name, identity, refs, "cpu")
    analyze({name: rows}, output_path=RUNTIME / "summary.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "search", "evaluate", "analyze",
                                           "all", "smoke", "ledger", "unseen"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--simulations", type=int)
    args = parser.parse_args()
    if args.action != "smoke" and (args.limit is not None or args.simulations is not None):
        parser.error("--limit and --simulations are only allowed for isolated smoke")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.action == "prepare":
        prepare()
    elif args.action == "search":
        samples = load_samples()
        spec = read_json(ART / "specification.json")
        for name, identity in spec["rounds"].items():
            refs = samples["rounds"][name]["train"] + samples["rounds"][name]["validation"]
            search_round(name, identity, refs, args.device,
                         simulations_override=args.simulations)
    elif args.action == "evaluate":
        samples = load_samples()
        spec = read_json(ART / "specification.json")
        rows = {name: evaluate_round(
            name, identity,
            samples["rounds"][name]["train"] + samples["rounds"][name]["validation"],
            args.device)
                for name, identity in spec["rounds"].items()}
        analyze(load_result_rows(spec["rounds"]))
    elif args.action == "analyze":
        spec = read_json(ART / "specification.json")
        analyze(load_result_rows(spec["rounds"]))
    elif args.action == "smoke":
        run_smoke(limit=args.limit or 8, simulations=args.simulations or 16)
    elif args.action == "ledger":
        run_ledger(args.device, simulations=args.simulations)
    elif args.action == "unseen":
        run_unseen(args.device)
    else:
        run_all(args.device, limit=args.limit, simulations=args.simulations)
