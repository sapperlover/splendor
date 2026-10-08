"""Formal replay-buffer learning-rate and step-dose response probe.

All arms restore the same committed checkpoint (parameters, AdamW moments and
RNG states) and replay the exact batch ledger of one completed formal
split-training round: policy samples from the most recent four collection
rounds, value samples 50/50 between the recent four rounds and the older
3000-game FIFO window.  Only the optimizer learning rate differs, so every
node is a directly comparable prefix of a shared data stream.

Axes:
- lr:      2.5e-5 / 5e-5 / 1e-4
- nodes:   25 / 50 / 100 optimizer steps (nine candidate nodes, lr1e4@100 is
            the frozen formal schedule control)

Diagnostics: stored-target holdout KL (formal recent/current validation),
eight-search empirical-mean bank KL, node-vs-start policy KL, top-1 action
change rate and value-function drift.  Strength panels are small (50 pairs,
S1024/N1, against the frozen start) and use the same DEV seeds as the earlier
step-dose and averaged-label probes.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from splendor.alphazero_checkpoint import (load_alphazero_checkpoint, model_state_sha256,
                                           move_optimizer_state)
from splendor.alphazero_recording import (append_json, environment_identity, read_json,
                                          sha256, source_identity, write_json)
from splendor.alphazero_replay import ReplayReference
from splendor.alphazero_train import AlphazeroLearner, LearnerConfig, _torch_batch
from splendor.evaluation import FixedEvaluationConfig, run_fixed_evaluation
from splendor.hybrid_train import SplitTrainingConfig, draw_split_batch, split_replay_pools
from splendor.offline_train import evaluate_validation, make_index
from splendor.rng import DeterministicRng
from splendor.selfplay_data import collate_selfplay_samples

ROOT = Path(__file__).resolve().parents[2]
ART = Path(__file__).resolve().parent
RUNTIME = ROOT / "runtime/entity_attention_formal_lr_step_20261007"
OUT = ROOT / "runs/entity_attention_formal_lr_step_20261007"

SOURCE_RUN = ROOT / "runs/entity_attention_hybrid_split_p4_v15_from_round50_seed1"
START = SOURCE_RUN / "rounds/round-019/checkpoints/final.pt"
START_SHA = "dbfbe066144a485d15ec633aa87c85ac27864f142183725b86562e4ba44ed2c4"
REPRO_START = SOURCE_RUN / "rounds/round-018/checkpoints/final.pt"
REPRO_START_SHA = "cab13feca700cdc15ad20e0d4ece731ae648f3d873e62b26933b8238b6eef3ca"
REPRO_REFERENCE = START
ROUND_STAGE = SOURCE_RUN / "rounds/round-019"
SNAPSHOT = ROUND_STAGE / "replay_snapshot.json"
USAGE = ROUND_STAGE / "usage"

HEADS = {"policy_recent": 256, "value_recent": 128, "value_history": 128}
STEPS = 100
NODES = (25, 50, 100)
ARMS = {"lr2p5e5": 2.5e-5, "lr5e5": 5.0e-5, "lr1e4": 1.0e-4}
REPRO_ARM = "repro_round019"

# Boundary extension: a second frozen ledger over the same pools and rule,
# drawn with an independent fixed seed, continues the 100-step trajectories.
EXT_SEED = 17030000
EXT_STEPS = 50
EXT_ARMS = ("lr2p5e5", "lr1e4")
EXT_NODES = (125, 150)

TRAIN_CACHE_BYTES = 6 * 1024 ** 3
EVAL_BATCH = 256
PANEL_PAIRS = 50
SEED_SETS = {"dev": {"game": 16830000, "policy": 16840000, "bootstrap": 16860000},
             "head": {"game": 16960000, "policy": 16970000, "bootstrap": 16980000}}

_avg_spec = importlib.util.spec_from_file_location(
    "late_stage_averaging_probe",
    ROOT / "artifacts/entity_attention_target_averaging_20261007/experiment.py")
avg = importlib.util.module_from_spec(_avg_spec)
_avg_spec.loader.exec_module(avg)


def case_paths():
    cases = {"start": START}
    for arm in ARMS:
        for step in NODES:
            cases[f"{arm}_{step}"] = checkpoint(arm, step)
    if (ART / "specification_extension.json").exists():
        for arm in EXT_ARMS:
            for step in EXT_NODES:
                cases[f"{arm}_{step}"] = checkpoint(arm, step)
    return cases


def emit(**kwargs):
    print(json.dumps(kwargs, ensure_ascii=False), flush=True)


def record(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def settings():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def checkpoint(arm, step):
    return OUT / arm / f"checkpoints/step-{step:06d}.pt"


def _tensor_digest(parts, digest):
    for name, value in parts:
        digest.update(f"{name}:".encode())
        if isinstance(value, torch.Tensor):
            digest.update(value.detach().to("cpu").contiguous().numpy().tobytes())
        else:
            digest.update(json.dumps(value, sort_keys=True, default=str).encode())


def optimizer_state_sha256(optimizer):
    state = optimizer.state_dict()
    digest = hashlib.sha256()
    for key in sorted(state["state"], key=str):
        for name in sorted(state["state"][key], key=str):
            _tensor_digest([(f"{key}:{name}", state["state"][key][name])], digest)
    # The learning rate is the controlled axis, never part of state identity.
    groups = [{k: v for k, v in group.items() if k not in ("params", "lr")}
              for group in state["param_groups"]]
    _tensor_digest([("param_groups", groups)], digest)
    return digest.hexdigest()


def rng_identity(payload, learner=None):
    sampling = asdict(learner.sampling_rng.snapshot()) if learner is not None \
        else asdict(payload.sampling_rng_state)
    cpu_state = torch.get_rng_state() if learner is not None else payload.torch_cpu_rng_state
    out = {"sampling_rng": sampling,
           "sampling_rng_sha256": hashlib.sha256(json.dumps(sampling, sort_keys=True).encode()).hexdigest(),
           "torch_cpu_sha256": hashlib.sha256(cpu_state.cpu().numpy().tobytes()).hexdigest()}
    if learner is not None and torch.cuda.is_available():
        cuda_states = [state.cpu() for state in torch.cuda.get_rng_state_all()]
    else:
        cuda_states = [state.cpu() for state in payload.torch_cuda_rng_states]
    out["torch_cuda_sha256"] = [hashlib.sha256(state.numpy().tobytes()).hexdigest()
                                for state in cuda_states]
    return out


def ledger_bins():
    paths = {}
    for head in HEADS:
        candidates = sorted((USAGE / head).glob("*.bin"))
        if len(candidates) != 1:
            raise ValueError(f"expected exactly one committed ledger for {head}: {candidates}")
        paths[head] = candidates[0]
    return paths


def load_ledger():
    ledger = {}
    for head, expected in HEADS.items():
        path = ledger_bins()[head]
        raw = path.read_bytes()
        if len(raw) % 32:
            raise ValueError(f"torn ledger {path}: {len(raw)} bytes")
        rows = np.frombuffer(raw, dtype="<i8").reshape(-1, 4)
        steps = np.unique(rows[:, 0])
        if list(steps) != list(range(1, STEPS + 1)):
            raise ValueError(f"ledger {head} does not cover committed steps 1..{STEPS}")
        counts = np.bincount(rows[:, 0], minlength=STEPS + 1)[1:]
        if counts.min() != expected or counts.max() != expected:
            raise ValueError(f"ledger {head} per-step counts are not {expected}")
        ledger[head] = rows
    return ledger


def build_index(cache_bytes):
    snapshot = read_json(SNAPSHOT)
    return make_index({"games": snapshot["games"]}, cache_bytes)


def restore(start, learning_rate, device="cuda", steps=STEPS):
    payload = load_alphazero_checkpoint(start, device=device)
    config = LearnerConfig(**payload.learner_config)
    if str(config.device) != str(device):
        config = dataclasses.replace(config, device=str(device))
    learner = AlphazeroLearner(payload.build_model(device=device), config,
                               sampling_rng=DeterministicRng.restore(payload.sampling_rng_state))
    learner.optimizer.load_state_dict(payload.optimizer_state_dict)
    move_optimizer_state(learner.optimizer, learner.device)
    learner.round_index = payload.round_index
    learner.total_train_step = payload.total_train_step
    learner.round_train_step = 0
    learner.round_steps_target = steps
    learner.parent_checkpoint = dict(payload.parent_checkpoint)
    learner.replay_identity = dict(payload.replay_identity)
    torch.set_rng_state(payload.torch_cpu_rng_state.cpu())
    if torch.cuda.is_available() and payload.torch_cuda_rng_states:
        torch.cuda.set_rng_state_all([state.cpu() for state in payload.torch_cuda_rng_states])
    for group in learner.optimizer.param_groups:
        group["lr"] = float(learning_rate)
    return learner, payload


def ledger_references(games):
    """Translate every ledger row into a checked ReplayReference."""
    ledger = load_ledger()
    refs, translated = {}, {}
    for head, rows in ledger.items():
        items = []
        for step, round_index, game_id, decision in rows:
            key = (int(round_index), int(game_id))
            game = games.get(key)
            if game is None:
                raise ValueError(f"ledger reference {key} is absent from the frozen snapshot")
            if game.split != "train":
                raise ValueError(f"ledger reference {key} is not a train game")
            items.append(ReplayReference(int(round_index), game.run_id, int(game_id), int(decision)))
        refs[head] = items
        translated[head] = rows[:, 1:]
    for head, rows in ledger.items():
        got = np.asarray([(r.round_index, r.game_id, r.decision_index) for r in refs[head]], dtype="<i8")
        if not np.array_equal(got, rows[:, 1:]):
            raise ValueError(f"ledger translation reordered {head}")
    per_step_refs = {head: {step: refs[head][(step - 1) * count:step * count]
                            for step in range(1, STEPS + 1)}
                     for head, count in HEADS.items()}
    return per_step_refs


def resolve_samples(index, games, references):
    return [index.cache.sample(games[(ref.round_index, ref.game_id)], ref.decision_index)
            for ref in references]


def train_arm(arm, *, start=START, learning_rate=None, steps=STEPS, nodes=(), tag=None):
    name = arm if tag is None else f"{arm}_{tag}"
    result_path = ART / f"train_{name}.json"
    if result_path.exists():
        emit(event="train_skip", arm=name, reason="result already exists")
        return
    lr = ARMS[arm] if learning_rate is None else float(learning_rate)
    spec = read_json(ART / "specification.json")
    learner, payload = restore(start, lr, steps=steps)
    start_optimizer_sha = optimizer_state_sha256(learner.optimizer)
    identity = rng_identity(payload, learner)
    expected = spec["rng_identity"] if Path(start).resolve() == START.resolve() else spec["repro_rng_identity"]
    if identity != expected:
        raise ValueError(f"{name}: restored RNG identity differs from the frozen record")
    state_sha = model_state_sha256(learner.model.state_dict())
    expected_model = spec["start"]["model_state_sha256"] if Path(start).resolve() == START.resolve() \
        else spec["repro_start"]["model_state_sha256"]
    if state_sha != expected_model:
        raise ValueError(f"{name}: restored model state differs from the frozen record")
    if start_optimizer_sha != (spec["start"]["optimizer_state_sha256"]
                               if Path(start).resolve() == START.resolve()
                               else spec["repro_start"]["optimizer_state_sha256"]):
        raise ValueError(f"{name}: restored AdamW state differs from the frozen record")
    index = build_index(TRAIN_CACHE_BYTES)
    games = {(game.round_index, game.game_id): game for game in index.games}
    per_step_refs = ledger_references(games)
    output = OUT / name
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "training_metrics.jsonl"
    if metrics_path.exists():
        raise ValueError(f"{metrics_path} exists; refusing to append to a previous trajectory")
    started = time.perf_counter()
    saved = []
    for step in range(1, steps + 1):
        policy_refs = per_step_refs["policy_recent"][step]
        value_refs = per_step_refs["value_recent"][step] + per_step_refs["value_history"][step]
        policy_batch = collate_selfplay_samples(resolve_samples(index, games, policy_refs))
        value_batch = collate_selfplay_samples(resolve_samples(index, games, value_refs))
        row = learner.train_split_step(policy_batch, value_batch,
                                       references=policy_refs + value_refs,
                                       policy_max_grad_norm=1.0, value_max_grad_norm=1.0)
        append_json(metrics_path, row)
        if step in nodes:
            path = checkpoint(name, step)
            learner.save(path, metadata={"protocol": "formal-lr-step-probe-v1", "arm": name,
                                         "learning_rate": lr, "ledger": spec["ledger"]["digest"],
                                         "start_checkpoint": record(start)})
            saved.append({"step": step, "checkpoint": record(path)})
        if step % 25 == 0:
            emit(event="train_progress", arm=name, step=step,
                 elapsed=round(time.perf_counter() - started, 1))
    result = {"protocol": "formal-lr-step-probe-v1", "arm": name, "learning_rate": lr,
              "steps": steps, "nodes": list(nodes), "start": record(start),
              "start_sha256": sha256(start), "rng_identity": identity,
              "start_model_state_sha256": state_sha,
              "start_optimizer_state_sha256": start_optimizer_sha,
              "round_index": learner.round_index, "total_train_step_end": learner.total_train_step,
              "checkpoints": saved, "metrics": record(metrics_path),
              "wall_seconds": time.perf_counter() - started,
              "environment": environment_identity()}
    result["end_model_state_sha256"] = model_state_sha256(learner.model.state_dict())
    write_json(result_path, result)
    del learner, payload, index
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    emit(event="train_done", arm=name, steps=steps, seconds=round(result["wall_seconds"], 2),
         nodes=[entry["step"] for entry in saved])


def train_all():
    for arm in ARMS:
        train_arm(arm, nodes=NODES)


def extension_rows():
    """Freeze a second ledger over the same pools and sampling rule."""
    root = RUNTIME / "ext_ledger"
    meta_path = root / "extension_ledger.json"
    if meta_path.exists():
        meta = read_json(meta_path)
        rows = {head: np.frombuffer((root / f"{head}.bin").read_bytes(), dtype="<i8").reshape(-1, 4)
                for head in HEADS}
        return rows, meta
    index = build_index(TRAIN_CACHE_BYTES)
    config = SimpleNamespace(split_training=SplitTrainingConfig(recent_rounds=4,
                                                                value_recent_positions=128))
    recent_pool, history_pool, recent_rounds = split_replay_pools(index, config)
    rng = DeterministicRng(EXT_SEED)
    collected = {head: [] for head in HEADS}
    for step in range(1, EXT_STEPS + 1):
        drawn = draw_split_batch(recent_pool, history_pool, rng, 256, 128)
        policy, value = drawn.references[:256], drawn.references[256:]
        for head, items in (("policy_recent", policy), ("value_recent", value[:128]),
                            ("value_history", value[128:])):
            collected[head].extend((step, ref.round_index, ref.game_id, ref.decision_index)
                                   for ref in items)
    rows = {head: np.asarray(items, dtype="<i8").reshape(-1, 4)
            for head, items in collected.items()}
    root.mkdir(parents=True, exist_ok=True)
    heads = {}
    for head, array in rows.items():
        path = root / f"{head}.bin"
        path.write_bytes(array.tobytes())
        rounds = Counter(int(value) for value in array[:, 1])
        heads[head] = {**record(path), "per_step": HEADS[head],
                       "rounds": {str(k): v for k, v in sorted(rounds.items())}}
    meta = {"protocol": "formal-lr-step-probe-extension-ledger-v1", "seed": EXT_SEED,
            "steps": EXT_STEPS, "heads": heads, "recent_rounds": sorted(recent_rounds),
            "pool_positions": {"recent": len(recent_pool), "history": len(history_pool)}}
    write_json(meta_path, meta)
    return rows, meta


def prepare_extension():
    if (ART / "specification_extension.json").exists():
        raise ValueError("extension specification already frozen")
    rows, meta = extension_rows()
    for head in ("policy_recent", "value_recent", "value_history"):
        if set(meta["heads"][head]["rounds"]) - {str(r) for r in meta["recent_rounds"]} and head != "value_history":
            raise ValueError(f"{head} leaves the recent rounds")
        if head == "value_history" and set(meta["heads"][head]["rounds"]) & {str(r) for r in meta["recent_rounds"]}:
            raise ValueError("value_history overlaps the recent rounds")
    spec = read_json(ART / "specification.json")
    output = {"protocol": "formal-lr-step-probe-extension-v1",
              "specification_sha256": sha256(ART / "specification.json"),
              "ledger": meta, "arms": {arm: {"learning_rate": ARMS[arm], "extends": "start+100",
                                              "steps": EXT_STEPS, "nodes": list(EXT_NODES)}
                                        for arm in EXT_ARMS},
              "base_ledger_digest": spec["ledger"]["digest"]}
    write_json(ART / "specification_extension.json", output)
    emit(event="extension_prepared", seed=EXT_SEED, arms=list(EXT_ARMS),
         heads={head: len(rows[head]) for head in rows})


def train_extension():
    ext_spec = read_json(ART / "specification_extension.json")
    rows, _ = extension_rows()
    index = build_index(TRAIN_CACHE_BYTES)
    games = {(game.round_index, game.game_id): game for game in index.games}
    per_step = {}
    for head, array in rows.items():
        items = [ReplayReference(int(rr), games[(int(rr), int(gg))].run_id, int(gg), int(dd))
                 for _, rr, gg, dd in array]
        got = np.asarray([(r.round_index, r.game_id, r.decision_index) for r in items], dtype="<i8")
        if not np.array_equal(got, array[:, 1:]):
            raise ValueError(f"extension ledger translation reordered {head}")
        per_step[head] = {step: items[(step - 1) * HEADS[head]:step * HEADS[head]]
                          for step in range(1, EXT_STEPS + 1)}
    for arm in EXT_ARMS:
        result_path = ART / f"train_ext_{arm}.json"
        if result_path.exists():
            emit(event="train_ext_skip", arm=arm)
            continue
        base = checkpoint(arm, STEPS)
        learner, payload = restore(base, ARMS[arm], steps=EXT_STEPS)
        if model_state_sha256(learner.model.state_dict()) != \
                read_json(ART / f"train_{arm}.json")["end_model_state_sha256"]:
            raise ValueError(f"{arm}: extension start differs from the 100-step trajectory")
        output = OUT / arm
        metrics_path = output / "training_metrics_ext.jsonl"
        if metrics_path.exists():
            raise ValueError(f"{metrics_path} exists; refusing to append")
        saved = []
        started = time.perf_counter()
        for step in range(1, EXT_STEPS + 1):
            policy_refs = per_step["policy_recent"][step]
            value_refs = per_step["value_recent"][step] + per_step["value_history"][step]
            policy_batch = collate_selfplay_samples(resolve_samples(index, games, policy_refs))
            value_batch = collate_selfplay_samples(resolve_samples(index, games, value_refs))
            row = learner.train_split_step(policy_batch, value_batch,
                                           references=policy_refs + value_refs,
                                           policy_max_grad_norm=1.0, value_max_grad_norm=1.0)
            append_json(metrics_path, row)
            global_step = STEPS + step
            if global_step in EXT_NODES:
                path = checkpoint(arm, global_step)
                learner.save(path, metadata={"protocol": "formal-lr-step-probe-extension-v1",
                                             "arm": arm, "learning_rate": ARMS[arm],
                                             "extends": str(base)})
                saved.append({"step": global_step, "checkpoint": record(path)})
        result = {"protocol": "formal-lr-step-probe-extension-v1", "arm": arm,
                  "learning_rate": ARMS[arm], "start": record(base), "steps": EXT_STEPS,
                  "nodes": list(EXT_NODES), "checkpoints": saved,
                  "metrics": record(metrics_path), "wall_seconds": time.perf_counter() - started,
                  "ledger": ext_spec["ledger"]["heads"], "environment": environment_identity()}
        write_json(result_path, result)
        emit(event="train_ext_done", arm=arm, nodes=len(saved),
             seconds=round(result["wall_seconds"], 2))


def repro_round019():
    """Bitwise control: replay round-019 training from its parent checkpoint."""
    result_path = ART / "repro_round019.json"
    if result_path.exists():
        emit(event="repro_skip", reason="result already exists")
        return
    train_arm(REPRO_ARM, start=REPRO_START, learning_rate=1e-4, steps=STEPS,
              nodes=(STEPS,), tag="from_round018")
    ours_path = OUT / f"{REPRO_ARM}_from_round018/checkpoints/step-{STEPS:06d}.pt"
    ours = load_alphazero_checkpoint(ours_path, device="cpu")
    theirs = load_alphazero_checkpoint(REPRO_REFERENCE, device="cpu")
    ours_state = ours.build_model().state_dict()
    theirs_state = theirs.build_model().state_dict()
    model_diff = {key: float(torch.max(torch.abs(ours_state[key].float() - theirs_state[key].float())))
                  for key in ours_state}
    optimizer_diff = {}
    mine = ours.optimizer_state_dict
    other = theirs.optimizer_state_dict
    for key in mine["state"]:
        for field in mine["state"][key]:
            a, b = mine["state"][key][field], other["state"][key][field]
            if isinstance(a, torch.Tensor):
                optimizer_diff[f"{key}:{field}"] = float(torch.max(torch.abs(a.float() - b.float())))
            else:
                optimizer_diff[f"{key}:{field}"] = 0.0 if a == b else float("inf")
    result = {"protocol": "formal-lr-step-probe-repro-v1",
              "start": record(REPRO_START), "reference": record(REPRO_REFERENCE),
              "model_max_abs_diff": max(model_diff.values()) if model_diff else None,
              "per_tensor_model_diff": model_diff,
              "optimizer_max_abs_diff": max(optimizer_diff.values()) if optimizer_diff else None,
              "identical": bool(max(model_diff.values()) == 0.0 and max(optimizer_diff.values()) == 0.0),
              "our_checkpoint": record(ours_path)}
    write_json(result_path, result)
    emit(event="repro_done", identical=result["identical"],
         model_max_abs_diff=result["model_max_abs_diff"],
         optimizer_max_abs_diff=result["optimizer_max_abs_diff"])


def raw_outputs(learner, index, games):
    """Per-position log-probabilities, top-1 action and value on one holdout set."""
    model = learner.model
    model.eval()
    widths, values, logps, offsets = [], [], [], [0]
    with torch.inference_mode():
        for game in games:
            start = 0
            while start < game.decision_count:
                chunk = [index.cache.sample(game, i)
                         for i in range(start, min(game.decision_count, start + EVAL_BATCH))]
                batch = collate_selfplay_samples(chunk)
                tensors = _torch_batch(batch, learner.device)
                logits, predicted = model(batch.decisions)
                mask = tensors["candidate_mask"]
                logp = torch.log_softmax(logits.masked_fill(~mask, -torch.inf), dim=-1).cpu().numpy()
                predicted = predicted.cpu().numpy()
                counts = mask.sum(dim=1).cpu().numpy()
                for row in range(len(chunk)):
                    width = int(counts[row])
                    widths.append(width)
                    logps.append(logp[row, :width].astype(np.float32))
                    values.append(float(predicted[row]))
                    offsets.append(offsets[-1] + width)
                start += len(chunk)
    model.eval()
    return {"width": np.asarray(widths, dtype=np.int32),
            "value": np.asarray(values, dtype=np.float32),
            "logp": np.concatenate(logps) if logps else np.zeros(0, dtype=np.float32),
            "offset": np.asarray(offsets, dtype=np.int64)}


def save_raw(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **raw)


def slice_logp(raw, index):
    start, stop = int(raw["offset"][index]), int(raw["offset"][index + 1])
    return raw["logp"][start:stop]


def load_raw(path):
    with np.load(path) as archive:
        return {name: archive[name] for name in archive.files}


def compare_raw(before, after):
    if not np.array_equal(before["width"], after["width"]):
        raise ValueError("holdout candidate widths differ between cases")
    kl_sum = kl_count = 0.0
    top1_all = top1_multi = multi = 0
    value_abs = []
    entropy_before = entropy_after = 0.0
    for i in range(len(before["width"])):
        a, b = slice_logp(after, i), slice_logp(before, i)
        p = np.exp(a.astype(np.float64))
        kl_sum += float(np.sum(p * (a - b)))
        kl_count += 1
        changed = int(np.argmax(a)) != int(np.argmax(b))
        top1_all += changed
        if len(a) > 1:
            multi += 1
            top1_multi += changed
        value_abs.append(abs(float(after["value"][i]) - float(before["value"][i])))
        entropy_before += float(np.sum(np.exp(b) * b))
        entropy_after += float(np.sum(p * a))
    return {"positions": int(kl_count),
            "policy_kl_after_before": kl_sum / kl_count if kl_count else None,
            "top1_change_rate": top1_all / kl_count if kl_count else None,
            "multi_candidate_positions": multi,
            "top1_change_rate_multi": top1_multi / multi if multi else None,
            "value_mean_abs_delta": float(np.mean(value_abs)) if value_abs else None,
            "value_rms_delta": float(np.sqrt(np.mean(np.square(value_abs)))) if value_abs else None,
            "policy_entropy_before": -entropy_before / kl_count if kl_count else None,
            "policy_entropy_after": -entropy_after / kl_count if kl_count else None}


def fit_bank(case_path):
    bank = read_json(avg.BANK)
    rows = bank["positions"]
    requests = [avg.decode_request(row["request"]) for row in rows]
    mean8 = np.load(avg.RAW)["policy_target"][3].mean(axis=0)
    model = avg.load_frozen_actor_critic_checkpoint(case_path, device="cuda").eval()
    evaluator = avg.FrozenSearchInference(model, device="cuda", config=avg.inference_config())
    results = evaluator.policy_value(requests)
    kls, top1 = [], []
    for i, row in enumerate(rows):
        logit = np.asarray(results[i][0], dtype=np.float64)
        logp = logit - np.logaddexp.reduce(logit)
        width = len(logit)
        pi = mean8[i][:width]
        mask = pi > 0
        kls.append(float(np.sum(pi[mask] * (np.log(pi[mask]) - logp[mask]))))
        top1.append(int(np.argmax(logit)) == int(np.argmax(pi)))
    del model, evaluator
    torch.cuda.empty_cache()
    return {"positions": len(rows), "mean_kl_to_mean8": float(np.mean(kls)),
            "top1_match_rate": float(np.mean(top1)),
            "per_root": [{"id": int(row["id"]), "kl_to_mean8": kls[i], "top1": bool(top1[i])}
                         for i, row in enumerate(rows)]}


def diagnostics():
    spec = read_json(ART / "specification.json")
    snapshot = read_json(SNAPSHOT)
    recent_rounds = list(snapshot["split_training"]["recent_round_indices"])
    current_round = max(recent_rounds)
    index = build_index(TRAIN_CACHE_BYTES)
    games = index.games

    def sub_index(rounds):
        sub = make_index({"games": [g.to_dict() for g in games if g.round_index in rounds]},
                         index.cache.max_bytes)
        sub.cache = index.cache
        return sub

    recent_index, current_index = sub_index(recent_rounds), sub_index([current_round])
    recent_games = sorted((g for g in recent_index.games if g.split == "validation"),
                          key=lambda g: g.key)
    current_games = sorted((g for g in current_index.games if g.split == "validation"),
                           key=lambda g: g.key)
    cases = case_paths()
    if not all(Path(path).exists() for path in cases.values()):
        missing = [name for name, path in cases.items() if not Path(path).exists()]
        raise ValueError(f"missing node checkpoints: {missing}")
    output = {}
    for name, path in cases.items():
        learner, _ = restore(path, 1e-4, steps=STEPS)
        recent_metrics = evaluate_validation(learner, recent_index)
        current_metrics = evaluate_validation(learner, current_index)
        raw_recent, raw_current = raw_outputs(learner, recent_index, recent_games), \
            raw_outputs(learner, current_index, current_games)
        baseline = {"kind": "stored_targets",
                    "recent_positions": recent_metrics["positions"],
                    "current_positions": current_metrics["positions"]}
        entry = {"checkpoint": record(path), "baseline": baseline,
                 "holdout_recent": recent_metrics, "holdout_current": current_metrics,
                 "fit_bank": fit_bank(path)}
        save_raw(ART / f"diagnostics/raw_{name}_recent.npz", raw_recent)
        save_raw(ART / f"diagnostics/raw_{name}_current.npz", raw_current)
        entry["raw_recent"] = record(ART / f"diagnostics/raw_{name}_recent.npz")
        entry["raw_current"] = record(ART / f"diagnostics/raw_{name}_current.npz")
        output[name] = entry
        del learner
        torch.cuda.empty_cache()
        emit(event="diagnostics_case", case=name,
             recent_kl=round(recent_metrics["policy_kl"], 5),
             current_kl=round(current_metrics["policy_kl"], 5),
             bank_kl=round(entry["fit_bank"]["mean_kl_to_mean8"], 5))
    for name in output:
        if name == "start":
            continue
        for horizon in ("recent", "current"):
            before = load_raw(ART / f"diagnostics/raw_start_{horizon}.npz")
            after = load_raw(ART / f"diagnostics/raw_{name}_{horizon}.npz")
            output[name][f"change_vs_start_{horizon}"] = compare_raw(before, after)
    write_json(ART / "diagnostics.json",
               {"protocol": "formal-lr-step-probe-diagnostics-v1",
                "snapshot": record(SNAPSHOT), "recent_rounds": recent_rounds,
                "current_round": current_round, "cases": output})
    emit(event="diagnostics_done", cases=list(output))


def panel_spec(arm, step, seeds, name):
    search = {"simulations": 1024, "determinizations": 1, "seed": 16850000,
              "root_noise_fraction": 0.0, "dirichlet_alpha": 0.3, "c_puct": 1.5,
              "fold_forced_phases": True, "max_simulation_transitions": 512}
    inference = {"mode": "eager", "actor_projection": False, "candidate_buckets": [16, 32, 48, 64]}
    return {"policy_a": {"kind": "mcts_checkpoint", "name": name,
                         "checkpoint": str(checkpoint(arm, step)),
                         "search_backend": "native", "search_config": search, "inference": inference},
            "policy_b": {"kind": "mcts_checkpoint", "name": "frozen_16500_native",
                         "checkpoint": str(START), "search_backend": "native",
                         "search_config": search.copy(), "inference": inference.copy()},
            "pairs": PANEL_PAIRS, "seed_start": seeds["game"], "policy_seed_base": seeds["policy"],
            "bootstrap_seed": seeds["bootstrap"], "bootstrap_resamples": 10000, "device": "cuda",
            "workers": 1, "max_completed_turns": 400, "adjudicate_deterministic_policy_cycles": True,
            "execution": {"backend": "native_batched", "active_games": 100, "max_batch_size": 200,
                          "candidate_buckets": [16, 32, 48, 64], "actor_projection": False,
                          "inference_mode": "eager", "max_batch_wait_ms": 0},
            "progress_every": 5, "summary_every": 5,
            "metadata": {"protocol": "formal-lr-step-probe-v1", "arm": arm, "step": step,
                         "seed_set": None}}


def evaluate(case, seed_set):
    arm, step = case.rsplit("_", 1)
    step = int(step)
    seeds = SEED_SETS[seed_set]
    name = f"{case}_{seed_set}_native"
    raw = panel_spec(arm, step, seeds, name)
    raw["metadata"]["seed_set"] = seed_set
    write_json(ART / f"configs/{name}.json", raw)
    config = FixedEvaluationConfig.from_dict(raw)
    output = ART / f"evaluations/{name}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    emit(event="evaluation_start", case=name, pairs=len(config.game_seeds))
    result = run_fixed_evaluation(config, output, resume=output.exists() or
                                  Path(str(output) + ".games").exists(),
                                  require_scored_pairs=True,
                                  progress_callback=lambda done, total: emit(
                                      event="evaluation_progress", case=name,
                                      completed_pairs=done, pairs=total))
    if not result.summary["passed"]:
        raise RuntimeError(f"evaluation contains engineering failures: {output}")
    write_json(ART / f"evaluations/{name}_recording.json",
               {"config": record(ART / f"configs/{name}.json"), "output": record(output),
                "checkpoint_a": record(Path(config.policy_a.checkpoint)),
                "checkpoint_b": record(Path(config.policy_b.checkpoint)),
                "environment": environment_identity()})
    emit(event="evaluation_done", case=name, score=result.summary["win_rate_score"])


def analysis():
    spec = read_json(ART / "specification.json")
    diagnostics_data = read_json(ART / "diagnostics.json")
    panels = {}
    for path in sorted((ART / "evaluations").glob("*_native.json")):
        result = read_json(path)
        if result.get("valid_games") != 2 * PANEL_PAIRS or not result.get("passed"):
            continue
        panels[path.stem] = {key: result[key] for key in
                             ("win_rate_score", "paired_bootstrap", "a_wins", "a_losses", "draws",
                              "valid_games", "elapsed_seconds_this_invocation")}
    differences = {}

    def diff(name_a, name_b, key):
        if name_a in panels and name_b in panels:
            a = read_json(ART / f"evaluations/{name_a}.json")
            b = read_json(ART / f"evaluations/{name_b}.json")
            seed_set = "head" if name_a.endswith("_head_native") else "dev"
            differences[key] = avg.bootstrap_difference(a, b, SEED_SETS[seed_set]["bootstrap"])

    for seed_set in SEED_SETS:
        suffix = f"_{seed_set}_native"
        control = f"lr1e4_100{suffix}"
        for name in panels:
            if name.endswith(suffix) and name != control:
                diff(name, control,
                     f"{name.removesuffix(suffix)}_minus_lr1e4_100_{seed_set}")
    cases = {}
    for name, case in diagnostics_data["cases"].items():
        entry = {"recent_kl": case["holdout_recent"]["policy_kl"],
                 "current_kl": case["holdout_current"]["policy_kl"],
                 "recent_value_loss": case["holdout_recent"]["value_loss"],
                 "bank_kl": case["fit_bank"]["mean_kl_to_mean8"],
                 "bank_top1": case["fit_bank"]["top1_match_rate"]}
        if f"change_vs_start_recent" in case:
            entry.update({"change_recent": case["change_vs_start_recent"],
                          "change_current": case["change_vs_start_current"]})
        cases[name] = entry
    output = {"protocol": "formal-lr-step-probe-analysis-v1", "cases": cases, "panels": panels,
              "panel_differences": differences,
              "limitations": [
                  "Single committed start (16500) and one frozen round ledger; one sampling stream, "
                  "no independent collection or training seed.",
                  "Strength panels are 50-pair DEV screens against the frozen start, not FINAL.",
                  "Value targets come from older teachers and the shared trunk still moves when only "
                  "the value branch is detached."]}
    write_json(ART / "analysis.json", output)
    emit(event="analysis_done", cases=list(cases), panels=list(panels),
         differences={k: round(v["difference"], 4) for k, v in differences.items()})


def verify():
    spec = read_json(ART / "specification.json")
    checks = {"start_sha256": sha256(START) == spec["start"]["sha256"],
              "repro_start_sha256": sha256(REPRO_START) == spec["repro_start"]["sha256"],
              "snapshot_sha256": sha256(SNAPSHOT) == spec["snapshot"]["sha256"]}
    for head, info in spec["ledger"]["heads"].items():
        path = ledger_bins()[head]
        checks[f"ledger_{head}"] = sha256(path) == info["sha256"]
    for arm in ARMS:
        result_path = ART / f"train_{arm}.json"
        checks[f"train_{arm}"] = result_path.exists()
        for step in NODES:
            checks[f"checkpoint_{arm}_{step}"] = checkpoint(arm, step).exists()
    if (ART / "specification_extension.json").exists():
        extension = read_json(ART / "specification_extension.json")
        for head, info in extension["ledger"]["heads"].items():
            checks[f"ext_ledger_{head}"] = sha256(Path(info["path"])) == info["sha256"]
        for arm in EXT_ARMS:
            checks[f"train_ext_{arm}"] = (ART / f"train_ext_{arm}.json").exists()
            for step in EXT_NODES:
                checks[f"checkpoint_{arm}_{step}"] = checkpoint(arm, step).exists()
    checks["repro"] = (ART / "repro_round019.json").exists()
    if checks["repro"]:
        checks["repro_identical"] = read_json(ART / "repro_round019.json")["identical"]
    diagnostics_path = ART / "diagnostics.json"
    checks["diagnostics"] = diagnostics_path.exists()
    if checks["diagnostics"]:
        cases = read_json(diagnostics_path)["cases"]
        checks["diagnostics_cases"] = len(cases) == len(case_paths())
        checks["diagnostics_change_fields"] = all(
            f"change_vs_start_{horizon}" in case
            for name, case in cases.items() if name != "start"
            for horizon in ("recent", "current"))
    for path in sorted((ART / "evaluations").glob("*_native.json")):
        result = read_json(path)
        checks[f"panel_{path.stem}"] = bool(result["passed"] and not result["exceptions"]
                                            and not result["illegal_actions"]
                                            and not result["invariant_failures"]
                                            and not result["engineering_truncations"])
    output = {"protocol": "formal-lr-step-probe-verify-v1", "passed": all(checks.values()),
              "checks": checks}
    write_json(ART / "verification.json", output)
    emit(event="verify_done", passed=output["passed"], checks=checks)


def prepare():
    if (ART / "specification.json").exists():
        raise ValueError("specification already frozen")
    if sha256(START) != START_SHA:
        raise ValueError("start checkpoint changed")
    if sha256(REPRO_START) != REPRO_START_SHA:
        raise ValueError("round-018 checkpoint changed")
    snapshot = read_json(SNAPSHOT)
    split = snapshot["split_training"]
    if split["config"]["recent_rounds"] != 4 or split["config"]["value_recent_positions"] != 128:
        raise ValueError("snapshot split-training protocol differs from the frozen description")
    if split["quotas"] != {"policy_recent": 256, "value_recent": 128, "value_history": 128}:
        raise ValueError("snapshot split-training quotas differ from the frozen description")
    recent_rounds = set(split["recent_round_indices"])
    if len(recent_rounds) != 4:
        raise ValueError("recent window is not four rounds")
    ledger = load_ledger()
    index = build_index(TRAIN_CACHE_BYTES)
    games = {(game.round_index, game.game_id): game for game in index.games}
    composition = {}
    for head, rows in ledger.items():
        rounds = Counter(int(value) for value in rows[:, 1])
        composition[head] = {str(round_index): count for round_index, count in sorted(rounds.items())}
        for round_index in rounds:
            for _, game_id, _ in rows[rows[:, 1] == round_index][:, 1:].astype(int):
                game = games.get((round_index, game_id))
                if game is None or game.split != "train":
                    raise ValueError(f"{head} draws outside the train split: {(round_index, game_id)}")
        if head in ("policy_recent", "value_recent") and not set(rounds) <= recent_rounds:
            raise ValueError(f"{head} leaves the recent four rounds")
        if head == "value_history" and set(rounds) & recent_rounds:
            raise ValueError("value_history overlaps the recent four rounds")
    RUNTIME.mkdir(parents=True, exist_ok=True)
    copies = {}
    for head, path in ledger_bins().items():
        target = RUNTIME / head / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        copies[head] = record(target)
    snapshot_copy = RUNTIME / "replay_snapshot.json"
    snapshot_copy.write_bytes(SNAPSHOT.read_bytes())
    source = source_identity(ROOT, RUNTIME / "source")
    payload = load_alphazero_checkpoint(START, device="cpu")
    repro_payload = load_alphazero_checkpoint(REPRO_START, device="cpu")
    start_model = model_state_sha256(payload.build_model().state_dict())
    start_optimizer = payload.optimizer_state_dict

    def optimizer_hash(state):
        digest = hashlib.sha256()
        for key in sorted(state["state"], key=str):
            for name in sorted(state["state"][key], key=str):
                _tensor_digest([(f"{key}:{name}", state["state"][key][name])], digest)
        groups = [{k: v for k, v in group.items() if k not in ("params", "lr")}
                  for group in state["param_groups"]]
        _tensor_digest([("param_groups", groups)], digest)
        return digest.hexdigest()

    spec = {"protocol": "formal-lr-step-probe-v1", "status": "prepared", "client_date": "2026-10-07",
            "start": {**record(START), "model_state_sha256": start_model,
                      "optimizer_state_sha256": optimizer_hash(start_optimizer),
                      "total_train_step": payload.total_train_step, "round_index": payload.round_index},
            "repro_start": {**record(REPRO_START),
                            "model_state_sha256": model_state_sha256(repro_payload.build_model().state_dict()),
                            "optimizer_state_sha256": optimizer_hash(repro_payload.optimizer_state_dict),
                            "total_train_step": repro_payload.total_train_step},
            "repro_reference": record(REPRO_REFERENCE),
            "source_run_config": record(SOURCE_RUN / "config.json"),
            "snapshot": {**record(SNAPSHOT), "digest": snapshot["digest"],
                         "recent_round_indices": sorted(recent_rounds),
                         "split_training": split},
            "ledger": {"heads": {head: {**record(path), "rows": int(len(ledger[head])),
                                        "per_step": HEADS[head]}
                                 for head, path in ledger_bins().items()},
                       "steps": STEPS, "digest": hashlib.sha256(
                           b"".join(ledger[head].tobytes() for head in sorted(HEADS))).hexdigest(),
                       "composition_by_round": composition,
                       "copies": copies, "snapshot_copy": record(snapshot_copy)},
            "rng_identity": rng_identity(payload),
            "repro_rng_identity": rng_identity(repro_payload),
            "arms": {arm: {"learning_rate": lr, "steps": STEPS, "nodes": list(NODES),
                           "start": "start"} for arm, lr in ARMS.items()},
            "repro_arm": {"name": REPRO_ARM, "learning_rate": 1e-4, "steps": STEPS},
            "panels": {"pairs": PANEL_PAIRS, "seed_sets": SEED_SETS,
                       "search": "S1024/N1 native, both sides old determinized search",
                       "opponent": "frozen 16500 start"},
            "source": source,
            "commands": {}}
    prefix = ("conda run --no-capture-output -n splendor env PYTHONPATH=src "
              "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u ")
    spec["commands"] = {name: prefix + str(Path(__file__).relative_to(ROOT)) + f" {name}"
                        for name in ("prepare", "train-all", "repro", "diagnostics", "analyze",
                                     "verify")}
    write_json(ART / "specification.json", spec)
    write_json(ART / "ledger_summary.json",
               {"protocol": "formal-lr-step-probe-ledger-v1", "heads": composition,
                "recent_round_indices": sorted(recent_rounds),
                "value_history_rounds": sorted(set(
                    int(r) for r in composition["value_history"]) - recent_rounds),
                "digest": spec["ledger"]["digest"]})
    emit(event="prepared", ledger=spec["ledger"]["digest"], arms=list(ARMS),
         recent_rounds=sorted(recent_rounds))


def smoke():
    """CPU smoke: ledger replay, split training, raw diagnostics on a few games."""
    assert sha256(START) == START_SHA
    learner, payload = restore(START, 1e-4, device="cpu", steps=STEPS)
    index = make_index({"games": read_json(SNAPSHOT)["games"]}, 256 * 1024 ** 2)
    games = {(game.round_index, game.game_id): game for game in index.games}
    per_step_refs = ledger_references(games)
    for step in (1, 2):
        policy_refs = per_step_refs["policy_recent"][step]
        value_refs = per_step_refs["value_recent"][step] + per_step_refs["value_history"][step]
        policy_batch = collate_selfplay_samples(resolve_samples(index, games, policy_refs))
        value_batch = collate_selfplay_samples(resolve_samples(index, games, value_refs))
        row = learner.train_split_step(policy_batch, value_batch, references=policy_refs + value_refs,
                                       policy_max_grad_norm=1.0, value_max_grad_norm=1.0)
        emit(event="smoke_step", step=step, loss=round(row["total_loss"], 6),
             policy_kl=round(row["policy_kl"], 6), value_zero_mse=round(row["value_zero_mse"], 6),
             policy_grad_clipped=row["policy_grad_clipped"], value_grad_clipped=row["value_grad_clipped"])
    holdout = [g for g in index.games if g.round_index == 78 and g.split == "validation"]
    raw = raw_outputs(learner, index, sorted(holdout, key=lambda g: g.key)[:2])
    emit(event="smoke_raw", positions=int(len(raw["width"])), width_max=int(raw["width"].max()),
         value_mean=round(float(raw["value"].mean()), 5))
    emit(event="smoke_done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "smoke", "train", "train-all", "repro",
                                           "prepare-extension", "train-extension", "diagnostics",
                                           "evaluate", "analyze", "verify"))
    parser.add_argument("--arm", choices=tuple(ARMS))
    parser.add_argument("--case")
    parser.add_argument("--seeds", choices=tuple(SEED_SETS), default="dev")
    args = parser.parse_args()
    settings()
    if args.action == "prepare":
        prepare()
    elif args.action == "smoke":
        smoke()
    elif args.action == "train":
        assert args.arm
        train_arm(args.arm, nodes=NODES)
    elif args.action == "train-all":
        train_all()
    elif args.action == "repro":
        repro_round019()
    elif args.action == "prepare-extension":
        prepare_extension()
    elif args.action == "train-extension":
        train_extension()
    elif args.action == "diagnostics":
        diagnostics()
    elif args.action == "evaluate":
        assert args.case
        evaluate(args.case, args.seeds)
    elif args.action == "analyze":
        analysis()
    elif args.action == "verify":
        verify()
