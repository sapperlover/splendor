"""M-independent-sample averaged-label probe on the frozen native 200-game teacher data.

Design:
- source: the 200 native games collected at the 16500-step checkpoint by the
  chance-training probe (``runs/entity_attention_chance_training_probe_20261007/native``).
- relabel: replay each archived game trace; at every searchable decision run M
  independent full-budget S1024/N1 determinized searches (world seed and root
  noise seed both vary) and average the visit distributions.  The stored single
  sample stays the M=1 control.
- train: the M=1 control is the existing native arm.  The averaged arm replays
  the native arm's exact batch ledger (same states, same value batches, same
  optimizer start) and replaces only the policy targets.  Value targets and the
  value branch are untouched.
- fit: KL of students to the eight-repeat empirical mean on the frozen
  200-root bank.
- strength: averaged student vs native student (same search both sides), each
  vs the frozen 16500 start, and greedy panels.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import shutil
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from splendor.alphazero_checkpoint import load_alphazero_checkpoint, move_optimizer_state, save_alphazero_checkpoint
from splendor.alphazero_recording import append_json, environment_identity, read_json, sha256, source_identity, write_json
from splendor.alphazero_train import AlphazeroLearner, LearnerConfig
from splendor.compact_trace import decode_request
from splendor.contracts import DecisionRequest, TerminalResult
from splendor.engine import GameEngine
from splendor.evaluation import FixedEvaluationConfig, run_fixed_evaluation
from splendor.mcts_policy import MCTSConfig
from splendor.native_policy import NativeBatchRunner, NativePUCTSearchPolicy
from splendor.offline_train import make_index
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.replay import to_primitive
from splendor.rng import DeterministicRng
from splendor.search_inference import FrozenSearchInference, InferenceConfig
from splendor.selfplay_data import collate_selfplay_samples, load_game

ROOT = Path(__file__).resolve().parents[2]
ART = Path(__file__).resolve().parent
RUNTIME = ROOT / "runtime/entity_attention_target_averaging_20261007"
OUT = ROOT / "runs/entity_attention_target_averaging_20261007"
PROBE = ROOT / "artifacts/entity_attention_chance_training_probe_20261007"
NATIVE_DATASET = PROBE / "dataset_native.json"
NATIVE_TRAIN = PROBE / "train_native.json"
NATIVE_GAMES = ROOT / "runs/entity_attention_chance_training_probe_20261007/native"
NATIVE_LEDGER = NATIVE_GAMES / "batch_references.npy"
NATIVE_CKPT = NATIVE_GAMES / "checkpoints"
BANK = ROOT / "artifacts/entity_attention_target_noise_20261007/root_bank.json"
RAW = ROOT / "artifacts/entity_attention_target_noise_20261007/raw_targets.npz"
INITIAL = ROOT / "runs/entity_attention_hybrid_split_p4_v15_from_round50_seed1/rounds/round-019/checkpoints/final.pt"
INITIAL_SHA = "dbfbe066144a485d15ec633aa87c85ac27864f142183725b86562e4ba44ed2c4"

REPEATS = 4
SIMULATIONS = 1024
WORLD_BASE = 16940000
NOISE_BASE = 16950000
DEV_SEEDS = {"game": 16830000, "policy": 16840000, "bootstrap": 16860000}
HEAD_SEEDS = {"game": 16960000, "policy": 16970000, "bootstrap": 16980000}
SEARCH = {"simulations": SIMULATIONS, "determinizations": 1, "c_puct": 1.5,
          "fold_forced_phases": True, "max_simulation_transitions": 512,
          "root_noise_fraction": 0.25, "dirichlet_alpha": 0.3}


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


def inference_config():
    return InferenceConfig(mode="eager", actor_projection=False, candidate_buckets=(16, 32, 48, 64))


def checkpoint_path(arm, step):
    return OUT / arm / f"checkpoints/step-{step:06d}.pt"


def load_labels(tag=""):
    path = RUNTIME / f"labels{tag}.npz"
    assert path.exists(), "run relabel first"
    z = np.load(path, allow_pickle=False)
    game_id = np.asarray(z["game_id"])
    decision_index = np.asarray(z["decision_index"])
    split = np.asarray(z["split"])
    candidate_count = np.asarray(z["candidate_count"])
    stored_target = np.asarray(z["stored_target"], dtype=np.float64)
    fresh_targets = np.asarray(z["fresh_targets"], dtype=np.float64)
    rows = {}
    for i in range(len(game_id)):
        key = (int(game_id[i]), int(decision_index[i]))
        width = int(candidate_count[i])
        rows[key] = {"split": str(split[i]), "width": width,
                     "stored": np.array(stored_target[i, :width], copy=True),
                     "fresh": np.array(fresh_targets[i, :, :width], copy=True)}
    return rows


def js(p, q):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def _js_batch(a, b, mask):
    a = np.where(mask, a, 0.0)
    b = np.where(mask, b, 0.0)
    m = 0.5 * (a + b)
    def kl(p, q):
        valid = (p > 0) & (q > 0)
        safe_p = np.where(valid, p, 1.0)
        safe_q = np.where(valid, q, 1.0)
        return np.where(valid, p * np.log(safe_p / safe_q), 0.0).sum(axis=1)
    return 0.5 * kl(a, m) + 0.5 * kl(b, m)


def label_summary_arrays(z):
    stored = np.asarray(z["stored_target"], dtype=np.float64)
    fresh = np.asarray(z["fresh_targets"], dtype=np.float64)
    counts = np.asarray(z["candidate_count"], dtype=np.int64)
    split = np.asarray(z["split"])
    n, repeats, width = fresh.shape
    mask = np.arange(width)[None, :] < counts[:, None]
    pair_js = []
    top1_pairs = []
    argmax = np.argmax(np.where(mask[:, None, :], fresh, -np.inf), axis=2)
    for i in range(repeats):
        for j in range(i + 1, repeats):
            pair_js.append(_js_batch(fresh[:, i], fresh[:, j], mask))
            top1_pairs.append(argmax[:, i] == argmax[:, j])
    stored_js = np.mean([_js_batch(stored, fresh[:, i], mask) for i in range(repeats)], axis=0)
    stored_top1 = np.mean([np.argmax(np.where(mask, stored, -np.inf), axis=1) == argmax[:, i]
                           for i in range(repeats)], axis=0)
    half = repeats // 2
    half_js = _js_batch(fresh[:, :half].mean(axis=1), fresh[:, half:].mean(axis=1), mask)
    half_top1 = (np.argmax(np.where(mask, fresh[:, :half].mean(axis=1), -np.inf), axis=1)
                 == np.argmax(np.where(mask, fresh[:, half:].mean(axis=1), -np.inf), axis=1))
    base = {"decisions": int(n), "repeats": int(repeats),
            "splits": {str(k): int(v) for k, v in zip(*np.unique(split, return_counts=True))},
            "mean_pairwise_fresh_js": float(np.mean(pair_js)),
            "mean_stored_vs_fresh_js": float(np.mean(stored_js)),
            "mean_two_half_average_js": float(np.mean(half_js)),
            "mean_fresh_top1_agreement": float(np.mean(top1_pairs)),
            "mean_stored_top1_agreement": float(np.mean(stored_top1)),
            "mean_two_half_average_top1_agreement": float(np.mean(half_top1))}
    for label in np.unique(split):
        pick = split == label
        base[f"split_{label}"] = {"decisions": int(pick.sum()),
                                  "mean_pairwise_fresh_js": float(np.mean([x[pick].mean() for x in pair_js])),
                                  "mean_stored_vs_fresh_js": float(np.mean(stored_js[pick])),
                                  "mean_two_half_average_js": float(np.mean(half_js[pick])),
                                  "mean_two_half_average_top1_agreement": float(np.mean(half_top1[pick]))}
    return base


def summarize():
    z = np.load(RUNTIME / "labels.npz", allow_pickle=False)
    summary = label_summary_arrays(z)
    summary.update({"simulations": SIMULATIONS, "world_base": WORLD_BASE, "noise_base": NOISE_BASE,
                    "labels": record(RUNTIME / "labels.npz")})
    write_json(ART / "labels_summary.json", summary)
    emit(event="labels_summary", **{k: summary[k] for k in ("decisions", "mean_pairwise_fresh_js",
        "mean_stored_vs_fresh_js", "mean_two_half_average_js", "mean_two_half_average_top1_agreement")})
    return summary


def replay_decisions(item):
    path = Path(item["path"])
    assert sha256(path) == item["sha256"], path
    arrays, meta = load_game(path)
    assert meta["game_id"] == item["game_id"] and meta["split"] == item["split"]
    assert meta["checkpoint_sha256"] == INITIAL_SHA
    engine = GameEngine.new(meta["game_seed"], first_player=meta["first_player"], mode="checked")
    offsets = arrays["candidate_offsets"]
    decisions = []
    for index, action in enumerate(meta["trace"]["actions"]):
        request = engine.decision()
        assert isinstance(request, DecisionRequest)
        assert action["decision_index"] == index
        chosen = int(action["candidate_index"])
        assert to_primitive(request.legal_actions[chosen]) == action["action"]
        width = len(request.legal_actions)
        assert width == int(offsets[index + 1]) - int(offsets[index])
        if bool(arrays["searchable"][index]) and width > 1:
            start, end = int(offsets[index]), int(offsets[index + 1])
            decisions.append({"game_id": int(meta["game_id"]), "decision_index": index,
                              "split": str(meta["split"]), "width": width,
                              "stored": np.asarray(arrays["policy_target"][start:end], dtype=np.float64),
                              "request": request})
        engine.apply(request.legal_actions[chosen])
    terminal = engine.decision()
    assert isinstance(terminal, TerminalResult)
    assert to_primitive(terminal) == meta["trace"]["terminal_result"]
    return decisions


def prepare():
    assert not (ART / "specification.json").exists(), "specification already frozen"
    dataset = read_json(NATIVE_DATASET)
    train = read_json(NATIVE_TRAIN)
    assert sha256(INITIAL) == INITIAL_SHA
    assert train["dataset"]["sha256"] == sha256(NATIVE_DATASET)
    assert train["batch_ledger"]["sha256"] == sha256(NATIVE_LEDGER)
    games = []
    for item in dataset["games"]:
        games.append({**record(item["path"]), "game_id": int(item["game_id"]), "split": str(item["split"]),
                      "decision_count": int(item["decision_count"]), "policy_samples": int(item["policy_samples"])})
    panels = panel_specs()
    for name, raw in panels.items():
        write_json(ART / f"configs/{name}.json", raw)
    source = source_identity(ROOT, RUNTIME / "source")
    extras = [ROOT / "src/splendor/native_search.cpp", ROOT / "setup.py",
              *[p for p in (ROOT / "data").glob("**/*") if p.is_file()]]
    source["additional_files"] = {}
    for path in extras:
        target = RUNTIME / "source" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        source["additional_files"][str(path.relative_to(ROOT))] = sha256(path)
    for path in ROOT.glob("src/splendor/_native_search*.so"):
        shutil.copyfile(path, RUNTIME / "source" / path.relative_to(ROOT))
    spec = {"protocol": "target-averaging-probe-v1", "status": "prepared", "client_date": "2026-10-07",
            "initial": record(INITIAL), "dataset": record(NATIVE_DATASET), "train_record": record(NATIVE_TRAIN),
            "ledger": record(NATIVE_LEDGER), "bank": record(BANK), "raw_targets": record(RAW),
            "games": games, "repeats": REPEATS, "simulations": SIMULATIONS,
            "mcts_config": SEARCH, "world_base": WORLD_BASE, "noise_base": NOISE_BASE,
            "dev_seeds": DEV_SEEDS, "head_seeds": HEAD_SEEDS, "source": source,
            "arms": {"control": "stored single-sample native labels (existing native arm)",
                     "avg4": "mean of four independent full-budget native searches per decision"},
            "panels": [{"id": name, "pairs": raw["pairs"], "policy_a": raw["policy_a"]["name"],
                        "policy_b": raw["policy_b"]["name"]} for name, raw in panels.items()],
            "scope": ("Replays the frozen 200-game native collection at the 16500 checkpoint. "
                      "The averaged arm replaces only policy visit-distribution targets; value labels, "
                      "value batches and optimizer settings are identical to the existing native arm. "
                      "Single start/collection seed; pilot strength panels."),
            "commands": {}}
    prefix = "conda run --no-capture-output -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u "
    spec["commands"] = {"prepare": prefix + str(Path(__file__).relative_to(ROOT)) + " prepare",
                        "relabel": prefix + str(Path(__file__).relative_to(ROOT)) + " relabel",
                        "train": prefix + str(Path(__file__).relative_to(ROOT)) + " train --arm avg4",
                        "fit": prefix + str(Path(__file__).relative_to(ROOT)) + " fit",
                        "analyze": prefix + str(Path(__file__).relative_to(ROOT)) + " analyze",
                        "verify": prefix + str(Path(__file__).relative_to(ROOT)) + " verify"}
    write_json(ART / "specification.json", spec)
    write_json(RUNTIME / "games.json", {"games": games})
    emit(event="prepared", games=len(games), searchable=sum(g["policy_samples"] for g in games), panels=len(panels))


def panel_specs():
    initial = str(INITIAL)
    native100 = str(NATIVE_CKPT / "step-000100.pt")
    avg100 = str(checkpoint_path("avg4", 100))
    avg300 = str(checkpoint_path("avg4", 300))
    def search_policy(name, path, seed):
        return {"kind": "mcts_checkpoint", "name": name, "checkpoint": path, "search_backend": "native",
                "search_config": dict(SEARCH, seed=seed, root_noise_fraction=0.0),
                "inference": {"mode": "eager", "actor_projection": False,
                              "candidate_buckets": [16, 32, 48, 64]}}
    def greedy_policy(name, path):
        return {"kind": "greedy_checkpoint", "name": name, "checkpoint": path}
    execution = {"backend": "native_batched", "max_batch_size": 200,
                 "candidate_buckets": [16, 32, 48, 64], "actor_projection": False,
                 "inference_mode": "eager", "max_batch_wait_ms": 0}
    raw = {
        "avg_100_native": {"policy_a": search_policy("avg4_100", avg100, 16850000), "policy_b": search_policy("frozen_16500_native", initial, 16850000),
                           "pairs": 50, "seed_start": DEV_SEEDS["game"], "policy_seed_base": DEV_SEEDS["policy"],
                           "bootstrap_seed": DEV_SEEDS["bootstrap"], "bootstrap_resamples": 10000, "device": "cuda",
                           "workers": 1, "max_completed_turns": 400, "adjudicate_deterministic_policy_cycles": True,
                           "execution": {**execution, "active_games": 100}, "progress_every": 5, "summary_every": 5,
                           "metadata": {"protocol": "target-averaging-probe-v1", "mode": "native", "step": 100}},
        "avg_300_native": {"policy_a": search_policy("avg4_300", avg300, 16850000), "policy_b": search_policy("frozen_16500_native", initial, 16850000),
                           "pairs": 50, "seed_start": DEV_SEEDS["game"], "policy_seed_base": DEV_SEEDS["policy"],
                           "bootstrap_seed": DEV_SEEDS["bootstrap"], "bootstrap_resamples": 10000, "device": "cuda",
                           "workers": 1, "max_completed_turns": 400, "adjudicate_deterministic_policy_cycles": True,
                           "execution": {**execution, "active_games": 100}, "progress_every": 5, "summary_every": 5,
                           "metadata": {"protocol": "target-averaging-probe-v1", "mode": "native", "step": 300}},
        "avg_100_greedy": {"policy_a": greedy_policy("avg4_100_greedy", avg100), "policy_b": greedy_policy("frozen_16500_greedy", initial),
                           "pairs": 50, "seed_start": DEV_SEEDS["game"], "policy_seed_base": DEV_SEEDS["policy"],
                           "bootstrap_seed": DEV_SEEDS["bootstrap"], "bootstrap_resamples": 10000, "device": "cuda",
                           "workers": 1, "max_completed_turns": 400, "adjudicate_deterministic_policy_cycles": True,
                           "progress_every": 5, "summary_every": 5,
                           "metadata": {"protocol": "target-averaging-probe-v1", "mode": "greedy", "step": 100}},
        "avg_100_vs_native100": {"policy_a": search_policy("avg4_100", avg100, 16990000), "policy_b": search_policy("native_100", native100, 16990000),
                                 "pairs": 100, "seed_start": HEAD_SEEDS["game"], "policy_seed_base": HEAD_SEEDS["policy"],
                                 "bootstrap_seed": HEAD_SEEDS["bootstrap"], "bootstrap_resamples": 10000, "device": "cuda",
                                 "workers": 1, "max_completed_turns": 400, "adjudicate_deterministic_policy_cycles": True,
                                 "execution": {**execution, "active_games": 200}, "progress_every": 5, "summary_every": 5,
                                 "metadata": {"protocol": "target-averaging-probe-v1", "mode": "native", "step": 100,
                                              "matchup": "avg4_vs_native100"}},
    }
    return raw


def relabel(limit_games=None, repeats=REPEATS, simulations=SIMULATIONS, tag=""):
    spec = read_json(ART / "specification.json")
    assert sha256(INITIAL) == INITIAL_SHA
    games = list(spec["games"])
    if limit_games:
        games = games[:limit_games]
    parts = RUNTIME / f"label_parts{tag}"
    parts.mkdir(parents=True, exist_ok=True)
    assert torch.cuda.is_available(), "relabel requires CUDA"
    model = load_frozen_actor_critic_checkpoint(INITIAL, device="cuda")
    evaluator = FrozenSearchInference(model, device="cuda", config=inference_config())
    runner = NativeBatchRunner(evaluator, NativePUCTSearchPolicy(model, evaluator=evaluator).catalog,
                               capacity=200, buckets=(16, 32, 48, 64), slots=1)
    write_json(RUNTIME / f"relabel_environment{tag}.json",
               {**environment_identity(), "tf32_matmul": False, "tf32_cudnn": False,
                "repeats": repeats, "simulations": simulations, "world_base": WORLD_BASE, "noise_base": NOISE_BASE})
    started = time.perf_counter()
    chunk = 8
    for start in range(0, len(games), chunk):
        block = games[start:start + chunk]
        path = parts / f"chunk-{start:04d}.npz"
        if path.exists():
            continue
        decisions = []
        for item in block:
            decisions.extend(replay_decisions(item))
        width = max(d["width"] for d in decisions)
        game_id = np.asarray([d["game_id"] for d in decisions], dtype=np.int64)
        decision_index = np.asarray([d["decision_index"] for d in decisions], dtype=np.int64)
        split = np.asarray([d["split"] for d in decisions])
        candidate_count = np.asarray([d["width"] for d in decisions], dtype=np.int32)
        stored = np.zeros((len(decisions), width), dtype=np.float64)
        fresh = np.zeros((len(decisions), repeats, width), dtype=np.float64)
        for i, d in enumerate(decisions):
            stored[i, :d["width"]] = d["stored"]
        for rep in range(repeats):
            policy = NativePUCTSearchPolicy(model, evaluator=evaluator,
                                            config=MCTSConfig(**dict(SEARCH, simulations=simulations, seed=WORLD_BASE + rep)))
            jobs = [policy.start_job(d["request"], add_root_noise=True, temperature=0.0,
                                     sampling_seed=NOISE_BASE + rep) for d in decisions]
            pending = jobs
            tick = time.perf_counter()
            while any(not job.done for job in pending):
                assert runner.run([job for job in pending if not job.done]) > 0
            for i, job in enumerate(jobs):
                result = job.result
                assert result.searchable and int(result.simulations) == simulations
                assert abs(float(np.sum(result.root_visit_counts)) - simulations) < 1e-9
                fresh[i, rep, :decisions[i]["width"]] = np.asarray(result.policy_target, dtype=np.float64)
            emit(event="relabel_repeat", chunk=start, repeat=rep, decisions=len(decisions),
                 seconds=round(time.perf_counter() - tick, 2), elapsed=round(time.perf_counter() - started, 1))
        np.savez_compressed(path, game_id=game_id, decision_index=decision_index, split=split,
                            candidate_count=candidate_count, stored_target=stored, fresh_targets=fresh,
                            world_seeds=np.asarray([WORLD_BASE + r for r in range(repeats)]),
                            noise_seeds=np.asarray([NOISE_BASE + r for r in range(repeats)]))
        emit(event="relabel_chunk_done", chunk=start, games=len(block), decisions=len(decisions),
             elapsed=round(time.perf_counter() - started, 1))
    rows_game, rows_decision, rows_split, rows_count, rows_stored, rows_fresh = [], [], [], [], [], []
    width = 0
    part_paths = sorted(parts.glob("chunk-*.npz"))
    assert len(part_paths) == (len(games) + chunk - 1) // chunk
    for path in part_paths:
        z = np.load(path, allow_pickle=False)
        width = max(width, z["stored_target"].shape[1])
    for path in part_paths:
        z = np.load(path, allow_pickle=False)
        n = len(z["game_id"])
        rows_game.append(z["game_id"]); rows_decision.append(z["decision_index"])
        rows_split.append(z["split"]); rows_count.append(z["candidate_count"])
        stored = np.zeros((n, width), dtype=np.float64); stored[:, :z["stored_target"].shape[1]] = z["stored_target"]
        fresh = np.zeros((n, repeats, width), dtype=np.float64); fresh[:, :, :z["fresh_targets"].shape[2]] = z["fresh_targets"]
        rows_stored.append(stored); rows_fresh.append(fresh)
    labels = {"game_id": np.concatenate(rows_game), "decision_index": np.concatenate(rows_decision),
              "split": np.concatenate(rows_split), "candidate_count": np.concatenate(rows_count),
              "stored_target": np.concatenate(rows_stored), "fresh_targets": np.concatenate(rows_fresh)}
    np.savez_compressed(RUNTIME / f"labels{tag}.npz", **labels)
    summary = label_summary_arrays(labels)
    summary.update({"simulations": simulations, "games": len(games),
                    "labels": record(RUNTIME / f"labels{tag}.npz"), "elapsed_seconds": time.perf_counter() - started})
    write_json(ART / (f"labels_summary{tag}.json" if tag else "labels_summary.json"), summary)
    emit(event="relabel_done", **{k: summary[k] for k in ("decisions", "mean_pairwise_fresh_js",
        "mean_stored_vs_fresh_js", "mean_fresh_top1_agreement", "mean_stored_top1_agreement")})


def restore():
    payload = load_alphazero_checkpoint(INITIAL, device="cuda")
    learner = AlphazeroLearner(payload.build_model(device="cuda"), LearnerConfig(**payload.learner_config),
                               sampling_rng=DeterministicRng.restore(payload.sampling_rng_state))
    learner.optimizer.load_state_dict(payload.optimizer_state_dict)
    move_optimizer_state(learner.optimizer, learner.device)
    learner.round_index = payload.round_index
    learner.total_train_step = payload.total_train_step
    learner.round_train_step = 0
    learner.round_steps_target = 500
    torch.set_rng_state(payload.torch_cpu_rng_state.cpu())
    torch.cuda.set_rng_state_all([s.cpu() for s in payload.torch_cuda_rng_states])
    return learner, payload


def replace_target(sample, target):
    arrays = dict(sample.arrays)
    updated = np.array(arrays["policy_target"], copy=True)
    updated[sample.candidate_slice] = target.astype(np.float32)
    arrays["policy_target"] = updated
    return dataclasses.replace(sample, arrays=arrays)


def train_arm(arm, steps=500, smoke=False, label_tag=""):
    spec = read_json(ART / "specification.json")
    assert sha256(INITIAL) == INITIAL_SHA
    labels = None if arm == "control" else load_labels(label_tag)
    dataset = read_json(NATIVE_DATASET)
    index = make_index(dataset, 1024 ** 3)
    pool = index.position_pool(split="train", round_index=0)
    if not smoke:
        assert len(pool) == 14477, len(pool)
    learner, payload = restore()
    native = read_json(NATIVE_TRAIN)
    assert asdict(learner.sampling_rng.snapshot()) == native["initial_sampling_rng"]
    torch_sha = hashlib.sha256(torch.get_rng_state().cpu().numpy().tobytes()).hexdigest()
    assert torch_sha == native["initial_torch_cpu_rng_sha256"]
    ledger = np.load(NATIVE_LEDGER)
    output = OUT / arm
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "training_metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()
    started = time.perf_counter()
    saved = []
    replaced = kept = 0
    for step in range(steps):
        draws = [[pool.sample(learner.sampling_rng) for _ in range(256)] for _ in range(2)]
        if not smoke:
            for head in range(2):
                got = [(int(x[1].game_id), int(x[1].decision_index)) for x in draws[head]]
                expected = [tuple(int(v) for v in row) for row in ledger[step, head]]
                assert got == expected, (step, head)
        policy_samples = [sample for sample, _ in draws[0]]
        if arm != "control":
            replaced_samples = []
            for sample, ref in draws[0]:
                key = (int(ref.game_id), int(ref.decision_index))
                row = labels.get(key)
                if row is None:
                    assert smoke or sample.candidate_slice.stop - sample.candidate_slice.start == 1
                    replaced_samples.append(sample)
                    kept += 1
                    continue
                target = row["fresh"].mean(axis=0)
                assert len(target) == sample.candidate_slice.stop - sample.candidate_slice.start
                replaced_samples.append(replace_target(sample, target))
                replaced += 1
            policy_samples = replaced_samples
        policy_batch = collate_selfplay_samples(policy_samples)
        value_batch = collate_selfplay_samples([sample for sample, _ in draws[1]])
        references = [ref for batch in draws for _, ref in batch]
        row = learner.train_split_step(policy_batch, value_batch, references=references,
                                       policy_max_grad_norm=1.0, value_max_grad_norm=1.0)
        append_json(metrics_path, row)
        if step + 1 in (100, 300, 500):
            path = checkpoint_path(arm, step + 1)
            save_alphazero_checkpoint(path, model=learner.model, optimizer=learner.optimizer,
                learner_config=asdict(learner.config), round_index=learner.round_index,
                round_train_step=step + 1, total_train_step=learner.total_train_step, round_steps_target=500,
                parent_checkpoint=record(INITIAL),
                replay_identity={"dataset_sha256": sha256(NATIVE_DATASET)} | ({} if arm == "control"
                                 else {"labels_sha256": sha256(RUNTIME / f"labels{label_tag}.npz")}),
                sampling_rng_state=learner.sampling_rng.snapshot(),
                metadata={"diagnostic": "target-averaging-probe-v1", "arm": arm,
                          "policy_targets": "stored_single_sample" if arm == "control" else "mean_of_4_searches",
                          "value_trunk_detached": True, "separate_gradient_clipping": True})
            saved.append(record(path))
        if (step + 1) % 100 == 0:
            emit(event="train_step", arm=arm, step=step + 1, elapsed=round(time.perf_counter() - started, 1))
    metrics = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    result = {"arm": arm, "steps": steps, "wall_seconds": time.perf_counter() - started,
              "metrics": record(metrics_path), "checkpoints": saved,
              "mean_policy_kl": float(np.mean([m["policy_kl"] for m in metrics])),
              "mean_value_mse": float(np.mean([m["value_loss"] for m in metrics])),
              "policy_clipped_steps": int(sum(m["policy_grad_clipped"] for m in metrics)),
              "value_clipped_steps": int(sum(m["value_grad_clipped"] for m in metrics)),
              "policy_targets_replaced": replaced, "single_candidate_kept": kept,
              "environment": environment_identity()}
    write_json(ART / f"train_{arm}.json", result)
    if arm == "control":
        compare = {}
        for step in (100, 300, 500):
            ours = load_alphazero_checkpoint(checkpoint_path("control", step), device="cpu").build_model().state_dict()
            theirs = load_alphazero_checkpoint(NATIVE_CKPT / f"step-{step:06d}.pt", device="cpu").build_model().state_dict()
            diffs = {key: float(torch.max(torch.abs(ours[key].float() - theirs[key].float()))) for key in ours}
            compare[str(step)] = {"max_abs_diff": max(diffs.values()),
                                  "mean_abs_diff": float(np.mean(list(diffs.values()))),
                                  "identical": bool(max(diffs.values()) == 0.0)}
        write_json(ART / "control_reproduction.json",
                   {"protocol": "target-averaging-probe-control-v1", "compare": compare,
                    "native_metrics": record(NATIVE_TRAIN)})
        emit(event="control_reproduction", **compare)
    emit(event="train_done", arm=arm, steps=steps, seconds=result["wall_seconds"])


def fit():
    bank = read_json(BANK)
    rows = bank["positions"]
    requests = [decode_request(row["request"]) for row in rows]
    raw = np.load(RAW)["policy_target"]
    mean8 = raw[3].mean(axis=0)
    single = raw[3][0]
    cases = {"initial": INITIAL,
             "native100": NATIVE_CKPT / "step-000100.pt",
             "native300": NATIVE_CKPT / "step-000300.pt",
             "native500": NATIVE_CKPT / "step-000500.pt",
             "avg100": checkpoint_path("avg4", 100),
             "avg300": checkpoint_path("avg4", 300),
             "avg500": checkpoint_path("avg4", 500)}
    output = {}
    for name, path in cases.items():
        if not Path(path).exists():
            continue
        model = load_frozen_actor_critic_checkpoint(path, device="cuda").eval()
        evaluator = FrozenSearchInference(model, device="cuda", config=inference_config())
        results = evaluator.policy_value(requests)
        per_root = []
        for i, row in enumerate(rows):
            logit, value = results[i]
            logit = np.asarray(logit, dtype=np.float64)
            logp = logit - np.logaddexp.reduce(logit)
            width = len(logit)
            pi = mean8[i][:width]
            one = single[i][:width]
            mask = pi > 0
            per_root.append({"id": int(row["id"]), "game_key": row["game_key"],
                             "kl_to_mean8": float(np.sum(pi[mask] * (np.log(pi[mask]) - logp[mask]))),
                             "kl_to_single": float(np.sum(one[one > 0] *
                                 (np.log(one[one > 0]) - logp[one > 0]))),
                             "top1_matches_mean8": bool(int(np.argmax(logit)) == int(np.argmax(pi))),
                             "value": float(np.asarray(value).reshape(-1)[0])})
        output[name] = {"checkpoint": record(path), "per_root": per_root,
                        "mean_kl_to_mean8": float(np.mean([r["kl_to_mean8"] for r in per_root])),
                        "mean_kl_to_single": float(np.mean([r["kl_to_single"] for r in per_root])),
                        "top1_match_rate": float(np.mean([r["top1_matches_mean8"] for r in per_root]))}
        emit(event="fit_done", name=name, kl=round(output[name]["mean_kl_to_mean8"], 5),
             top1=round(output[name]["top1_match_rate"], 4))
    write_json(ART / "fit.json", {"protocol": "target-averaging-probe-fit-v1",
                                  "bank": record(BANK), "raw_targets": record(RAW),
                                  "cases": output})


def evaluate(case):
    raw = read_json(ART / f"configs/{case}.json")
    config = FixedEvaluationConfig.from_dict(raw)
    output = ART / f"evaluations/{case}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    emit(event="evaluation_start", case=case, pairs=len(config.game_seeds))
    result = run_fixed_evaluation(config, output, resume=output.exists(), require_scored_pairs=True,
        progress_callback=lambda done, total: emit(event="evaluation_progress", case=case,
                                                   completed_pairs=done, pairs=total))
    assert result.summary["passed"]
    write_json(ART / f"evaluations/{case}_recording.json",
               {"config": record(ART / f"configs/{case}.json"), "output": record(output),
                "checkpoint_a": record(Path(config.policy_a.checkpoint)),
                "checkpoint_b": record(Path(config.policy_b.checkpoint)),
                "environment": environment_identity()})
    emit(event="evaluation_done", case=case, score=result.summary["win_rate_score"])


def bootstrap_difference(a, b, seed):
    def scores(result):
        payoff = {"a_win": 1.0, "a_loss": 0.0, "draw": 0.5}
        return np.asarray([np.mean([payoff[g["outcome_for_a"]] for g in pair["games"]]) for pair in result["pairs"]])
    assert [p["game_seed"] for p in a["pairs"]] == [p["game_seed"] for p in b["pairs"]]
    delta = scores(a) - scores(b)
    rng = np.random.default_rng(seed)
    values = delta[rng.integers(0, len(delta), size=(10000, len(delta)))].mean(axis=1)
    return {"difference": float(delta.mean()), "lower": float(np.quantile(values, 0.025)),
            "upper": float(np.quantile(values, 0.975)), "pairs": int(len(delta))}


def analyze():
    spec = read_json(ART / "specification.json")
    labels = read_json(ART / "labels_summary.json")
    fit_data = read_json(ART / "fit.json")
    control = read_json(ART / "control_reproduction.json")
    panels = {}
    for panel in spec["panels"]:
        path = ART / f"evaluations/{panel['id']}.json"
        if path.exists():
            result = read_json(path)
            assert result["passed"] and result["valid_games"] == 2 * panel["pairs"]
            panels[panel["id"]] = {k: result[k] for k in ("win_rate_score", "paired_bootstrap", "a_wins",
                "a_losses", "draws", "valid_games", "elapsed_seconds_this_invocation")}
    differences = {}
    for name, ref, seed in (("avg_100_native", "native_100_native", DEV_SEEDS["bootstrap"]),
                            ("avg_100_greedy", "native_100_greedy", DEV_SEEDS["bootstrap"]),
                            ("avg_300_native", "native_500_native", DEV_SEEDS["bootstrap"])):
        ours = ART / f"evaluations/{name}.json"
        theirs = PROBE / f"evaluations/{ref}.json"
        if ours.exists() and theirs.exists():
            differences[f"{name}_minus_{ref}"] = bootstrap_difference(read_json(ours), read_json(theirs), seed)
    fit_deltas = {}
    for step in (100, 300, 500):
        avg = fit_data["cases"].get(f"avg{step}")
        nat = fit_data["cases"].get(f"native{step}")
        if not avg or not nat:
            continue
        by_id = {r["id"]: r for r in nat["per_root"]}
        delta = np.asarray([r["kl_to_mean8"] - by_id[r["id"]]["kl_to_mean8"] for r in avg["per_root"]])
        rng = np.random.default_rng(HEAD_SEEDS["bootstrap"])
        values = delta[rng.integers(0, len(delta), size=(10000, len(delta)))].mean(axis=1)
        fit_deltas[str(step)] = {"mean_kl_delta_avg_minus_native": float(delta.mean()),
                                 "lower": float(np.quantile(values, 0.025)),
                                 "upper": float(np.quantile(values, 0.975)), "roots": int(len(delta))}
    output = {"protocol": "target-averaging-probe-analysis-v1", "labels": labels,
              "control_reproduction": control["compare"], "fit": {k: {kk: vv for kk, vv in v.items()
              if kk in ("mean_kl_to_mean8", "mean_kl_to_single", "top1_match_rate")}
              for k, v in fit_data["cases"].items()}, "fit_deltas": fit_deltas,
              "panels": panels, "panel_differences": differences,
              "limitations": ["Single start/collection seed; DEV and pilot panels.",
                               "Four-search averaging estimates the belief-mean target; it does not change value labels.",
                               "The direct averaged-vs-native panel uses a fresh 100-pair seed domain."]}
    write_json(ART / "analysis.json", output)
    emit(event="analysis_done", panels=list(panels), fit_deltas=fit_deltas)


def verify():
    spec = read_json(ART / "specification.json")
    checks = {}
    checks["initial"] = sha256(INITIAL) == INITIAL_SHA
    checks["dataset"] = sha256(NATIVE_DATASET) == spec["dataset"]["sha256"]
    checks["ledger"] = sha256(NATIVE_LEDGER) == spec["ledger"]["sha256"]
    labels = load_labels()
    searchable = set()
    for item in read_json(NATIVE_DATASET)["games"]:
        if item["split"] != "train":
            continue
        arrays, meta = load_game(Path(item["path"]))
        offsets = arrays["candidate_offsets"]
        for i in range(int(item["decision_count"])):
            if bool(arrays["searchable"][i]) and int(offsets[i + 1]) - int(offsets[i]) > 1:
                searchable.add((int(item["game_id"]), i))
    checks["train_searchable_covered"] = searchable.issubset(labels.keys())
    checks["label_sums"] = all(abs(row["fresh"].sum(axis=-1) - 1).max() < 1e-6 for row in labels.values())
    checks["control_reproduction"] = all(v["max_abs_diff"] == 0.0 for v in read_json(ART / "control_reproduction.json")["compare"].values())
    avg_train = read_json(ART / "train_avg4.json")
    checks["avg4_targets_replaced"] = int(avg_train["policy_targets_replaced"]) > 0
    for panel in spec["panels"]:
        path = ART / f"evaluations/{panel['id']}.json"
        if path.exists():
            result = read_json(path)
            checks[f"panel_{panel['id']}"] = bool(result["passed"] and result["valid_games"] == 2 * panel["pairs"]
                                                  and not result["exceptions"] and not result["illegal_actions"]
                                                  and not result["invariant_failures"] and not result["engineering_truncations"])
    output = {"protocol": "target-averaging-probe-verify-v1", "passed": all(checks.values()), "checks": checks,
              "labels": record(RUNTIME / "labels.npz")}
    write_json(ART / "verification.json", output)
    emit(event="verify_done", passed=output["passed"], checks=checks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "relabel", "summarize", "train", "fit", "evaluate", "analyze", "verify"))
    parser.add_argument("--arm", choices=("control", "avg4"))
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--case")
    parser.add_argument("--limit-games", type=int)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--simulations", type=int, default=SIMULATIONS)
    parser.add_argument("--tag", default="")
    parser.add_argument("--label-tag", default="")
    args = parser.parse_args()
    settings()
    if args.action == "prepare":
        prepare()
    elif args.action == "relabel":
        relabel(limit_games=args.limit_games, repeats=args.repeats, simulations=args.simulations, tag=args.tag)
    elif args.action == "summarize":
        summarize()
    elif args.action == "train":
        assert args.arm
        train_arm(args.arm, steps=args.steps, smoke=bool(args.limit_games), label_tag=args.label_tag)
    elif args.action == "fit":
        fit()
    elif args.action == "evaluate":
        assert args.case
        evaluate(args.case)
    elif args.action == "analyze":
        analyze()
    elif args.action == "verify":
        verify()
