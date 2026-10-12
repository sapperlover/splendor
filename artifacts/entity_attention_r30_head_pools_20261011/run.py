"""Round29 endpoint -> frozen round30 data: B/C/D joint head-pool experiments."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from splendor.alphazero_checkpoint import load_alphazero_checkpoint
from splendor.alphazero_recording import ReferenceWriter, environment_identity, read_json, sha256, source_identity, write_json
from splendor.alphazero_replay import PositionPool
from splendor.alphazero_train import AlphazeroLearner, DrawnBatch, LearnerConfig, run_training_round
from splendor.compact_game_cache import CompactGameCache
from splendor.evaluation import FixedEvaluationConfig, run_fixed_evaluation
from splendor.offline_train import _digest, make_index
from splendor.pool_fitting import RoundFitting
from splendor.rng import DeterministicRng

ART = Path(__file__).resolve().parent
ROOT = ART.parents[1]
RUN, RUNTIME = ROOT / "runs" / ART.name, ROOT / "runtime" / ART.name
ROUNDS = ROOT / "runs/entity_attention_r9_online_lr2e5_1000_forever_20261011/rounds"
ORIGINAL = ROUNDS / "round-020"
INITIAL, PARENT = ORIGINAL / "checkpoints/initial.pt", ROUNDS / "round-019/checkpoints/final.pt"
POINTS = (0, 100, 250, 500, 1000)
BRANCHES = {"B": (3000, 6000), "C": (6000, 9000), "D": (3000, 9000)}
PLAN = ART / "plan.json"
CACHE_BYTES = 8 * 1024**3


def identity(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}


def freeze(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Frozen experiment changed: {path}")
    else:
        write_json(path, value)


def fitting(index, stage):
    return RoundFitting({"steps": list(POINTS), "seed": 33400000,
                         "train_positions_per_generation": 2048}, RUN, stage, None, index, None,
                        inherited_references=read_json(ORIGINAL / "fitting/train_references.json")["references"])


def prepare():
    if sha256(PARENT) != "8918f75bc1dd098553fe4e1dbb9b4b6898620504b202ebeccaca869e4c918227":
        raise ValueError("round29 endpoint changed")
    payload, parent = load_alphazero_checkpoint(INITIAL), load_alphazero_checkpoint(PARENT)
    config = LearnerConfig(**payload.learner_config)
    if (payload.round_index, payload.round_train_step, payload.total_train_step, payload.round_steps_target) != (30, 0, 58898, 1000):
        raise ValueError("wrong round30 initial checkpoint")
    if payload.model_sha256 != parent.model_sha256:
        raise ValueError("round30 initial differs from round29 endpoint")
    if (config.learning_rate, config.batch_size, config.policy_loss_weight, config.value_loss_weight,
            config.max_grad_norm) != (2e-5, 256, 1., 1., 1.):
        raise ValueError("original learner settings changed")
    sources = [ROUNDS / "round-017/replay_snapshot.json", ORIGINAL / "replay_snapshot.json"]
    games = {}
    for path in sources:
        for game in read_json(path)["games"]:
            key = game["round_index"], game["game_id"]
            if key in games and games[key] != game:
                raise ValueError("overlapping snapshots disagree")
            games[key] = game
    records = [games[k] for k in sorted(games)]
    if len(records) != 9000 or sorted({g["round_index"] for g in records}) != list(range(21, 30)):
        raise ValueError("expected absolute rounds22-30, 9000 games")
    generations = []
    for generation in range(21, 30):
        selected = [g for g in records if g["round_index"] == generation]
        manifest = Path(selected[0]["path"]).parents[1] / "manifest.json"
        raw = read_json(manifest)
        search = raw["config"]["search_config"]
        if len(selected) != 1000 or (search["simulations"], search["determinizations"]) != (1024, 1):
            raise ValueError("generation source differs from S1024/N1, 1000 games")
        generations.append({"absolute_round": generation + 1, "manifest": identity(manifest),
            "teacher": raw["checkpoint_identity"], "splits": dict(Counter(g["split"] for g in selected)),
            "positions": {split: sum(g["decision_count"] for g in selected if g["split"] == split)
                          for split in ("train", "validation")}})
    dataset = {"games": records, "digest": _digest(records), "sources": [identity(p) for p in sources]}
    freeze(ART / "dataset.json", dataset)
    index = make_index(dataset, CACHE_BYTES)
    common = fitting(index, ART / "common")
    refs = read_json(ART / "common/fitting/train_references.json")
    arena = read_json(ORIGINAL / "evaluation/greedy_vs_start.json")
    if not arena["passed"] or arena["completed_pairs"] != 200:
        raise ValueError("original A greedy panel incomplete")
    plan = {"experiment": ART.name, "round29_endpoint": identity(PARENT), "initial": identity(INITIAL),
        "base_learner": asdict(config), "original_A_final": identity(ORIGINAL / "checkpoints/final.pt"),
        "dataset": identity(ART / "dataset.json"), "common_train_references": identity(ART / "common/fitting/train_references.json"),
        "generations": generations, "steps": list(POINTS), "branches": {},
        "greedy_template": arena["config"], "original_A_greedy": identity(ORIGINAL / "evaluation/greedy_vs_start.json"),
        "runner": identity(Path(__file__)), "analyzer": identity(ART / "analyze.py"),
        "learner_source": identity(ROOT / "src/splendor/alphazero_train.py"),
        "sampling": "Uniform Train positions. Both head RNGs fork the original per-step RNG state; advance the persisted stream exactly as original 6k A. Equal pools yield identical samples. B value and C policy match original A at every step; B/D policy and C/D value match each other. RNG state alone suffices for resume.",
        "training": "Separate 256-position batches per head, normal shared-encoder gradients, one loss1:1 sum, one global clip1 and one AdamW step. Original optimizer/RNG retained. No detached-value or independent-head clipping.",
        "evaluation": "Fixed 2048 Train positions per generation plus complete original whole-game validation22-30; common references, per-generation/stage. 200-pair greedy vs P4V4 on original A panel. DEV trend, single training seed; no search or new collection.",
    }
    for name, (p, v) in BRANCHES.items():
        replay = {"digest": _digest({"dataset": dataset["digest"], "policy_games": p, "value_games": v,
                                    "sampling": plan["sampling"], "training": plan["training"]}),
                  "dataset": str(ART / "dataset.json"), "policy_games": p, "value_games": v,
                  "reference_layout": "first256 policy, then256 value"}
        plan["branches"][name] = {"policy_games": p, "value_games": v, "replay_identity": replay}
    if len(refs["references"]) != 9 * 2048:
        raise ValueError("unexpected common Train diagnostic population")
    freeze(PLAN, plan)
    RUN.mkdir(parents=True, exist_ok=True)
    RUNTIME.mkdir(parents=True, exist_ok=True)
    return plan


def draw_batches(policy_pool, value_pool, canonical_pool, rng, batch_size):
    """Couple head draws to the original stream without adding resumable RNGs."""
    state = rng.snapshot()
    samples, refs = [], []
    counts = Counter()
    for head, pool in (("policy", policy_pool), ("value", value_pool)):
        head_rng = DeterministicRng.restore(state)
        drawn = [pool.sample(head_rng) for _ in range(batch_size)]
        samples.append(tuple(s for s, _ in drawn))
        refs.extend(r for _, r in drawn)
        counts.update(f"{head}_round_{r.round_index + 1:03d}" for _, r in drawn)
    for _ in range(batch_size):
        rng.randbelow(len(canonical_pool))
    return DrawnBatch(samples[0], tuple(refs), 2 * batch_size, 0, dict(counts), samples[1])


def committed_references(branch):
    batches = {}
    for path in sorted((RUN / branch / "usage").glob("*.bin")):
        raw = path.read_bytes()
        data = np.frombuffer(raw[:len(raw)//32*32], dtype="<i8").reshape(-1, 4)
        for step in np.unique(data[:, 0]):
            block = data[data[:, 0] == step]
            if len(block) == 512 and step <= 1000:
                if int(step) in batches and not np.array_equal(batches[int(step)], block):
                    raise ValueError("resumed sampling changed")
                batches[int(step)] = block
    if sorted(batches) != list(range(1, 1001)):
        raise ValueError("reference ledger incomplete")
    return batches


def verify(branch, plan):
    batches = committed_references(branch)
    original_rows = {r["round_train_step"]: r for r in map(json.loads, (ORIGINAL / "training_metrics.jsonl").read_text().splitlines()) if r["event"] == "train_step"}
    spec = plan["branches"][branch]
    hashes = {"policy": {}, "value": {}}
    coverage = {}
    for head, sl, count in (("policy", slice(0, 256), spec["policy_games"]), ("value", slice(256, 512), spec["value_games"])):
        all_refs = np.concatenate([batches[s][sl] for s in sorted(batches)])
        if not set(all_refs[:, 1]).issubset(set(range(30-count//1000, 30))):
            raise ValueError("draw outside its head pool")
        for step, block in batches.items():
            digest = hashlib.sha256(block[sl].tobytes()).hexdigest()
            hashes[head][str(step)] = digest
            if count == 6000 and _digest([digest]) != original_rows[step]["reference_sha256"]:
                raise ValueError("unchanged 6k head differs from original A samples")
        coverage[head] = {"draws": len(all_refs), "unique_positions": len(np.unique(all_refs[:, 1:], axis=0)),
            "draws_by_absolute_round": {str(int(g)+1): int((all_refs[:, 1] == g).sum()) for g in np.unique(all_refs[:, 1])}}
    rows = [r for r in map(json.loads, (RUN / branch / "training_metrics.jsonl").read_text().splitlines()) if r["event"] == "train_step"]
    if len(rows) != 1000 or any(r["training_mode"] != "joint_separate_batches" for r in rows):
        raise ValueError("training update count or mode differs")
    final = RUN / branch / "checkpoints/final.pt"
    write_json(ART / branch / "head_reference_hashes.json", hashes)
    write_json(ART / branch / "training_verification.json", {"checkpoint": identity(final), "steps": 1000,
        "coverage": coverage, "unchanged_6k_heads_match_original_A": True,
        "mean_grad_norm": sum(r["grad_norm"] for r in rows)/1000,
        "clip_fraction": sum(r["grad_clipped"] for r in rows)/1000})


def train(branch, plan):
    spec = plan["branches"][branch]
    config = LearnerConfig(**plan["base_learner"])
    checkpoints = RUN / branch / "checkpoints"
    saved = sorted(checkpoints.glob("step-*.pt"))
    start = saved[-1] if saved else checkpoints / "initial.pt"
    if start.exists():
        learner = AlphazeroLearner.resume(start, config, replay_identity=spec["replay_identity"], round_index=30, round_steps_target=1000)
    else:
        original = load_alphazero_checkpoint(INITIAL)
        learner = AlphazeroLearner.resume(INITIAL, config, replay_identity=original.replay_identity, round_index=30, round_steps_target=1000)
        learner.replay_identity = spec["replay_identity"]
        learner.save(start, metadata={"experiment": ART.name, "branch": branch, "source": plan["initial"]})
    index = make_index(read_json(ART / "dataset.json"), CACHE_BYTES)
    index.cache = CompactGameCache(CACHE_BYTES)
    common = fitting(index, ART / "common")
    if not (ART / "common/fitting/step-000000000.json").exists():
        zero = AlphazeroLearner.resume(INITIAL, config, replay_identity=load_alphazero_checkpoint(INITIAL).replay_identity, round_index=30, round_steps_target=1000)
        common(zero, INITIAL)
        del zero
    diagnostic = RoundFitting(common.config, RUN, ART / branch, None, index, None,
                              inherited_references=common.train_fixed["references"])
    initial = checkpoints / "initial.pt"
    freeze(ART / branch / "fitting/step-000000000.json", {
        **read_json(ART / "common/fitting/step-000000000.json"), "checkpoint": str(initial),
        "checkpoint_sha256": sha256(initial), "shared_zero_step": identity(ART / "common/fitting/step-000000000.json")})
    # The extra common evaluation has no sampling RNG effect; resume restores
    # the committed Torch/AdamW/sampling state as well.
    learner = AlphazeroLearner.resume(start, config, replay_identity=spec["replay_identity"], round_index=30, round_steps_target=1000)
    if learner.round_train_step:
        diagnostic(learner, start)
    pools = {n: PositionPool([g for g in index.split_games("train") if g.round_index >= 30-n//1000], index.cache, str(n)) for n in (3000, 6000, 9000)}
    session = str(time.time_ns())
    writer = ReferenceWriter(RUN / branch / "usage", session, 512, spec["replay_identity"])
    try:
        for stop in (250, 1000):
            if learner.round_train_step >= stop:
                continue
            stats = run_training_round(learner, steps_target=1000, current_pool=pools[9000], history_pool=None,
                validation_samples=(), validation_games=0, checkpoint_dir=checkpoints,
                metrics_path=RUN / branch / "training_metrics.jsonl", reference_writer=writer,
                checkpoint_callback=diagnostic, max_steps_this_session=stop-learner.round_train_step,
                joint_separate_batches=True,
                batch_drawer=lambda rng: draw_batches(pools[spec["policy_games"]], pools[spec["value_games"]], pools[6000], rng, 256),
                progress_callback=lambda r: print(json.dumps({"event": "train", "branch": branch,
                    "step": r["round_train_step"], "policy_kl": r["policy_kl"], "value_mse": r["value_loss"]}), flush=True) if r["round_train_step"] % 100 == 0 else None)
            write_json(ART / branch / "training_sessions" / f"{session}-step{stop}.json", stats)
    finally:
        writer.close()
    shutil.copyfile(checkpoints / "step-000001000.pt", checkpoints / "final.pt")
    verify(branch, plan)


def arena(branch, plan):
    config = {**plan["greedy_template"], "policy_a": {"kind": "greedy_checkpoint", "name": branch,
        "checkpoint": str(RUN / branch / "checkpoints/final.pt")}}
    freeze(ART / branch / "greedy_config.json", config)
    output = ART / branch / "greedy.json"
    run_fixed_evaluation(FixedEvaluationConfig.from_dict(config), output,
        resume=output.exists() or output.with_suffix(".json.games").exists(), progress_every=50, summary_every=50,
        progress_callback=lambda done, total: print(json.dumps({"event": "greedy", "branch": branch, "pairs": done}), flush=True),
        failure_dir=RUNTIME / branch / "failures")
    result = read_json(output)
    if not result["passed"] or result["completed_pairs"] != 200:
        raise ValueError("greedy evaluation incomplete or invalid")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "train", "arena", "analyze"), nargs="?", default="run")
    parser.add_argument("--branch", choices=tuple(BRANCHES))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    plan = prepare()
    if args.action == "prepare":
        print(json.dumps({"plan": str(PLAN)}), flush=True)
        return
    if args.action in ("train", "arena"):
        if not args.branch:
            parser.error("--branch required")
        if args.action == "train" and not torch.cuda.is_available():
            raise RuntimeError("CUDA required for training")
        (train if args.action == "train" else arena)(args.branch, plan)
        return
    if args.action == "analyze":
        subprocess.run([sys.executable, str(ART / "analyze.py")], cwd=ROOT, check=True)
        return
    with (RUNTIME / "run.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        session = str(time.time_ns())
        destination = RUNTIME / session
        source = source_identity(ROOT, destination)
        for path in (Path(__file__), ART / "analyze.py", PLAN):
            shutil.copyfile(path, destination / path.name)
        write_json(ART / "execution_sessions" / f"{session}.json", {"source": source,
            "environment": environment_identity(), "argv": sys.argv, "plan": identity(PLAN), "time_unix": time.time()})
        live = {"pid": __import__("os").getpid(), "status": "running", "plan": identity(PLAN)}
        try:
            for branch in BRANCHES:
                for phase in ("train", "arena"):
                    marker = ART / branch / ("training_verification.json" if phase == "train" else "greedy.json")
                    if marker.exists():
                        if phase == "train" or (read_json(marker)["passed"] and read_json(marker)["completed_pairs"] == 200):
                            continue
                    live.update(branch=branch, phase=phase, time_unix=time.time())
                    write_json(RUNTIME / "live.json", live)
                    print(json.dumps({"event": "start", "branch": branch, "phase": phase}), flush=True)
                    subprocess.run([sys.executable, "-u", str(Path(__file__)), phase, "--branch", branch], cwd=ROOT, check=True)
            subprocess.run([sys.executable, str(ART / "analyze.py")], cwd=ROOT, check=True)
            live.update(status="complete", time_unix=time.time())
        except BaseException:
            live.update(status="interrupted_or_failed", time_unix=time.time())
            raise
        finally:
            write_json(RUNTIME / "live.json", live)


if __name__ == "__main__":
    main()
