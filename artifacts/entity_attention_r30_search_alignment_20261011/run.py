"""Current round30 policy versus fresh, noise-free S1024/N1 on full round30 holdout."""
from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from splendor.alphazero_recording import environment_identity, read_json, sha256, source_identity, write_json
from splendor.features import encode_decision
from splendor.mcts_policy import MCTSConfig
from splendor.native_policy import NativeBatchRunner, NativePUCTSearchPolicy
from splendor.offline_train import _digest
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.policy_averaging import replay_roots
from splendor.replay import to_primitive
from splendor.search_inference import FrozenSearchInference, InferenceConfig
from splendor.selfplay_data import STATE_FIELDS, load_game

ART = Path(__file__).resolve().parent
ROOT = ART.parents[1]
RUNTIME = ROOT / "runtime" / ART.name
ORIGINAL = ROOT / "runs/entity_attention_r9_online_lr2e5_1000_forever_20261011/rounds/round-020"
CHECKPOINT = ORIGINAL / "checkpoints/final.pt"
PLAN = ART / "plan.json"
SEEDS = (33510000, 33510001)
CAPACITY = 200


def identity(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}


def freeze(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Frozen input changed: {path}")
    else:
        write_json(path, value)


def prepare():
    if sha256(CHECKPOINT) != "9fc258b211954cf32cd7b688d5d534a502441304ea7cc92a7a35e059a7c91b0f":
        raise ValueError("original round30 checkpoint changed")
    snapshot = read_json(ORIGINAL / "replay_snapshot.json")
    games = sorted((g for g in snapshot["games"] if g["round_index"] == 29 and g["split"] == "validation"), key=lambda g: g["game_id"])
    refs, sources = [], []
    for game in games:
        path = Path(game["path"])
        arrays, meta = load_game(path)
        if meta["split"] != "validation" or meta["round_index"] != 29:
            raise ValueError("source is not original round30 holdout")
        roots = replay_roots(arrays, meta)
        if len(roots) != meta["policy_samples"]:
            raise ValueError("effective policy root count differs")
        for index, request in roots:
            encoded = encode_decision(request)
            a, b = arrays["candidate_offsets"][index:index+2]
            if any(not np.array_equal(getattr(encoded, name), arrays[name][index]) for name in STATE_FIELDS):
                raise ValueError("public request does not reproduce archived state features")
            if not np.array_equal(encoded.candidate_features, arrays["candidate_features"][a:b]):
                raise ValueError("candidate features/order differ")
            o = request.observation
            refs.append({"root_id": len(refs), "game_id": game["game_id"], "game_seed": meta["game_seed"],
                "decision_index": index, "phase": int(arrays["phase"][index]), "candidate_count": len(request.legal_actions),
                "stage": "late" if max(o.self_public.score, o.opponent_public.score) >= 10 else "early_mid",
                "scores": [o.self_public.score, o.opponent_public.score],
                "opponent_blind_reserved": sum(r.card is None for r in o.opponent_public.reserved_cards),
                "request_sha256": _digest(to_primitive(request))})
        sources.append({**identity(path), "game_id": game["game_id"], "game_seed": meta["game_seed"],
                        "decisions": meta["decision_count"], "policy_roots": len(roots)})
    if len(games) != 124 or len(refs) != sum(g["policy_samples"] for g in games):
        raise ValueError("complete round30 holdout population differs")
    freeze(ART / "root_references.json", {"references": refs, "sources": sources})
    raw = read_json(ORIGINAL / "selfplay/manifest.json")["config"]["search_config"]
    search = {**raw, "simulations": 1024, "determinizations": 1, "root_noise_fraction": 0.0}
    search.pop("seed", None)
    plan = {"experiment": ART.name, "checkpoint": identity(CHECKPOINT), "absolute_round": 30, "global_step": 59898,
        "root_references": identity(ART / "root_references.json"), "snapshot": identity(ORIGINAL / "replay_snapshot.json"),
        "source_manifest": identity(ORIGINAL / "selfplay/manifest.json"), "source_teacher": snapshot["teacher"],
        "positions": len(refs), "games": len(games), "all_decisions": sum(g["decision_count"] for g in games),
        "by_stage": dict(Counter(r["stage"] for r in refs)), "by_phase": dict(Counter(str(r["phase"]) for r in refs)),
        "search": search, "world_seeds": list(SEEDS), "capacity": CAPACITY, "bootstrap_seed": 33520000,
        "inference": InferenceConfig(mode="eager", actor_projection=False, candidate_buckets=(16, 32, 48, 64)).to_dict(),
        "runner": identity(Path(__file__)), "analyzer": identity(ART / "analyze.py"),
        "scope": "Original round30 complete whole-game validation, generated by round29 S1024/N1 self-play; model30 was not trained on these games. Fresh model30 S1024/N1 targets, no root noise. All searchable multi-candidate phases, exclude forced decisions. Two independent N1 world seeds are reported separately and as mean metrics, not a shared N2 tree. Measures residual policy/search alignment, not playing strength or a before/after learning effect."}
    freeze(PLAN, plan)
    RUNTIME.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"event": "prepared", "positions": len(refs), "games": len(games), "stages": plan["by_stage"]}), flush=True)


def requests(bank):
    result = []
    for source in bank["sources"]:
        path = ROOT / source["path"]
        if sha256(path) != source["sha256"]:
            raise ValueError("source NPZ changed")
        arrays, meta = load_game(path)
        for index, request in replay_roots(arrays, meta):
            ref = bank["references"][len(result)]
            if (ref["game_id"], ref["decision_index"], ref["request_sha256"]) != (meta["game_id"], index, _digest(to_primitive(request))):
                raise ValueError("reconstructed request differs from frozen reference")
            result.append(request)
    if len(result) != len(bank["references"]):
        raise ValueError("reconstructed root count differs")
    return result


def run():
    plan = read_json(PLAN)
    for entry in (plan["runner"], plan["analyzer"], plan["checkpoint"], plan["root_references"]):
        if sha256(ROOT / entry["path"]) != entry["sha256"]:
            raise ValueError("frozen experiment identity changed")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this diagnostic")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with (RUNTIME / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        session = str(time.time_ns())
        source = source_identity(ROOT, RUNTIME / session)
        write_json(ART / "execution_sessions" / f"{session}.json", {"source": source,
            "environment": environment_identity(), "argv": sys.argv, "plan": identity(PLAN),
            "tf32_matmul": False, "tf32_cudnn": False})
        bank = read_json(ART / "root_references.json")
        roots = requests(bank)
        model = load_frozen_actor_critic_checkpoint(CHECKPOINT, device="cuda")
        inference = FrozenSearchInference(model, device="cuda", config=InferenceConfig(**plan["inference"]))
        begun = time.perf_counter()
        for repeat, seed in enumerate(plan["world_seeds"]):
            policy = NativePUCTSearchPolicy(model, evaluator=inference, config=MCTSConfig(**plan["search"], seed=seed))
            runner = NativeBatchRunner(inference, policy.catalog, capacity=CAPACITY, buckets=(16, 32, 48, 64), slots=1)
            for start in range(0, len(roots), CAPACITY):
                stop = min(start+CAPACITY, len(roots))
                path = ART / "raw" / f"repeat-{repeat}-roots-{start:06d}-{stop:06d}.npz"
                if path.exists():
                    with np.load(path, allow_pickle=False) as saved:
                        if not np.array_equal(saved["root_ids"], np.arange(start, stop)) or int(saved["seed"]) != seed:
                            raise ValueError("completed search chunk identity differs")
                    continue
                tick = time.perf_counter()
                batch = roots[start:stop]
                width = max(len(r.legal_actions) for r in batch)
                log_policy = np.full((len(batch), width), -np.inf, dtype=np.float64)
                for i, (logits, _) in enumerate(inference.policy_value(batch)):
                    x = np.asarray(logits, dtype=np.float64)
                    x = x-x.max()
                    log_policy[i, :len(x)] = x-np.log(np.exp(x).sum())
                jobs = [policy.start_job(request, add_root_noise=False, temperature=0.) for request in batch]
                while any(not job.done for job in jobs):
                    if runner.run([job for job in jobs if not job.done]) <= 0:
                        raise RuntimeError("native search made no progress")
                visits = np.zeros((len(batch), width), dtype=np.int32)
                prior = np.zeros((len(batch), width), dtype=np.float64)
                chosen = np.zeros(len(batch), dtype=np.int32)
                terminal_fraction = np.zeros(len(batch), dtype=np.float64)
                cap_hits = np.zeros(len(batch), dtype=np.int32)
                depth = np.zeros(len(batch), dtype=np.int32)
                for i, job in enumerate(jobs):
                    result = job.result
                    if (not result.searchable or sum(result.root_visit_counts) != 1024
                            or tuple(result.simulation_budgets) != (1024,)
                            or tuple(result.root_network_prior) != tuple(result.root_used_prior)):
                        raise ValueError("search is not noise-free complete S1024/N1")
                    n = len(batch[i].legal_actions)
                    visits[i, :n], prior[i, :n] = result.root_visit_counts, result.root_network_prior
                    if not np.allclose(prior[i, :n], np.exp(log_policy[i, :n]), rtol=2e-5, atol=2e-7):
                        raise ValueError("search root prior differs from current policy forward")
                    chosen[i], cap_hits[i], depth[i] = result.chosen_candidate_index, result.transition_cap_hits, result.max_tree_depth
                    terminal_fraction[i] = job.core.stats()["terminal_hits"]/1024
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".npz.tmp")
                with temporary.open("wb") as handle:
                    np.savez_compressed(handle, root_ids=np.arange(start, stop), seed=seed, visits=visits,
                        log_policy=log_policy, root_prior=prior, chosen=chosen, terminal_fraction=terminal_fraction,
                        cap_hits=cap_hits, max_depth=depth)
                    handle.flush()
                    __import__("os").fsync(handle.fileno())
                temporary.replace(path)
                elapsed = time.perf_counter()-tick
                write_json(RUNTIME / "live.json", {"status": "running", "repeat": repeat, "seed": seed,
                    "completed_roots_this_repeat": stop, "total_roots": len(roots), "seconds": time.perf_counter()-begun})
                write_json(ART / "timings" / f"repeat-{repeat}-start-{start:06d}.json", {"repeat": repeat, "seed": seed,
                    "start": start, "stop": stop, "seconds": elapsed, "raw": identity(path)})
                if start % 1000 == 0 or stop == len(roots):
                    print(json.dumps({"event": "search", "repeat": repeat, "roots": stop, "total": len(roots), "seconds": time.perf_counter()-begun}), flush=True)
                del jobs
            del runner
        subprocess.run([sys.executable, str(ART / "analyze.py")], cwd=ROOT, check=True)
        write_json(RUNTIME / "live.json", {"status": "complete", "seconds_this_invocation": time.perf_counter()-begun})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "analyze"), nargs="?", default="run")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "run":
        run()
    else:
        subprocess.run([sys.executable, str(ART / "analyze.py")], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
