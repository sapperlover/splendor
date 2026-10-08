"""Four-arm, frozen-model target-noise diagnostic; no training or game collection."""
from __future__ import annotations

import argparse
import itertools
import json
import shutil
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from splendor.alphazero_recording import environment_identity, read_json, sha256, source_identity, write_json
from splendor.compact_trace import decode_request
from splendor.contracts import DecisionRequest, TerminalResult
from splendor.engine import GameEngine
from splendor.mcts_policy import MCTSConfig
from splendor.native_policy import NativeBatchRunner, NativePUCTSearchPolicy
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.replay import to_primitive
from splendor.search_inference import FrozenSearchInference, InferenceConfig
from splendor.selfplay_data import load_game

ROOT = Path(__file__).resolve().parents[2]
ART = Path(__file__).resolve().parent
RUNTIME = ROOT / "runtime/entity_attention_target_noise_20261007"
RUN = ROOT / "runs/entity_attention_hybrid_split_p4_v15_from_round50_seed1"
ARMS = ("A", "B", "C", "D")
REPEATS = 8
POSITIONS = 200
SEEDS = {"selection": 16700000, "world": 16710000, "noise": 16720000, "bootstrap": 16730000}


def record(path):
    path = Path(path)
    return {"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size}


def prepare():
    assert not (ART / "manifest.json").exists(), "use the existing frozen input rather than reselecting it"
    state = read_json(RUN / "hybrid_state.json")
    checkpoint = Path(state["latest_checkpoint"]["path"])
    assert sha256(checkpoint) == state["latest_checkpoint"]["sha256"]
    stage = checkpoint.parents[1]
    snapshot_path = stage / "replay_snapshot.json"
    snapshot = read_json(snapshot_path)
    current_round = int(snapshot["round_index"])
    recent_rounds = list(range(current_round - 3, current_round + 1))
    games = sorted((g for g in snapshot["games"] if g["round_index"] in recent_rounds
                    and g["split"] == "validation"), key=lambda g: (g["round_index"], g["game_id"]))
    assert games and all(g["run_id"].startswith(RUN.name + "-round-") for g in games)
    candidates, inputs = [], []
    for g in games:
        path = Path(g["path"])
        arrays, meta = load_game(path)
        assert meta["split"] == "validation" and meta["round_index"] == g["round_index"]
        assert meta["run_id"] == g["run_id"] and meta["game_id"] == g["game_id"]
        eligible = [i for i in range(meta["decision_count"]) if int(arrays["phase"][i]) == 0
                    and int(arrays["candidate_offsets"][i + 1] - arrays["candidate_offsets"][i]) > 1]
        choices = {eligible[int((len(eligible) - 1) * q)]: label
                   for q, label in ((0.15, "early"), (0.5, "middle"), (0.85, "late"))}
        assert len(choices) == 3
        engine = GameEngine.new(meta["game_seed"], first_player=meta["first_player"], mode="checked")
        assert len(meta["trace"]["actions"]) == meta["decision_count"]
        for i, action in enumerate(meta["trace"]["actions"]):
            request = engine.decision()
            assert isinstance(request, DecisionRequest)
            assert action["decision_index"] == i
            assert to_primitive(request.legal_actions[action["candidate_index"]]) == action["action"]
            assert len(request.legal_actions) == arrays["candidate_offsets"][i + 1] - arrays["candidate_offsets"][i]
            if i in choices:
                candidates.append({"game_key": f"{g['run_id']}/{g['game_id']}", "game_id": g["game_id"],
                    "round_index": g["round_index"], "teacher_sha256": meta["checkpoint_sha256"],
                    "decision_index": i, "stage": choices[i], "request": to_primitive(request),
                    "candidate_count": len(request.legal_actions),
                    "opponent_blind_reserved": sum(r.card is None for r in request.observation.opponent_public.reserved_cards),
                    "deck_counts": list(request.observation.deck_counts),
                    "completed_player_turns": request.observation.own_completed_turns + request.observation.opponent_completed_turns})
            engine.apply(request.legal_actions[action["candidate_index"]])
        terminal = engine.decision()
        assert isinstance(terminal, TerminalResult) and to_primitive(terminal) == meta["trace"]["terminal_result"]
        inputs.append(record(path))
    rng = np.random.default_rng(SEEDS["selection"])
    rows = []
    for label, count in (("early", 67), ("middle", 67), ("late", 66)):
        pool = [r for r in candidates if r["stage"] == label]
        assert len(pool) >= count
        rows.extend(pool[int(i)] for i in rng.choice(len(pool), count, replace=False))
    rows.sort(key=lambda r: (r["round_index"], r["game_id"], r["decision_index"]))
    for i, row in enumerate(rows):
        row["id"] = i
    assert len({(r["game_key"], r["decision_index"]) for r in rows}) == POSITIONS
    bank = {"positions": rows, "inputs": inputs, "recent_round_indices": recent_rounds,
            "selection_seed": SEEDS["selection"], "scope": "balanced early/middle/late main actions from recent-four validation games; at most three roots per game"}
    write_json(ART / "root_bank.json", bank)
    source = source_identity(ROOT, RUNTIME / "source")
    extra_paths = [ROOT / "src/splendor/native_search.cpp", ROOT / "setup.py", *sorted((ROOT / "data").glob("**/*"))]
    source["additional_files"] = {}
    for path in extra_paths:
        if path.is_file():
            source["additional_files"][str(path.relative_to(ROOT))] = sha256(path)
            target = RUNTIME / "source" / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    cfg = read_json(RUN / "config.json")["selfplay"]["search_config"]
    cfg.update(simulations=1024, determinizations=1)
    manifest = {"protocol": "attention-target-noise-abcd-v1", "status": "prepared", "client_date": "2026-10-07",
        "checkpoint": record(checkpoint), "total_train_step": state["latest_checkpoint"]["total_train_step"],
        "completed_new_round": len(state["completed_rounds"]), "seeds": SEEDS,
        "search_config": cfg, "positions": POSITIONS, "repeats": REPEATS, "arms": {
            "A": {"world_varies": False, "root_noise": False},
            "B": {"world_varies": False, "root_noise": True},
            "C": {"world_varies": True, "root_noise": False},
            "D": {"world_varies": True, "root_noise": True}},
        "source": source, "inputs": [record(RUN / "config.json"), record(snapshot_path), record(ART / "root_bank.json")],
        "script": record(__file__), "commands": {
            "prepare": "conda run --no-capture-output -n splendor env PYTHONPATH=src python artifacts/entity_attention_target_noise_20261007/probe.py prepare",
            "run": "conda run --no-capture-output -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u artifacts/entity_attention_target_noise_20261007/probe.py run",
            "analyze": "conda run --no-capture-output -n splendor env PYTHONPATH=src python artifacts/entity_attention_target_noise_20261007/probe.py analyze",
            "verify": "conda run --no-capture-output -n splendor env PYTHONPATH=src python artifacts/entity_attention_target_noise_20261007/probe.py verify"},
        "interpretation": "A tests exact repetition, not budget adequacy. B/C are marginal probes at fixed complementary factors; D is joint variation, not an additive variance decomposition. Empirical mean of eight targets is not a true target or a strength label."}
    write_json(ART / "manifest.json", manifest)
    print(json.dumps({"event": "prepared", "round": manifest["completed_new_round"], "step": manifest["total_train_step"],
        "roots": len(rows), "games": len(set(r["game_key"] for r in rows)), "source_validation_games": len(games),
        "blind_reserved_roots": sum(r["opponent_blind_reserved"] > 0 for r in rows)}, ensure_ascii=False), flush=True)


def run(smoke=False):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.is_available()
    manifest = read_json(ART / "manifest.json")
    checkpoint = Path(manifest["checkpoint"]["path"])
    assert sha256(checkpoint) == manifest["checkpoint"]["sha256"]
    assert sha256(Path(__file__)) == manifest["script"]["sha256"]
    bank = read_json(ART / "root_bank.json")["positions"]
    if smoke:
        bank = bank[:8]
    else:
        assert not list((RUNTIME / "parts").glob("*.npz")), "formal run must start with no existing parts"
    requests = [decode_request(r["request"]) for r in bank]
    model = load_frozen_actor_critic_checkpoint(checkpoint, device="cuda")
    config = InferenceConfig(mode="eager", actor_projection=False, candidate_buckets=(16, 32, 48, 64))
    evaluator = FrozenSearchInference(model, device="cuda", config=config)
    runner = NativeBatchRunner(evaluator, NativePUCTSearchPolicy(model, evaluator=evaluator).catalog,
                               capacity=200, buckets=(16, 32, 48, 64), slots=1)
    write_json(RUNTIME / ("smoke_environment.json" if smoke else "environment.json"), {
        **environment_identity(), "inference_config": config.to_dict(), "tf32_matmul": False, "tf32_cudnn": False})
    root_count, width = len(bank), max(len(r.legal_actions) for r in requests)
    started = time.perf_counter()
    tasks = [("A", i) for i in range(2)] if smoke else list(itertools.product(ARMS, range(REPEATS)))
    timing, smoke_targets = [], []
    for arm, rep in tasks:
        world_seed = SEEDS["world"] + (rep if manifest["arms"][arm]["world_varies"] else 0)
        noise_seed = SEEDS["noise"] + (rep if manifest["arms"][arm]["root_noise"] else 0)
        search_cfg = dict(manifest["search_config"], seed=world_seed)
        if smoke:
            search_cfg["simulations"] = 64
        policy = NativePUCTSearchPolicy(model, evaluator=evaluator, config=MCTSConfig(**search_cfg))
        jobs = [policy.start_job(r, add_root_noise=manifest["arms"][arm]["root_noise"],
                                temperature=0, sampling_seed=noise_seed) for r in requests]
        tick = time.perf_counter()
        while any(not j.done for j in jobs):
            assert runner.run([j for j in jobs if not j.done]) > 0
        elapsed = time.perf_counter() - tick
        floats = {k: np.zeros((root_count, width), dtype=np.float64)
                  for k in ("policy_target", "root_network_prior", "root_used_prior")}
        counts = np.zeros((root_count, width), dtype=np.int32)
        stats = {k: np.zeros(root_count, dtype=np.int32) for k in (
            "chosen_candidate_index", "max_tree_depth", "transition_cap_hits", "simulations")}
        for root_index, job in enumerate(jobs):
            result = job.result
            assert result.searchable and tuple(result.simulation_budgets) == (search_cfg["simulations"],)
            assert sum(result.root_visit_counts) == result.simulations == search_cfg["simulations"]
            assert tuple(result.root_visit_counts) == tuple(result.per_world_root_visit_counts[0])
            size = len(requests[root_index].legal_actions)
            for key, value in floats.items():
                value[root_index, :size] = getattr(result, key)
            counts[root_index, :size] = result.root_visit_counts
            for key, value in stats.items():
                value[root_index] = getattr(result, key)
        row = {"arm": arm, "repeat": rep, "world_seed": world_seed, "noise_seed": noise_seed,
               "seconds": elapsed, "roots": root_count, "search_stats": asdict(policy.stats)}
        timing.append(row)
        if smoke:
            smoke_targets.append(floats["policy_target"])
        else:
            output = RUNTIME / "parts" / f"{arm}_{rep}.npz"
            output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(output, **floats, **stats, root_visit_counts=counts)
            write_json(RUNTIME / "timing.json", timing)
        print(json.dumps({"event": "search_done", "arm": arm, "repeat": rep, "seconds": elapsed,
                          "elapsed_total": time.perf_counter() - started}, ensure_ascii=False), flush=True)
    assert sha256(checkpoint) == manifest["checkpoint"]["sha256"]
    if smoke:
        assert np.array_equal(*smoke_targets)
        write_json(ART / "smoke.json", {"passed": True, "roots": root_count, "simulations": 64,
            "repeats": 2, "identical_targets": True, "timing": timing,
            "environment": read_json(RUNTIME / "smoke_environment.json")})
    else:
        manifest.update(status="search_complete", search_seconds=time.perf_counter() - started,
                        environment=read_json(RUNTIME / "environment.json"), timing=timing)
        write_json(ART / "manifest.json", manifest)


def kl(p, q):
    mask = p > 0
    assert np.all(q[mask] > 0)
    return float(np.sum(p[mask] * (np.log(p[mask]) - np.log(q[mask]))))


def js(p, q):
    mean = (p + q) / 2
    return (kl(p, mean) + kl(q, mean)) / 2


def top_indices(p, prior):
    return np.lexsort((np.arange(len(p)), -prior, -p))


def describe(values):
    a = np.asarray(values)
    return {"mean": float(a.mean()), "median": float(np.median(a)),
            "p90": float(np.quantile(a, 0.9)), "max": float(a.max())}


def collect_raw():
    keys = ("policy_target", "root_network_prior", "root_used_prior", "root_visit_counts",
            "chosen_candidate_index", "max_tree_depth", "transition_cap_hits", "simulations")
    raw = {}
    for key in keys:
        values = []
        for arm in ARMS:
            reps = []
            for rep in range(REPEATS):
                with np.load(RUNTIME / "parts" / f"{arm}_{rep}.npz", allow_pickle=False) as archive:
                    reps.append(archive[key].copy())
            values.append(np.stack(reps))
        raw[key] = np.stack(values)
    return raw


def compute(raw, bank):
    pairings = list(itertools.combinations(range(REPEATS), 2))
    per_root = []
    for ai, arm in enumerate(ARMS):
        for ri, root in enumerate(bank):
            size = root["candidate_count"]
            targets = raw["policy_target"][ai, :, ri, :size]
            prior = raw["root_network_prior"][ai, 0, ri, :size]
            mean = targets.mean(axis=0)
            orders = [top_indices(p, prior) for p in targets]
            tops = np.asarray([o[0] for o in orders])
            assert np.array_equal(tops, raw["chosen_candidate_index"][ai, :, ri])
            k = min(3, size)
            agreements = [tops[i] == tops[j] for i, j in pairings]
            overlaps = [len(set(orders[i][:k]) & set(orders[j][:k])) / k for i, j in pairings]
            single_kl = float(np.mean([kl(p, prior) for p in targets]))
            mean_kl = kl(mean, prior)
            variation = float(np.mean([kl(p, mean) for p in targets]))
            gaps = [float(p[o[0]] - p[o[1]]) for p, o in zip(targets, orders)]
            row = {"arm": arm, "root_id": ri, "game_key": root["game_key"], "stage": root["stage"],
                "opponent_blind_reserved": root["opponent_blind_reserved"], "candidate_count": size,
                "pairwise_js": float(np.mean([js(targets[i], targets[j]) for i, j in pairings])),
                "top1_agreement": float(np.mean(agreements)), "top3_overlap": float(np.mean(overlaps)),
                "unique_top1": int(len(set(tops))), "target_kl_to_network": single_kl,
                "mean_target_kl_to_network": mean_kl, "multiway_js": variation,
                "identity_residual": single_kl - mean_kl - variation,
                "mean_top1_gap": float(np.mean(gaps)),
                "decisive_flip_fraction": float(np.mean([tops[i] != tops[j] and min(gaps[i], gaps[j]) >= 0.1
                                                         for i, j in pairings])),
                "network_top1_agreement": float(np.mean(tops == top_indices(prior, prior)[0])),
                "mean_target": mean.tolist(), "network_prior": prior.tolist()}
            per_root.append(row)
    metric_keys = ("pairwise_js", "top1_agreement", "top3_overlap", "unique_top1", "target_kl_to_network",
                   "mean_target_kl_to_network", "multiway_js", "mean_top1_gap", "decisive_flip_fraction",
                   "network_top1_agreement")
    summaries = {}
    for arm in ARMS:
        rows = [r for r in per_root if r["arm"] == arm]
        summary = {k: describe([r[k] for r in rows]) for k in metric_keys}
        summary["randomness_fraction_of_mean_kl"] = summary["multiway_js"]["mean"] / summary["target_kl_to_network"]["mean"]
        summary["roots_with_top1_changes"] = sum(r["unique_top1"] > 1 for r in rows)
        summary["strata"] = {}
        for label, chosen in [*( (stage, [r for r in rows if r["stage"] == stage]) for stage in ("early", "middle", "late")),
                              ("opponent_blind", [r for r in rows if r["opponent_blind_reserved"] > 0]),
                              ("no_opponent_blind", [r for r in rows if r["opponent_blind_reserved"] == 0])]:
            summary["strata"][label] = {"roots": len(chosen), **{k: float(np.mean([r[k] for r in chosen]))
                                           for k in metric_keys}} if chosen else {"roots": 0}
        summaries[arm] = summary
    # Cluster-bootstrap games: up to three correlated roots originate from one trajectory.
    game_keys = sorted(set(r["game_key"] for r in bank))
    game_indices = [np.asarray([i for i, r in enumerate(bank) if r["game_key"] == g]) for g in game_keys]
    rng = np.random.default_rng(SEEDS["bootstrap"])
    choices = rng.integers(len(game_keys), size=(10000, len(game_keys)))
    weights = np.stack([np.bincount(c, minlength=len(game_keys)) for c in choices]).astype(np.float64)
    denominators = weights @ np.asarray([len(i) for i in game_indices])
    differences = {}
    for left, right in (("C", "B"), ("D", "C"), ("D", "B")):
        diff = {}
        for metric in ("pairwise_js", "top1_agreement", "multiway_js", "target_kl_to_network", "mean_target_kl_to_network"):
            a = np.asarray([r[metric] for r in per_root if r["arm"] == left])
            b = np.asarray([r[metric] for r in per_root if r["arm"] == right])
            sums = np.asarray([(a[i] - b[i]).sum() for i in game_indices])
            sampled = weights @ sums / denominators
            lo, hi = np.quantile(sampled, [0.025, 0.975])
            diff[metric] = {"mean": float((a - b).mean()), "ci95": [float(lo), float(hi)]}
        differences[f"{left}_minus_{right}"] = diff
    max_residual = max(abs(r["identity_residual"]) for r in per_root)
    assert max_residual < 1e-10
    return {"groups": summaries, "paired_game_cluster_bootstrap": {"resamples": 10000,
        "seed": SEEDS["bootstrap"], "game_clusters": len(game_keys), "differences": differences},
        "max_kl_identity_residual": max_residual}, per_root


def checks(raw, bank):
    assert raw["policy_target"].shape[:3] == (4, REPEATS, POSITIONS)
    assert np.all(raw["simulations"] == 1024)
    assert np.all(raw["root_visit_counts"].sum(axis=-1) == 1024)
    np.testing.assert_allclose(raw["policy_target"], raw["root_visit_counts"] / 1024, rtol=0, atol=0)
    assert np.all(raw["transition_cap_hits"] == 0)
    np.testing.assert_allclose(raw["policy_target"].sum(axis=-1), 1, atol=1e-12, rtol=0)
    for ri, root in enumerate(bank):
        size = root["candidate_count"]
        assert np.all(raw["root_network_prior"][:, :, ri, :size] > 0)
        assert np.all(raw["policy_target"][:, :, ri, size:] == 0)
    reference = np.broadcast_to(raw["root_network_prior"][0, 0], raw["root_network_prior"].shape)
    assert np.array_equal(raw["root_network_prior"], reference)
    for key in raw:
        assert np.array_equal(raw[key][0], np.broadcast_to(raw[key][0, 0], raw[key][0].shape)), ("A not reproducible", key)
        assert np.array_equal(raw[key][0, 0], raw[key][2, 0]), ("A/C first seed differs", key)
        assert np.array_equal(raw[key][1, 0], raw[key][3, 0]), ("B/D first seed differs", key)
    assert np.array_equal(raw["root_used_prior"][0], raw["root_network_prior"][0])
    assert np.array_equal(raw["root_used_prior"][2], raw["root_network_prior"][2])
    assert np.array_equal(raw["root_used_prior"][1], raw["root_used_prior"][3])
    return {"passed": True, "searches": 4 * REPEATS * POSITIONS, "simulations": int(raw["simulations"].sum()),
            "A_bitwise_identical": True, "prior_bitwise_identical": True, "common_first_seed_verified": True,
            "independent_world_and_noise_controls_verified": True, "transition_cap_hits": 0}


def analyze():
    bank = read_json(ART / "root_bank.json")["positions"]
    raw = collect_raw()
    verification = checks(raw, bank)
    summary, per_root = compute(raw, bank)
    summary.update(protocol="attention-target-noise-abcd-v1", positions=POSITIONS, repeats=REPEATS,
                   units="natural logarithm / nats", verification=verification,
                   metric_definitions={"pairwise_js": "mean over all 28 repeat pairs, then equally across roots",
                       "top1_agreement": "fraction of 28 pairs choosing the same top action; prior then index breaks ties",
                       "top3_overlap": "intersection size / min(3, candidate count), mean over pairs",
                       "multiway_js": "mean KL(target_i || mean_target), finite-eight empirical dispersion",
                       "target_kl_to_network": "mean KL(target_i || frozen network prior)",
                       "mean_target_kl_to_network": "KL(mean_target || frozen network prior)",
                       "decisive_flip_fraction": "pairwise top1 disagreement with both targets' top1-top2 probability gap >= 0.1"})
    np.savez_compressed(ART / "raw_targets.npz", **raw)
    write_json(ART / "per_root.json", per_root)
    write_json(ART / "summary.json", summary)
    write_json(ART / "verification.json", verification)
    manifest = read_json(ART / "manifest.json")
    manifest.update(status="complete", outputs=[record(ART / f) for f in (
        "root_bank.json", "raw_targets.npz", "per_root.json", "summary.json", "verification.json", "smoke.json")])
    write_json(ART / "manifest.json", manifest)
    print(json.dumps({"verification": verification, "groups": {a: {k: v["mean"] for k, v in g.items()
          if isinstance(v, dict) and "mean" in v} for a, g in summary["groups"].items()}}, ensure_ascii=False), flush=True)


def verify():
    manifest = read_json(ART / "manifest.json")
    assert manifest["status"] == "complete"
    for item in [manifest["checkpoint"], manifest["script"], *manifest["inputs"], *manifest["outputs"],
                 *read_json(ART / "root_bank.json")["inputs"]]:
        path = Path(item["path"])
        assert sha256(path) == item["sha256"] and path.stat().st_size == item["bytes"], path
    with np.load(ART / "raw_targets.npz", allow_pickle=False) as archive:
        raw = {k: archive[k] for k in archive.files}
    bank = read_json(ART / "root_bank.json")["positions"]
    checks(raw, bank)
    recomputed, per_root = compute(raw, bank)
    assert per_root == read_json(ART / "per_root.json")
    saved = read_json(ART / "summary.json")
    assert recomputed["groups"] == saved["groups"]
    assert recomputed["paired_game_cluster_bootstrap"] == saved["paired_game_cluster_bootstrap"]
    print(json.dumps({"passed": True, "raw_searches": 6400, "recomputed_metrics": True,
                      "input_and_output_hashes_verified": True}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "analyze", "verify"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.command == "run":
        run(args.smoke)
    else:
        globals()[args.command]()
