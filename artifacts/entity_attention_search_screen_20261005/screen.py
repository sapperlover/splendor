"""Bounded S/N screening of one frozen attention checkpoint; no training."""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import shutil
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from splendor.alphazero_recording import read_json, sha256, source_identity, write_json
from splendor.compact_trace import decode_request, replay_trace
from splendor.contracts import DecisionRequest
from splendor.engine import GameEngine
from splendor.mcts_policy import MCTSConfig
from splendor.native_policy import NativeBatchRunner, NativePUCTSearchPolicy
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.paired_statistics import paired_bootstrap_difference, serialized_pair_scores
from splendor.replay import to_primitive
from splendor.search_inference import FrozenSearchInference
from splendor.selfplay_data import load_game

ROOT = Path(__file__).resolve().parents[2]
ART = Path(__file__).resolve().parent
RUNTIME = ROOT / "runtime/entity_attention_search_screen_20261005"
RUN = ROOT / "runs/entity_attention_hybrid_static_graph_seed1"
STAGE = RUN / "rounds/round-008"
CASES = ((512, 1), (1024, 1), (512, 2), (1024, 2))
SEEDS = (14830000, 14830001, 14830002)


def name(s, n):
    return f"s{s}_n{n}"


def record(path):
    return {"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size}


def prepare():
    ART.mkdir(parents=True, exist_ok=True)
    RUNTIME.mkdir(parents=True, exist_ok=True)
    base = read_json(STAGE / "evaluation/config.json")
    baseline = STAGE / "evaluation/mcts_vs_theta5000.json"
    assert read_json(baseline)["passed"]
    cfg = read_json(RUN / "config.json")
    manifest = {"protocol": "attention-sn-screen-v1", "status": "prepared", "cases": CASES,
                "checkpoint": record(Path(base["policy_a"]["checkpoint"])),
                "opponent": record(Path(base["policy_b"]["checkpoint"])),
                "source": source_identity(ROOT, RUNTIME / "source"),
                "inputs": [record(baseline), record(STAGE / "evaluation/config.json"), record(RUN / "config.json")],
                "root_seeds": SEEDS,
                "selection_scope": "fixed-model strength, target sensitivity and production short-window cost; no training-optimum claim",
                "commands": []}
    for s, n in CASES:
        case = name(s, n)
        evaluation = copy.deepcopy(base)
        evaluation["policy_a"]["search_config"].update(simulations=s, determinizations=n)
        write_json(ART / "configs" / f"eval_{case}.json", evaluation)
        if (s, n) == (1024, 1):
            shutil.copyfile(baseline, ART / f"eval_{case}.json")
            shutil.copyfile(Path(str(baseline) + ".games/header.json"), ART / f"eval_{case}_header.json")
        else:
            manifest["commands"].append(
                "conda run --no-capture-output -n splendor env PYTHONPATH=src python -u "
                f"scripts/run_recorded_evaluation.py --config {ART.relative_to(ROOT)}/configs/eval_{case}.json "
                f"--output {RUNTIME.relative_to(ROOT)}/eval_{case}.json --progress-every 10")
        collection = copy.deepcopy(cfg["selfplay"])
        collection.update(run_id=f"attention_sn_screen_{case}", round_index=0, num_games=128,
                          checkpoint=base["policy_a"]["checkpoint"], seed=14800000, split_seed=14810000)
        collection["search_config"].update(simulations=s, determinizations=n, seed=14820000)
        collection["execution"].update(total_active_games=100, max_batch_size=128,
                                        batch_buckets=[16, 32, 64, 128])
        write_json(ART / "configs" / f"collect_{case}.json", collection)
    write_json(ART / "manifest.json", manifest)
    print(json.dumps({"cases": [name(*c) for c in CASES], "checkpoint": manifest["checkpoint"]}, ensure_ascii=False), flush=True)


def bank():
    snapshot = read_json(STAGE / "replay_snapshot.json")
    games = [g for g in snapshot["games"] if g["round_index"] == 8 and g["split"] == "validation"][:8]
    assert len(games) == 8
    rows, inputs = [], []
    for game in games:
        path = Path(game["path"])
        arrays, meta = load_game(path)
        trace = meta["trace"]
        replay_trace(trace)  # Checked full-trajectory verification, independent of search/model.
        engine = GameEngine.new(meta["game_seed"], first_player=meta["first_player"], mode="checked")
        eligible = [i for i in range(meta["decision_count"]) if int(arrays["phase"][i]) == 0
                    and int(arrays["candidate_offsets"][i + 1] - arrays["candidate_offsets"][i]) > 1]
        choices = {eligible[int((len(eligible) - 1) * q)]: label
                   for q, label in ((0.15, "early"), (0.5, "middle"), (0.85, "late"))}
        assert len(choices) == 3
        for i, action in enumerate(trace["actions"]):
            request = engine.decision()
            assert isinstance(request, DecisionRequest)
            assert to_primitive(request.legal_actions[action["candidate_index"]]) == action["action"]
            assert len(request.legal_actions) == int(arrays["candidate_offsets"][i + 1] - arrays["candidate_offsets"][i])
            if i in choices:
                rows.append({"id": len(rows), "game_id": game["game_id"], "decision_index": i,
                             "stage": choices[i], "request": to_primitive(request),
                             "opponent_blind_reserved": sum(r.card is None for r in request.observation.opponent_public.reserved_cards),
                             "completed_player_turns": request.observation.own_completed_turns + request.observation.opponent_completed_turns})
            engine.apply(request.legal_actions[action["candidate_index"]])
        inputs.append(record(path))
    assert len(rows) == 24
    write_json(ART / "root_bank.json", {"positions": rows, "inputs": inputs,
               "scope": "8 held-out current-round games, 3 main-action positions per game; checked trace reconstruction"})
    print(json.dumps({"root_positions": len(rows), "blind_reserved_positions": sum(r["opponent_blind_reserved"] > 0 for r in rows)}, ensure_ascii=False), flush=True)


def js(p, q):
    p, q = np.asarray(p, dtype=np.float64), np.asarray(q, dtype=np.float64)
    m = (p + q) / 2
    def kl(a):
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / m[mask])))
    return (kl(p) + kl(q)) / 2


def roots():
    torch.set_num_threads(1)
    checkpoint = Path(read_json(ART / "manifest.json")["checkpoint"]["path"])
    model = load_frozen_actor_critic_checkpoint(checkpoint, device="cpu")
    evaluator = FrozenSearchInference(model, device="cpu")
    bank_rows = read_json(ART / "root_bank.json")["positions"]
    requests = [decode_request(r["request"]) for r in bank_rows]
    runner = NativeBatchRunner(evaluator, NativePUCTSearchPolicy(model).catalog,
                               capacity=64, buckets=(16, 32, 48, 64), slots=1)
    rows = []
    started = time.perf_counter()
    for noise in (False, True):
        for s, n in CASES:
            for seed in SEEDS:
                policy = NativePUCTSearchPolicy(model, evaluator=evaluator,
                    config=MCTSConfig(simulations=s, determinizations=n, seed=seed, root_noise_fraction=0.25))
                jobs = [policy.start_job(r, add_root_noise=noise, temperature=0, sampling_seed=14840000) for r in requests]
                tick = time.perf_counter()
                while any(not j.done for j in jobs):
                    assert runner.run([j for j in jobs if not j.done]) > 0
                elapsed = time.perf_counter() - tick
                for root, job in zip(bank_rows, jobs):
                    result = job.result
                    assert sum(result.simulation_budgets) == s
                    assert np.isclose(sum(result.policy_target), 1)
                    rows.append({"case": name(s, n), "seed": seed, "root_noise": noise,
                                 "root_id": root["id"], **asdict(result)})
                print(json.dumps({"root_case": name(s, n), "seed": seed, "root_noise": noise,
                                  "roots": len(jobs), "cpu_seconds": elapsed}, ensure_ascii=False), flush=True)
                write_json(ART / "root_results.json", {"device": "cpu", "mode": "eager", "dtype": "float32", "threads": 1,
                           "checkpoint": record(checkpoint), "rows": rows, "elapsed_seconds": time.perf_counter() - started,
                           "timing_scope": "CPU root diagnostic only; not production CUDA collection cost"})
    print("all fixed-root searches complete", flush=True)


def digest():
    manifest = read_json(ART / "manifest.json")
    panels = {}
    for s, n in CASES:
        case = name(s, n)
        if (s, n) != (1024, 1):
            source = RUNTIME / f"eval_{case}.json"
            shutil.copyfile(source, ART / source.name)
            shutil.copyfile(Path(str(source) + ".games/header.json"), ART / f"eval_{case}_header.json")
            ledger = RUNTIME / "recording" / f"eval_{case}" / "recording"
            for folder in ("sessions", "identities"):
                for path in (ledger / folder).glob("*.json"):
                    target = ART / "recording" / f"eval_{case}" / folder / path.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, target)
        summary = read_json(ART / f"eval_{case}.json")
        assert summary["passed"] and summary["valid_games"] == 100
        assert summary["checkpoint_identity"]["policy_a"]["sha256"] == manifest["checkpoint"]["sha256"]
        panels[case] = summary
    strengths = []
    baseline_data = serialized_pair_scores(panels["s1024_n1"])
    for case, summary in panels.items():
        delta = paired_bootstrap_difference(*serialized_pair_scores(summary), *baseline_data,
                                           resamples=10000, seed=14720000)
        stats = summary["policy_a_stats"]
        strengths.append({"case": case, "score": summary["win_rate_score"], "wins": summary["a_wins"],
                          "losses": summary["a_losses"], "draws": summary["draws"],
                          "score_ci": summary["paired_bootstrap"], "difference_vs_s1024_n1": delta.to_dict(),
                          "mean_depth_per_simulation": stats["tree_depth_sum"] / stats["simulations"],
                          "terminal_hit_fraction": stats["terminal_hits"] / stats["simulations"],
                          "changed_action_fraction": stats["changed_action_count"] / stats["searchable_decisions"],
                          "transition_cap_hits": stats["transition_cap_hits"],
                          "evaluation_seconds": summary["elapsed_seconds_this_invocation"],
                          "timing_scope": "mixed candidate/fixed-opponent panel; not pure candidate collection cost"})
    all_differences = {f"{a}_minus_{b}": paired_bootstrap_difference(
                        *serialized_pair_scores(panels[a]), *serialized_pair_scores(panels[b]),
                        resamples=10000, seed=14720000).to_dict()
                       for a, b in itertools.combinations(panels, 2)}
    root_rows = read_json(ART / "root_results.json")["rows"]
    root_lookup = {(r["case"], r["seed"], r["root_noise"], r["root_id"]): r for r in root_rows}
    sensitivity = []
    for noise in (False, True):
        for s, n in CASES:
            case = name(s, n)
            group = [r for r in root_rows if r["case"] == case and r["root_noise"] == noise]
            changes, divergences, stability, seed_changes, entropy, prior_js, world_js = [], [], [], [], [], [], []
            for r in group:
                reference = root_lookup[("s1024_n1", r["seed"], noise, r["root_id"])]
                changes.append(r["chosen_candidate_index"] != reference["chosen_candidate_index"])
                divergences.append(js(r["policy_target"], reference["policy_target"]))
                p = np.asarray(r["policy_target"])
                entropy.append(float(-np.sum(p[p > 0] * np.log(p[p > 0]))))
                prior_js.append(js(r["policy_target"], r["root_network_prior"]))
                if n == 2:
                    worlds = [np.asarray(v, dtype=np.float64) / budget for v, budget in
                              zip(r["per_world_root_visit_counts"], r["simulation_budgets"])]
                    world_js.append(js(*worlds))
            for root_id in range(24):
                for left, right in itertools.combinations(SEEDS, 2):
                    a = root_lookup[(case, left, noise, root_id)]
                    b = root_lookup[(case, right, noise, root_id)]
                    stability.append(js(a["policy_target"], b["policy_target"]))
                    seed_changes.append(a["chosen_candidate_index"] != b["chosen_candidate_index"])
            sensitivity.append({"case": case, "root_noise": noise,
                                "choice_change_vs_s1024_n1": float(np.mean(changes)),
                                "mean_js_vs_s1024_n1": float(np.mean(divergences)),
                                "mean_js_across_world_seeds": float(np.mean(stability)),
                                "choice_change_across_world_seeds": float(np.mean(seed_changes)),
                                "mean_target_entropy": float(np.mean(entropy)),
                                "mean_js_to_network_prior": float(np.mean(prior_js)),
                                "mean_js_between_worlds": float(np.mean(world_js)) if world_js else None})
    write_json(ART / "screening_summary.json", {"strengths": strengths, "paired_differences": all_differences,
                                               "root_sensitivity": sensitivity})
    manifest["status"] = "strength_and_roots_complete"
    manifest["analysis_script"] = record(Path(__file__))
    write_json(ART / "manifest.json", manifest)
    print(json.dumps(strengths, ensure_ascii=False), flush=True)


def costs():
    windows = []
    for s, n in CASES:
        case = name(s, n)
        for rep in (1, 2):
            panel = f"window_{case}_rep{rep}"
            path = RUNTIME / f"{panel}.json"
            result = read_json(path)
            assert result["simulations"] == s and result["determinizations"] == n
            assert result["execution"]["total_active_games"] == 100
            assert result["execution"]["shared_slots"] == 2 and result["execution"]["pipeline_slots"] == 1
            assert result["clock"]["owner_measured_seconds"] >= 35
            for compile_stats in result["worker_compile"]:
                assert not compile_stats["compile_failures"] and compile_stats["compile_failure_fallbacks"] == 0
            shutil.copyfile(path, ART / path.name)
            ledger = RUNTIME / "recording" / panel / "recording"
            for folder in ("sessions", "identities"):
                for identity in (ledger / folder).glob("*.json"):
                    target = ART / "recording" / panel / folder / identity.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(identity, target)
            clock = result["clock"]
            windows.append({"case": case, "repeat": rep, "order": result["screening_order"],
                            "simulations_per_second": clock["rates"]["simulations_per_second"],
                            "searchable_decisions_per_second_estimate": clock["rates"]["simulations_per_second"] / s,
                            "completed_decisions_per_second": sum(w["root_decisions"] / w["measured_seconds"] for w in result["workers"]),
                            "network_positions_per_second": clock["rates"]["network_positions_per_second"],
                            "mean_batch_size": clock["rates"]["mean_batch_size"],
                            "owner_window_seconds": clock["owner_measured_seconds"],
                            "common_seconds": clock["common_seconds"], "process_seconds": result["process_seconds"],
                            "gpu_owner_compile": result["worker_compile"]})
    base = {w["repeat"]: w["searchable_decisions_per_second_estimate"] for w in windows if w["case"] == "s1024_n1"}
    grouped = []
    for s, n in CASES:
        case = name(s, n)
        group = [w for w in windows if w["case"] == case]
        speeds = [w["searchable_decisions_per_second_estimate"] for w in group]
        ratios = [w["searchable_decisions_per_second_estimate"] / base[w["repeat"]] for w in group]
        grouped.append({"case": case, "mean_searchable_decisions_per_second_estimate": float(np.mean(speeds)),
                        "min_searchable_decisions_per_second_estimate": min(speeds),
                        "max_searchable_decisions_per_second_estimate": max(speeds),
                        "mean_speed_ratio_vs_s1024_n1": float(np.mean(ratios)),
                        "ratio_range": [min(ratios), max(ratios)],
                        "mean_batch_size": float(np.mean([w["mean_batch_size"] for w in group]))})
    write_json(ART / "cost_comparison.json", {"windows": windows, "cases": grouped,
               "scope": "two 35s windows per case after graph warmup and 10s real-game warmup; order reversed for repeat 2; 100 active games",
               "limitations": "simulations/S estimates searchable decision throughput, not completed games/s or full-round wall time; opening/early-game only"})
    manifest = read_json(ART / "manifest.json")
    manifest["window_command"] = "conda run --no-capture-output -n splendor env PYTHONPATH=src python -u artifacts/entity_attention_search_screen_20261005/windows.py --seconds 35 --warmup-seconds 10"
    manifest["window_script"] = record(ART / "windows.py")
    manifest["analysis_script"] = record(Path(__file__))
    manifest["status"] = "measurements_complete"
    manifest["outputs"] = [record(p) for p in sorted(ART.rglob("*")) if p.is_file() and p.name != "manifest.json" and "__pycache__" not in p.parts]
    write_json(ART / "manifest.json", manifest)
    print(json.dumps(grouped, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "bank", "roots", "digest", "costs"))
    args = parser.parse_args()
    {"prepare": prepare, "bank": bank, "roots": roots, "digest": digest, "costs": costs}[args.phase]()
