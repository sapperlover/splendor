"""Capture and recompute the completed run's report evidence; never run models."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
RUN = REPO / "runs/entity_attention_p4_pool_iteration_20261010"
EVIDENCE = HERE / "completed_evidence.json.gz"
ANALYSIS = HERE / "completed_analysis.json"
RESAMPLES = 10000
SEED_BASE = 314165
METRICS = ("positions", "policy_weight", "policy_kl", "value_mse", "value_correlation",
           "teacher_top1_agreement", "target_entropy", "policy_entropy", "value_bias")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seed(label):
    return SEED_BASE ^ int(hashlib.sha256(label.encode()).hexdigest()[:8], 16)


def capture():
    hashes = {}

    def read(path):
        hashes[str(path.relative_to(REPO))] = sha(path)
        return json.loads(path.read_text())

    complete = read(RUN / "complete.json")
    state = read(RUN / "hybrid_state.json")
    assert state["completed_rounds"] == list(range(9))
    assert complete["checkpoint"] == state["latest_checkpoint"]
    for identity in [state["anchor_checkpoint"], complete["checkpoint"], complete["control"]]:
        assert sha(Path(identity["path"])) == identity["sha256"]
    plan = read(RUN / "plan.json")
    # The completion marker hashes the original CLI config, not its canonical
    # JSON reserialization written to RUN/plan.json.
    plan_source = REPO / "configs/entity_attention_v1/p4_pool_iteration_20261010.json"
    assert read(plan_source) == plan and sha(plan_source) == complete["plan_sha256"]
    config = read(RUN / "config.json")
    source_path, = (RUN / "source").glob("*/identity.json")
    source = read(source_path)
    rounds, battles = [], {}
    counts = ("completed_games", "samples", "policy_samples", "draws", "failed_games",
              "exceptions", "engineering_truncations", "wall_seconds", "cpu_peak_rss_kib",
              "gpu_peak_bytes", "simulations", "data_bytes")

    def add_battle(path):
        d = read(path)
        assert d["passed"] and d["completed_pairs"] == d["total_pairs"] == 200
        for name in ("exceptions", "illegal_actions", "invariant_failures", "engineering_truncations"):
            assert d[name] == 0
        pair_scores = []
        for pair in d["pairs"]:
            assert len(pair["games"]) == 2
            assert {g["first_player"] for g in pair["games"]} == {0, 1}
            scores = [{"a_win": 1., "a_loss": 0., "draw": .5}[g["outcome_for_a"]]
                      for g in pair["games"]]
            pair_scores.append([pair["pair_id"], pair["game_seed"], *scores])
        assert abs(np.mean([np.mean(p[2:]) for p in pair_scores]) - d["win_rate_score"]) < 1e-12
        is_search = d["config"]["policy_a"]["kind"] == "mcts_checkpoint"
        if is_search:
            for side in ("a", "b"):
                cfg = d["config"][f"policy_{side}"]["search_config"]
                assert cfg["simulations"] == 1024 and cfg["determinizations"] == 1
                stats = d[f"policy_{side}_stats"]
                assert stats["simulations"] == 1024 * stats["searchable_decisions"]
                assert stats["determinizations"] == stats["searchable_decisions"]
                assert stats["transition_cap_hits"] == 0
        keep = ("win_rate_score", "paired_bootstrap", "a_wins", "a_losses", "draws", "a_first",
                "a_second", "policy_cycle_draws", "checkpoint_identity", "config",
                "policy_a_stats", "policy_b_stats", "elapsed_seconds_this_invocation", "updated_at")
        battles[str(path.relative_to(RUN))] = {k: d[k] for k in keep}
        battles[str(path.relative_to(RUN))]["pair_scores"] = pair_scores

    for i in range(9):
        stage = RUN / "rounds" / f"round-{i:03d}"
        collection = read(stage / "selfplay/summary.json")
        assert collection["completed_games"] == 1000 and collection["missing_games"] == 0
        assert all(collection[k] == 0 for k in ("failed_games", "exceptions", "engineering_truncations"))
        snapshot = read(stage / "replay_snapshot.json")
        games = [g for g in snapshot["games"] if g["round_index"] == i]
        assert len(games) == 1000
        split = {s: {"games": sum(g["split"] == s for g in games),
                     "positions": sum(g["decision_count"] for g in games if g["split"] == s),
                     "policy_positions": sum(g["policy_samples"] for g in games if g["split"] == s)}
                 for s in ("train", "validation")}
        points = []
        for step in (0, 100, 250, 500):
            d = read(stage / "fitting" / f"step-{step:09d}.json")
            point = {"step": step, "checkpoint_sha256": d["checkpoint_sha256"],
                     "offline_fixed": d["offline_fixed"]["pool_weighted"], "seconds": d["seconds"]}
            for scope in ("online_train_fixed", "online_validation_full"):
                point[scope] = {key: {k: row[k] for k in METRICS}
                                for key, row in d[scope]["metrics"].items()}
            if step in (0, 500):
                point["per_game"] = {key: {k: row[k] for k in ("kl", "policy_weight", "sq_error", "positions")}
                                     for key, row in d["online_validation_full"]["per_game"].items()
                                     if key.endswith("/all")}
                whole = point["online_validation_full"]["__all_observed__/validation/all"]
                for metric, numerator, denominator in (("policy_kl", "kl", "policy_weight"),
                                                        ("value_mse", "sq_error", "positions")):
                    ratio = sum(v[numerator] for v in point["per_game"].values()) / sum(v[denominator] for v in point["per_game"].values())
                    assert abs(ratio-whole[metric]) < 1e-12
            points.append(point)
        metrics_path = stage / "training_metrics.jsonl"
        hashes[str(metrics_path.relative_to(REPO))] = sha(metrics_path)
        rows = [json.loads(line) for line in metrics_path.read_text().splitlines()]
        train = [row for row in rows if row["event"] == "train_step"]
        assert len(train) == 500 and train[-1]["round_train_step"] == 500
        usage = read(stage / "source_usage.json")
        assert sum(usage[k]["draws"] for k in ("offline", "online")) == 500*256
        timings = {"collection_start": (stage / "selfplay/manifest.json").stat().st_mtime,
                   "training_initial": (stage / "checkpoints/initial.pt").stat().st_mtime,
                   "fitting_end": (stage / "fitting/step-000000500.json").stat().st_mtime,
                   "round_end": (stage / "evaluation/greedy_vs_start.json").stat().st_mtime}
        rounds.append({"round": i + 1, "split": split, "collection": {k: collection[k] for k in counts},
                       "usage": usage, "points": points, "timings": timings,
                       "cache": train[-1]["replay_cache"],
                       "training": {"steps": len(train),
                                    "clipped_steps": sum(row["grad_clipped"] for row in train),
                                    "mean_grad_norm": float(np.mean([row["grad_norm"] for row in train]))}})
        add_battle(stage / "evaluation/greedy_vs_start.json")
        if (i + 1) % 3 == 0:
            add_battle(stage / "evaluation/mcts_vs_fixed_anchor.json")
    for name in ("online_vs_offline_control", "online_vs_P12V4"):
        add_battle(RUN / "final_evaluation" / f"{name}.json")
    old_full = {}
    paths = {"P4": REPO / "artifacts/entity_attention_recent_pool_20261009/full_holdout_step_000034398.json",
             "online": RUN / "final_fitting/online_old_pool_full.json",
             "offline_control": RUN / "final_fitting/offline_control_old_pool_full.json"}
    for name, path in paths.items():
        d = read(path)["evaluation"]
        old_full[name] = {"pool_weighted": d["pool_weighted"], "metrics": d["metrics"],
                          "per_game": {k: {n: v[n] for n in ("kl", "policy_weight", "sq_error", "positions")}
                                       for k, v in d["per_game"].items() if k.endswith("/all")}}
    control = [read(p) for p in sorted((RUN / "offline_control/fitting").glob("*.json"))]
    times = {"start": source_path.stat().st_mtime, "complete": (RUN / "complete.json").stat().st_mtime,
             "control_initial": (RUN / "offline_control/checkpoints/initial.pt").stat().st_mtime,
             "control_final": (RUN / "offline_control/checkpoints/final.pt").stat().st_mtime}
    result = {"protocol": "p4-iteration-report-evidence-v1", "complete": complete, "state": state,
              "plan": plan, "config": config, "source_identity": source, "input_sha256": hashes,
              "rounds": rounds, "battles": battles, "old_full": old_full,
              "control_fitting": control, "timings": times}
    data = json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    EVIDENCE.write_bytes(gzip.compress(data, mtime=0))


def ratio_change(a, b, label):
    assert a.keys() == b.keys()
    keys = sorted(a)
    rng = np.random.default_rng(seed(label))
    result = {"games": len(keys), "seed": seed(label), "resamples": RESAMPLES}
    for metric, numerator, denominator in (("kl", "kl", "policy_weight"), ("mse", "sq_error", "positions")):
        weights = np.array([a[k][denominator] for k in keys])
        assert all(a[k][denominator] == b[k][denominator] for k in keys)
        diffs = np.array([b[k][numerator] - a[k][numerator] for k in keys])
        bootstrap = []
        # Bound temporary memory even for the complete 2935-game old holdout.
        for start in range(0, RESAMPLES, 256):
            ix = rng.integers(0, len(keys), size=(min(256, RESAMPLES-start), len(keys)))
            bootstrap.extend((diffs[ix].sum(1) / weights[ix].sum(1)).tolist())
        result[metric] = {"delta": float(diffs.sum()/weights.sum()),
                          "ci95": np.quantile(bootstrap, [.025, .975]).tolist()}
    return result


def derive(e):
    rounds = e["rounds"]
    matrices = {name: [[None]*9 for _ in range(9)] for name in ("kl", "mse", "delta_kl", "delta_mse")}
    within, cumulative = {}, {}
    for row in rounds:
        i = row["round"]-1
        a, b = row["points"][0], row["points"][-1]
        for j in range(max(0, i-5), i+1):
            first = rounds[j]["points"][0]["online_validation_full"][f"g{j+1:03d}/validation/all"]
            final = b["online_validation_full"][f"g{j+1:03d}/validation/all"]
            for name, metric in (("kl", "policy_kl"), ("mse", "value_mse")):
                matrices[name][i][j] = final[metric]
                matrices[f"delta_{name}"][i][j] = final[metric]-first[metric]
        within[str(i+1)] = ratio_change({k:v for k,v in a["per_game"].items() if k.startswith(f"{i}:")},
                                      {k:v for k,v in b["per_game"].items() if k.startswith(f"{i}:")}, f"within-{i+1}")
    last = rounds[-1]["points"][-1]
    for j in range(3,9):
        a = {k:v for k,v in rounds[j]["points"][0]["per_game"].items() if k.startswith(f"{j}:")}
        b = {k:v for k,v in last["per_game"].items() if k.startswith(f"{j}:")}
        cumulative[str(j+1)] = within["9"] if j == 8 else ratio_change(a, b, f"cumulative-{j+1}")
    active = ratio_change(rounds[-1]["points"][0]["per_game"], last["per_game"], "r9-active")
    old_changes = {f"{b}-minus-{a}": ratio_change(e["old_full"][a]["per_game"], e["old_full"][b]["per_game"], f"old-{b}-{a}")
                   for a,b in (("P4","online"),("P4","offline_control"),("offline_control","online"))}
    contrasts = {}
    for a,b,mode in ((9,6,"mcts_vs_fixed_anchor"),(9,3,"mcts_vs_fixed_anchor"),
                     (9,7,"greedy_vs_start"),(7,6,"greedy_vs_start"),(8,6,"greedy_vs_start")):
        label = f"r{a}-minus-r{b}-{mode}"
        x = e["battles"][f"rounds/round-{a-1:03d}/evaluation/{mode}.json"]["pair_scores"]
        y = e["battles"][f"rounds/round-{b-1:03d}/evaluation/{mode}.json"]["pair_scores"]
        x,y = sorted(x),sorted(y)
        assert all(p[:2] == q[:2] for p,q in zip(x,y))
        diffs = np.array([np.mean(p[2:])-np.mean(q[2:]) for p,q in zip(x,y)])
        rng = np.random.default_rng(seed(label))
        ix = rng.integers(0,200,size=(RESAMPLES,200))
        contrasts[label] = {"delta":float(diffs.mean()), "ci95":np.quantile(diffs[ix].mean(1),[.025,.975]).tolist(),
                            "seed":seed(label),"resamples":RESAMPLES,"unit":"matched opening pair; scores vs common P4"}
    totals = {split: {key: sum(r["split"][split][key] for r in rounds) for key in ("games","positions","policy_positions")}
              for split in ("train","validation")}
    totals.update({"old_draws":sum(r["usage"]["offline"]["draws"] for r in rounds),
                   "online_draws":sum(r["usage"]["online"]["draws"] for r in rounds),
                   "greedy_pairs":1800,"search_pairs":1000,"evaluation_games":5600})
    times = {"total_seconds":e["timings"]["complete"]-e["timings"]["start"],
             "main_seconds":rounds[-1]["timings"]["round_end"]-e["timings"]["start"],
             "collection_seconds":sum(r["collection"]["wall_seconds"] for r in rounds),
             "main_training_and_fitting_seconds":sum(r["timings"]["fitting_end"]-r["timings"]["training_initial"] for r in rounds),
             "control_initial_to_final_seconds":e["timings"]["control_final"]-e["timings"]["control_initial"],
             "search_evaluation_seconds":sum(b["elapsed_seconds_this_invocation"] for b in e["battles"].values() if b["config"]["policy_a"]["kind"]=="mcts_checkpoint"),
             "greedy_evaluation_seconds":sum(b["elapsed_seconds_this_invocation"] for b in e["battles"].values() if b["config"]["policy_a"]["kind"]=="greedy_checkpoint")}
    return {"protocol":"p4-iteration-completed-analysis-v1","evidence_sha256":sha(EVIDENCE),
            "bootstrap":{"seed_base":SEED_BASE,"seed_rule":"base XOR first 32 bits of SHA256(label)",
                         "resamples":RESAMPLES,"unit":"whole game for fitting; whole pair for battle; conditional DEV intervals, no multiplicity correction"},
            "totals":totals,"timings":times,"matrices":matrices,"within_round":within,
            "cumulative_to_round9":cumulative,"round9_active":active,"old_full_changes":old_changes,
            "battle_contrasts":contrasts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture",action="store_true",help="capture frozen local run evidence before recomputing")
    parser.add_argument("--verify",action="store_true",help="recompute from retained evidence and compare without writing")
    args = parser.parse_args()
    assert not (args.capture and args.verify)
    if args.capture:
        capture()
    evidence = json.loads(gzip.decompress(EVIDENCE.read_bytes()))
    result = derive(evidence)
    if args.verify:
        assert result == json.loads(ANALYSIS.read_text())
        print("completed analysis verified from retained evidence; no training or evaluation")
    else:
        ANALYSIS.write_text(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+"\n")
        print(json.dumps({"status":"complete","rounds":9,"panels":len(evidence['battles']),
                          "evidence_bytes":EVIDENCE.stat().st_size,"analysis":str(ANALYSIS)},ensure_ascii=False))


if __name__ == "__main__":
    main()
