"""Combine the scale-up pieces into one results file.

Reads:
* `nodes/` (this artifact) for the 2.5e-5 arm stored-target diagnostics;
* the published `entity_attention_policy_history_20261009/nodes/` A/B/C series;
* per-game holdout rows in both runtime directories for paired intervals;
* `expanded_summary.json`, `fresh_summary.json`, `battle_summary.json` when present.

Usage: python artifacts/entity_attention_scaleup_20261009/analyze.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import common

PH = common.hist_module()
DELTA = common.delta_module()
ARM = "A_lr2.5e-5"


def game_kl(row):
    return (row["policy_ce_sum"] - row["target_entropy_sum"]) / row["policy_weight_sum"]


def game_value(row):
    return row["value_sq_sum"] / row["value_weight_sum"]


def load_games(path: Path):
    rows = common.read_json(path)
    return {(row["round"], row["game_id"]): row for row in rows}


def node_table():
    table = {}
    for key in common.KEYS:
        table[key] = {"A": {}, ARM: {}}
        for step in common.LR_NODES:
            for arm, path in (("A", PH.ART / "nodes" / f"{key}_A_{step:06d}.json"),
                              (ARM, common.ART / "nodes" / f"{key}_{ARM}_{step:06d}.json")):
                if path.exists():
                    node = common.read_json(path)
                    table[key][arm][step] = {
                        "latest_train_kl": node["latest"]["train_fixed_subset"]["policy_kl"],
                        "latest_holdout_kl": node["latest"]["holdout_full"]["policy_kl"],
                        "latest_holdout_value_mse": node["latest"]["holdout_full"]["value_loss"],
                        "recent_train_kl": node["recent_four"]["train_fixed_subset"]["policy_kl"],
                        "recent_holdout_kl": node["recent_four"]["holdout_full"]["policy_kl"],
                    }
    return table


def paired_vs_A(step):
    values, clusters, strata = [], [], []
    per_key = {}
    for key in common.KEYS:
        a_path = PH.RUNTIME / key / "A" / f"holdout_games_{step:06d}.json"
        b_path = common.LR_RUNTIME / key / f"holdout_games_{step:06d}.json"
        if not (a_path.exists() and b_path.exists()):
            continue
        a_rows, b_rows = load_games(a_path), load_games(b_path)
        if set(a_rows) != set(b_rows):
            raise ValueError(f"{key}@{step}: holdout games differ")
        local = []
        for identity, row in sorted(a_rows.items()):
            delta = game_kl(b_rows[identity]) - game_kl(row)
            values.append(delta)
            clusters.append(f"{key}:{row['round']}:{row['game_id']}")
            strata.append(key)
            local.append(delta)
        per_key[key] = {"games": len(local), "estimate": float(np.mean(local))}
    if not values:
        return None
    boot = DELTA.bootstrap_mean(values, clusters, DELTA.BOOTSTRAP_RESAMPLES,
                                DELTA.BOOTSTRAP_SEED, strata=strata)
    return {**boot, "games": len(clusters),
            "scope": "recent_four_holdout", "weighting": "equal_game",
            "cluster": "start_key:teacher_round:game_id", "strata": "start_key",
            "improved_fraction": float(np.mean(np.asarray(values) < 0)),
            "per_key": per_key}


def fresh_validation_vs_start():
    """Pool net changes on the latest whole-game holdout, not differences to A."""
    rows = [json.loads(line) for line in
            (common.LR_RUNTIME / "fresh_rows.jsonl").read_text().splitlines()]
    rows = [row for row in rows
            if row["tag"] == "train" and row["set"] == "validation"]
    starts = {(row["key"], row["game_id"], row["decision_index"]): row["kl_fresh"]
              for row in rows if row["arm"] == "start"}
    groups = defaultdict(list)
    for row in rows:
        if row["arm"] != "start":
            groups[(row["arm"], row["step"])].append(row)
    arms = {}
    for (arm, step), selected in sorted(groups.items()):
        values = [row["kl_fresh"] - starts[
            (row["key"], row["game_id"], row["decision_index"])] for row in selected]
        clusters = [f"{row['key']}:{row['game_id']}" for row in selected]
        strata = [row["key"] for row in selected]
        boot = DELTA.bootstrap_mean(values, clusters, DELTA.BOOTSTRAP_RESAMPLES,
                                    DELTA.BOOTSTRAP_SEED, strata=strata)
        arms.setdefault(arm, {})[step] = {
            **boot, "positions": len(values),
            "per_key": {key: float(np.mean([v for v, s in zip(values, strata) if s == key]))
                        for key in sorted(set(strata))},
        }
    return {"scope": "latest_validation", "metric": "KL(fresh || student) after - start",
            "weighting": "equal_position", "cluster": "start_key:game_id",
            "strata": "start_key", "arms": arms}


def compact(boot):
    if not boot:
        return None
    return {key: boot[key] for key in
            ("estimate", "lower", "upper", "positions", "games",
             "improved_fraction", "progress") if key in boot}


def expanded_digest():
    path = common.ART / "expanded_summary.json"
    if not path.exists():
        return None
    payload = common.read_json(path)
    return {
        "sets": {kind: compact(entry) for kind, entry in payload["sets"].items()},
        "rounds": {name: {kind: compact(entry) for kind, entry in sets.items()}
                   for name, sets in payload["rounds"].items()},
        "reproduction_of_published": payload["reproduction_of_published"],
    }


def fresh_digest():
    path = common.ART / "fresh_summary.json"
    if not path.exists():
        return None
    payload = common.read_json(path)
    means = {tag: {name: {k: entry[k] for k in ("positions", "mean_kl", "top1_agree")}
                   for name, entry in entries.items()}
             for tag, entries in payload["sets"].items()}
    deltas = {tag: {name: compact(entry)
                    for name, entry in entries.items()}
              for tag, entries in payload["deltas_vs_start"].items()}
    comparisons = {tag: {name: compact(entry)
                         for name, entry in entries.items()}
                   for tag, entries in payload["paired_arm_comparisons"].items()}
    by_set = {tag: {split: {
                        "positions": view["positions"],
                        "deltas_vs_start": {name: compact(entry)
                                            for name, entry in view["deltas_vs_start"].items()
                                            if entry},
                        "paired_arm_comparisons": {name: compact(entry)
                                                   for name, entry in view[
                                                       "paired_arm_comparisons"].items()
                                                   if entry},
                    } for split, view in splits.items()}
              for tag, splits in payload.get("by_set", {}).items()}
    return {"mean_kl": means, "deltas_vs_start": deltas,
            "paired_arm_comparisons": comparisons, "by_set": by_set}


def fresh_stored_digest():
    path = common.ART / "fresh_stored_summary.json"
    if not path.exists():
        return None
    payload = common.read_json(path)
    output = {}
    for tag, entries in payload["deltas_vs_start"].items():
        output[tag] = {}
        for name, entry in entries.items():
            output[tag][name] = {
                "step": entry["step"], "arm": entry["arm"],
                "delta_stored": compact(entry["delta_stored"]),
                "delta_fresh": compact(entry["delta_fresh"]),
                "by_set": {split: {"positions": view["positions"],
                                   "delta_stored": compact(view["delta_stored"]),
                                   "delta_fresh": compact(view["delta_fresh"])}
                           for split, view in entry.get("by_set", {}).items()},
            }
    return output


def battle_digest():
    path = common.ART / "battle_summary.json"
    if not path.exists():
        return None
    payload = common.read_json(path)
    return {"pooled": payload["pooled"], "panels": payload["panels"],
            "matched_contrasts": payload.get("matched_contrasts", {})}


def main():
    output = {
        "protocol": "entity-attention-scaleup-results-v1",
        "client_date": "2026-10-09",
        "stored_nodes": node_table(),
        "stored_paired_lr_minus_A": {step: paired_vs_A(step)
                                     for step in (50, 200, 500)},
        "fresh_validation_delta_vs_start": fresh_validation_vs_start(),
        "expanded_roots": expanded_digest(),
        "fresh_labels": fresh_digest(),
        "fresh_labels_same_position": fresh_stored_digest(),
        "battles": battle_digest(),
        "source": common.git_identity([Path(__file__), common.ART / "common.py"]),
    }
    common.write_json(common.ART / "results.json", output)
    print(json.dumps({"paired": output["stored_paired_lr_minus_A"]}, ensure_ascii=False,
                     indent=2)[:2000])


if __name__ == "__main__":
    main()
