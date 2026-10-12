"""Scale-up 4: extend the behavior+M4 strength panels.

Two operations:

* `make-config` writes extension configs for round-4 / round-8 checkpoints with
  +200 fresh pairs (new game seeds 26500100+, policy seeds 26600200+, disjoint
  from every published panel which used 26500000+ / 26600000+).
* `analyze` merges the published 100-pair panels with their extension panels and
  bootstraps the pooled pair-level score, keeping panel strata fixed.

Usage (from the repository root):

    python artifacts/entity_attention_scaleup_20261009/battle_extend.py make-config
    python artifacts/entity_attention_scaleup_20261009/battle_extend.py analyze
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import common

EXT_PAIRS = 200
EXT_SEED_START = 26500100
EXT_POLICY_BASE = 26600200
EXT_BOOTSTRAP_SEED = 26800200
PUBLISHED = {
    "r004": common.BEHAVIOR_DIR / "evaluations/seed1_avg4_r004.json",
    "r008": common.BEHAVIOR_DIR / "evaluations/seed1_avg4_r008.json",
    "r012": common.BEHAVIOR_DIR / "evaluations/seed1_avg4_r012.json",
}
EXTENSIONS = {
    "r004": common.ART / "battles/seed1_avg4_r004_ext200.json",
    "r008": common.ART / "battles/seed1_avg4_r008_ext200.json",
    "r012": common.ART / "battles/seed1_avg4_r012_ext200.json",
}
CONFIG_DIR = common.ART / "battles/configs"


def emit(event, **values):
    print(json.dumps({"event": event, **values}, ensure_ascii=False), flush=True)


def make_configs():
    for key, target in EXTENSIONS.items():
        source = common.BEHAVIOR_DIR / f"evaluation_configs/seed1_avg4_{key}.json"
        config = common.read_json(source)
        config["pairs"] = EXT_PAIRS
        config["seed_start"] = EXT_SEED_START
        config["policy_seed_base"] = EXT_POLICY_BASE
        config["bootstrap_seed"] = EXT_BOOTSTRAP_SEED
        config["metadata"] = {
            "scaleup_job": {
                "source_config": str(source.relative_to(common.ROOT)),
                "id": f"seed1_avg4_{key}_ext{EXT_PAIRS}",
                "pairs": EXT_PAIRS,
                "note": "disjoint seed domain extension of the published 100-pair panel",
            }}
        config_path = CONFIG_DIR / target.name
        if config_path.exists() and common.read_json(config_path) != config:
            raise ValueError(f"{config_path} exists with different content")
        common.write_json(config_path, config)
        emit("config", path=str(config_path.relative_to(common.ROOT)), pairs=EXT_PAIRS)


def pair_score(games):
    score = 0.0
    valid = 0
    for game in games:
        if not game.get("normal_end"):
            raise ValueError(f"non-normal end in pair: {game.get('error_type')}")
        outcome = game["outcome_for_a"]
        score += {"a_win": 1.0, "draw": 0.5, "a_loss": 0.0}[outcome]
        valid += 1
    return score / valid


def panel_scores(path: Path):
    data = common.read_json(path)
    if not data.get("passed") or data["completed_pairs"] != data["total_pairs"]:
        raise ValueError(f"{path}: panel incomplete (passed={data.get('passed')}, "
                         f"{data.get('completed_pairs')}/{data.get('total_pairs')})")
    return {int(pair["game_seed"]): pair_score(pair["games"]) for pair in data["pairs"]}


def matched_contrast(scores_a, scores_b, label_a, label_b, rng_seed):
    """Paired score difference on shared game seeds (b - a), bootstrap by pair."""
    shared = sorted(set(scores_a) & set(scores_b))
    if not shared:
        return None
    differences = np.asarray([scores_b[seed] - scores_a[seed] for seed in shared])
    rng = np.random.default_rng(rng_seed)
    chosen = rng.integers(0, len(differences), size=(10000, len(differences)))
    means = differences[chosen].mean(axis=1)
    return {"comparison": f"{label_b} - {label_a}", "pairs": len(shared),
            "estimate": float(differences.mean()),
            "lower": float(np.percentile(means, 2.5)),
            "upper": float(np.percentile(means, 97.5)),
            "confidence": 0.95, "resamples": 10000, "seed": rng_seed}


def analyze(confidence=0.95, resamples=10000, seed=20261013):
    output = {"protocol": "avg4-scaleup-battle-extension-v1",
              "client_date": "2026-10-09", "panels": {}, "pooled": {},
              "matched_contrasts": {"published": {}, "extension": {}}}
    published_scores = {}
    for key in PUBLISHED:
        panels = {}
        panels["published100"] = panel_scores(PUBLISHED[key])
        published_scores[key] = panels["published100"]
        output["panels"][f"{key}_published100"] = {
            "pairs": len(panels["published100"]),
            "score": float(np.mean(list(panels["published100"].values()))),
            "source": str(PUBLISHED[key].relative_to(common.ROOT)),
        }
        if EXTENSIONS[key].exists() and EXTENSIONS[key].with_suffix(".json.games").exists():
            try:
                panels["extension200"] = panel_scores(EXTENSIONS[key])
                output["panels"][f"{key}_extension200"] = {
                    "pairs": len(panels["extension200"]),
                    "score": float(np.mean(list(panels["extension200"].values()))),
                    "source": str(EXTENSIONS[key].relative_to(common.ROOT)),
                }
            except ValueError as error:
                output["panels"][f"{key}_extension200"] = {"error": str(error)}
        arrays = np.concatenate([np.asarray(list(v.values()), dtype=np.float64)
                                 for v in panels.values()])
        strata = np.concatenate([np.full(len(v), name) for name, v in panels.items()])
        rng = np.random.default_rng(seed)
        means = np.zeros(resamples)
        for stratum in np.unique(strata):
            mask = strata == stratum
            values = arrays[mask]
            chosen = rng.integers(0, len(values), size=(resamples, len(values)))
            means += (mask.sum() / len(arrays)) * values[chosen].mean(axis=1)
        output["pooled"][key] = {
            "pairs": len(arrays),
            "score": float(arrays.mean()),
            "lower": float(np.percentile(means, 2.5)),
            "upper": float(np.percentile(means, 97.5)),
            "confidence": confidence, "resamples": resamples, "seed": seed,
            "strata": {name: len(v) for name, v in panels.items()},
        }
    keys = sorted(published_scores)
    for i, left in enumerate(keys):
        for right in keys[i + 1:]:
            label = f"{right}_minus_{left}_published100"
            contrast = matched_contrast(published_scores[left], published_scores[right],
                                        left, right, seed + i * 100 + keys.index(right))
            if contrast:
                output["matched_contrasts"]["published"][label] = contrast
    extension_scores = {}
    for key in keys:
        path = EXTENSIONS[key]
        if path.exists() and path.with_suffix(".json.games").exists():
            try:
                extension_scores[key] = panel_scores(path)
            except ValueError:
                continue
    extensions = sorted(extension_scores)
    for i, left in enumerate(extensions):
        for right in extensions[i + 1:]:
            label = f"{right}_minus_{left}_ext200"
            contrast = matched_contrast(extension_scores[left], extension_scores[right],
                                        left, right, seed + 5000 + i * 100 + extensions.index(right))
            if contrast:
                output["matched_contrasts"]["extension"][label] = contrast
    common.write_json(common.ART / "battle_summary.json", output)
    print(json.dumps(output["pooled"], ensure_ascii=False))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("make-config", "analyze"))
    args = parser.parse_args()
    if args.action == "make-config":
        make_configs()
    else:
        analyze()


if __name__ == "__main__":
    main()
