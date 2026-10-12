"""Scale-up 1: expand the two crossing-zero fresh-M4 samples 256 -> 1024 roots.

The published `entity_attention_avg4_delta_kl_20261009` pooled estimates for
`train_ledger` (-0.00367 [-0.01003, +0.00264]) and `train_unseen`
(+0.00415 [-0.00154, +0.00980]) cross zero at 256 roots per round.  This module
adds 768 new roots per round per set with independent sample/search seeds,
keeping teacher, before/after checkpoints, 4 x S1024/N1 budget, root-noise
schedule and cluster-bootstrap protocol unchanged.  The existing 256 roots are
reused unchanged, so the pooled sample is a superset of the published one.

Usage (from the repository root):

    python -u artifacts/entity_attention_scaleup_20261009/expand_roots.py freeze --device cuda
    python -u artifacts/entity_attention_scaleup_20261009/expand_roots.py run --device cuda
    python -u artifacts/entity_attention_scaleup_20261009/expand_roots.py summarize
    python -u artifacts/entity_attention_scaleup_20261009/expand_roots.py smoke
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

import common

DELTA = common.delta_module()

EXPAND_COUNT = 768
SAMPLE_SEED_BASE_EXPAND = 20261012
EXPAND_WORLD_SEED = 26900000
EXPAND_NOISE_SEED = 27000000
KINDS = {
    "ledger": {"set": "train_ledger", "tag": "ledger", "stream": "ledger_expand_sample",
               "label": "same-round Train roots actually drawn by the round policy ledger"},
    "unseen": {"set": "train_unseen", "tag": "unseen", "stream": "unseen_expand_sample",
               "label": "same-round Train roots never drawn by the round policy ledger"},
}
SPEC_PATH = common.ART / "expand_specification.json"


def emit(event, **values):
    print(json.dumps({"event": event, **values}, ensure_ascii=False), flush=True)


def existing_rows(name: str, tag: str):
    path = DELTA.RUNTIME / name / (f"positions_{tag}.jsonl" if tag else "positions.jsonl")
    return [json.loads(line) for line in path.read_text().splitlines()]


def pool_for(name: str, identity: dict, kind: str):
    """Recompute the frozen candidate pool exactly as the source experiment did."""
    current, ledger = DELTA.current_policy_draws(name, identity)
    games = DELTA.load_round_roots(name)
    by_game = {int(game["meta"]["game_id"]): game for game in games}
    if kind == "unseen":
        seen = {(int(game), int(decision)) for _, _, game, decision in current}
        pool = [{"game_id": int(game["meta"]["game_id"]), "decision_index": int(i),
                 "file": game["path"].name, "split": "train_unseen",
                 "round": name, "round_index": int(identity["round_index"])}
                for game in games if str(game["meta"]["split"]) == "train"
                for i in game["roots"] if (int(game["meta"]["game_id"]), int(i)) not in seen]
    else:
        unique = sorted({(int(game_id), int(decision))
                         for _, _, game_id, decision in current})
        pool = []
        for game_id, decision in unique:
            game = by_game.get(game_id)
            if (game is None or str(game["meta"]["split"]) != "train"
                    or decision not in game["roots"]):
                continue
            pool.append({"game_id": game_id, "decision_index": decision,
                         "file": game["path"].name, "split": "train_ledger",
                         "round": name, "round_index": int(identity["round_index"])})
    stats = {"draws_current_round": int(len(current)), "pool": len(pool),
             "ledger": DELTA.record(ledger),
             "game_archives_sha256": DELTA.hashlib.sha256(json.dumps(
                 [(g["path"].name, DELTA.sha256(g["path"])) for g in games],
                 separators=(",", ":")).encode()).hexdigest()}
    return pool, stats


def freeze(device: str):
    spec = DELTA.read_json(DELTA.ART / "specification.json")
    saved = common.read_json(SPEC_PATH) if SPEC_PATH.exists() else None
    rounds = {}
    for name, identity in spec["rounds"].items():
        entry = {"identity": identity, "sets": {}}
        for kind, meta in KINDS.items():
            pool, stats = pool_for(name, identity, kind)
            entry["ledger"] = stats["ledger"]
            entry["game_archives_sha256"] = stats["game_archives_sha256"]
            entry["draws_current_round"] = stats["draws_current_round"]
            rows = existing_rows(name, meta["tag"])
            existing = [{"game_id": int(row["game_id"]), "decision_index": int(row["decision_index"])}
                        for row in rows]
            existing_keys = {(ref["game_id"], ref["decision_index"]) for ref in existing}
            pool_keys = {(ref["game_id"], ref["decision_index"]) for ref in pool}
            if not existing_keys <= pool_keys:
                raise ValueError(f"{name}/{kind}: published roots absent from recomputed pool")
            remaining = [ref for ref in pool
                         if (ref["game_id"], ref["decision_index"]) not in existing_keys]
            rng = np.random.default_rng(DELTA.derive_seed(
                SAMPLE_SEED_BASE_EXPAND, identity["round_index"], 0, meta["stream"]))
            count = min(EXPAND_COUNT, len(remaining))
            chosen = sorted(int(i) for i in rng.choice(len(remaining), size=count, replace=False))
            refs = [{**{k: ref[k] for k in ("game_id", "decision_index", "file", "split")}}
                    for ref in (remaining[i] for i in chosen)]
            entry["sets"][kind] = {
                "set": meta["set"], "tag": meta["tag"], "label": meta["label"],
                "published": len(existing), "pool_size": len(pool),
                "available_after_published": len(remaining), "new_roots": len(refs),
                "sampled_games": len({ref["game_id"] for ref in refs}),
                "sample_seed": DELTA.derive_seed(SAMPLE_SEED_BASE_EXPAND,
                                                 identity["round_index"], 0, meta["stream"]),
                "refs": refs,
            }
        rounds[name] = entry
    payload = {
        "protocol": "avg4-scaleup-expanded-roots-v1",
        "client_date": "2026-10-09",
        "published_source": str(DELTA.ART.relative_to(common.ROOT)),
        "published_per_round": DELTA.SAMPLES_PER_SET,
        "expansion_per_round": EXPAND_COUNT,
        "fresh_samples": DELTA.FRESH_SAMPLES,
        "expand_world_seed": EXPAND_WORLD_SEED, "expand_noise_seed": EXPAND_NOISE_SEED,
        "sample_seed_base": SAMPLE_SEED_BASE_EXPAND,
        "bootstrap": {"cluster": "source game", "strata": "round",
                      "resamples": DELTA.BOOTSTRAP_RESAMPLES, "seed": DELTA.BOOTSTRAP_SEED},
        "source": common.git_identity([Path(__file__), common.ART / "common.py"]),
        "execution": {"device": device}, "rounds": rounds,
    }
    if saved is not None:
        ignore = {"source", "execution"}
        if {k: v for k, v in saved.items() if k not in ignore} != {
                k: v for k, v in payload.items() if k not in ignore}:
            raise ValueError("expansion specification changed; use a new experiment")
    if saved is None:
        common.write_json(SPEC_PATH, payload)
    emit("frozen", rounds=list(rounds))
    return payload


def run(device: str, kinds=("ledger", "unseen")):
    spec = common.read_json(SPEC_PATH)
    DELTA.FRESH_WORLD_SEED = EXPAND_WORLD_SEED
    DELTA.FRESH_NOISE_SEED = EXPAND_NOISE_SEED
    for name, entry in spec["rounds"].items():
        for kind in kinds:
            meta = KINDS[kind]
            info = entry["sets"][kind]
            refs = [{**ref, "split": info["set"]} for ref in info["refs"]]
            emit("expand_search_start", round=name, kind=kind, roots=len(refs))
            DELTA.search_round(name, entry["identity"], refs, device, tag=f"{meta['tag']}_expand")
            DELTA.evaluate_round(name, entry["identity"], refs, device, tag=f"{meta['tag']}_expand")
            emit("expand_done", round=name, kind=kind)


def _stats(rows):
    if not rows:
        return {"positions": 0}
    delta_values = [r["kl_fresh_after"] - r["kl_fresh_before"] for r in rows]
    clusters = [f"{r['round']}:{r['game_id']}" for r in rows]
    strata = [r["round"] for r in rows]
    boot = DELTA.bootstrap_mean(delta_values, clusters, DELTA.BOOTSTRAP_RESAMPLES,
                                DELTA.BOOTSTRAP_SEED, strata=strata)
    return {
        **boot, "positions": len(rows),
        "games": len({c for c in clusters}),
        "improved_fraction": float(np.mean([d < 0 for d in delta_values])),
        "median_delta": float(np.median(delta_values)),
        "mean_before": float(np.mean([r["kl_fresh_before"] for r in rows])),
        "mean_after": float(np.mean([r["kl_fresh_after"] for r in rows])),
        "top1_before": float(np.mean([r["top1_fresh_before"] for r in rows])),
        "top1_after": float(np.mean([r["top1_fresh_after"] for r in rows])),
    }


def summarize(write: bool = True):
    output = {"protocol": "avg4-scaleup-expanded-roots-summary-v1",
              "client_date": "2026-10-09",
              "bootstrap": {"cluster": "source game", "strata": "round",
                            "resamples": DELTA.BOOTSTRAP_RESAMPLES, "seed": DELTA.BOOTSTRAP_SEED},
              "sets": {}, "rounds": {}, "reproduction_of_published": {}}
    published_summary = common.read_json(DELTA.ART / "summary.json")
    for kind, meta in KINDS.items():
        rows = []
        for name in DELTA.ROUNDS:
            rows += existing_rows(name, meta["tag"])
            rows += existing_rows(name, f"{meta['tag']}_expand")
        output["sets"][kind] = {"set": meta["set"], "positions": len(rows), **_stats(rows)}
        for name in DELTA.ROUNDS:
            round_rows = [r for r in rows if r["round"] == name]
            output["rounds"][name] = output["rounds"].get(name, {})
            output["rounds"][name][kind] = _stats(round_rows)
        # Reproduce the published 256-root pooled estimate as a merge check.
        published_rows = [r for name in DELTA.ROUNDS for r in existing_rows(name, meta["tag"])]
        got = _stats(published_rows)
        want = published_summary["pooled"][meta["set"]]
        output["reproduction_of_published"][kind] = {
            "positions": len(published_rows),
            "estimate": got["estimate"], "published_estimate": want["estimate"],
            "max_abs_diff": max(abs(got["estimate"] - want["estimate"]),
                                abs(got["lower"] - want["lower"]),
                                abs(got["upper"] - want["upper"])),
        }
    if write:
        common.write_json(common.ART / "expanded_summary.json", output)
    return output


def smoke(device="cpu"):
    """Small-budget CPU check in an isolated runtime directory."""
    spec = common.read_json(SPEC_PATH) if SPEC_PATH.exists() else freeze("cpu")
    DELTA.FRESH_WORLD_SEED = EXPAND_WORLD_SEED
    DELTA.FRESH_NOISE_SEED = EXPAND_NOISE_SEED
    DELTA.RUNTIME = DELTA.RUNTIME / "smoke-expand"
    name = "r004"
    entry = spec["rounds"][name]
    for kind in ("ledger", "unseen"):
        info = entry["sets"][kind]
        refs = [{**ref, "split": info["set"]} for ref in info["refs"][:2]]
        DELTA.search_round(name, entry["identity"], refs, device,
                           simulations_override=16, tag=f"{kind}_smoke")
        DELTA.evaluate_round(name, entry["identity"], refs, device, tag=f"{kind}_smoke")
    emit("smoke_done")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run", "summarize", "smoke"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--kinds", nargs="+", choices=tuple(KINDS), default=tuple(KINDS))
    args = parser.parse_args()
    if args.action == "freeze":
        freeze(args.device)
    elif args.action == "run":
        run(args.device, kinds=tuple(args.kinds))
        summarize()
    elif args.action == "summarize":
        print(json.dumps(summarize(), ensure_ascii=False)[:2000])
    else:
        smoke()


if __name__ == "__main__":
    main()
