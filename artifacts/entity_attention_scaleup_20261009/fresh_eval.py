"""Scale-up 3: score every fixed-data checkpoint against the fresh M4 labels.

The published A/B/C comparison only used the stored behavior+3 M4 supervision,
and the round-12 `C@500` and `B-vs-A` statements were explicitly left
"not independently verified".  This module evaluates the published A/B/C nodes
and the new 2.5e-5 arm on the *fresh* M4 targets of
`entity_attention_avg4_delta_kl_20261009` (train / validation / ledger / unseen
plus the 1024-root expansions when present), using the identical paired
Delta-KL protocol: KL(fresh_M4 || student) before/after on the same positions,
cluster bootstrap by source game.

Usage (from the repository root):

    python -u artifacts/entity_attention_scaleup_20261009/fresh_eval.py freeze
    python -u artifacts/entity_attention_scaleup_20261009/fresh_eval.py evaluate --device cuda
    python -u artifacts/entity_attention_scaleup_20261009/fresh_eval.py summarize
    python -u artifacts/entity_attention_scaleup_20261009/fresh_eval.py smoke
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import common

DELTA = common.delta_module()
PH = common.hist_module()
SPEC_PATH = common.ART / "fresh_eval_specification.json"
ROWS_PATH = common.LR_RUNTIME / "fresh_rows.jsonl"
ROWS_BOTH_PATH = common.LR_RUNTIME / "fresh_rows_both.jsonl"

TAGS = (None, "ledger", "unseen", "ledger_expand", "unseen_expand")


def emit(event, **values):
    print(json.dumps({"event": event, **values}, ensure_ascii=False), flush=True)


def rows_path(key: str, tag):
    name = "positions.jsonl" if tag is None else f"positions_{tag}.jsonl"
    return DELTA.RUNTIME / key / name


def has_tag(key: str, tag) -> bool:
    return rows_path(key, tag).exists()


def default_checkpoints():
    """Published A/B/C nodes plus the new 2.5e-5 arm; step 0 is the shared start."""
    spec = common.read_json(PH.ART / "specification.json")
    checkpoints = {}
    for key in common.KEYS:
        start = Path(spec["subjects"][key]["start"]["path"])
        checkpoints[f"{key}|start|000000"] = {
            "key": key, "arm": "start", "step": 0, "path": str(start)}
        for arm in ("A", "B", "C"):
            for step in (50, 200, 500):
                path = common.ROOT / f"runs/entity_attention_policy_history_20261009/{key}/{arm}/checkpoints/step-{step:06d}.pt"
                if path.exists():
                    checkpoints[f"{key}|{arm}|{step:06d}"] = {
                        "key": key, "arm": arm, "step": step, "path": str(path)}
        for step in (50, 100, 200, 500):
            path = common.LR_RUN / key / "checkpoints" / f"step-{step:06d}.pt"
            if path.exists():
                checkpoints[f"{key}|lr2.5e-5|{step:06d}"] = {
                    "key": key, "arm": "lr2.5e-5", "step": step, "path": str(path)}
    return checkpoints


def freeze():
    checkpoints = default_checkpoints()
    payload = {
        "protocol": "avg4-scaleup-fresh-eval-v1",
        "client_date": "2026-10-09",
        "fresh_source": str(DELTA.ART.relative_to(common.ROOT)),
        "metric": "KL(fresh_mean_M4 || student) per position; paired before/after on one label",
        "checkpoints": {name: {**info, "sha256": common.sha256(Path(info["path"]))}
                        for name, info in checkpoints.items()},
        "source": common.git_identity([Path(__file__), common.ART / "common.py"]),
    }
    if SPEC_PATH.exists():
        saved = common.read_json(SPEC_PATH)
        saved_wo = {k: v for k, v in saved.items() if k != "source"}
        payload_wo = {k: v for k, v in payload.items() if k != "source"}
        if saved_wo["checkpoints"].keys() != payload_wo["checkpoints"].keys():
            raise ValueError("checkpoint set changed; use a new experiment")
        if any(saved_wo["checkpoints"][k]["sha256"] != payload_wo["checkpoints"][k]["sha256"]
               for k in saved_wo["checkpoints"]):
            raise ValueError("checkpoint content changed; use a new experiment")
    else:
        common.write_json(SPEC_PATH, payload)
    emit("frozen", checkpoints=len(payload["checkpoints"]))
    return payload


_game_cache: dict[str, dict[int, dict]] = {}


def requests_for(key: str, refs):
    if key not in _game_cache:
        games = DELTA.load_round_roots(key)
        _game_cache[key] = {int(game["meta"]["game_id"]): game for game in games}
    by_game = _game_cache[key]
    return [by_game[int(ref["game_id"])]["roots"][int(ref["decision_index"])]
            for ref in refs]


_stored_cache: dict[tuple[str, str], object] = {}


def stored_targets(key: str, refs):
    """Stored behavior+3 M4 target for the same positions (delta-run slicing)."""
    if key not in _game_cache:
        requests_for(key, [])
    by_game = _game_cache[key]
    stage = DELTA.round_stage(key)
    out = []
    for ref in refs:
        game = by_game[int(ref["game_id"])]
        file = game["path"].name
        cache_key = (key, file)
        if cache_key not in _stored_cache:
            with np.load(stage / "policy_labels/games" / file, allow_pickle=False) as archive:
                _stored_cache[cache_key] = np.asarray(archive["target_mean"], dtype=np.float64)
        target = _stored_cache[cache_key]
        offsets = game["arrays"]["candidate_offsets"]
        decision = int(ref["decision_index"])
        a, b = offsets[decision], offsets[decision + 1]
        out.append(target[a:b])
    return out


def evaluate_both(device: str, batch: int = 128):
    """Re-evaluate every checkpoint with fresh and stored labels on one pass."""
    spec = common.read_json(SPEC_PATH)
    common.LR_RUNTIME.mkdir(parents=True, exist_ok=True)
    progress_path = ROWS_BOTH_PATH.with_name("fresh_progress_both.json")
    done = common.read_json(progress_path) if progress_path.exists() else {}
    with ROWS_BOTH_PATH.open("a" if ROWS_BOTH_PATH.exists() else "w") as handle:
        for name, info in spec["checkpoints"].items():
            key = info["key"]
            missing = []
            for tag in TAGS:
                if not has_tag(key, tag):
                    continue
                expected = len(refs_for(key, tag))
                if done.get(f"{name}|{tag or 'train'}", 0) != expected:
                    missing.append(tag)
            if not missing:
                continue
            model = DELTA.load_frozen_actor_critic_checkpoint(info["path"], device=device)
            inference = DELTA.FrozenSearchInference(
                model, device=device,
                config=DELTA.InferenceConfig(mode="eager", compile_failure_policy="fail_fast"))
            for tag in missing:
                refs = refs_for(key, tag)
                requests = requests_for(key, refs)
                stored = stored_targets(key, refs)
                with np.load(DELTA.RUNTIME / key /
                             ("fresh_targets.npz" if tag is None else f"fresh_targets_{tag}.npz"),
                             allow_pickle=False) as archive:
                    targets = archive["target_mean"]
                pending = []
                for start in range(0, len(requests), batch):
                    chunk = requests[start:start + batch]
                    values = inference.policy_value(chunk)
                    for offset in range(len(chunk)):
                        index = start + offset
                        request = chunk[offset]
                        width = len(request.legal_actions)
                        pi_fresh = np.asarray(targets[index, :width], dtype=np.float64)
                        pi_stored = np.asarray(stored[index][:width], dtype=np.float64)
                        p = DELTA.softmax(values[offset][0])
                        pending.append({
                            "checkpoint": name, "key": key, "arm": info["arm"],
                            "step": info["step"], "tag": tag or "train",
                            "game_id": int(refs[index]["game_id"]),
                            "decision_index": int(refs[index]["decision_index"]),
                            "kl_fresh": DELTA.kl(pi_fresh, p),
                            "kl_stored": DELTA.kl(pi_stored, p),
                            "entropy_student": DELTA.entropy(p),
                            "top1_fresh": int(int(np.argmax(pi_fresh)) == int(np.argmax(p))),
                            "top1_stored": int(int(np.argmax(pi_stored)) == int(np.argmax(p))),
                        })
                for row in pending:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
                done[f"{name}|{tag or 'train'}"] = len(pending)
                common.write_json(progress_path, done)
                emit("evaluated_both", checkpoint=name, key=key, tag=tag or "train",
                     positions=len(pending))
            del inference, model
            if device == "cuda":
                import torch
                torch.cuda.empty_cache()


def summarize_both():
    spec = common.read_json(SPEC_PATH)
    rows = [json.loads(line) for line in ROWS_BOTH_PATH.read_text().splitlines()]
    key_of = lambda r: (r["checkpoint"], r["key"], r["tag"], r["game_id"],
                        r["decision_index"])
    set_of = {key_of(row): row["set"]
              for row in map(json.loads, ROWS_PATH.read_text().splitlines())}
    output = {"protocol": "avg4-scaleup-fresh-stored-sameposition-v1",
              "client_date": "2026-10-09",
              "metric": "KL(label || student), labels scored on identical positions",
              "bootstrap": {"cluster": "source game", "strata": "round",
                            "resamples": DELTA.BOOTSTRAP_RESAMPLES, "seed": DELTA.BOOTSTRAP_SEED},
              "deltas_vs_start": {}, "stored_minus_fresh_delta": {}, "stored": {}}
    tags = sorted({row["tag"] for row in rows})
    for tag in tags:
        output["deltas_vs_start"][tag] = {}
        output["stored_minus_fresh_delta"][tag] = {}
        output["stored"][tag] = {}
        for name, info in spec["checkpoints"].items():
            base_name = f"{info['key']}|start|000000"
            now = [row for row in rows if row["checkpoint"] == name and row["tag"] == tag]
            if not now:
                continue
            stored_mean = float(np.mean([row["kl_stored"] for row in now]))
            fresh_mean = float(np.mean([row["kl_fresh"] for row in now]))
            output["stored"][tag][name] = {"kl_stored": stored_mean, "kl_fresh": fresh_mean,
                                           "positions": len(now)}
            if name == base_name:
                continue
            base = [row for row in rows if row["checkpoint"] == base_name and row["tag"] == tag]
            paired = _paired_keyed(base, now)
            if paired:
                per_set = {}
                splits = sorted({set_of.get(key_of(row)) for row in now} - {None})
                for split in splits:
                    now_split = [row for row in now
                                 if set_of.get(key_of(row)) == split]
                    base_split = [row for row in base
                                  if set_of.get(key_of(row)) == split]
                    sub = _paired_keyed(base_split, now_split)
                    if sub["stored"]:
                        per_set[split] = {
                            "positions": len(sub["stored"]),
                            "delta_stored": _stats_like(sub["stored"], sub["clusters"],
                                                        sub["strata"]),
                            "delta_fresh": _stats_like(sub["fresh"], sub["clusters"],
                                                       sub["strata"]),
                        }
                output["deltas_vs_start"][tag][name] = {
                    "step": info["step"], "arm": info["arm"],
                    "delta_stored": _stats_like(paired["stored"], paired["clusters"],
                                                paired["strata"]),
                    "delta_fresh": _stats_like(paired["fresh"], paired["clusters"],
                                               paired["strata"]),
                    "stored_minus_fresh": _stats_like(
                        np.asarray(paired["stored"]) - np.asarray(paired["fresh"]),
                        paired["clusters"], paired["strata"]),
                    "by_set": per_set,
                }
    common.write_json(common.ART / "fresh_stored_summary.json", output)
    emit("summarized_both", tags=tags)
    return output


def _paired_keyed(base_rows, now_rows):
    keyed = {(r["key"], r["tag"], r["game_id"], r["decision_index"]): r for r in base_rows}
    stored, fresh, clusters, strata = [], [], [], []
    for row in now_rows:
        k = (row["key"], row["tag"], row["game_id"], row["decision_index"])
        if k not in keyed:
            continue
        base = keyed[k]
        stored.append(row["kl_stored"] - base["kl_stored"])
        fresh.append(row["kl_fresh"] - base["kl_fresh"])
        clusters.append(f"{row['key']}:{row['game_id']}")
        strata.append(row["key"])
    return {"stored": stored, "fresh": fresh, "clusters": clusters, "strata": strata}


def refs_for(key: str, tag):
    rows = [json.loads(line) for line in rows_path(key, tag).read_text().splitlines()]
    return [{"game_id": int(row["game_id"]), "decision_index": int(row["decision_index"]),
             "set": row["set"]} for row in rows]


def evaluate(device: str, only: list[str] | None = None, batch: int = 128):
    spec = common.read_json(SPEC_PATH)
    common.LR_RUNTIME.mkdir(parents=True, exist_ok=True)
    progress_path = ROWS_PATH.with_name("fresh_progress.json")
    done = common.read_json(progress_path) if progress_path.exists() else {}
    names = list(spec["checkpoints"])
    if only:
        names = [name for name in names if any(name.startswith(prefix) for prefix in only)]
    with ROWS_PATH.open("a" if ROWS_PATH.exists() else "w") as handle:
        for name in names:
            info = spec["checkpoints"][name]
            key = info["key"]
            missing = []
            for tag in TAGS:
                if not has_tag(key, tag):
                    continue
                expected = len(refs_for(key, tag))
                if done.get(f"{name}|{tag or 'train'}", 0) != expected:
                    missing.append(tag)
            if not missing:
                continue
            model = DELTA.load_frozen_actor_critic_checkpoint(info["path"], device=device)
            inference = DELTA.FrozenSearchInference(
                model, device=device,
                config=DELTA.InferenceConfig(mode="eager", compile_failure_policy="fail_fast"))
            for tag in missing:
                refs = refs_for(key, tag)
                if done.get(f"{name}|{tag or 'train'}", 0) not in (0, None):
                    raise ValueError(
                        f"{name}|{tag}: partial rows without completion marker; "
                        "rerun with a fresh rows file")
                requests = requests_for(key, refs)
                with np.load(DELTA.RUNTIME / key /
                             ("fresh_targets.npz" if tag is None else f"fresh_targets_{tag}.npz"),
                             allow_pickle=False) as archive:
                    targets = archive["target_mean"]
                    game_ids = archive["game_id"]
                    decision_index = archive["decision_index"]
                    if not (np.array_equal(game_ids, [r["game_id"] for r in refs])
                            and np.array_equal(decision_index,
                                               [r["decision_index"] for r in refs])):
                        raise ValueError(f"{name}: fresh targets and rows differ ({key}/{tag})")
                widths = [len(request.legal_actions) for request in requests]
                pending = []
                for start in range(0, len(requests), batch):
                    chunk = requests[start:start + batch]
                    values = inference.policy_value(chunk)
                    for offset in range(len(chunk)):
                        index = start + offset
                        request = chunk[offset]
                        width = widths[index]
                        pi = np.asarray(targets[index, :width], dtype=np.float64)
                        p = DELTA.softmax(values[offset][0])
                        pending.append({
                            "checkpoint": name, "key": key, "arm": info["arm"],
                            "step": info["step"], "tag": tag or "train",
                            "set": refs[index]["set"],
                            "game_id": int(refs[index]["game_id"]),
                            "decision_index": int(refs[index]["decision_index"]),
                            "kl_fresh": DELTA.kl(pi, p),
                            "entropy_student": DELTA.entropy(p),
                            "entropy_fresh": DELTA.entropy(pi),
                            "top1_agree": int(int(np.argmax(pi)) == int(np.argmax(p))),
                        })
                for row in pending:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
                done[f"{name}|{tag or 'train'}"] = len(pending)
                common.write_json(progress_path, done)
                emit("evaluated", checkpoint=name, key=key, tag=tag or "train",
                     positions=len(refs))
            del inference, model
            if device == "cuda":
                import torch
                torch.cuda.empty_cache()


def _stats(rows):
    values = np.asarray([row["kl_fresh"] for row in rows], dtype=np.float64)
    clusters = [f"{row['key']}:{row['game_id']}" for row in rows]
    strata = [row["key"] for row in rows]
    return {
        "positions": len(rows),
        "games": len(set(clusters)),
        "mean_kl": float(values.mean()),
        "mean_kl_bootstrap": DELTA.bootstrap_mean(
            values, clusters, DELTA.BOOTSTRAP_RESAMPLES, DELTA.BOOTSTRAP_SEED,
            strata=strata),
        "top1_agree": float(np.mean([row["top1_agree"] for row in rows])),
        "entropy_student": float(np.mean([row["entropy_student"] for row in rows])),
        "entropy_fresh": float(np.mean([row["entropy_fresh"] for row in rows])),
    }


def _paired(rows_a, rows_b, label_a, label_b):
    """Paired per-position difference (b - a) with the shared cluster bootstrap."""
    keyed = {(r["key"], r["tag"], r["game_id"], r["decision_index"]): r["kl_fresh"]
             for r in rows_a}
    values, clusters, strata = [], [], []
    for row in rows_b:
        k = (row["key"], row["tag"], row["game_id"], row["decision_index"])
        if k not in keyed:
            continue
        values.append(row["kl_fresh"] - keyed[k])
        clusters.append(f"{row['key']}:{row['game_id']}")
        strata.append(row["key"])
    if not values:
        return None
    return {"comparison": f"{label_b} - {label_a}", "positions": len(values),
            **_stats_like(values, clusters, strata)}


def _stats_like(values, clusters, strata):
    boot = DELTA.bootstrap_mean(values, clusters, DELTA.BOOTSTRAP_RESAMPLES,
                                DELTA.BOOTSTRAP_SEED, strata=strata)
    return {**boot, "improved_fraction": float(np.mean(np.asarray(values) < 0))}


def summarize():
    spec = common.read_json(SPEC_PATH)
    rows = [json.loads(line) for line in ROWS_PATH.read_text().splitlines()]
    output = {
        "protocol": "avg4-scaleup-fresh-eval-summary-v1",
        "client_date": "2026-10-09",
        "metric": "KL(fresh_mean_M4 || student)",
        "bootstrap": {"cluster": "source game", "strata": "round",
                      "resamples": DELTA.BOOTSTRAP_RESAMPLES, "seed": DELTA.BOOTSTRAP_SEED},
        "checkpoints": spec["checkpoints"],
        "sets": {}, "deltas_vs_start": {}, "paired_arm_comparisons": {},
        "by_set": {},
    }
    tags = sorted({row["tag"] for row in rows})
    for tag in tags:
        output["sets"][tag] = {}
        for name in spec["checkpoints"]:
            subset = [row for row in rows if row["checkpoint"] == name and row["tag"] == tag]
            if subset:
                output["sets"][tag][name] = _stats(subset)
    # Delta-KL of every checkpoint relative to its own start checkpoint.
    for tag in tags:
        output["deltas_vs_start"][tag] = {}
        for name, info in spec["checkpoints"].items():
            base_name = f"{info['key']}|start|000000"
            if name == base_name:
                continue
            base = [row for row in rows if row["checkpoint"] == base_name and row["tag"] == tag]
            now = [row for row in rows if row["checkpoint"] == name and row["tag"] == tag]
            paired = _paired(base, now, "start", name)
            if paired:
                paired["step"] = info["step"]
                paired["arm"] = info["arm"]
                output["deltas_vs_start"][tag][name] = paired
    # Matched-step arm comparisons on identical positions.
    for tag in tags:
        entries = {}
        for step in (50, 100, 200, 500):
            by_arm = {}
            for name, info in spec["checkpoints"].items():
                if info["step"] == step and info["arm"] in ("A", "B", "C", "lr2.5e-5"):
                    by_arm.setdefault(name, info)
            per_key = {}
            for name, info in by_arm.items():
                rows_here = [row for row in rows
                             if row["checkpoint"] == name and row["tag"] == tag]
                if rows_here:
                    per_key.setdefault(info["arm"], {})[info["key"]] = rows_here
            arm_rows = {arm: [row for subset in keys.values() for row in subset]
                        for arm, keys in per_key.items()}
            for arm in ("B", "C", "lr2.5e-5"):
                if "A" in arm_rows and arm in arm_rows:
                    result = _paired(arm_rows["A"], arm_rows[arm], f"A@{step}",
                                     f"{arm}@{step}")
                    if result:
                        entries[f"{arm}_minus_A@{step}"] = result
            if "B" in arm_rows and "C" in arm_rows:
                entries[f"C_minus_B@{step}"] = _paired(arm_rows["B"], arm_rows["C"],
                                                       f"B@{step}", f"C@{step}")
        output["paired_arm_comparisons"][tag] = entries
        splits = sorted({row["set"] for row in rows if row["tag"] == tag})
        if len(splits) > 1:
            output["by_set"][tag] = {}
            for split in splits:
                subset = [row for row in rows if row["tag"] == tag and row["set"] == split]
                output["by_set"][tag][split] = {
                    "positions": len(subset),
                    "deltas_vs_start": {},
                    "paired_arm_comparisons": {},
                }
                for name, info in spec["checkpoints"].items():
                    base_name = f"{info['key']}|start|000000"
                    if name == base_name:
                        continue
                    base = [row for row in subset if row["checkpoint"] == base_name]
                    now = [row for row in subset if row["checkpoint"] == name]
                    paired = _paired(base, now, "start", name)
                    if paired:
                        paired["step"] = info["step"]
                        paired["arm"] = info["arm"]
                        output["by_set"][tag][split]["deltas_vs_start"][name] = paired
                for step in (50, 100, 200, 500):
                    arm_rows = {}
                    for name, info in spec["checkpoints"].items():
                        if info["step"] == step and info["arm"] in ("A", "B", "C", "lr2.5e-5"):
                            arm_rows.setdefault(info["arm"], []).extend(
                                [row for row in subset if row["checkpoint"] == name])
                    for arm in ("B", "C", "lr2.5e-5"):
                        if "A" in arm_rows and arm in arm_rows:
                            result = _paired(arm_rows["A"], arm_rows[arm], f"A@{step}",
                                             f"{arm}@{step}")
                            if result:
                                output["by_set"][tag][split]["paired_arm_comparisons"][
                                    f"{arm}_minus_A@{step}"] = result
                    if "B" in arm_rows and "C" in arm_rows:
                        output["by_set"][tag][split]["paired_arm_comparisons"][
                            f"C_minus_B@{step}"] = _paired(arm_rows["B"], arm_rows["C"],
                                                           f"B@{step}", f"C@{step}")
    common.write_json(common.ART / "fresh_summary.json", output)
    emit("summarized", tags=tags)
    return output


def smoke():
    """CPU mini-run: evaluate the start and one A node on two positions per tag."""
    spec = common.read_json(SPEC_PATH) if SPEC_PATH.exists() else freeze()
    model_names = [name for name in spec["checkpoints"]
                   if name.endswith("start|000000") or name.endswith("|A|000500")]
    rows = []
    for name in model_names[:2]:
        info = spec["checkpoints"][name]
        key = info["key"]
        model = DELTA.load_frozen_actor_critic_checkpoint(info["path"], device="cpu")
        inference = DELTA.FrozenSearchInference(
            model, device="cpu",
            config=DELTA.InferenceConfig(mode="eager", compile_failure_policy="fail_fast"))
        for tag in TAGS:
            if not has_tag(key, tag):
                continue
            refs = refs_for(key, tag)[:2]
            requests = requests_for(key, refs)
            with np.load(DELTA.RUNTIME / key /
                         ("fresh_targets.npz" if tag is None else f"fresh_targets_{tag}.npz"),
                         allow_pickle=False) as archive:
                targets = archive["target_mean"]
            values = inference.policy_value(requests)
            for offset, request in enumerate(requests):
                width = len(request.legal_actions)
                rows.append({"checkpoint": name, "tag": tag or "train",
                             "kl": DELTA.kl(np.asarray(targets[offset, :width]),
                                            DELTA.softmax(values[offset][0]))})
    emit("smoke_done", rows=len(rows), sample=rows[:2])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "evaluate", "summarize", "smoke",
                                           "evaluate-both", "summarize-both"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--batch", type=int, default=128)
    args = parser.parse_args()
    if args.action == "freeze":
        freeze()
    elif args.action == "evaluate":
        if not SPEC_PATH.exists():
            freeze()
        evaluate(args.device, only=args.only, batch=args.batch)
    elif args.action == "summarize":
        summarize()
    elif args.action == "evaluate-both":
        if not SPEC_PATH.exists():
            freeze()
        evaluate_both(args.device, batch=args.batch)
    elif args.action == "summarize-both":
        summarize_both()
    else:
        smoke()


if __name__ == "__main__":
    main()
