"""Frozen independent real-game roots for checkpoint decision-drift diagnostics.

This measures changes in search decisions, not improvement in playing strength.
Build a bank once, evaluate checkpoints with shared search seeds, then compare
the saved results. No reference committee or strength calibration is used.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import platform
import random
import resource
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .catalog import load_default_catalog
from .compact_trace import append_action, decode_request, new_trace, replay_trace
from .contracts import DecisionRequest
from .engine import GameEngine
from .features import FEATURE_SCHEMA_VERSION, encode_decision
from .heuristic_policy import DevelopmentPolicy, ScorePolicy
from .mcts_policy import MCTSConfig
from .model import DecisionPhase
from .native_policy import (ChanceWideningConfig, NativeBatchRunner,
                            NativeChancePUCTSearchPolicy, NativePUCTSearchPolicy)
from .opponent_pool import load_frozen_actor_critic_checkpoint
from .replay import RULESET_ID, to_primitive
from .result import TerminalResult
from .search_inference import FrozenSearchInference, InferenceConfig

PROTOCOL = "splendor-decision-probe-v1"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BANK = "artifacts/decision_probe_v1_20261009/bank.json"
DEFAULT_SEEDS = (2026101500, 2026101501)
PAIRS = (("score", "score"), ("score", "development"),
         ("development", "score"), ("development", "development"))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def object_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def read(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")
    temp.replace(path)


@contextmanager
def locked(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "lock").open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def code_identity() -> dict[str, Any]:
    names = ("decision_probe.py", "engine.py", "rules.py", "heuristic_policy.py",
             "compact_trace.py", "features.py", "native_policy.py", "native_search.cpp",
             "search_inference.py", "opponent_pool.py", "actor_critic.py", "entity_attention.py",
             "alphazero_checkpoint.py", "mcts_policy.py", "search_policy.py", "catalog.py",
             "contracts.py", "model.py", "actions.py", "rng.py", "replay.py")
    return {"git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                                text=True).strip(),
            "files": {name: digest(ROOT / "src/splendor" / name) for name in names},
            "native_extensions": {p.name: digest(p) for p in (ROOT / "src/splendor").glob("_native_search*.so")}}


def same_execution_code(a: dict[str, Any], b: dict[str, Any]) -> bool:
    # A documentation-only commit must not invalidate a saved search reference.
    return a["files"] == b["files"] and a["native_extensions"] == b["native_extensions"]


def same_identity(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return (same_execution_code(a["code"], b["code"])
            and {k: v for k, v in a.items() if k != "code"}
            == {k: v for k, v in b.items() if k != "code"})


def stage_of(request: DecisionRequest, threshold: int) -> str:
    o = request.observation
    return "late" if max(o.self_public.score, o.opponent_public.score) >= threshold else "early_mid"


def collect_game(game_id: int, seed: int, selection_seed: int, threshold: int) -> dict[str, Any]:
    """One genuinely played game, followed by one predeclared stage sample."""
    names = PAIRS[game_id % len(PAIRS)]
    policies = tuple(ScorePolicy() if n == "score" else DevelopmentPolicy() for n in names)
    first = (game_id // 8) % 2
    engine = GameEngine.new(seed, first_player=first, mode="checked")
    trace = new_trace(game_id, seed, first, engine.catalog)
    wanted_stage = "early_mid" if (game_id // 4) % 2 == 0 else "late"
    eligible = []
    for _ in range(2000):
        request = engine.decision()
        if isinstance(request, TerminalResult):
            trace["terminal_result"] = to_primitive(request)
            break
        index = len(trace["actions"])
        if (request.observation.phase is DecisionPhase.MAIN_ACTION
                and len(request.legal_actions) > 1 and stage_of(request, threshold) == wanted_stage):
            eligible.append({"decision_index": index, "actor": engine.current_player,
                             "request": to_primitive(request)})
        action = policies[engine.current_player].choose(request)
        append_action(trace, request, engine.current_player, request.legal_actions.index(action))
        engine.apply(action)
    else:
        raise RuntimeError(f"source game {game_id} did not terminate within 2000 decisions")
    selected = random.Random(selection_seed + game_id).choice(eligible) if eligible else None
    return {"game_id": game_id, "game_seed": seed, "policy_pair": list(names), "trace": trace,
            "selected": selected, "stage": wanted_stage, "eligible_positions": len(eligible)}


def load_bank(path: Path) -> dict[str, Any]:
    bank = read(path)
    catalog = load_default_catalog()
    if (bank["protocol"] != PROTOCOL or bank["ruleset_id"] != RULESET_ID
            or bank["feature_schema_version"] != FEATURE_SCHEMA_VERSION
            or bank["catalog_fingerprint"] != catalog.data_fingerprint):
        raise ValueError("bank protocol, rules, catalog or feature schema mismatch")
    rows = bank["positions"]
    if not rows or [r["id"] for r in rows] != list(range(len(rows))):
        raise ValueError("bank position ids must be contiguous and nonempty")
    if len({r["game_seed"] for r in rows}) != len(rows):
        raise ValueError("bank must contain one position per independent source game")
    for row in rows:
        request = decode_request(row["request"])
        if (request.observation.phase is not DecisionPhase.MAIN_ACTION or len(request.legal_actions) < 2
                or row["candidate_count"] != len(request.legal_actions)
                or row["stage"] != stage_of(request, bank["late_score_threshold"])
                or row["request_sha256"] != object_digest(row["request"])):
            raise ValueError(f"invalid bank root {row['id']}")
    return bank


def build_bank(path: Path, *, positions: int = 256, seed_start: int = 2026100900,
               selection_seed: int = 2026101300, threshold: int = 10,
               resume: bool = False) -> dict[str, Any]:
    if positions <= 0 or positions % 8 or not 0 < threshold < 15:
        raise ValueError("positions must be a positive multiple of 8; threshold must lie in [1,14]")
    catalog = load_default_catalog()
    identity = {"protocol": PROTOCOL, "positions": positions, "seed_start": seed_start,
                "selection_seed": selection_seed, "late_score_threshold": threshold,
                "sampling": "one root per stage-eligible game; preserve stage-ineligible games, then advance by positions",
                "policy_pairs": PAIRS, "policy_weights": [asdict(ScorePolicy().weights),
                                                           asdict(DevelopmentPolicy().weights)],
                "catalog_fingerprint": catalog.data_fingerprint, "code": code_identity()}
    # JSON round trip gives tuples their on-disk representation before comparison.
    identity = json.loads(json.dumps(identity))
    sidecar = path.parent / "collection.games"
    with locked(sidecar):
        manifest = sidecar / "manifest.json"
        if manifest.exists():
            if not resume:
                raise FileExistsError("collection exists; use --resume or a new bank directory")
            if not same_identity(read(manifest), identity):
                raise ValueError("collection identity changed; cannot resume")
        else:
            write(manifest, identity)
        if path.exists():
            return verify_bank(path)
        records = []
        for i in range(positions):
            record_path = sidecar / f"game-{i:06d}.json"
            if record_path.exists():
                record = read(record_path)
                if record["root_id"] != i or record["game_seed"] != seed_start + record["game_id"]:
                    raise ValueError("source-game identity mismatch")
            else:
                source_id = i
                while True:
                    excluded_path = sidecar / f"excluded-{source_id:06d}.json"
                    if excluded_path.exists():
                        source_id += positions
                        continue
                    record = collect_game(source_id, seed_start + source_id, selection_seed, threshold)
                    record["root_id"] = i
                    if record["selected"] is not None:
                        break
                    write(excluded_path, {**record, "reason": "no_multi_candidate_main_action_in_requested_stage"})
                    source_id += positions
                write(record_path, record)
            records.append(record)
            if (i + 1) % 32 == 0:
                print(json.dumps({"event": "bank_progress", "games": i + 1, "total": positions}), flush=True)
        rows = []
        for record in records:
            selected = record["selected"]
            request = decode_request(selected["request"])
            o = request.observation
            rows.append({"id": record["root_id"], "source_game_id": record["game_id"], "game_seed": record["game_seed"],
                         "stage": record["stage"], "policy_pair": record["policy_pair"],
                         "decision_index": selected["decision_index"], "actor": selected["actor"],
                         "candidate_count": len(request.legal_actions),
                         "completed_player_turns": o.own_completed_turns + o.opponent_completed_turns,
                         "scores": [o.self_public.score, o.opponent_public.score],
                         "opponent_blind_reserved": sum(r.card is None for r in o.opponent_public.reserved_cards),
                         "final_round_triggered": o.final_round_triggered,
                         "request_sha256": object_digest(selected["request"]), "request": selected["request"]})
        traces = path.parent / "source_games.jsonl"
        temp = traces.with_suffix(".jsonl.tmp")
        with temp.open("w", encoding="utf-8") as f:
            for record in records:
                source = {k: v for k, v in record.items() if k != "selected"}
                source["selected_decision_index"] = record["selected"]["decision_index"]
                f.write(json.dumps(source, ensure_ascii=False, separators=(",", ":")) + "\n")
        temp.replace(traces)
        excluded = [read(p) for p in sorted(sidecar.glob("excluded-*.json"))]
        excluded_file = path.parent / "excluded_source_games.jsonl"
        excluded_file.write_text("".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n"
                                         for r in excluded), encoding="utf-8")
        write(path, {"protocol": PROTOCOL, "purpose": "persistent_diagnostic_test_never_training_data",
                     "ruleset_id": RULESET_ID, "feature_schema_version": FEATURE_SCHEMA_VERSION,
                     "catalog_fingerprint": catalog.data_fingerprint, "late_score_threshold": threshold,
                     "collection": identity, "source_games": {"file": traces.name, "sha256": digest(traces)},
                     "excluded_source_games": {"file": excluded_file.name, "sha256": digest(excluded_file),
                                               "count": len(excluded)},
                     "command": sys.argv, "environment": {"python": sys.version, "platform": platform.platform()},
                     "positions": rows})
        return verify_bank(path)


def verify_bank(path: Path) -> dict[str, Any]:
    bank = load_bank(path)
    traces = path.parent / bank["source_games"]["file"]
    if digest(traces) != bank["source_games"]["sha256"]:
        raise ValueError("source trace file changed")
    sources = [json.loads(line) for line in traces.read_text(encoding="utf-8").splitlines()]
    if len(sources) != len(bank["positions"]):
        raise ValueError("source-game count mismatch")
    for row, source in zip(bank["positions"], sources, strict=True):
        trace = source["trace"]
        if (source["root_id"] != row["id"] or source["game_id"] != row["source_game_id"]
                or trace["game_seed"] != row["game_seed"]):
            raise ValueError("root/source identity mismatch")
        replay_trace(trace)
        prefix = {**trace, "actions": trace["actions"][:row["decision_index"]]}
        if to_primitive(replay_trace(prefix, require_terminal=False)) != row["request"]:
            raise ValueError(f"saved root {row['id']} does not reproduce its real-game decision")
    excluded_path = path.parent / bank["excluded_source_games"]["file"]
    if digest(excluded_path) != bank["excluded_source_games"]["sha256"]:
        raise ValueError("excluded source trace file changed")
    excluded = [json.loads(line) for line in excluded_path.read_text(encoding="utf-8").splitlines()]
    if len(excluded) != bank["excluded_source_games"]["count"]:
        raise ValueError("excluded source count mismatch")
    all_seeds = [r["game_seed"] for r in sources + excluded]
    if len(all_seeds) != len(set(all_seeds)):
        raise ValueError("accepted/excluded source seeds overlap")
    for source in excluded:
        replay_trace(source["trace"])
    result = {"protocol": PROTOCOL, "bank_sha256": digest(path), "positions": len(sources),
              "independent_games": len({r["game_seed"] for r in bank["positions"]}),
              "excluded_games": len(excluded), "source_games_played": len(all_seeds),
              "by_stage": {s: sum(r["stage"] == s for r in bank["positions"])
                           for s in ("early_mid", "late")}, "all_real_roots_replayed": True}
    write(path.parent / "verification.json", result)
    return result


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2
    def kl(x):
        positive = x > 0
        return np.sum(x[positive] * np.log(x[positive] / m[positive]))
    return float((kl(p) + kl(q)) / 2)


def rank_changes(p: np.ndarray, q: np.ndarray, *, high_mass: float = .8) -> dict[str, Any]:
    """Baseline-defined high-mass actions; only strictly ordered pairs count.

    Candidate ties count as half an inversion. A one-action high-mass set has
    no ranking pairs and returns None, never a misleading perfect score.
    """
    p, q = np.asarray(p), np.asarray(q)
    a = np.argsort(-p, kind="stable")
    b = np.argsort(-q, kind="stable")
    k = min(3, len(p))
    n = int(np.searchsorted(np.cumsum(p[a]), high_mass, side="left")) + 1
    selected = a[:min(n, len(p))]
    inversions = []
    for i, left in enumerate(selected):
        for right in selected[i + 1:]:
            if abs(p[left] - p[right]) <= 1e-12:
                continue
            inversions.append(1. if q[left] < q[right] - 1e-12
                              else .5 if abs(q[left] - q[right]) <= 1e-12 else 0.)
    ranks = []
    for index in selected:
        # Midranks make ties explicit instead of relying on candidate ordering.
        def rank(v):
            return 1 + np.sum(v > v[index] + 1e-12) + .5 * (np.sum(np.abs(v - v[index]) <= 1e-12) - 1)
        ranks.append(abs(float(rank(q) - rank(p))))
    return {"top3_overlap": len(set(a[:k]) & set(b[:k])) / k,
            "top3_order_changed": float(not np.array_equal(a[:k], b[:k])),
            "high_visit_actions": len(selected), "high_visit_ordered_pairs": len(inversions),
            "high_visit_rank_inversion": float(np.mean(inversions)) if inversions else None,
            "high_visit_mean_rank_displacement": float(np.mean(ranks)),
            "mass_on_baseline_top3_change": float(q[a[:k]].sum() - p[a[:k]].sum())}


def root_comparison(a: dict[str, Any], b: dict[str, Any], *, high_mass: float = .8) -> dict[str, Any]:
    p, q = np.array(a["policy_target"]), np.array(b["policy_target"])
    if len(p) != len(q) or a["root_id"] != b["root_id"] or a["seed"] != b["seed"]:
        raise ValueError("paired roots/candidates/seeds do not match")
    action = a["chosen_candidate_index"]
    ordered = np.sort(p)[::-1]
    gap = float(ordered[0] - ordered[1])
    changed = float(action != b["chosen_candidate_index"])
    return {"action_changed": changed, "visit_jsd": jsd(p, q),
            "prior_jsd": jsd(np.array(a["root_network_prior"]), np.array(b["root_network_prior"])),
            "root_value_change": b["root_value"] - a["root_value"],
            "root_value_absolute_change": abs(b["root_value"] - a["root_value"]),
            "baseline_visit_margin": gap, "baseline_confident": gap >= .2,
            "baseline_near_tie": gap < .05,
            "baseline_chosen_mass_change": float(q[action] - p[action]),
            **rank_changes(p, q, high_mass=high_mass)}


def summarize(values: list[float | None]) -> dict[str, Any]:
    v = np.array([x for x in values if x is not None], dtype=float)
    return {"defined_values": len(v), "mean": float(v.mean()) if len(v) else None,
            "median": float(np.median(v)) if len(v) else None,
            "p90": float(np.quantile(v, .9)) if len(v) else None}


def load_score(directory: Path) -> tuple[dict[str, Any], list[list[dict[str, Any]]]]:
    summary = read(directory / "summary.json")
    if summary["status"] != "complete":
        raise ValueError("incomplete score cannot be compared")
    data = directory / "searches.json"
    if digest(data) != summary["searches_sha256"]:
        raise ValueError("saved search results changed")
    return summary, read(data)


def compare_scores(a_dir: Path, b_dir: Path, output: Path, *, high_mass: float = .8,
                   allow_search_change: bool = False) -> dict[str, Any]:
    if not 0 < high_mass <= 1:
        raise ValueError("high_mass must lie in (0,1]")
    a_summary, a = load_score(a_dir)
    b_summary, b = load_score(b_dir)
    for key in ("bank_sha256", "repeat_seeds", "root_identity_sha256", "feature_schema_version"):
        if a_summary[key] != b_summary[key]:
            raise ValueError(f"incompatible comparison: {key}")
    search_changed = a_summary["search"] != b_summary["search"]
    if search_changed and not allow_search_change:
        raise ValueError("search configurations differ; use matching references or --allow-search-change")
    code_changed = not same_execution_code(a_summary["code"], b_summary["code"])
    if code_changed and not allow_search_change:
        raise ValueError("search/source code changed; create a new reference or explicitly allow the change")
    if len(a) != len(b) or len(a) != len(a_summary["repeat_seeds"]):
        raise ValueError("search repeat count mismatch")
    if a_summary["positions"] != b_summary["positions"] or any(
            len(run) != len(a_summary["positions"]) for run in (*a, *b)):
        raise ValueError("search position identities/counts mismatch")
    position_rows = []
    numeric = None
    for i, info in enumerate(a_summary["positions"]):
        repeats = [root_comparison(left[i], right[i], high_mass=high_mass)
                   for left, right in zip(a, b, strict=True)]
        numeric = [k for k in repeats[0] if k not in ("baseline_confident", "baseline_near_tie")]
        means = {k: summarize([r[k] for r in repeats])["mean"] for k in numeric}
        confident = [r["action_changed"] for r in repeats if r["baseline_confident"]]
        near_tie = [r["action_changed"] for r in repeats if r["baseline_near_tie"]]
        position_rows.append({**info, "metrics": means, "repeats": repeats,
                              "confident_changes": confident, "near_tie_changes": near_tie})
    groups = {}
    for stage in ("all", "early_mid", "late"):
        rows = [r for r in position_rows if stage == "all" or r["stage"] == stage]
        groups[stage] = {"positions": len(rows),
                        "metrics": {k: summarize([r["metrics"][k] for r in rows]) for k in numeric},
                        "confident_action_change": summarize([x for r in rows for x in r["confident_changes"]]),
                        "near_tie_action_change": summarize([x for r in rows for x in r["near_tie_changes"]])}
    report = {"protocol": PROTOCOL, "kind": "checkpoint_decision_comparison", "a": str(a_dir.resolve()),
              "b": str(b_dir.resolve()), "checkpoints": [a_summary["checkpoint"], b_summary["checkpoint"]],
              "bank_sha256": a_summary["bank_sha256"], "repeat_seeds": a_summary["repeat_seeds"],
              "high_visit_mass": high_mass, "search_configuration_changed": search_changed,
              "code_changed": code_changed,
              "execution": [{k: s[k] for k in ("device", "inference", "active_roots", "environment")}
                            for s in (a_summary, b_summary)],
              "within_checkpoint_seed_variation": [a_summary["within_checkpoint_seed_variation"],
                                                     b_summary["within_checkpoint_seed_variation"]],
              "by_stage": groups,
              "positions": position_rows,
              "interpretation": "Decision drift, not action correctness or strength. Lower JSD is not stronger. "
                                "Compare with each checkpoint's within-seed variation; do not subtract a noise floor."}
    write(output, report)
    return report


def evaluate(bank_path: Path, checkpoint: Path, output: Path, *, search: dict[str, Any],
             seeds: tuple[int, ...] = DEFAULT_SEEDS, device: str = "cpu", active_roots: int = 64,
             inference_mode: str = "eager", resume: bool = False) -> dict[str, Any]:
    bank = load_bank(bank_path)
    if not seeds or len(set(seeds)) != len(seeds) or active_roots <= 0:
        raise ValueError("repeat seeds must be nonempty/unique and active_roots positive")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if inference_mode == "compiled" and device != "cuda":
        raise ValueError("compiled native CUDA Graph inference requires CUDA")
    config_values = {k: v for k, v in search.items() if k not in ("backend", "chance_widening")}
    config = MCTSConfig(**config_values, seed=seeds[0])
    if config.root_noise_fraction != 0:
        raise ValueError("persistent decision probe uses no root noise")
    backend = search["backend"]
    if backend not in ("native", "native_chance"):
        raise ValueError("supported backends: native, native_chance")
    if search.get("chance_widening") and backend != "native_chance":
        raise ValueError("chance widening requires native_chance")
    buckets = (16, 32, 48, 64)
    inference = InferenceConfig(mode=inference_mode, candidate_buckets=buckets,
                               compile_mode="native-cudagraph" if inference_mode == "compiled" else None,
                               batch_buckets=tuple(sorted(set((16, 32, 64, active_roots)))) if inference_mode == "compiled" else (),
                               compile_failure_policy="fail_fast")
    positions = [{k: r[k] for k in ("id", "stage", "game_seed", "candidate_count", "scores",
                                    "completed_player_turns", "opponent_blind_reserved", "final_round_triggered")}
                 for r in bank["positions"]]
    identity = {"protocol": PROTOCOL, "bank_sha256": digest(bank_path),
                "root_identity_sha256": object_digest([r["request_sha256"] for r in bank["positions"]]),
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "checkpoint": {"path": str(checkpoint.resolve()), "sha256": digest(checkpoint)},
                "search": search, "repeat_seeds": list(seeds), "code": code_identity(),
                "device": device, "inference": inference.to_dict(), "active_roots": active_roots,
                "torch_version": torch.__version__}
    sidecar = output / "work.games"
    with locked(sidecar):
        manifest = sidecar / "manifest.json"
        if manifest.exists():
            if not resume:
                raise FileExistsError("score exists; use --resume or a new output directory")
            if not same_identity(read(manifest), identity):
                raise ValueError("score identity changed; cannot resume")
        else:
            write(manifest, identity)
        if (output / "summary.json").exists():
            return load_score(output)[0]
        model = load_frozen_actor_critic_checkpoint(checkpoint, device=device)
        evaluator = FrozenSearchInference(model, device=device, config=inference)
        policy_cls = NativePUCTSearchPolicy if backend == "native" else NativeChancePUCTSearchPolicy
        extra = {"chance_widening": ChanceWideningConfig(**search["chance_widening"])} if search.get("chance_widening") else {}
        requests = [decode_request(r["request"]) for r in bank["positions"]]
        root_values = []
        for start in range(0, len(requests), active_roots):
            encoded = tuple(encode_decision(r) for r in requests[start:start + active_roots])
            root_values.extend(v for _, v in evaluator.policy_value_encoded(encoded))
        started = time.perf_counter()
        all_runs = []
        for seed in seeds:
            policy = policy_cls(model, evaluator=evaluator, config=MCTSConfig(**config_values, seed=seed), **extra)
            runner = NativeBatchRunner(evaluator, policy.catalog, capacity=active_roots, buckets=buckets)
            pending = [i for i in range(len(requests)) if not (sidecar / f"seed-{seed}-root-{i:06d}.json").exists()]
            for start in range(0, len(pending), active_roots):
                indices = pending[start:start + active_roots]
                jobs = [policy.start_job(requests[i], add_root_noise=False, temperature=0.) for i in indices]
                while any(not j.done for j in jobs):
                    if runner.run([j for j in jobs if not j.done]) <= 0:
                        raise RuntimeError("native search made no progress")
                for i, job in zip(indices, jobs, strict=True):
                    result = job.result
                    if not result.searchable or sum(result.root_visit_counts) != config.simulations:
                        raise ValueError("search result budget/searchability mismatch")
                    row = {"root_id": i, "seed": seed, "root_value": root_values[i],
                           **result.to_dict(), "stats": job.core.stats()}
                    write(sidecar / f"seed-{seed}-root-{i:06d}.json", row)
                del jobs
                print(json.dumps({"event": "search_progress", "seed": seed, "new_roots": start + len(indices),
                                  "pending_roots": len(pending), "seconds": time.perf_counter() - started}), flush=True)
            all_runs.append([read(sidecar / f"seed-{seed}-root-{i:06d}.json") for i in range(len(requests))])
        noise = {}
        absolute = {}
        for stage in ("all", "early_mid", "late"):
            indices = [i for i, p in enumerate(positions) if stage == "all" or p["stage"] == stage]
            noise_rows = []
            root_rows = []
            for i in indices:
                runs = [run[i] for run in all_runs]
                seed_changes, seed_js = [], []
                for left in range(len(runs)):
                    for right in range(left + 1, len(runs)):
                        seed_changes.append(float(runs[left]["chosen_candidate_index"] != runs[right]["chosen_candidate_index"]))
                        seed_js.append(jsd(np.array(runs[left]["policy_target"]), np.array(runs[right]["policy_target"])))
                if seed_changes:
                    noise_rows.append((float(np.mean(seed_changes)), float(np.mean(seed_js))))
                for r in runs:
                    p = np.array(r["policy_target"])
                    positive = p > 0
                    entropy = float(-np.sum(p[positive] * np.log(p[positive])))
                    sorted_mass = np.sort(p)[::-1]
                    root_rows.append({"root_value": r["root_value"], "visit_entropy": entropy,
                                      "normalized_visit_entropy": entropy / np.log(len(p)),
                                      "effective_actions": float(np.exp(entropy)), "top1_mass": float(sorted_mass[0]),
                                      "top1_margin": float(sorted_mass[0] - sorted_mass[1]),
                                      "search_changes_prior_action": float(r["chosen_candidate_index"] != np.argmax(r["root_network_prior"])),
                                      "search_prior_jsd": jsd(p, np.array(r["root_network_prior"])),
                                      "mean_tree_depth": r["stats"]["tree_depth_sum"] / config.simulations,
                                      "terminal_hit_fraction": r["stats"]["terminal_hits"] / config.simulations,
                                      "transition_cap_hits": r["transition_cap_hits"]})
            noise[stage] = {"positions": len(noise_rows), "action_disagreement": summarize([r[0] for r in noise_rows]),
                            "visit_jsd": summarize([r[1] for r in noise_rows])}
            absolute[stage] = {"positions": len(indices),
                               "metrics": {k: summarize([r[k] for r in root_rows]) for k in root_rows[0]}}
        write(output / "searches.json", all_runs)
        summary = {**identity, "status": "complete", "positions": positions,
                   "searches_sha256": digest(output / "searches.json"), "within_checkpoint_seed_variation": noise,
                   "by_stage": absolute, "search_seconds_this_invocation": time.perf_counter() - started,
                   "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   "environment": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
                                   "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None},
                   "command": sys.argv, "interpretation": "Decision behavior only; no win-rate/Elo calibration."}
        write(output / "summary.json", summary)
        return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="play independent source games and freeze the bank")
    build.add_argument("--bank", type=Path, default=Path(DEFAULT_BANK))
    build.add_argument("--positions", type=int, default=256)
    build.add_argument("--seed-start", type=int, default=2026100900)
    build.add_argument("--selection-seed", type=int, default=2026101300)
    build.add_argument("--late-score-threshold", type=int, default=10)
    build.add_argument("--resume", action="store_true")
    verify = commands.add_parser("verify")
    verify.add_argument("--bank", type=Path, default=Path(DEFAULT_BANK))
    score = commands.add_parser("evaluate", help="score a checkpoint and optionally compare with a saved reference")
    score.add_argument("--bank", type=Path, default=Path(DEFAULT_BANK))
    score.add_argument("--checkpoint", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--compare-to", type=Path)
    score.add_argument("--simulations", type=int, default=1024)
    score.add_argument("--determinizations", type=int, default=1)
    score.add_argument("--c-puct", type=float, default=1.5)
    score.add_argument("--max-simulation-transitions", type=int, default=512)
    score.add_argument("--backend", choices=("native", "native_chance"), default="native")
    score.add_argument("--chance-pw-k", type=float)
    score.add_argument("--chance-pw-alpha", type=float, default=.5)
    score.add_argument("--repeat-seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    score.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    score.add_argument("--inference-mode", choices=("eager", "compiled"), default="eager")
    score.add_argument("--active-roots", type=int, default=64)
    score.add_argument("--resume", action="store_true")
    score.add_argument("--allow-search-change", action="store_true")
    compare = commands.add_parser("compare", help="CPU-only reanalysis of completed saved scores")
    compare.add_argument("--a", type=Path, required=True)
    compare.add_argument("--b", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--high-visit-mass", type=float, default=.8)
    compare.add_argument("--allow-search-change", action="store_true")
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.command == "build":
        result = build_bank(args.bank, positions=args.positions, seed_start=args.seed_start,
                            selection_seed=args.selection_seed, threshold=args.late_score_threshold, resume=args.resume)
    elif args.command == "verify":
        result = verify_bank(args.bank)
    elif args.command == "compare":
        result = compare_scores(args.a, args.b, args.output, high_mass=args.high_visit_mass,
                                allow_search_change=args.allow_search_change)
    else:
        device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
        search = {"backend": args.backend, "simulations": args.simulations,
                  "determinizations": args.determinizations, "c_puct": args.c_puct,
                  "fold_forced_phases": True, "max_simulation_transitions": args.max_simulation_transitions,
                  "root_noise_fraction": 0.}
        if args.chance_pw_k is not None:
            search["chance_widening"] = {"k": args.chance_pw_k, "alpha": args.chance_pw_alpha}
        result = evaluate(args.bank, args.checkpoint, args.output, search=search,
                          seeds=tuple(args.repeat_seeds), device=device, active_roots=args.active_roots,
                          inference_mode=args.inference_mode, resume=args.resume)
        if args.compare_to:
            result = compare_scores(args.compare_to, args.output, args.output / "comparison.json",
                                    allow_search_change=args.allow_search_change)
    console = {"status": result.get("status", "complete"), "kind": result.get("kind", args.command),
                      "positions": len(result["positions"]) if isinstance(result.get("positions"), list) else result.get("positions"),
                      "bank_sha256": result.get("bank_sha256")}
    if result.get("kind") == "checkpoint_decision_comparison":
        console["by_stage"] = {s: {k: result["by_stage"][s]["metrics"][k]["mean"]
                                  for k in ("action_changed", "visit_jsd", "high_visit_rank_inversion")}
                               for s in ("all", "early_mid", "late")}
    print(json.dumps(console, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
