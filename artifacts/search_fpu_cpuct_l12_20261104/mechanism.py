"""FPU mechanism check on real L12 roots: no arena, no training.

``positions`` plays fixed greedy L12 self-play games and keeps searchable roots.
``search`` reruns S128N1 on those roots with the historical search and with each
FPU reduction; ``--extension`` loads another build of the native module so the
historical default can be compared bit for bit against the build before FPU.
``report`` bins roots by the historical root Q and writes mechanism.json.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUNTIME = ROOT / "runtime" / OUT.name
L12 = ROOT / "runs/entity_attention_policy140221_r31_r37_p60k_iteration_20261025/rounds/round-011/checkpoints/final.pt"
L12_SHA = "2a6304e3470cab3dad78ec0a2789ee892892e741f3731cafc2197cde9bd2d45d"
GAME_SEEDS = range(47400000, 47400024)
KEEP_EVERY = 3
SEARCH_SEED = 47410000
REDUCTIONS = (None, 0.1, 0.2, 0.3)
BINS = ((-1.0, -0.6), (-0.6, -0.3), (-0.3, -0.1), (-0.1, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 1.0))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_model(device: str):
    from splendor.opponent_pool import load_frozen_actor_critic_checkpoint

    if sha256(L12) != L12_SHA:
        raise RuntimeError("L12 checkpoint changed")
    return load_frozen_actor_critic_checkpoint(L12, device=device)


def positions(args) -> None:
    from splendor.contracts import DecisionRequest
    from splendor.engine import GameEngine
    from splendor.neural_policy import GreedyActorCriticPolicy

    policy = GreedyActorCriticPolicy(load_model(args.device), device=args.device)
    kept, games = [], []
    for seed in GAME_SEEDS:
        engine = GameEngine.new(seed, first_player=seed % 2, mode="fast")
        decision, searchable = engine.decision(), 0
        while isinstance(decision, DecisionRequest):
            if len(decision.legal_actions) > 1:
                if searchable % KEEP_EVERY == 0:
                    kept.append(decision)
                searchable += 1
            decision = engine.apply(policy.choose(decision))
        games.append({"seed": seed, "searchable_decisions": searchable})
    RUNTIME.mkdir(parents=True, exist_ok=True)
    (RUNTIME / "positions.pkl").write_bytes(pickle.dumps(kept))
    print(json.dumps({"positions": len(kept), "games": games}))


def search(args) -> None:
    if args.extension:
        spec = importlib.util.spec_from_file_location("splendor._native_search", args.extension)
        module = importlib.util.module_from_spec(spec)
        sys.modules["splendor._native_search"] = module
        spec.loader.exec_module(module)
    from splendor.mcts_policy import MCTSConfig
    from splendor.native_policy import NativePUCTSearchPolicy

    reduction = None if args.reduction == "none" else float(args.reduction)
    config = MCTSConfig(simulations=128, determinizations=1, seed=SEARCH_SEED,
                        root_noise_fraction=0.0, fpu_reduction=reduction)
    policy = NativePUCTSearchPolicy(load_model(args.device), config=config, device=args.device,
                                    model_version=L12_SHA)
    rows = []
    for request in pickle.loads((RUNTIME / "positions.pkl").read_bytes()):
        result = policy.search(request)
        rows.append({"visits": list(result.per_world_root_visit_counts[0]),
                     "value_sums": list(result.per_world_root_value_sums[0]),
                     "prior": list(result.root_network_prior),
                     "chosen": result.chosen_candidate_index})
    import splendor._native_search as native
    payload = {"reduction": reduction, "extension": native.__file__,
               "extension_sha256": sha256(Path(native.__file__)), "rows": rows}
    (RUNTIME / f"search_{args.label}.json").write_text(json.dumps(payload))
    print(args.label, len(rows))


def _entropy(p):
    return -sum(x * math.log(x) for x in p if x > 0)


def _kl(p, q):
    return sum(x * math.log(x / y) for x, y in zip(p, q) if x > 0)


def report(args) -> None:
    runs = {label: json.loads((RUNTIME / f"search_{label}.json").read_text()) for label in args.labels}
    base = runs[args.labels[0]]["rows"]
    root_q = [sum(r["value_sums"]) / sum(r["visits"]) for r in base]
    summary = {"checkpoint": str(L12.relative_to(ROOT)), "checkpoint_sha256": L12_SHA,
               "game_seeds": [GAME_SEEDS.start, GAME_SEEDS.stop - 1], "keep_every": KEEP_EVERY,
               "search": "S128N1 total128, c_puct 1.5, no root noise, T0", "search_seed": SEARCH_SEED,
               "positions": len(base), "runs": {}, "bins": []}
    for label, run in runs.items():
        rows = run["rows"]
        same = [a["visits"] == b["visits"] and a["value_sums"] == b["value_sums"] for a, b in zip(rows, base)]
        summary["runs"][label] = {"reduction": run["reduction"], "extension_sha256": run["extension_sha256"],
                                  "identical_to_first": sum(same),
                                  "chosen_differs_from_first": sum(a["chosen"] != b["chosen"] for a, b in zip(rows, base))}
    for lo, hi in BINS:
        index = [i for i, q in enumerate(root_q) if lo <= q < hi or (hi == 1.0 and q == 1.0)]
        entry = {"root_q": [lo, hi], "positions": len(index), "legal": None, "by_run": {}}
        if index:
            entry["legal"] = sum(len(base[i]["visits"]) for i in index) / len(index)
        for label, run in runs.items():
            if not index:
                continue
            rows = [run["rows"][i] for i in index]
            target = [[v / sum(r["visits"]) for v in r["visits"]] for r in rows]
            entry["by_run"][label] = {
                "visited": sum(sum(v > 0 for v in r["visits"]) for r in rows) / len(rows),
                "target_entropy": sum(map(_entropy, target)) / len(rows),
                "kl_target_prior": sum(_kl(t, r["prior"]) for t, r in zip(target, rows)) / len(rows),
                "root_q": sum(sum(r["value_sums"]) / sum(r["visits"]) for r in rows) / len(rows),
            }
        summary["bins"].append(entry)
    (OUT / "mechanism.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False)[:4000])


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("positions")
    p.add_argument("--device", default="cuda")
    p = sub.add_parser("search")
    p.add_argument("--reduction", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--extension")
    p.add_argument("--device", default="cuda")
    p = sub.add_parser("report")
    p.add_argument("labels", nargs="+")
    args = parser.parse_args()
    {"positions": positions, "search": search, "report": report}[args.action](args)


if __name__ == "__main__":
    main()
