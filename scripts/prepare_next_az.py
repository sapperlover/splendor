"""Prepare/verify frozen next-AZ configurations; never execute an experiment."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch
from splendor.alphazero_loop import LoopConfig
from splendor.alphazero_recording import environment_identity, read_json, sha256, source_identity, write_json
from splendor.evaluation import FixedEvaluationConfig
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.selfplay import derive_seed

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/az_theta5_10x1000"
ARTIFACT = ROOT / "artifacts/az_theta5_10x1000_seed1"
INITIAL = "runs/az_loop_seed1/rounds/round-004/checkpoints/final.pt"
PPO = "runs/5c_main_seed1/checkpoints/local_005000000_lifetime_005512112.pt"
EXPECTED_CHECKPOINT_SHA = {
    "old_theta5": "e0cba7733de4977cd0a606d35d2f328f47f285a0be81a1bd3895633dd105f97b",
    "ppo_latest": "009d17d2e9bc0d879142d7ca3ff1777a6f8caa9dda9243c6a2d5eb3538b5d5b0",
}


def evaluation_search_seed(start: int) -> int:
    # Distinct high-bit domains survive XOR with the low-range policy seeds.
    bit = {11400000: 32, 11600000: 33, 11700000: 34, 11800000: 35,
           11900000: 36, 12000000: 37, 12100000: 38, 12400000: 40}[start]
    return (1 << bit) + start + 30000


def configurations() -> dict[str, dict]:
    main = read_json(ROOT / "configs/speed_2026-10-03/alphazero_cached_example.json")
    main.update(run_id="az_theta5_10x1000_seed1", num_rounds=10, new_games_per_round=1000,
                initial_checkpoint=INITIAL, replay_max_games=6000, replay_cache_bytes=4294967296,
                seeds={"collection": 11300000, "split": 11310000, "sampling": 11320000},
                diagnostics={"artifact_dir": "artifacts/az_theta5_10x1000_seed1", "fixed_positions": 2048,
                             "fixed_seed": 11340000, "resource_interval_seconds": 5.0})
    main["selfplay"]["search_config"]["seed"] = 11330000
    for name, start in (("mcts_vs_parent", 11400000), ("greedy_vs_initial", 11500000)):
        panel = main["evaluation"][name]
        panel.update(pairs=50, seed_start=start, policy_seed_base=start + 10000,
                     bootstrap_resamples=10000, bootstrap_seed=start + 20000)
        if name == "mcts_vs_parent":
            panel["search_config"]["seed"] = evaluation_search_seed(start)
    main["learner"]["max_steps_per_round"] = None
    files = {"formal.json": main}
    smoke = copy.deepcopy(main)
    smoke.update(run_id="az_theta5_diagnostics_smoke", num_rounds=2, new_games_per_round=6, replay_max_games=12,
                 replay_cache_bytes=64 * 1024 * 1024, seeds={"collection": 12300000, "split": 12310002, "sampling": 12320000},
                 diagnostics={"artifact_dir": "runtime/az_theta5_diagnostics_smoke/artifacts", "fixed_positions": 16,
                              "fixed_seed": 12340000, "resource_interval_seconds": 5.0})
    smoke["selfplay"]["validation_fraction"] = 0.5
    smoke["selfplay"]["search_config"].update(simulations=4, seed=12330000)
    smoke["selfplay"]["execution"].update(search_workers=1, total_active_games=4, max_batch_size=4, local_microbatch=4)
    smoke["learner"].update(batch_size=16, history_batch_size=8, max_steps_per_round=2, checkpoint_every_steps=1, validation_batch_size=16)
    smoke["evaluation"]["mcts_execution"].update(active_games=2, max_batch_size=4)
    for n, name in enumerate(("mcts_vs_parent", "greedy_vs_initial")):
        panel = smoke["evaluation"][name]
        start = 12400000 + n * 100000
        panel.update(pairs=1, seed_start=start, policy_seed_base=start + 10000, bootstrap_resamples=32, bootstrap_seed=start + 20000)
        if panel.get("search_config"):
            panel["search_config"].update(simulations=4, seed=evaluation_search_seed(start))
    files["smoke.json"] = smoke
    template = read_json(ROOT / "configs/speed_2026-10-03/evaluation_optimized.json")
    for r in (2, 4, 9):
        for anchor, opponent, start in (("theta5", INITIAL, 11800000 if r == 9 else 11600000), ("ppo", PPO, 11900000 if r == 9 else 11700000)):
            panel = copy.deepcopy(template)
            checkpoint = f"runs/{main['run_id']}/rounds/round-{r:03d}/checkpoints/final.pt"
            panel.update(pairs=200, seed_start=start, policy_seed_base=start + 10000,
                         bootstrap_seed=start + 20000, metadata={"panel_id": f"round-{r:03d}-vs-{anchor}",
                           "purpose": "RESERVED_FINAL" if r == 9 else "STAGE_DEV repeated at rounds 3 and 5", "run_id": main["run_id"],
                           "schedule": "external, GPU idle, serial; loop does not pause or schedule this panel"})
            for name, path, label in (("policy_a", checkpoint, f"new_round_{r:03d}"), ("policy_b", opponent, f"fixed_{anchor}")):
                panel[name]["checkpoint"] = path
                panel[name]["name"] = label
                panel[name]["search_config"]["seed"] = evaluation_search_seed(start)
            files[f"round-{r:03d}-vs-{anchor}.json"] = panel
            if r == 9:
                extended = copy.deepcopy(panel)
                fresh = 12000000 if anchor == "theta5" else 12100000
                extended.update(pairs=500, seed_start=fresh, policy_seed_base=fresh + 10000, bootstrap_seed=fresh + 20000)
                extended["metadata"].update(panel_id=f"round-009-vs-{anchor}-extension500", purpose="independent extension; record prior look and expansion decision")
                for name in ("policy_a", "policy_b"):
                    extended[name]["search_config"]["seed"] = evaluation_search_seed(fresh)
                files[f"round-009-vs-{anchor}-extension500.json"] = extended
    return files


def seed_inventory(main: dict, panels: dict[str, dict]) -> dict:
    config = LoopConfig.from_dict(main)
    rows = []
    games = set()
    splits = set()
    worlds = set()
    sampling = set()
    search_base = main["selfplay"]["search_config"]["seed"]
    for r in range(config.num_rounds):
        c, s, t = config.seeds.collection_for(r), config.seeds.split_for(r), config.seeds.sampling_for(r)
        game_values = [derive_seed(c, r, g, "game") for g in range(config.new_games_per_round)]
        split_values = [derive_seed(s, r, g, "split") for g in range(config.new_games_per_round)]
        world_values = [search_base ^ derive_seed(c, r, g, "search_world", p) for g in range(config.new_games_per_round) for p in (0, 1)]
        rows.append({"human_round": r + 1, "round_index": r, "collection": c, "split": s, "sampling": t,
                     "first_game_seed": game_values[0], "first_split_draw": split_values[0], "first_game_search_world_seeds": world_values[:2],
                     "expected_train_games": sum(v / (1 << 64) >= config.selfplay.validation_fraction for v in split_values),
                     "expected_validation_games": sum(v / (1 << 64) < config.selfplay.validation_fraction for v in split_values)})
        games.update(game_values); splits.update(split_values); worlds.update(world_values); sampling.add(t)
    evaluation = {}
    for name, panel in {**main["evaluation"], **panels}.items():
        if not isinstance(panel, dict) or "seed_start" not in panel:
            continue
        policies = [panel["policy_seed_base"] + 2 * pair + p for pair in range(panel["pairs"]) for p in (0, 1)]
        search = panel.get("search_config") or panel.get("policy_a", {}).get("search_config")
        evaluation[name] = {"pairs": panel["pairs"], "game_seed_start": panel["seed_start"], "game_seed_end_inclusive": panel["seed_start"] + panel["pairs"] - 1,
                            "policy_seed_start": policies[0], "policy_seed_end_inclusive": policies[-1],
                            "bootstrap_resamples": panel["bootstrap_resamples"], "bootstrap_seed": panel["bootstrap_seed"],
                            "search_seed_base": search["seed"] if search else None,
                            "search_world_first_pair": [search["seed"] ^ p for p in policies[:2]] if search else None}
    historical = []
    legacy_games = set()
    legacy_worlds = set()
    old_path = ROOT / "runs/az_loop_seed1/config.json"
    if old_path.exists():
        old = LoopConfig.from_dict(read_json(old_path)["config"])
        for r in range(old.num_rounds):
            c = old.seeds.collection_for(r)
            legacy_games.update(derive_seed(c, r, g, "game") for g in range(old.new_games_per_round))
            legacy_worlds.update(old.selfplay.search_config["seed"] ^ derive_seed(c, r, g, "search_world", p) for g in range(old.new_games_per_round) for p in (0, 1))
        historical.append({"path": str(old_path.relative_to(ROOT)), "sha256": sha256(old_path)})
    prior_ranges = []
    legacy_eval_worlds = set()
    for path in sorted((ROOT / "configs").rglob("*.json")):
        if CONFIG in path.parents:
            continue
        raw = read_json(path)
        sections = [raw] if isinstance(raw, dict) else []
        if isinstance(raw, dict):
            sections += list(raw.get("evaluation", {}).values())
        for section in sections:
            if isinstance(section, dict) and "seed_start" in section and "pairs" in section:
                prior_ranges.append({"path": str(path.relative_to(ROOT)), "start": section["seed_start"], "end": section["seed_start"] + section["pairs"] - 1})
                searches = [section["search_config"]] if section.get("search_config") else [section[p]["search_config"] for p in ("policy_a", "policy_b") if isinstance(section.get(p), dict) and section[p].get("search_config")]
                for search in searches:
                    legacy_eval_worlds.update(search.get("seed", 0) ^ (section["policy_seed_base"] + 2 * pair + player)
                                             for pair in range(section["pairs"]) for player in (0, 1))
        if isinstance(raw, dict) and "execution" in raw and "num_games" in raw and "seed" in raw and "round_index" in raw:
            c, r = raw["seed"], raw["round_index"]
            legacy_games.update(derive_seed(c, r, g, "game") for g in range(raw["num_games"]))
            if "search_config" in raw:
                legacy_worlds.update(raw["search_config"]["seed"] ^ derive_seed(c, r, g, "search_world", p) for g in range(raw["num_games"]) for p in (0, 1))
            historical.append({"path": str(path.relative_to(ROOT)), "sha256": sha256(path)})
    collisions = []
    panel_worlds = {name: {panel["search_seed_base"] ^ p for p in range(panel["policy_seed_start"], panel["policy_seed_end_inclusive"] + 1)}
                    for name, panel in evaluation.items() if panel["search_seed_base"] is not None}
    intentional_reuse = []
    names = list(panel_worlds)
    for i, name in enumerate(names):
        inputs = panel_worlds[name]
        if inputs & (legacy_eval_worlds | legacy_worlds | worlds):
            collisions.append(f"{name} search RNG overlaps historical or training search RNG")
        for other in names[i + 1:]:
            overlap = inputs & panel_worlds[other]
            if not overlap:
                continue
            if name.startswith(("round-002-", "round-004-")) and other.startswith(("round-002-", "round-004-")) and name.split("-vs-")[1] == other.split("-vs-")[1]:
                intentional_reuse.append({"panels": [name, other], "search_inputs_reused": len(overlap), "reason": "same STAGE_DEV anchor at rounds 3 and 5"})
            else:
                collisions.append(f"search XOR inputs overlap: {name} / {other}")
    if games & legacy_games or worlds & legacy_worlds:
        collisions.append("derived collection/search seeds overlap historical configurations")
    for name, panel in evaluation.items():
        for old in prior_ranges:
            if max(panel["game_seed_start"], old["start"]) <= min(panel["game_seed_end_inclusive"], old["end"]):
                collisions.append(f"{name} overlaps {old['path']}")
        if games & set(range(panel["game_seed_start"], panel["game_seed_end_inclusive"] + 1)):
            collisions.append(f"{name} overlaps derived training game seeds")
    if collisions or len(games) != config.num_rounds * config.new_games_per_round or len(worlds) != 2 * config.num_rounds * config.new_games_per_round:
        raise ValueError(f"seed audit failed: {collisions}")
    return {"protocol": "az-seed-inventory-v1", "base": main["seeds"], "training": rows, "evaluation": evaluation,
            "derivation": "LoopSeeds uses derive_seed(base, round, 0, 'splendor-alphazero-loop-seed-v1:<domain>'); game/split use derive_seed(derived_base, round, game, domain); search world = search_config.seed XOR derive_seed(collection, round, game, 'search_world', player); behavior = derive_seed(collection, round, game, 'search_sampling', player, decision). Evaluation policy = base+2*pair+player; native search = search seed XOR policy seed.",
            "audit": {"collisions": collisions, "unique_training_game_seeds": len(games), "unique_training_split_draws": len(splits), "unique_search_world_seeds": len(worlds),
                      "historical_configs": historical, "historical_evaluation_game_ranges": prior_ranges,
                      "intentional_development_panel_reuse": intentional_reuse,
                      "search_xor_inputs_audited": {name: {"count": len(values), "min": min(values), "max": max(values)} for name, values in panel_worlds.items()},
                      "scope": "derived finite game/search inputs and discovered configuration game ranges; behavior decision counts exist only after collection; distinct seeds do not prove statistical independence. Stage DEV intentionally reuses seeds at rounds 3/5; final inputs are reserved, never viewed during tuning."}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write frozen configs and preparation manifest; no experiment")
    args = parser.parse_args()
    files = configurations()
    if args.write:
        for name, raw in files.items():
            write_json(CONFIG / name, raw)
    else:
        for name, raw in files.items():
            if read_json(CONFIG / name) != raw:
                raise ValueError(f"frozen preparation config differs: {name}")
    for name, raw in files.items():
        if name in ("formal.json", "smoke.json"):
            LoopConfig.from_dict(raw)
        else:
            FixedEvaluationConfig.from_dict(raw)
    seeds = seed_inventory(files["formal.json"], {k: v for k, v in files.items() if k not in ("formal.json", "smoke.json")})
    smoke = seed_inventory(files["smoke.json"], {})
    if not all(row["expected_train_games"] and row["expected_validation_games"] for row in smoke["training"]):
        raise ValueError("smoke needs both train and validation games in each round")
    checkpoints = {}
    with torch.random.fork_rng(devices=[]):
        for label, name in (("old_theta5", INITIAL), ("ppo_latest", PPO)):
            path = ROOT / name
            if sha256(path) != EXPECTED_CHECKPOINT_SHA[label]:
                raise ValueError(f"frozen {label} checkpoint SHA changed: {path}")
            model = load_frozen_actor_critic_checkpoint(path, device="cpu")
            checkpoints[label] = {"path": str(path), "sha256": sha256(path), "model_config": asdict(model.config), "parameter_count": sum(p.numel() for p in model.parameters())}
    manifest = {"protocol": "az-training-preparation-v1", "state": "prepared_not_started", "run_id": files["formal.json"]["run_id"],
                "checkpoints": checkpoints, "config_sha256": {name: sha256(CONFIG / name) for name in files},
                "source": source_identity(ROOT), "environment": environment_identity(), "seed_inventory": "seeds.json",
                "formal_output_exists": (ROOT / "runs" / files["formal.json"]["run_id"]).exists(),
                "not_executed": ["formal collection/training", "stage/final/extension panels", "fixed-data 4/8", "separate GPU smoke"],
                "future_checkpoint_validation": "stage configs parsed; new round-002/004/009 checkpoint identities must be checked when generated"}
    if args.write:
        write_json(ARTIFACT / "seeds.json", seeds)
        write_json(ARTIFACT / "smoke_seeds.json", smoke)
        write_json(ARTIFACT / "preparation_manifest.json", manifest)
    print(json.dumps({"state": "prepared_not_started", "configs_verified": len(files), "initial_checkpoints": checkpoints, "formal_output_exists": manifest["formal_output_exists"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
