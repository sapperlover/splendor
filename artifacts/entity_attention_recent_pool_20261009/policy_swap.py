"""Final policy plus exposure-4 value: verify both branches, then 200 DEV pairs."""
from __future__ import annotations

import argparse
import fcntl
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import torch

from compact_cache import CompactGameCache
from train import evaluate, load_records
from splendor.actor_critic import MODEL_WEIGHTS_FORMAT_VERSION, TorchStateBatch, tensorize_batch
from splendor.alphazero_recording import environment_identity, read_json, sha256, source_identity, write_json
from splendor.alphazero_train import LearnerConfig
from splendor.entity_attention import EntityAttentionPolicyValue
from splendor.evaluation import FixedEvaluationConfig
from splendor.features import FEATURE_SCHEMA_VERSION
from splendor.offline_train import make_index
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.paired_statistics import paired_bootstrap_difference, paired_bootstrap_mean, serialized_pair_scores
from splendor.replay import RULESET_ID
from splendor.search_inference import CompiledForward
from splendor.selfplay_data import collate_selfplay_samples

ART = Path(__file__).resolve().parent
ROOT = ART.parents[1]
RUNTIME = ROOT / "runtime/entity_attention_recent_pool_20261009/policy_swap"
CHECKPOINT = ROOT / "runs/entity_attention_recent_pool_20261009/compositions/final_policy_early_value.pt"
CONFIG = ART / "policy_swap_config.json"
VERIFICATION = ART / "policy_swap_verification.json"
HOLDOUT = ART / "policy_swap_holdout.json"
OUTPUT = ART / "battles/final_policy_early_value_vs_theta5000.json"
SOURCES = {
    "policy": ("runs/entity_attention_recent_pool_20261009/checkpoints/final.pt",
               "9faa66af917c597fbc62b823eeffea14e8403391415ce552d7a4114481d2784a", 103194),
    "value": ("runs/entity_attention_recent_pool_20261009/checkpoints/step-000034398.pt",
              "94f4ce89d27f42ec0c39e72055b460405ae36b8f9931051cbe360ee145ace5ec", 34398),
}


def prepare():
    identities = {}
    for name, (path, expected, step) in SOURCES.items():
        if sha256(ROOT / path) != expected:
            raise ValueError(f"{name} source changed")
        identities[name] = {"checkpoint": path, "sha256": expected, "step": step}
    metadata = {"rules_version": RULESET_ID, "composition": "policy(final103194), value(step34398); each retains its own complete encoder",
                "sources": identities, "inference_only": True}
    if not CHECKPOINT.exists():
        model = EntityAttentionPolicyValue().eval().requires_grad_(False)
        for name, identity in identities.items():
            parent = load_frozen_actor_critic_checkpoint(ROOT / identity["checkpoint"])
            getattr(model, f"{name}_model").load_state_dict(parent.state_dict(), strict=True)
        CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"checkpoint_schema": "splendor-frozen-actor-critic-v1",
                    "format_version": MODEL_WEIGHTS_FORMAT_VERSION,
                    "feature_schema_version": FEATURE_SCHEMA_VERSION,
                    "config": asdict(model.config), "state_dict": model.state_dict(), "metadata": metadata}, CHECKPOINT)
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    if payload["metadata"] != metadata:
        raise ValueError("composition identity changed")
    model = load_frozen_actor_critic_checkpoint(CHECKPOINT)
    for name, identity in identities.items():
        parent = load_frozen_actor_critic_checkpoint(ROOT / identity["checkpoint"])
        actual, expected = getattr(model, f"{name}_model").state_dict(), parent.state_dict()
        if actual.keys() != expected.keys() or any(not torch.equal(actual[k], expected[k]) for k in expected):
            raise ValueError(f"composition {name} branch weights changed")
    config = read_json(ART / "early_battle_config.json")
    config["policy_a"]["checkpoint"] = str(CHECKPOINT)
    config["policy_a"]["name"] = "final_policy103194_value34398"
    config["metadata"] = {"experiment": "entity_attention_recent_pool_20261009", "panel": "final_policy_early_value_vs_theta5000",
                          "sources": identities, "candidate_sha256": sha256(CHECKPOINT),
                          "template_sha256": sha256(ART / "early_battle_config.json"),
                          "seeds": "shared DEV with original step34398 and final103194 panels"}
    FixedEvaluationConfig.from_dict(config)
    if CONFIG.exists() and read_json(CONFIG) != config:
        raise ValueError("frozen composition battle configuration changed")
    write_json(CONFIG, config)
    return config


def verify_and_evaluate(config):
    training = read_json(ART / "config.json")
    manifest = Path(training["manifest"])
    if sha256(manifest) != training["manifest_sha256"]:
        raise ValueError("frozen heldout manifest changed")
    records = load_records(manifest)
    validation = [r for r in records if r["split"] == "validation"]
    for record in validation:
        stat = Path(record["path"]).stat()
        if (stat.st_size, stat.st_mtime_ns) != (record["size"], record["mtime_ns"]):
            raise ValueError(f"validation archive changed: {record['path']}")
    index = make_index({"games": records}, training["cache_mib"] * 1024**2)
    index.cache = CompactGameCache(training["cache_mib"] * 1024**2)
    cfg = LearnerConfig(**training["learner"])
    model = load_frozen_actor_critic_checkpoint(CHECKPOINT, device=cfg.device)
    if not VERIFICATION.exists():
        games = index.split_games("validation")
        selected = [games[len(games) * i // 32] for i in range(32)]
        refs = [(g, j) for g in selected for j in (0, g.decision_count - 1)]
        samples = [index.cache.sample(g, j) for g, j in refs]
        batch = tensorize_batch(collate_selfplay_samples(samples).decisions, device=cfg.device)
        state = TorchStateBatch(**{k: getattr(batch, k) for k in TorchStateBatch.__dataclass_fields__})
        policy = load_frozen_actor_critic_checkpoint(ROOT / SOURCES["policy"][0], device=cfg.device)
        value = load_frozen_actor_critic_checkpoint(ROOT / SOURCES["value"][0], device=cfg.device)
        compiled = CompiledForward(model, projection=None, buckets=(64,), compile_mode=None, failure_policy="fail_fast")
        errors = {}
        with torch.inference_mode():
            expected_policy, expected_value = policy(batch)[0], value(batch)[1]
            eager = model(batch)
            for name, actual, expected in (("eager_policy", eager[0], expected_policy), ("eager_value", eager[1], expected_value)):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                errors[name] = float((actual - expected).abs().max())
            actual_policy, actual_value = compiled.policy_value(batch)
            for name, actual, expected in (
                ("compiled_policy", actual_policy[:, :expected_policy.shape[1]], expected_policy),
                ("compiled_value", actual_value, expected_value),
                ("compiled_policy_only", compiled.policy_only(batch)[:, :expected_policy.shape[1]], expected_policy),
                ("compiled_value_only", compiled.value_only(state), expected_value),
            ):
                torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
                errors[name] = float((actual - expected).abs().max())
        if compiled.stats.failures or compiled.stats.failure_fallbacks or compiled.stats.fallbacks:
            raise ValueError("compiled verification fell back")
        write_json(VERIFICATION, {"checkpoint_sha256": sha256(CHECKPOINT), "sources": config["metadata"]["sources"],
            "references": [{"round_index": g.round_index, "game_id": g.game_id, "decision_index": j} for g, j in refs],
            "device": cfg.device, "eager_exact": True, "compiled_tolerance": {"rtol": 2e-5, "atol": 2e-5},
            "maximum_absolute_errors": errors, "compile_stats": compiled.stats.to_dict(),
            "verification": "complete source branch weights exact; source outputs exact in eager; actual CUDA compiled policy/value, policy-only and value-only agree"})
        del policy, value, compiled
        print(f"两路网络及 CUDA 编译前向核验通过: {errors}", flush=True)
    if read_json(VERIFICATION)["checkpoint_sha256"] != sha256(CHECKPOINT):
        raise ValueError("verification checkpoint mismatch")
    if not HOLDOUT.exists():
        started = time.perf_counter()
        learner = SimpleNamespace(model=model, device=torch.device(cfg.device), config=cfg)
        result = evaluate(learner, index, records)
        write_json(HOLDOUT, {"checkpoint": str(CHECKPOINT.relative_to(ROOT)), "checkpoint_sha256": sha256(CHECKPOINT),
            "sources": config["metadata"]["sources"], "manifest_sha256": training["manifest_sha256"],
            "evaluation": result, "seconds": time.perf_counter() - started})
    holdout = read_json(HOLDOUT)
    if (holdout["checkpoint_sha256"], holdout["manifest_sha256"]) != (sha256(CHECKPOINT), training["manifest_sha256"]):
        raise ValueError("heldout identity mismatch")
    result = holdout["evaluation"]
    for branch, metric, step in (("policy", "policy_kl", 103194), ("value", "value_mse", 34398)):
        parent = read_json(ART / f"full_holdout_step_{step:09d}.json")["evaluation"]
        if result["per_game"].keys() != parent["per_game"].keys():
            raise ValueError("heldout population mismatch")
        for key, actual in result["metrics"].items():
            expected = parent["metrics"][key]
            if actual["positions"] != expected["positions"] or actual["policy_weight"] != expected["policy_weight"]:
                raise ValueError("heldout metric population mismatch")
            if abs(actual[metric] - expected[metric]) > 1e-7:
                raise ValueError(f"heldout {branch} output mismatch: {key}")
    print(f"完整留出核验通过: {result['pool_weighted']['validation']}", flush=True)


def analyze():
    raw = read_json(OUTPUT)
    seeds, scores = serialized_pair_scores(raw)
    config = FixedEvaluationConfig.from_dict(read_json(CONFIG)).to_dict()
    if seeds != tuple(range(14900000, 14900200)) or raw["completed_games"] != 400 or raw["config"] != config:
        raise ValueError("battle population/config mismatch")
    for name, identity in raw["checkpoint_identity"].items():
        if identity["path"] != config[name]["checkpoint"] or sha256(Path(identity["path"])) != identity["sha256"]:
            raise ValueError("battle model identity mismatch")
    ci = paired_bootstrap_mean(scores, resamples=10000, seed=14920000).to_dict()
    if ci != raw["paired_bootstrap"]:
        raise ValueError("battle CI recomputation mismatch")
    result = {"protocol": "recent-pool-final-policy-early-value-v1", "status": "complete", "pairs": 200, "games": 400,
        "sources": read_json(CONFIG)["metadata"]["sources"], "score": raw["win_rate_score"], "ci95": ci,
        "completed_at": raw["updated_at"], "elapsed_seconds": raw["elapsed_seconds_this_invocation"],
        **{k: raw[k] for k in ("a_wins", "draws", "a_losses", "a_first", "a_second", "engineering_truncations", "exceptions", "illegal_actions", "invariant_failures", "policy_cycle_draws")},
        "output": str(OUTPUT.relative_to(ROOT)), "output_sha256": sha256(OUTPUT),
        "config_sha256": sha256(CONFIG), "verification_sha256": sha256(VERIFICATION), "holdout_sha256": sha256(HOLDOUT),
        "heldout": read_json(HOLDOUT)["evaluation"]["pool_weighted"]["validation"], "matched_contrasts": {},
        "scope": "shared DEV seeds; same opponent/search budget; differences compare scores against theta5000, not direct candidate battles; each network keeps its own encoder; no additional training"}
    for i, (name, path) in enumerate((
        ("early34398", ART / "battles/early_step000034398_vs_theta5000.json"),
        ("final103194", ART / "battles/final103194_vs_theta5000.json"),
    )):
        old = read_json(path)
        old_seeds, old_scores = serialized_pair_scores(old)
        if old["checkpoint_identity"]["policy_b"] != raw["checkpoint_identity"]["policy_b"]:
            raise ValueError("matched opponent identity mismatch")
        for key in ("policy_b", "execution", "game_seeds", "policy_seed_base", "max_completed_turns", "adjudicate_deterministic_policy_cycles"):
            if old["config"][key] != config[key]:
                raise ValueError(f"matched battle config differs: {key}")
        if old["config"]["policy_a"]["search_config"] != config["policy_a"]["search_config"]:
            raise ValueError("matched candidate search differs")
        result["matched_contrasts"][f"mixed_minus_{name}"] = {
            **paired_bootstrap_difference(seeds, scores, old_seeds, old_scores, resamples=10000, seed=30400000+i).to_dict(),
            "source": str(path.relative_to(ROOT)), "source_sha256": sha256(path)}
    write_json(ART / "policy_swap_summary.json", result)
    print(__import__("json").dumps(result, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "evaluate", "run", "analyze"), default="run", nargs="?")
    args = parser.parse_args()
    RUNTIME.mkdir(parents=True, exist_ok=True)
    with (RUNTIME / "run.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        config = prepare()
        if args.action == "prepare":
            return
        if args.action == "analyze":
            analyze()
            return
        session = RUNTIME / str(time.time_ns())
        identity = source_identity(ROOT, session)
        for name in ("policy_swap.py", "policy_swap_config.json", "train.py", "compact_cache.py"):
            shutil.copyfile(ART / name, session / name)
        write_json(session / "identity.json", {**identity, "environment": environment_identity(),
            "command": [sys.executable, *sys.argv], "start_unix": time.time(), "config_sha256": sha256(CONFIG)})
        verify_and_evaluate(config)
        if args.action == "evaluate":
            return
        command = [sys.executable, "-u", "-m", "splendor.evaluation", "--config", str(CONFIG), "--output", str(OUTPUT),
                   "--progress-every", "10", "--summary-every", "10"]
        if OUTPUT.exists() or OUTPUT.with_suffix(OUTPUT.suffix + ".games").exists():
            command.append("--resume")
        write_json(session / "battle_command.json", {"command": command, "start_unix": time.time()})
        try:
            subprocess.run(command, cwd=ROOT, check=True)
        except subprocess.CalledProcessError as error:
            if error.returncode == 130:
                raise SystemExit(130) from None
            raise
        analyze()


if __name__ == "__main__":
    main()
