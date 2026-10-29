"""Four real games, four CPU updates, exact shared-worker resume, base-only inference."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys

import torch

from splendor.alphazero_checkpoint import load_alphazero_checkpoint
from splendor.alphazero_recording import read_json, sha256, write_json
from splendor.engine import GameEngine
from splendor.mcts_policy import MCTSConfig
from splendor.native_policy import NativePUCTSearchPolicy
from splendor.offline_early_stopping import EarlyStoppingOfflineConfig, run
from splendor.offline_scratch import initialize_learner

ROOT = Path.cwd()
NAME = "entity_attention_128x2_noaux_r21_r37_20261026"
ART, WORK = ROOT / "artifacts" / NAME, ROOT / "runtime" / NAME / "cpu_validation"


def main():
    torch.set_num_threads(1)
    formal = EarlyStoppingOfflineConfig.from_dict(read_json(ROOT /
        "configs/entity_attention_v1/offline_noaux_r21_r37_20261026.json"))
    rows = [json.loads(line) for line in Path(formal.manifest).read_text().splitlines()]
    selected = [row for split in ("train", "validation")
        for row in [r for r in rows if r["round_index"] == 79 and r["split"] == split
                    and r["policy_samples"] > 0][:2]]
    assert len(selected) == 4
    WORK.mkdir(parents=True, exist_ok=True)
    manifest = WORK / "games.jsonl"
    manifest.write_text("".join(json.dumps(r, sort_keys=True)+"\n" for r in selected))
    train_positions = sum(r["decision_count"] for r in selected if r["split"] == "train")
    cfg = replace(formal, manifest=str(manifest), manifest_sha256=sha256(manifest), first_round=80,
        last_round=80, expected_games_per_round=4, cache_mib=8, training_workers=0,
        draw_exposures=(4*8-.01)/train_positions, validation_positions=16, validation_percent_step=50,
        learner=replace(formal.learner, device="cpu", batch_size=8, history_batch_size=4,
                        validation_batch_size=32, checkpoint_every_steps=2))
    fresh = initialize_learner(cfg, {"engineering_check": True}, 4)
    assert fresh.total_train_step == 0 and not fresh.optimizer.state
    assert fresh.model.config.architecture == "entity_attention_v1"
    assert fresh.model.trainable_parameter_count() == 583682
    assert not any("head" in key for key in fresh.model.state_dict())
    serial, resumed = WORK / "serial", WORK / "resumed"
    run(cfg, serial)
    parallel = replace(cfg, training_workers=4)
    run(parallel, resumed, max_steps_this_session=2)
    # Exercise the actual supervisor adapter's module routing in both phases.
    spec = importlib.util.spec_from_file_location("noaux_validation_entry", ART / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    entry = module.configured_entry()
    entry.RUNTIME = WORK
    execution_config = WORK / "config.json"
    from dataclasses import asdict
    write_json(execution_config, asdict(parallel))
    command = [sys.executable, "-u", "-m", "splendor.offline_continuation", "--config",
               str(execution_config), "--output", str(resumed)]
    source = ROOT / "runtime" / NAME / "source"
    entry.run_child(command, "train", source)
    entry.run_child(command + ["--select-only"], "select", source)
    a, b = [load_alphazero_checkpoint(p / "checkpoints/final.pt") for p in (serial, resumed)]
    assert a.total_train_step == b.total_train_step == 4
    assert a.replay_identity == b.replay_identity and a.sampling_rng_state == b.sampling_rng_state
    torch.testing.assert_close(a.torch_cpu_rng_state, b.torch_cpu_rng_state, rtol=0, atol=0)
    for key, value in a.model_state_dict.items():
        torch.testing.assert_close(value, b.model_state_dict[key], rtol=0, atol=0)
    assert a.optimizer_state_dict["param_groups"] == b.optimizer_state_dict["param_groups"]
    for pid, state in a.optimizer_state_dict["state"].items():
        for key, value in state.items():
            torch.testing.assert_close(value, b.optimizer_state_dict["state"][pid][key], rtol=0, atol=0)
    metrics = read_json(resumed / "full_validation/step-000000004.json")["metrics"]
    assert metrics == read_json(serial / "full_validation/step-000000004.json")["metrics"]
    assert not any(k.startswith(("opponent_", "shortterm_", "score_ownership_", "progress_")) for k in metrics)
    prefetch = read_json(resumed / "prefetch_progress.json")
    assert prefetch["workers"] == 4 and 0 < prefetch["arena_bytes"] < 8*1024**2
    selection = read_json(resumed / "selection.json")
    assert selection["selection_objective"] == min(
        p["metrics"]["policy_kl"]+p["metrics"]["value_loss"] for p in selection["candidates"])
    model = b.build_model()
    assert model.config.architecture == "entity_attention_v1"
    searches = []
    for seed in (902611, 902612, 902613):
        result = NativePUCTSearchPolicy(model, config=MCTSConfig(simulations=8, determinizations=1,
            seed=seed, root_noise_fraction=0.)).search(GameEngine.new(seed, mode="fast").decision())
        assert result.simulations == 8
        searches.append({"seed": seed, "simulations": result.simulations})
    write_json(ART / "cpu_validation.json", {"scope": "Finite CPU engineering check, no strength conclusion",
        "formal_config_sha256": sha256(ROOT / "configs/entity_attention_v1/offline_noaux_r21_r37_20261026.json"),
        "games": [{**r, "sha256": sha256(Path(r["path"]))} for r in selected],
        "fresh_model_and_adamw": True, "parameters": 583682, "auxiliary_heads": 0,
        "serial_vs_four_worker_resume_bitwise": ["model", "AdamW", "sampling_rng", "Torch_rng", "full_validation"],
        "steps_per_path": 4, "supervisor_train_and_select_module": "splendor.offline_early_stopping",
        "prefetch": prefetch, "metrics": metrics, "selection": selection, "native_searches": searches,
        "imported_modules": {name: __import__("splendor."+name, fromlist=[name]).__file__ for name in
                             ("offline_early_stopping", "offline_objectives", "training_prefetch", "alphazero_train")}})
    print({"status": "passed", "games": 4, "steps_per_path": 4, "native_simulations": 24})


if __name__ == "__main__":
    main()
