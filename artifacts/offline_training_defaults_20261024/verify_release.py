"""Verify execution releases and all original frozen identities, without training."""
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import json
import importlib.util

from splendor.alphazero_recording import read_json, sha256, write_json
from splendor.offline_continuation import ContinuationOfflineConfig, prepare_inputs
from run_variant import configured_entry

ROOT = Path(__file__).resolve().parents[2]
ART = Path(__file__).resolve().parent


def main():
    variants = {}
    for key in ("both104k", "value44k"):
        entry = configured_entry(key)
        baseline_plan = read_json(entry.ART / "plan.json")
        original_cfg = read_json(Path(baseline_plan["config"]["path"]))
        plan = entry.prepare()
        source = entry.frozen_source()
        actual = read_json(entry.CONFIG)
        comparison = dict(actual)
        assert comparison.pop("training_workers") == 4
        comparison["cache_mib"] = original_cfg["cache_mib"]
        assert comparison == original_cfg
        assert actual["cache_mib"] == 18432
        config = ContinuationOfflineConfig.from_dict(actual)
        index, _, identity, fixed, _ = prepare_inputs(config, entry.OUTPUT)
        original_run = read_json(entry.OUTPUT / "plan.json")
        assert identity == original_run["replay_identity"]
        assert fixed["digest"] == baseline_plan["fixed_validation_digest"]
        assert plan == baseline_plan
        record = read_json(ART / key / "source.json")
        assert entry.identity(entry.ART / "source.json")["sha256"] == sha256(ART / key / "source.json")
        for relative in record["changes"]:
            assert sha256(ROOT / relative) == sha256(source / relative)
        variants[key] = dict(source=str(source), config=str(entry.CONFIG), games=plan["games"],
            steps_target=plan["steps_target"], cache_mib=config.cache_mib, training_workers=config.training_workers,
            snapshot_files_verified=len(record["snapshot_files"]), original_plan_unchanged=True,
            replay_identity_unchanged=True, fixed_validation_digest=fixed["digest"],
            production_checkpoints=len(list((entry.OUTPUT / "checkpoints").glob("*.pt"))))
        del index
    before = read_json(ART / "before.json")
    policy_wrapper = ROOT / "artifacts/training_streaming_20261024/resume_policy.py"
    spec = importlib.util.spec_from_file_location("existing_policy_execution", policy_wrapper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    policy = module.configured_entry()
    policy.prepare()
    policy_source = policy.frozen_source()
    changed = [p for p, expected in before["protected"].items() if sha256(ROOT / p) != expected]
    assert not changed, changed
    # Both adds three shortterm arrays (12+1+1 bytes/position) to the measured
    # policy-only arena. Allow up to 63 padding bytes for each new array/game.
    ready = read_json(ROOT / "runs/entity_attention_128x2_opponent_policy_r21_r37_20261024/prefetch_progress.json")
    train = read_json(ROOT / "artifacts/entity_attention_128x2_continuation_r21_r37_20261024/plan.json")["splits"]["train"]
    upper = ready["arena_bytes"] + 14*train["positions"] + 3*63*train["games"]
    assert upper < 18*1024**3
    result = dict(time=datetime.now(ZoneInfo("Asia/Singapore")).isoformat(),
        status="prepared_verified_no_production_start", variants=variants,
        existing_policy_entry=dict(source=str(policy_source),
            checkpoint_step=read_json(policy.OUTPUT / "progress.json")["step"], prepared_verified=True),
        protected_files_unchanged=len(before["protected"]),
        full_pool_capacity_bound=dict(measured_policy_arena_bytes=ready["arena_bytes"],
            both_upper_bound_bytes=upper, chosen_capacity_bytes=18*1024**3,
            scope="Includes added shortterm arrays/alignment. The 44k value-only subset removes games/opponent arrays and also fits this conservative bound; not a process-RAM cap."),
        queue_command="bash scripts/run_attention128_auxiliary_queue_20261024.sh")
    write_json(ART / "verification.json", result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
