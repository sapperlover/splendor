"""Verify frozen resume identities without running training or arenas."""
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import json

from splendor.alphazero_checkpoint import load_alphazero_checkpoint
from splendor.alphazero_recording import read_json, sha256, write_json
from resume_policy import configured_entry

ROOT = Path(__file__).resolve().parents[2]
ART = Path(__file__).resolve().parent


def main():
    entry = configured_entry()
    plan = entry.prepare()
    source = entry.frozen_source()
    progress = read_json(entry.OUTPUT / "progress.json")
    checkpoint = Path(progress["checkpoint"])
    saved = load_alphazero_checkpoint(checkpoint)
    run_plan = read_json(entry.OUTPUT / "plan.json")
    assert progress["step"] == saved.round_train_step == 115000
    assert run_plan["replay_identity"] == progress["replay_identity"] == saved.replay_identity
    assert saved.round_steps_target == progress["steps_target"] == plan["steps_target"]
    fixed = read_json(entry.OUTPUT / "fixed_validation.json")
    assert fixed["digest"] == plan["fixed_validation_digest"]
    execution = read_json(ART / "source.json")
    assert entry.CONFIG == ROOT / execution["config"]
    assert entry.identity(entry.ART / "source.json")["sha256"] == sha256(ART / "source.json")
    for relative in execution["changes"]:
        assert sha256(ROOT / relative) == sha256(source / relative)
    before = read_json(ART / "before.json")
    changed = [relative for relative, expected in before["protected"].items() if sha256(ROOT / relative) != expected]
    assert not changed, changed
    result = dict(time=datetime.now(ZoneInfo("Asia/Singapore")).isoformat(),
        status="verified_prepared_only", production_restart=False, checkpoint=str(checkpoint),
        checkpoint_sha256=sha256(checkpoint), step=115000, source=str(source),
        replay_identity_unchanged=True, fixed_validation_digest=fixed["digest"],
        protected_files_unchanged=len(before["protected"]), snapshot_files_verified=len(execution["snapshot_files"]),
        cache_mib=execution["cache_mib"], training_workers=execution["training_workers"],
        launcher_sha256=sha256(ROOT / "scripts/run_attention128_policy104k_20261024.sh"),
        commands=dict(current_chain="bash scripts/run_attention128_policy104k_20261024.sh",
            original_serial_queue="bash scripts/run_attention128_auxiliary_queue_20261024.sh"))
    write_json(ART / "verification.json", result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
