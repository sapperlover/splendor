"""Freeze the streaming-statistics execution release; never starts training."""
from pathlib import Path
import json
import shutil

from splendor.alphazero_recording import sha256

ROOT = Path(__file__).resolve().parents[2]
ART = Path(__file__).resolve().parent
NAME = "entity_attention_128x2_opponent_policy_r21_r37_20261024"
BASE = ROOT / "runtime" / NAME / "source_prefetch_20261024"
DESTINATION = ROOT / "runtime" / NAME / "source_streaming_20261024"
CONFIG = "configs/entity_attention_v1/offline_opponent_policy_r21_r37_prefetch_20261024.json"
MODULES = ("src/splendor/alphazero_train.py", "src/splendor/continuation_auxiliary.py")


def main():
    if DESTINATION.exists():
        raise ValueError("Execution release already exists")
    shutil.copytree(BASE, DESTINATION, ignore=shutil.ignore_patterns("__pycache__"))
    for relative in MODULES:
        shutil.copyfile(ROOT / relative, DESTINATION / relative)
    record = dict(protocol="training-execution-release-v1", source=str(DESTINATION), base_source=str(BASE),
        config=CONFIG, cache_mib=18432, training_workers=4, resume_step=115000,
        changes={rel: dict(before=sha256(BASE / rel), after=sha256(DESTINATION / rel)) for rel in MODULES},
        snapshot_files={str(p.relative_to(DESTINATION)): sha256(p) for p in sorted(DESTINATION.rglob("*"))
                        if p.is_file() and "__pycache__" not in p.parts},
        scope="policy104k only. Streaming round summary; state-only auxiliary value transfer; batched scalar reads. Algorithm/config/optimizer/RNG/validation unchanged. Prepared only; production training remains stopped.")
    (ART / "source.json").write_text(json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2)+"\n")
    print(json.dumps({k: v for k, v in record.items() if k != "snapshot_files"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
