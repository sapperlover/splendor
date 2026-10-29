"""Create an execution-only release without editing the frozen experiment."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[2]
ART = Path(__file__).resolve().parent
NAME = "entity_attention_128x2_opponent_policy_r21_r37_20261024"
BASE = ROOT / "runtime" / NAME / "source"
DESTINATION = ROOT / "runtime" / NAME / "source_prefetch_20261024"
CONFIG = Path("configs/entity_attention_v1/offline_opponent_policy_r21_r37_prefetch_20261024.json")
MODULES = ("src/splendor/training_prefetch.py", "src/splendor/alphazero_train.py",
           "src/splendor/offline_objectives.py", "src/splendor/offline_continuation.py")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if DESTINATION.exists():
        raise ValueError("Execution release already exists")
    shutil.copytree(BASE, DESTINATION, ignore=shutil.ignore_patterns("__pycache__"))
    for relative in MODULES:
        shutil.copyfile(ROOT / relative, DESTINATION / relative)
    raw = json.loads((ROOT / "configs/entity_attention_v1/offline_opponent_policy_r21_r37_20261024.json").read_text())
    raw.update(cache_mib=18432, training_workers=4)
    (ROOT / CONFIG).write_text(json.dumps(raw, sort_keys=True, indent=2)+"\n")
    shutil.copyfile(ROOT / CONFIG, DESTINATION / CONFIG)
    record = {"protocol": "training-execution-release-v1", "source": str(DESTINATION),
        "base_source": str(BASE), "config": str(CONFIG), "cache_mib": 18432, "training_workers": 4,
        "changes": {rel: {"before": sha(BASE / rel) if (BASE / rel).exists() else None,
                           "after": sha(DESTINATION / rel)} for rel in MODULES},
        "snapshot_files": {str(p.relative_to(DESTINATION)): sha(p) for p in sorted(DESTINATION.rglob("*"))
                           if p.is_file() and "__pycache__" not in p.parts},
        "scope": "Current policy104k only; original snapshots/config/checkpoints retained. Same labels, RNG sequence, optimizer and validation schedule. Cache/CPU batch preparation are execution settings."}
    (ART / "source.json").write_text(json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2)+"\n")
    print(json.dumps({k: v for k, v in record.items() if k != "snapshot_files"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
