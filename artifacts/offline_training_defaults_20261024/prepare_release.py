"""Prepare both remaining queue releases without starting training."""
from pathlib import Path
import json
import shutil

from splendor.alphazero_recording import read_json, sha256, write_json

ROOT = Path(__file__).resolve().parents[2]
ART = Path(__file__).resolve().parent
VARIANTS = {
    "both104k": ("entity_attention_128x2_continuation_r21_r37_20261024", "offline_continuation_r21_r37"),
    "value44k": ("entity_attention_128x2_shortterm_value_r21_r31_vs_r37_20261024", "offline_shortterm_value_r21_r31_vs_r37"),
}
MODULES = ("alphazero_train", "training_prefetch", "offline_objectives", "offline_continuation",
    "offline_scratch", "offline_train", "progress_auxiliary", "continuation_auxiliary", "hybrid_train")


def main():
    releases = {}
    for key, (name, stem) in VARIANTS.items():
        base = ROOT / "runtime" / name / "source"
        destination = ROOT / "runtime" / name / "source_offline_defaults_20261024"
        config = Path(f"configs/entity_attention_v1/{stem}_prefetch_20261024.json")
        if destination.exists() or (ROOT / config).exists():
            raise ValueError(f"Execution release already exists: {key}")
        raw = read_json(ROOT / f"configs/entity_attention_v1/{stem}_20261024.json")
        raw.update(cache_mib=18432, training_workers=4)
        write_json(ROOT / config, raw)
        shutil.copytree(base, destination, ignore=shutil.ignore_patterns("__pycache__"))
        changes = {}
        for module in MODULES:
            relative = f"src/splendor/{module}.py"
            shutil.copyfile(ROOT / relative, destination / relative)
            changes[relative] = dict(before=sha256(base / relative) if (base / relative).exists() else None,
                                     after=sha256(destination / relative))
        shutil.copyfile(ROOT / config, destination / config)
        record = dict(protocol="training-execution-release-v1", variant=key, source=str(destination),
            base_source=str(base), config=str(config), cache_mib=18432, training_workers=4,
            original_plan_sha256=sha256(ROOT / "artifacts" / name / "plan.json"), changes=changes,
            snapshot_files={str(p.relative_to(destination)): sha256(p) for p in sorted(destination.rglob("*"))
                if p.is_file() and "__pycache__" not in p.parts},
            scope="Execution release only: streaming summary, 4 readonly workers, state-only value transfer, batched scalar reads. Same seeds/data/split/losses/optimizer/validation/arenas. Prepared only; no production launch.")
        write_json(ART / key / "source.json", record)
        releases[key] = {k: v for k, v in record.items() if k not in ("snapshot_files", "changes")}
    write_json(ART / "releases.json", releases)
    print(json.dumps(releases, ensure_ascii=False))


if __name__ == "__main__":
    main()
