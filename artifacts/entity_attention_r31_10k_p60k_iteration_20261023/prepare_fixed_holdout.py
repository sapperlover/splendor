"""Freeze the user-approved R26-R31 holdout and a new diagnostic runtime release."""
from pathlib import Path
import shutil

from splendor.alphazero_recording import read_json, sha256, source_identity, write_json
from splendor.fixed_holdout import PROTOCOL, check_disjoint
from splendor.offline_train import _digest

ROOT = Path(__file__).resolve().parents[2]
ART = Path(__file__).resolve().parent
RUN = ROOT / "runs" / ART.name
DEST = ROOT / "runtime" / ART.name / "source_fixed_holdout_20261023"
REPORT = ART / "fixed_holdout_20261023"


def freeze(path, value):
    if path.exists() and read_json(path) != value:
        raise ValueError(f"frozen input changed: {path}")
    if not path.exists():
        write_json(path, value)


def copy_frozen(source, destination):
    if destination.exists():
        if sha256(destination) != sha256(source):
            raise ValueError(f"runtime release is immutable: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def main():
    config_path = ROOT / "configs/entity_attention_v1/random_budget_r31_10k_p60k_20261023.json"
    config = read_json(config_path)
    history = Path(config["history_manifest"]["path"])
    if sha256(history) != config["history_manifest"]["sha256"]:
        raise ValueError("original history manifest changed")
    import json
    games = [json.loads(line) for line in history.read_text().splitlines()]
    # Historical data round_index = lineage R + 42.
    selected = [g for g in games if 68 <= g["round_index"] <= 73 and g["split"] == "validation"]
    if {g["round_index"] for g in selected} != set(range(68, 74)):
        raise ValueError("R26-R31 validation rounds are incomplete")
    sources = []
    parent = ROOT / "runs/entity_attention_r20_s256n4_noble12_iteration_20261022"
    for lineage in range(26, 32):
        stage = parent / "rounds" / f"round-{lineage-1:03d}"
        index, collection = stage / "collection_index.json", stage / "selfplay/manifest.json"
        sources.append({"lineage": lineage,
            "collection_index": {"path": str(index), "sha256": sha256(index)},
            "collection_manifest": {"path": str(collection), "sha256": sha256(collection)},
            "collection_config": read_json(collection)["config"]})
    records = [{**g, "archive_sha256": sha256(Path(g["path"]))} for g in selected]
    manifest = {"protocol": PROTOCOL, "lineage_rounds": list(range(26, 32)),
        "selection": "all original whole-game validation; fixed R26-R31 for all model rounds, no resampling",
        "history_manifest": config["history_manifest"], "sources": sources, "games": records,
        "games_count": len(records), "value_positions": sum(g["decision_count"] for g in records),
        "policy_positions": sum(g["policy_samples"] for g in records)}
    manifest["digest"] = _digest(manifest)
    for row in read_json(RUN / "state.json")["completed_rounds"]:
        check_disjoint(manifest, read_json(Path(row["snapshot"]["path"]))["games"])
    manifest_path = REPORT / "manifest.json"
    freeze(manifest_path, manifest)
    settings = {"manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)}, "batch_size": 512}
    # Keep the original runtime immutable; change only the diagnostic module/driver.
    old_source = ROOT / "runtime" / ART.name / "source"
    original_source = read_json(ART / "source.json")
    before = {"config": {"path": str(config_path), "sha256": sha256(config_path)},
              "state": {"path": str(RUN / "state.json"), "sha256": sha256(RUN / "state.json")}}
    replacements = {"src/splendor/random_budget_iteration.py", "src/splendor/fixed_holdout.py"}
    for relative, expected in original_source["snapshot_files"].items():
        source = old_source / relative
        if sha256(source) != expected:
            raise ValueError(f"original runtime changed: {relative}")
        if relative in replacements:
            continue
        destination = DEST / relative
        copy_frozen(source, destination)
    for relative in replacements:
        destination = DEST / relative
        copy_frozen(ROOT / relative, destination)
    files = {str(p.relative_to(DEST)): sha256(p) for p in sorted(DEST.rglob("*"))
             if p.is_file() and "__pycache__" not in p.parts}
    freeze(REPORT / "runtime_source.json", {"base_source": {"path": str(ART / "source.json"),
        "sha256": sha256(ART / "source.json")}, "runtime": str(DEST), "snapshot_files": files,
        "changed_files": sorted(replacements), "scope": "original frozen learner/model/native/data, diagnostic integration only"})
    if "fixed_holdout" in config and config["fixed_holdout"] != settings:
        raise ValueError("requested holdout differs from configured panel")
    if "fixed_holdout" not in config:
        freeze(REPORT / "before_integration.json", before)
        config["fixed_holdout"] = settings
        write_json(config_path, config)
    if not (REPORT / "preparation.json").exists():
        freeze(REPORT / "preparation.json", {"fixed_holdout": settings,
            "validation_steps": config["validation_steps"], "training_disjoint_completed_rounds": 6,
            "workspace": source_identity(ROOT), "command": "conda run -n splendor env PYTHONPATH=src python artifacts/" + ART.name + "/prepare_fixed_holdout.py"})
    print({k: manifest[k] for k in ("games_count", "value_positions", "policy_positions", "digest")}, flush=True)


if __name__ == "__main__":
    main()
