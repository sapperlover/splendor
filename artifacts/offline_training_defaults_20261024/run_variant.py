"""Use a new execution release while preserving the original frozen plan."""
from pathlib import Path
import importlib.util
import sys

from splendor.alphazero_recording import read_json, sha256

ROOT = Path(__file__).resolve().parents[2]
ART = Path(__file__).resolve().parent


def configured_entry(key):
    path = ROOT / "artifacts/entity_attention_auxiliary_queue_20261024/run_experiment.py"
    spec = importlib.util.spec_from_file_location(f"offline_defaults_{key}_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    entry = module.configure(key)
    baseline_config, baseline_prepare = entry.CONFIG, entry.prepare
    record_path = ART / key / "source.json"
    execution = read_json(record_path)
    execution_config = ROOT / execution["config"]
    entry.CONFIG = execution_config

    def prepare():
        entry.CONFIG = baseline_config
        try:
            return baseline_prepare()
        finally:
            entry.CONFIG = execution_config

    def frozen_source():
        source = Path(execution["source"])
        for relative, expected in execution["snapshot_files"].items():
            if sha256(source / relative) != expected:
                raise ValueError(f"Offline execution source changed: {relative}")
        if sha256(execution_config) != execution["snapshot_files"][execution["config"]]:
            raise ValueError("Offline execution config changed")
        return source

    original_identity = entry.identity

    def identity(path):
        return original_identity(record_path if path == entry.ART / "source.json" else path)

    entry.prepare, entry.frozen_source, entry.identity = prepare, frozen_source, identity
    return entry


if __name__ == "__main__":
    variant = sys.argv.pop(1)
    raise SystemExit(configured_entry(variant).main())
