"""Resume the existing policy104k chain using the recorded prefetch release."""
from pathlib import Path
import importlib.util

from splendor.alphazero_recording import read_json, sha256

ROOT = Path(__file__).resolve().parents[2]
ART = Path(__file__).resolve().parent


def configured_entry():
    path = ROOT / "artifacts/entity_attention_auxiliary_queue_20261024/run_experiment.py"
    spec = importlib.util.spec_from_file_location("policy_prefetch_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    entry = module.configure("policy104k")
    baseline_config, baseline_prepare = entry.CONFIG, entry.prepare
    execution = read_json(ART / "source.json")
    execution_config = ROOT / execution["config"]
    entry.CONFIG = execution_config

    def prepare():
        # Original plan/algorithm identities remain intact and reviewable.
        entry.CONFIG = baseline_config
        try:
            return baseline_prepare()
        finally:
            entry.CONFIG = execution_config

    def frozen_source():
        source = Path(execution["source"])
        for relative, expected in execution["snapshot_files"].items():
            if sha256(source / relative) != expected:
                raise ValueError(f"Prefetch execution source changed: {relative}")
        if sha256(execution_config) != execution["snapshot_files"][execution["config"]]:
            raise ValueError("Prefetch execution config changed")
        return source

    original_identity = entry.identity

    def identity(path):
        if path == entry.ART / "source.json":
            return original_identity(ART / "source.json")
        return original_identity(path)

    entry.prepare, entry.frozen_source, entry.identity = prepare, frozen_source, identity
    return entry


if __name__ == "__main__":
    raise SystemExit(configured_entry().main())
