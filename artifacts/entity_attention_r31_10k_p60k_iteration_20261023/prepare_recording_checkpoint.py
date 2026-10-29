"""Release only the recording/cadence patch over the frozen six-round driver."""
from pathlib import Path
import fcntl
import shutil
import subprocess

from splendor.alphazero_recording import read_json, sha256, write_json
from splendor.random_budget_iteration import Config, _algorithm_config

ROOT = Path(__file__).resolve().parents[2]
ART = Path(__file__).resolve().parent
REPORT = ART / "recording_checkpoint_20261023"
RUN = ROOT / "runs" / ART.name
BASE = ART / "six_round_holdout_20261023/runtime_source.json"
DEST = ROOT / "runtime" / ART.name / "source_recording_checkpoint_20261023"


def main():
    with (RUN / ".iteration.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        raw = read_json(ROOT / "configs/entity_attention_v1/random_budget_r31_10k_p60k_20261023.json")
        config = Config.from_dict(raw)
        if config.learner.checkpoint_every_steps != 500 or config.validation_steps != (0, 100, 250, 500, 1000, 2000, 3000):
            raise ValueError("expected 500-step recovery plus seven unchanged diagnostic nodes")
        if _algorithm_config(raw) != _algorithm_config(read_json(RUN / "config.json")):
            raise ValueError("training algorithm changed")
        record = REPORT / "runtime_source.json"
        if record.exists():
            release = read_json(record)
            for relative, expected in release["snapshot_files"].items():
                if sha256(DEST / relative) != expected:
                    raise ValueError(f"runtime release changed: {relative}")
            return
        if DEST.exists():
            raise ValueError("unregistered runtime destination already exists")
        base = read_json(BASE)
        for relative, expected in base["snapshot_files"].items():
            source = Path(base["runtime"]) / relative
            if sha256(source) != expected:
                raise ValueError(f"base runtime changed: {relative}")
            target = DEST / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        patches = read_json(REPORT / "patches.json")
        for relative, patch in patches.items():
            subprocess.run(["patch", "--batch", "--fuzz=0", "--no-backup-if-mismatch", "-p1"],
                           input=patch, text=True, cwd=DEST, check=True, capture_output=True)
        files = {str(p.relative_to(DEST)): sha256(p) for p in sorted(DEST.rglob("*")) if p.is_file()}
        if set(files) != set(base["snapshot_files"]):
            raise ValueError("release file set changed")
        changed = {name for name in files if files[name] != base["snapshot_files"][name]}
        if changed != set(patches):
            raise ValueError("release differs outside the authorized patch")
        write_json(record, {"runtime": str(DEST),
            "base_source": {"path": str(BASE), "sha256": sha256(BASE)},
            "patches_sha256": sha256(REPORT / "patches.json"),
            "snapshot_files": files, "changed_files": sorted(changed),
            "scope": "hash-only sampling records; 500-step recovery plus seven diagnostic nodes; unchanged model/native/catalog/evaluator"})
        print({"runtime": str(DEST), "changed_files": sorted(changed), "files": len(files)}, flush=True)


if __name__ == "__main__":
    main()
