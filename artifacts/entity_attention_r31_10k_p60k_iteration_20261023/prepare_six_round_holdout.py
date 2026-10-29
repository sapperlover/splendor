"""Release the previous-six-round schedule without modifying historical results."""
from pathlib import Path
import fcntl
import shutil

from splendor.alphazero_recording import read_json, sha256, source_identity, write_json
from splendor.random_budget_iteration import Config, _algorithm_config
from splendor.six_round_holdout import HoldoutSchedule

ROOT = Path(__file__).resolve().parents[2]
ART = Path(__file__).resolve().parent
RUN = ROOT / "runs" / ART.name
REPORT = ART / "six_round_holdout_20261023"
BASE_REPORT = ART / "fixed_holdout_20261023"
DEST = ROOT / "runtime" / ART.name / "source_six_round_holdout_20261023"


def freeze(path, value):
    if path.exists() and read_json(path) != value:
        raise ValueError(f"frozen release changed: {path}")
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
    with (RUN / ".iteration.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config_path = ROOT / "configs/entity_attention_v1/random_budget_r31_10k_p60k_20261023.json"
        raw = read_json(config_path)
        if "six_round_holdout" not in raw:
            freeze(REPORT / "before_update.json", {
                "config": {"path": str(config_path), "sha256": sha256(config_path), "content": raw},
                "state": {"path": str(RUN / "state.json"), "sha256": sha256(RUN / "state.json")}})
            raw["six_round_holdout"] = {**raw.pop("fixed_holdout"), "block_rounds": 6}
        config = Config.from_dict(raw)
        if _algorithm_config(raw) != _algorithm_config(read_json(RUN / "config.json")):
            raise ValueError("training algorithm changed")
        base = read_json(BASE_REPORT / "runtime_source.json")
        base_root = Path(base["runtime"])
        replacements = {"src/splendor/random_budget_iteration.py", "src/splendor/six_round_holdout.py"}
        for relative, expected in base["snapshot_files"].items():
            source = base_root / relative
            if sha256(source) != expected:
                raise ValueError(f"previous runtime changed: {relative}")
            if relative not in replacements:
                copy_frozen(source, DEST / relative)
        for relative in replacements:
            copy_frozen(ROOT / relative, DEST / relative)
        files = {str(p.relative_to(DEST)): sha256(p) for p in sorted(DEST.rglob("*"))
                 if p.is_file() and "__pycache__" not in p.parts}
        freeze(REPORT / "runtime_source.json", {"runtime": str(DEST),
            "base_source": {"path": str(BASE_REPORT / "runtime_source.json"),
                            "sha256": sha256(BASE_REPORT / "runtime_source.json")},
            "snapshot_files": files, "changed_files": sorted(replacements),
            "scope": "same frozen model/learner/native/catalog and unchanged fixed evaluator; add six-round scheduling"})
        schedule = HoldoutSchedule(config.six_round_holdout, RUN, config.learner.device, six_rounds=True)
        state = read_json(RUN / "state.json")
        for row in state["completed_rounds"]:
            observer = schedule.for_iteration(row["iteration"])
            from splendor.fixed_holdout import check_disjoint
            snapshot = Path(row["snapshot"]["path"])
            if sha256(snapshot) != row["snapshot"]["sha256"]:
                raise ValueError("completed replay snapshot changed")
            check_disjoint(observer.manifest, read_json(snapshot)["games"])
        next_iteration = len(state["completed_rounds"])+1
        upcoming = schedule.for_iteration(next_iteration)
        freeze(REPORT / "next_block_manifest.json", upcoming.manifest)
        write_json(config_path, raw)
        if not (REPORT / "preparation.json").exists():
            freeze(REPORT / "preparation.json", {"workspace": source_identity(ROOT),
                "six_round_holdout": config.six_round_holdout,
                "next_local_iteration": next_iteration, "next_lineage": 31+next_iteration,
                "next_manifest_digest": upcoming.manifest["digest"],
                "next_block_population": {k: upcoming.manifest[k]
                    for k in ("games_count", "policy_positions", "value_positions")},
                "historical_backfill": "cancelled by user; no R20-R31 forward evaluations launched",
                "command": "conda run -n splendor env PYTHONPATH=src python artifacts/" + ART.name + "/prepare_six_round_holdout.py"})
        print({"next_iteration": next_iteration, "source_iterations": upcoming.manifest["source_iterations"],
            "games": upcoming.manifest["games_count"]}, flush=True)


if __name__ == "__main__":
    main()
