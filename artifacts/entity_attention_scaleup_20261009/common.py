"""Shared helpers for the 2026-10-09 scale-up experiments.

The scale-up reuses the verified execution code of the two source experiments:

* `entity_attention_avg4_delta_kl_20261009` (fresh-M4 paired Delta-KL search/eval)
* `entity_attention_policy_history_20261009` (fixed-M4 A/B/C training harness)

Both are imported as modules; nothing in their directories is modified except that
new artifacts are written into ignored `runs/` and `runtime/` subdirectories and
new small summaries into this artifact directory.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = Path(__file__).resolve().parent
DELTA_DIR = ROOT / "artifacts/entity_attention_avg4_delta_kl_20261009"
HIST_DIR = ROOT / "artifacts/entity_attention_policy_history_20261009"
BEHAVIOR_DIR = ROOT / "artifacts/entity_attention_avg4_behavior_20261009"

LR_RUN = ROOT / "runs/entity_attention_scaleup_20261009"
LR_RUNTIME = ROOT / "runtime/entity_attention_scaleup_20261009"
LR_IDENTIFIER = "entity_attention_scaleup_20261009"
LR_VALUE = 2.5e-5
LR_STEPS = 500
LR_NODES = (0, 50, 100, 200, 500)
KEYS = ("r004", "r008", "r012")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def delta_module():
    return load_module("scaleup_delta_run", DELTA_DIR / "run.py")


def hist_module():
    return load_module("scaleup_hist_run", HIST_DIR / "run.py")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
                         encoding="utf-8")
    temporary.replace(path)


def git_identity(script_paths) -> dict:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                          capture_output=True, check=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, text=True,
                            capture_output=True, check=True).stdout
    return {
        "git_head": head,
        "git_status_sha256": hashlib.sha256(status.encode()).hexdigest(),
        "command": sys.argv,
        "scripts": {str(Path(p).relative_to(ROOT)): sha256(Path(p)) for p in script_paths},
    }
