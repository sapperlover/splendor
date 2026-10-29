"""Two independent R37-initialized104k offline runs with the original offline settings."""
from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import importlib.util
from pathlib import Path
import shutil
import subprocess

from splendor.actor_critic import model_config_from_dict
from splendor.alphazero_checkpoint import load_alphazero_checkpoint
from splendor.alphazero_recording import read_json, sha256
from artifacts.entity_attention_r37_104k_offline_lr_20261027.train import R37OfflineConfig, prepare_inputs

ROOT = Path(__file__).resolve().parents[2]
SHARED = Path(__file__).resolve().parent
BASE_ART = ROOT / "artifacts/entity_attention_128x2_noaux_r21_r37_20261026"
BASE_CONFIG = ROOT / "configs/entity_attention_v1/offline_noaux_r21_r37_20261026.json"
TRAIN_MODULE = "artifacts.entity_attention_r37_104k_offline_lr_20261027.train"
VARIANTS = {"lr1e4": 1e-4, "lr2e5": 2e-5}


def configured_entry(key):
    reference = read_json(BASE_ART / "plan.json")
    template = ROOT / reference["reference_runner"]["path"]
    if sha256(template) != reference["reference_runner"]["sha256"]:
        raise ValueError("Reference offline supervisor changed")
    spec = importlib.util.spec_from_file_location("r37_offline_pipeline", template)
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    entry.__doc__ = f"R37 weights, fresh offline optimizer, {key}; train/select and original R37 arenas."
    name = f"entity_attention_r37_104k_{key}_20261027"
    entry.ART, entry.OUTPUT, entry.RUNTIME = ROOT / "artifacts" / name, ROOT / "runs" / name, ROOT / "runtime" / name
    entry.CONFIG = ROOT / f"configs/entity_attention_v1/offline_r37_104k_{key}_20261027.json"
    original_child, original_evaluation, original_report = entry.run_child, entry.evaluation_config, entry.report

    def prepare():
        anchor = reference["opponents"]["R37"]
        expected = copy.deepcopy(read_json(BASE_CONFIG))
        expected.update(initial_checkpoint=anchor["path"], initial_checkpoint_sha256=anchor["sha256"])
        expected["learner"]["learning_rate"] = VARIANTS[key]
        raw = read_json(entry.CONFIG)
        if raw != expected:
            raise ValueError("Only R37 initialization and the requested LR may differ from offline defaults")
        if sha256(Path(anchor["path"])) != anchor["sha256"]:
            raise ValueError("Raw R37 checkpoint changed")
        parent = load_alphazero_checkpoint(anchor["path"])
        if asdict(model_config_from_dict(parent.model_config)) != asdict(model_config_from_dict(raw["model_config"])):
            raise ValueError("R37/offline model configuration differs")
        config = R37OfflineConfig.from_dict(raw)
        _, _, _, fixed, training = prepare_inputs(config, entry.OUTPUT)
        if (training["splits"] != reference["splits"] or training["validation_steps"] != reference["validation_steps"]
                or read_json(entry.OUTPUT / "dataset.json")["digest"] != reference["dataset_digest"]
                or fixed["digest"] != reference["fixed_validation_digest"]):
            raise ValueError("Original104k pool, split or validation schedule differs")
        entry.freeze(entry.ART / "evaluation_templates.json", read_json(BASE_ART / "evaluation_templates.json"))
        for filename, expected_sha in reference["catalog_files"].items():
            source = BASE_ART / "catalog" / filename
            if sha256(source) != expected_sha or sha256(ROOT / "data" / filename) != expected_sha:
                raise ValueError("Catalog changed")
            target = entry.ART / "catalog" / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copyfile(source, target)
            if sha256(target) != expected_sha:
                raise ValueError("Frozen catalog changed")
        plan = {**reference, "experiment": name, "variant": key, "learning_rate": VARIANTS[key],
            "config": entry.identity(entry.CONFIG), "runner": entry.identity(Path(__file__)),
            "launcher": entry.identity(entry.ART / "run.py"),
            "training_entry": entry.identity(SHARED / "train.py"), "training_module": TRAIN_MODULE,
            "baseline_config": entry.identity(BASE_CONFIG), "reference_plan": entry.identity(BASE_ART / "plan.json"),
            "initial_checkpoint": anchor, "source_total_train_step": parent.total_train_step,
            "initialization": {"kind": "checkpoint_weights_only", "optimizer": "fresh AdamW",
                "preserve_optimizer": False, "sampling_seed": config.sampling_seed,
                "initialization_seed": config.initialization_seed, "local_step_starts_at_zero": True},
            "evaluation_templates": entry.identity(entry.ART / "evaluation_templates.json"),
            "scope": "R37 weights only; fresh AdamW and original offline RNG seeds. All auxiliaries off. Original104k NPZ/splits/main labels,lr as requested; all other offline fields preserved including18GiB,4shared workers,streaming/state-only/batched scalars,12value exposures,350552maxsteps,5% full mainKL+MSE3-increase stop/minimum selection. Default original R37greedy1000pair then searchA500pair bothS1024N1 total1024; train action only trains/selects. Command-only delivery; no queue or execution started."}
        entry.freeze(entry.ART / "plan.json", plan)
        return plan

    def frozen_source():
        destination, record_path = entry.RUNTIME / "source", entry.ART / "source.json"
        if not record_path.exists():
            # Scope the patch to code/configuration, excluding staged historical raw-data removals.
            code_paths = ["src", "scripts", "configs", str(SHARED.relative_to(ROOT))]
            patch = subprocess.check_output(["git", "diff", "HEAD", "--", *code_paths], cwd=ROOT)
            files = set(ROOT.glob("src/splendor/**/*.py")) | set(ROOT.glob("scripts/*.py"))
            files.update([entry.CONFIG, BASE_CONFIG, template, Path(__file__), entry.ART / "run.py", SHARED / "train.py", SHARED / "validate.py",
                ROOT / f"scripts/run_r37_104k_offline_{key}_20261027.sh", ROOT / "src/splendor/native_search.cpp",
                *ROOT.glob("src/splendor/_native_search*.so")])
            for source in sorted(files):
                target = destination / source.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
            for source in (entry.ART / "catalog").iterdir():
                target = destination / "data" / source.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
            (destination / "dirty.patch").write_bytes(patch)
            record = {"git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "git_status_scoped": subprocess.check_output(["git", "status", "--short", "--", *code_paths], cwd=ROOT, text=True),
                "patch_scope": code_paths, "dirty_diff_sha256": hashlib.sha256(patch).hexdigest(),
                "snapshot_files": {str(p.relative_to(destination)): sha256(p) for p in sorted(destination.rglob("*"))
                                   if p.is_file() and "__pycache__" not in p.parts}}
            entry.freeze(record_path, record)
        for relative, expected in read_json(record_path)["snapshot_files"].items():
            if sha256(destination / relative) != expected:
                raise ValueError(f"Frozen source changed: {relative}")
        return destination

    def run_child(command, phase, source):
        command = list(command)
        if phase in ("train", "select"):
            position = command.index("-m") + 1
            assert command[position] == "splendor.offline_continuation"
            command[position] = TRAIN_MODULE
        return original_child(command, phase, source)

    def evaluation_config(plan, selection, panel):
        result = original_evaluation(plan, selection, panel)
        result["policy_a"]["name"] = name
        result["metadata"].update(initialization="R37 weights only; fresh offline AdamW/RNG", learning_rate=VARIANTS[key])
        return result

    def report(plan):
        result = original_report(plan)
        path = entry.ART / "results.md"
        path.write_text(path.read_text().replace("第21–37轮固定池短期value离线训练",
            f"R37初值／104k池／lr={VARIANTS[key]:g}无辅助离线训练"), encoding="utf-8")
        return result

    entry.prepare, entry.frozen_source, entry.run_child = prepare, frozen_source, run_child
    entry.evaluation_config, entry.report = evaluation_config, report
    return entry
