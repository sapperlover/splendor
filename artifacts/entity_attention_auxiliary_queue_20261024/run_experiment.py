"""Three continuation ablations, sharing the existing train/select/arena pipeline."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import shutil

from splendor.alphazero_recording import read_json, sha256, source_identity, write_json
from splendor.offline_continuation import ContinuationOfflineConfig, initialize_learner, prepare_inputs

ROOT = Path(__file__).resolve().parents[2]
ART = Path(__file__).resolve().parent
PARENT_NAME = "entity_attention_128x2_shortterm_value_r21_r37_20261023"
PARENT_ART = ROOT / "artifacts" / PARENT_NAME
PARENT_CONFIG = ROOT / "configs/entity_attention_v1/offline_shortterm_value_r21_r37_20261023.json"
VARIANTS = {
    "policy104k": {"name": "entity_attention_128x2_opponent_policy_r21_r37_20261024",
        "config": "offline_opponent_policy_r21_r37_20261024.json", "value": 0., "policy": .15,
        "games": 104000, "parameters": 784778, "label": "R21–R37仅policy辅助"},
    "both104k": {"name": "entity_attention_128x2_continuation_r21_r37_20261024",
        "config": "offline_continuation_r21_r37_20261024.json", "value": .25, "policy": .15,
        "games": 104000, "parameters": 851853, "label": "R21–R37双辅助"},
    "value44k": {"name": "entity_attention_128x2_shortterm_value_r21_r31_vs_r37_20261024",
        "config": "offline_shortterm_value_r21_r31_vs_r37_20261024.json", "value": .25, "policy": 0.,
        "games": 44000, "parameters": 650757, "label": "R21–R31的44k池仅value辅助"},
}


def template():
    path = PARENT_ART / "run.py"
    spec = importlib.util.spec_from_file_location("continuation_parent_pipeline", path)
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    return entry


def configure(key):
    arm = VARIANTS[key]
    entry = template()
    entry.ART = ROOT / "artifacts" / arm["name"]
    entry.OUTPUT = ROOT / "runs" / arm["name"]
    entry.RUNTIME = ROOT / "runtime" / arm["name"]
    entry.CONFIG = ROOT / "configs/entity_attention_v1" / arm["config"]
    entry.BASE_CONFIG = PARENT_CONFIG
    parent_plan = read_json(PARENT_ART / "plan.json")
    parent_source = read_json(PARENT_ART / "source.json")
    if sha256(PARENT_CONFIG) != parent_plan["config"]["sha256"]:
        raise ValueError("Reference experiment configuration changed")
    if sha256(PARENT_ART / "run.py") != parent_plan["runner"]["sha256"]:
        raise ValueError("Reference pipeline changed")
    original_evaluation = entry.evaluation_config
    original_report = entry.report

    def prepare():
        raw = copy.deepcopy(read_json(PARENT_CONFIG))
        raw["continuation_auxiliary"].update(shortterm_value_weight=arm["value"], opponent_policy_weight=arm["policy"])
        if arm["games"] == 44000:
            rows = [json.loads(line) for line in Path(raw["manifest"]).read_text().splitlines()]
            rows = [row for row in rows if row["round_index"] <= 73]
            if len(rows) != 44000:
                raise ValueError("Expected exactly the original R21-R31 pool")
            manifest = entry.OUTPUT / "inputs/pool_games.jsonl"
            content = "".join(json.dumps(row, sort_keys=True, separators=(",", ":"))+"\n" for row in rows)
            manifest.parent.mkdir(parents=True, exist_ok=True)
            if manifest.exists() and manifest.read_text() != content:
                raise ValueError("Frozen44k pool changed")
            if not manifest.exists():
                manifest.write_text(content)
            raw.update(manifest=str(manifest), manifest_sha256=sha256(manifest), last_round=74,
                       expected_games_per_round=4000)
        entry.freeze(entry.CONFIG, raw)
        config = ContinuationOfflineConfig.from_dict(raw)
        _, _, _, fixed, training = prepare_inputs(config, entry.OUTPUT)
        if training["games"] != arm["games"]:
            raise ValueError("Ablation pool population differs")
        entry.freeze(entry.ART / "evaluation_templates.json", read_json(PARENT_ART / "evaluation_templates.json"))
        for filename, expected in parent_plan["catalog_files"].items():
            source = PARENT_ART / "catalog" / filename
            if sha256(source) != expected:
                raise ValueError("Reference catalog changed")
            target = entry.ART / "catalog" / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copyfile(source, target)
            if sha256(target) != expected:
                raise ValueError("Frozen ablation catalog changed")
        for opponent in parent_plan["opponents"].values():
            if sha256(Path(opponent["path"])) != opponent["sha256"]:
                raise ValueError("Raw R37 changed")
        plan = {**parent_plan, "experiment": arm["name"], "variant": key, "label": arm["label"],
            "reference_plan": entry.identity(PARENT_ART / "plan.json"),
            "reference_source": entry.identity(PARENT_ART / "source.json"),
            "config": entry.identity(entry.CONFIG), "runner": entry.identity(entry.ART / "run.py"),
            "shared_runner": entry.identity(Path(__file__)), "baseline_config": entry.identity(PARENT_CONFIG),
            "source_iterations": [21, 31 if arm["games"] == 44000 else 37],
            "source_rounds": [r for r in parent_plan["source_rounds"] if arm["games"] != 44000 or r["lineage_round"] <= 31],
            "absolute_rounds": [config.first_round, config.last_round], "games": training["games"],
            "splits": training["splits"], "dataset_digest": read_json(entry.OUTPUT / "dataset.json")["digest"],
            "fixed_validation_digest": fixed["digest"], "parameters": arm["parameters"],
            "steps_target": training["steps_target"], "validation_steps": training["validation_steps"],
            "sampling": training["sampling"], "continuation_auxiliary": raw["continuation_auxiliary"],
            "evaluation_templates": entry.identity(entry.ART / "evaluation_templates.json"),
            "scope": "Independent fresh model/AdamW, original reference seeds/learner/12value-exposures/mainKL+MSE stopping-selection/latest lightweight recording. Only auxiliary switches and requested44k pool differ. Each arm runs R37greedy1000pair then searchA500pair bothS1024N1 total1024; no inherited candidate weights. Disabled heads/targets/losses/diagnostics are absent. Current reference run and its frozen implementation are not changed."}
        entry.freeze(entry.ART / "plan.json", plan)
        return plan

    def frozen_source():
        destination, record_path = entry.RUNTIME / "source", entry.ART / "source.json"
        if not record_path.exists():
            record = source_identity(ROOT, destination)
            sources = [entry.CONFIG, entry.ART / "run.py", Path(__file__), PARENT_ART / "run.py",
                       ROOT / "src/splendor/native_search.cpp", *ROOT.glob("src/splendor/_native_search*.so")]
            for path in sources:
                target = destination / path.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
            for path in (entry.ART / "catalog").iterdir():
                target = destination / "data" / path.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
            record["snapshot_files"] = {str(p.relative_to(destination)): sha256(p)
                for p in sorted(destination.rglob("*")) if p.is_file() and "__pycache__" not in p.parts}
            record["reference_source_sha256"] = sha256(PARENT_ART / "source.json")
            record["changed_reference_modules"] = {p: {"reference": expected, "current": sha256(destination/p)}
                for p, expected in parent_source["snapshot_files"].items()
                if p.startswith("src/") and (destination/p).exists() and sha256(destination/p) != expected}
            entry.freeze(record_path, record)
        for relative, expected in read_json(record_path)["snapshot_files"].items():
            if sha256(destination / relative) != expected:
                raise ValueError(f"Frozen ablation source changed: {relative}")
        return destination

    def evaluation_config(plan, selection, panel):
        result = original_evaluation(plan, selection, panel)
        result["policy_a"]["name"] = arm["name"]
        result["metadata"].update(variant=key, continuation_auxiliary=plan["continuation_auxiliary"])
        return result

    def report(plan):
        result = original_report(plan)
        path = entry.ART / "results.md"
        text = path.read_text()
        text = text.replace("第21–37轮固定池短期value离线训练", arm["label"]+"离线训练")
        path.write_text(text)
        return result

    entry.prepare, entry.frozen_source = prepare, frozen_source
    entry.evaluation_config, entry.report = evaluation_config, report
    return entry


def main(key):
    return configure(key).main()
