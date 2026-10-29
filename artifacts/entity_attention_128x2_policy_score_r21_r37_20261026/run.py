"""Fresh attention128, opponent policy + score ownership, original 104k pool."""
from __future__ import annotations

import copy
from dataclasses import asdict
import importlib.util
from pathlib import Path
import shutil

from splendor.alphazero_recording import read_json, sha256, source_identity
from splendor.offline_continuation import ContinuationOfflineConfig, model_config, prepare_inputs

ROOT = Path(__file__).resolve().parents[2]
ART = Path(__file__).resolve().parent
REFERENCE = ROOT / "artifacts/entity_attention_128x2_shortterm_value_r21_r37_20261023"
BASE_CONFIG = ROOT / "configs/entity_attention_v1/offline_opponent_policy_r21_r37_prefetch_20261024.json"
CONFIG = ROOT / "configs/entity_attention_v1/offline_policy_score_r21_r37_20261026.json"


def configured_entry():
    # Reuse the established train/select/greedy/search supervisor and its locks.
    path = REFERENCE / "run.py"
    reference_plan = read_json(REFERENCE / "plan.json")
    if sha256(path) != reference_plan["runner"]["sha256"]:
        raise ValueError("Reference experiment runner changed")
    spec = importlib.util.spec_from_file_location("policy_score_pipeline", path)
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    entry.ART, entry.CONFIG, entry.BASE_CONFIG = ART, CONFIG, BASE_CONFIG
    entry.OUTPUT = ROOT / "runs" / ART.name
    entry.RUNTIME = ROOT / "runtime" / ART.name
    original_evaluation, original_report = entry.evaluation_config, entry.report

    def prepare():
        baseline = read_json(BASE_CONFIG)
        expected = copy.deepcopy(baseline)
        expected["continuation_auxiliary"]["score_ownership_weight"] = .25
        raw = read_json(CONFIG)
        if raw != expected:
            raise ValueError("Only the requested score auxiliary may differ from policy104k")
        for record in reference_plan["source_rounds"]:
            for key in ("index", "manifest", "summary"):
                if sha256(ROOT / record[key]["path"]) != record[key]["sha256"]:
                    raise ValueError("Original 104k collection identity changed")
        config = ContinuationOfflineConfig.from_dict(raw)
        _, _, _, fixed, training = prepare_inputs(config, entry.OUTPUT)
        dataset_digest = read_json(entry.OUTPUT / "dataset.json")["digest"]
        if (training["games"] != 104000 or training["splits"] != reference_plan["splits"]
                or dataset_digest != reference_plan["dataset_digest"]
                or fixed["digest"] != reference_plan["fixed_validation_digest"]):
            raise ValueError("Original 104k data, splits or validation subset changed")
        # The templates already contain configuration A and the original seeds.
        entry.freeze(ART / "evaluation_templates.json", read_json(REFERENCE / "evaluation_templates.json"))
        for filename, expected_sha in reference_plan["catalog_files"].items():
            source = REFERENCE / "catalog" / filename
            if sha256(source) != expected_sha or sha256(ROOT / "data" / filename) != expected_sha:
                raise ValueError("Reference or current catalog changed")
            target = ART / "catalog" / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copyfile(source, target)
            if sha256(target) != expected_sha:
                raise ValueError("Frozen experiment catalog changed")
        for opponent in reference_plan["opponents"].values():
            if sha256(Path(opponent["path"])) != opponent["sha256"]:
                raise ValueError("Raw R37 opponent changed")
        # One additional int16[8] array per train position, aligned per game.
        cache_upper_bound = (17869150492 + training["splits"]["train"]["positions"] * 16
                             + training["splits"]["train"]["games"] * 63)
        if cache_upper_bound > config.cache_mib * 1024**2:
            raise ValueError("Shared training arena capacity is insufficient")
        plan = {**reference_plan, "experiment": ART.name,
            "config": entry.identity(CONFIG), "runner": entry.identity(Path(__file__)),
            "baseline_config": entry.identity(BASE_CONFIG),
            "reference_plan": entry.identity(REFERENCE / "plan.json"),
            "reference_runner": entry.identity(path),
            "model_config": asdict(model_config(config)), "parameters": 852498,
            "continuation_auxiliary": raw["continuation_auxiliary"],
            "initialization": "Fresh attention128 and both auxiliary heads; fresh AdamW; original seeds; no checkpoint inheritance",
            "steps_target": training["steps_target"], "validation_steps": training["validation_steps"],
            "sampling": training["sampling"], "cache_mib": config.cache_mib,
            "training_workers": config.training_workers,
            "cache_capacity": {"same_pool_policy_arena_bytes": 17869150492,
                "additional_label_bytes_per_position": 16, "alignment_upper_bound_per_game": 63,
                "combined_arena_upper_bound_bytes": cache_upper_bound,
                "configured_bytes": config.cache_mib * 1024**2, "full_pool_warmed": False},
            "evaluation_templates": entry.identity(ART / "evaluation_templates.json"),
            "scope": "Original R21-R37 104k games/splits/main labels, fresh attention128, opponent policy .15 plus remaining-score-by-player-source-v1 .25; shortterm value off. All policy104k offline settings unchanged, including lr1e-4, 18GiB, 4 shared read-only workers, streaming statistics/state-only transfer/batched scalars. Stop/select only by main full-holdout KL+MSE. Independent run, not attached to prior queue. Default: train/select, raw R37 greedy1000pair, then searchA500pair both S1024N1 total1024. No formal training started."}
        entry.freeze(ART / "plan.json", plan)
        return plan

    def frozen_source():
        destination, record_path = entry.RUNTIME / "source", ART / "source.json"
        if not record_path.exists():
            record = source_identity(ROOT, destination)
            sources = [CONFIG, BASE_CONFIG, Path(__file__), path,
                ART / "validate.py", ROOT / "scripts/run_attention128_policy_score104k_20261026.sh",
                ROOT / "src/splendor/native_search.cpp", *ROOT.glob("src/splendor/_native_search*.so")]
            for source in sources:
                target = destination / source.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
            for source in (ART / "catalog").iterdir():
                target = destination / "data" / source.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
            record["snapshot_files"] = {str(p.relative_to(destination)): sha256(p)
                for p in sorted(destination.rglob("*")) if p.is_file() and "__pycache__" not in p.parts}
            record["reference_runner"] = entry.identity(path)
            entry.freeze(record_path, record)
        for relative, expected in read_json(record_path)["snapshot_files"].items():
            if sha256(destination / relative) != expected:
                raise ValueError(f"Frozen source changed: {relative}")
        return destination

    def evaluation_config(plan, selection, panel):
        result = original_evaluation(plan, selection, panel)
        result["policy_a"]["name"] = ART.name
        result["metadata"]["continuation_auxiliary"] = plan["continuation_auxiliary"]
        return result

    def report(plan):
        result = original_report(plan)
        path = ART / "results.md"
        path.write_text(path.read_text().replace("第21–37轮固定池短期value离线训练",
            "第21–37轮固定池policy＋得分归属辅助离线训练"), encoding="utf-8")
        return result

    entry.prepare, entry.frozen_source = prepare, frozen_source
    entry.evaluation_config, entry.report = evaluation_config, report
    return entry


if __name__ == "__main__":
    raise SystemExit(configured_entry().main())
