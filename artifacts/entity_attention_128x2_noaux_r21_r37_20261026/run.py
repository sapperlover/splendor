"""Fresh base attention128 without auxiliary targets on the original 104k pool."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import shutil

from splendor.alphazero_recording import read_json, sha256, source_identity
from splendor.offline_early_stopping import EarlyStoppingOfflineConfig
from splendor.offline_mixed_budget import prepare_inputs

ROOT = Path(__file__).resolve().parents[2]
ART = Path(__file__).resolve().parent
REFERENCE = ROOT / "artifacts/entity_attention_128x2_policy_score_r21_r37_20261026"
TEMPLATE = ROOT / "artifacts/entity_attention_128x2_shortterm_value_r21_r37_20261023/run.py"
BASE_CONFIG = ROOT / "configs/entity_attention_v1/offline_policy_score_r21_r37_20261026.json"
CONFIG = ROOT / "configs/entity_attention_v1/offline_noaux_r21_r37_20261026.json"


def configured_entry():
    reference = read_json(REFERENCE / "plan.json")
    if sha256(TEMPLATE) != reference["reference_runner"]["sha256"]:
        raise ValueError("Reference supervisor changed")
    spec = importlib.util.spec_from_file_location("noaux104k_pipeline", TEMPLATE)
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    entry.__doc__ = __doc__
    entry.ART, entry.CONFIG, entry.BASE_CONFIG = ART, CONFIG, BASE_CONFIG
    entry.OUTPUT = ROOT / "runs" / ART.name
    entry.RUNTIME = ROOT / "runtime" / ART.name
    original_evaluation, original_report, original_child = entry.evaluation_config, entry.report, entry.run_child

    def prepare():
        expected = copy.deepcopy(read_json(BASE_CONFIG))
        expected.pop("continuation_auxiliary")
        expected.pop("encoder_grad_norm_every_steps")  # Auxiliary-only diagnostics.
        raw = read_json(CONFIG)
        if raw != expected:
            raise ValueError("Only auxiliary targets/diagnostics may differ from the predecessor")
        for record in reference["source_rounds"]:
            for key in ("index", "manifest", "summary"):
                if sha256(ROOT / record[key]["path"]) != record[key]["sha256"]:
                    raise ValueError("Original collection identity changed")
        config = EarlyStoppingOfflineConfig.from_dict(raw)
        _, _, _, fixed, training = prepare_inputs(config, entry.OUTPUT)
        if (training["games"] != 104000 or training["splits"] != reference["splits"]
                or read_json(entry.OUTPUT / "dataset.json")["digest"] != reference["dataset_digest"]
                or fixed["digest"] != reference["fixed_validation_digest"]
                or training["validation_steps"] != reference["validation_steps"]):
            raise ValueError("Original pool, splits or validation schedule changed")
        entry.freeze(ART / "evaluation_templates.json", read_json(REFERENCE / "evaluation_templates.json"))
        for filename, expected_sha in reference["catalog_files"].items():
            source = REFERENCE / "catalog" / filename
            if sha256(source) != expected_sha or sha256(ROOT / "data" / filename) != expected_sha:
                raise ValueError("Reference or current catalog changed")
            target = ART / "catalog" / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copyfile(source, target)
            if sha256(target) != expected_sha:
                raise ValueError("Frozen catalog changed")
        for opponent in reference["opponents"].values():
            if sha256(Path(opponent["path"])) != opponent["sha256"]:
                raise ValueError("Raw R37 opponent changed")
        # No auxiliary arrays: the same-pool measured policy arena is an upper bound.
        if 17869150492 > config.cache_mib * 1024**2:
            raise ValueError("Shared training arena capacity is insufficient")
        plan = {**reference, "experiment": ART.name, "config": entry.identity(CONFIG),
            "runner": entry.identity(Path(__file__)), "baseline_config": entry.identity(BASE_CONFIG),
            "reference_plan": entry.identity(REFERENCE / "plan.json"),
            "reference_runner": entry.identity(TEMPLATE), "model_config": config.model_config,
            "parameters": 583682, "continuation_auxiliary": None,
            "initialization": "Fresh base attention128, fresh AdamW, original initialization/sampling/validation seeds; no inherited checkpoint or auxiliary heads",
            "training_module": "splendor.offline_early_stopping",
            "steps_target": training["steps_target"], "validation_steps": training["validation_steps"],
            "sampling": training["sampling"], "cache_mib": config.cache_mib,
            "training_workers": config.training_workers,
            "cache_capacity": {"same_pool_policy_arena_upper_bound_bytes": 17869150492,
                "auxiliary_arrays": 0, "configured_bytes": config.cache_mib * 1024**2, "full_pool_warmed": False},
            "evaluation_templates": entry.identity(ART / "evaluation_templates.json"),
            "scope": "Original R21-R37 104k pool/splits/main labels, fresh base attention128 and AdamW. All auxiliary targets/heads/labels/losses disabled; main policy and terminal value remain. Other offline settings identical to predecessor: lr1e-4,18GiB,4 shared read-only workers,streaming/state-only/batched scalars,12 exposures,5% main KL+MSE three-increase stopping/minimum selection. RawR37greedy1000pair then searchA500pair bothS1024N1 total1024. Scheduled only after the predecessor's entire successful chain."}
        entry.freeze(ART / "plan.json", plan)
        return plan

    def frozen_source():
        destination, record_path = entry.RUNTIME / "source", ART / "source.json"
        if not record_path.exists():
            record = source_identity(ROOT, destination)
            for source in [CONFIG, BASE_CONFIG, Path(__file__), TEMPLATE, ART / "validate.py",
                    ROOT / "scripts/run_attention128_noaux104k_20261026.sh",
                    ROOT / "src/splendor/native_search.cpp", *ROOT.glob("src/splendor/_native_search*.so")]:
                target = destination / source.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
            for source in (ART / "catalog").iterdir():
                target = destination / "data" / source.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
            record["snapshot_files"] = {str(p.relative_to(destination)): sha256(p)
                for p in sorted(destination.rglob("*")) if p.is_file() and "__pycache__" not in p.parts}
            entry.freeze(record_path, record)
        for relative, expected in read_json(record_path)["snapshot_files"].items():
            if sha256(destination / relative) != expected:
                raise ValueError(f"Frozen source changed: {relative}")
        return destination

    def run_child(command, phase, source):
        # The shared supervisor hardcodes continuation; this arm uses the base fitter.
        command = list(command)
        if phase in ("train", "select"):
            module_index = command.index("-m") + 1
            assert command[module_index] == "splendor.offline_continuation"
            command[module_index] = "splendor.offline_early_stopping"
        return original_child(command, phase, source)

    def evaluation_config(plan, selection, panel):
        result = original_evaluation(plan, selection, panel)
        result["policy_a"]["name"] = ART.name
        result["metadata"]["auxiliary_targets"] = "none"
        return result

    def report(plan):
        result = original_report(plan)
        path = ART / "results.md"
        path.write_text(path.read_text().replace("第21–37轮固定池短期value离线训练",
            "第21–37轮固定池无辅助目标离线训练"), encoding="utf-8")
        return result

    entry.prepare, entry.frozen_source, entry.run_child = prepare, frozen_source, run_child
    entry.evaluation_config, entry.report = evaluation_config, report
    return entry


if __name__ == "__main__":
    raise SystemExit(configured_entry().main())
