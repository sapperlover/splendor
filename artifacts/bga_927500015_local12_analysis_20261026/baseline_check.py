"""Finite GPU replay of 16 old-R20 roots using the current execution stack."""
import importlib.util
from pathlib import Path
import json

import numpy as np
import torch

from splendor.alphazero_recording import sha256, write_json
from splendor.compact_trace import decode_request
from splendor.native_policy import NativeBatchRunner
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.search_inference import FrozenSearchInference, InferenceConfig

ART = Path(__file__).resolve().parent
ROOT = ART.parents[1]


def main():
    spec = importlib.util.spec_from_file_location("local12_analysis", ART / "analyze.py")
    analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analysis)
    plan = json.loads((ART / "plan.json").read_text())
    analysis.identity_check(plan)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    old = ROOT / "artifacts/bga_927500015_r20_analysis_20261020"
    original_plan = json.loads((old / "plan.json").read_text())
    checkpoint = ROOT / original_plan["checkpoint"]
    model = load_frozen_actor_critic_checkpoint(checkpoint, device="cuda")
    config = dict(plan["inference"])
    for key in ("candidate_buckets", "batch_buckets"):
        config[key] = tuple(config[key])
    evaluator = FrozenSearchInference(model, device="cuda", config=InferenceConfig(**config))
    panel, seed = plan["panels"][0], plan["seeds"][0]
    rows = [r for r in analysis.records() if len(r["request"]["legal_actions"]) > 1][:16]
    priors = json.loads((old / "greedy.json").read_text())["rows"]
    previous_path = old / "search" / panel["name"] / f"{seed}.json"
    previous = {r["decision_index"]: r for r in json.loads(previous_path.read_text())["rows"]}
    runner = NativeBatchRunner(evaluator, analysis.load_default_catalog(), capacity=plan["capacity"],
                               buckets=(16, 32, 48, 64), slots=1)
    policy = analysis.make_policy(model, evaluator, panel, seed, plan)
    jobs = [policy.start_job(decode_request(r["request"]), add_root_noise=False, temperature=0.) for r in rows]
    while any(not job.done for job in jobs):
        assert runner.run([job for job in jobs if not job.done]) > 0
    checks = []
    for row, job in zip(rows, jobs, strict=True):
        i = row["decision_index"]
        now = analysis.search_row(job, row, priors[i]["prior"], panel, seed, panel["simulations"])
        before = previous[i]
        assert now["world_seeds"] == before["world_seeds"]
        checks.append({"decision_index": i, "choice_equal": now["choice"] == before["choice"],
                       "visit_l1": int(np.abs(np.asarray(now["visits"])-before["visits"]).sum()),
                       "prior_linf": float(np.max(np.abs(np.asarray(now["prior"])-before["prior"])))})
    write_json(ART / "baseline_check.json", {"scope": "16 old-R20 roots; finite stack compatibility check, not a new quality panel.",
        "checkpoint_sha256": sha256(checkpoint), "script_sha256": sha256(Path(__file__)),
        "old_raw_sha256": sha256(previous_path), "checks": checks,
        "same_choices": sum(c["choice_equal"] for c in checks),
        "identical_visits": sum(c["visit_l1"] == 0 for c in checks), "inference": evaluator.stats.to_dict()})
    print(json.dumps({"checked": len(checks), "same_choices": sum(c["choice_equal"] for c in checks),
                      "identical_visits": sum(c["visit_l1"] == 0 for c in checks)}))


if __name__ == "__main__":
    main()
