"""CPU-only comparison on the original 88 information-safe human-game requests."""
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import sys
import subprocess
import time

import numpy as np
import torch

from splendor.compact_trace import decode_request
from splendor.features import encode_decision
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.replay import to_primitive
from splendor.search_inference import FrozenSearchInference, InferenceConfig

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CAPTURE = ROOT / "artifacts/bga_927500015_20261020"
OLD = ROOT / "artifacts/bga_927500015_r20_analysis_20261020"
RUN = ROOT / "runs/entity_attention_policy140221_r31_r37_p60k_iteration_20261025"
MODELS = {
    "old_mixed_R20": ROOT / "runs/entity_attention_37423_random_budget_p6k_v24k_20261019/rounds/round-019/checkpoints/final.pt",
    "current_local12": RUN / "rounds/round-011/checkpoints/final.pt",
    "current_local13": RUN / "rounds/round-012/checkpoints/final.pt",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(records):
    return {"decisions": len(records), "top1_count": sum(r["actual_rank"] == 1 for r in records),
            "top3_count": sum(r["actual_rank"] <= 3 for r in records),
            "mean_actual_probability": float(np.mean([r["actual_probability"] for r in records])),
            "mean_rank": float(np.mean([r["actual_rank"] for r in records])),
            "mean_actual_nll": float(np.mean([r["actual_nll"] for r in records]))}


def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    original = json.loads((OLD / "greedy.json").read_text())["rows"]
    rows = [json.loads(line) for line in (CAPTURE / "model_decisions.jsonl").read_text().splitlines()]
    assert len(rows) == len(original) == 88
    encoded, requests = [], []
    with np.load(CAPTURE / "model_features.npz", allow_pickle=False) as stored:
        for i, row in enumerate(rows):
            request = decode_request(row["request"])
            assert i == row["decision_index"]
            assert to_primitive(request.legal_actions[row["candidate_index"]]) == row["action"]
            features = encode_decision(request)
            for key in stored.files:
                if key.startswith(f"{i:03d}/"):
                    np.testing.assert_array_equal(getattr(features, key.split("/", 1)[1]), stored[key])
            encoded.append(features)
            requests.append(request)
    plan = {"table": 927500015, "device": "cpu", "threads": 1,
            "scope": "Greedy only on the original 88 saved requests; no MCTS, training or browser actions.",
            "command": sys.argv, "torch": torch.__version__,
            "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                                       capture_output=True, text=True).stdout.strip(),
            "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / "src/splendor").glob("*.py"))},
            "inputs_sha256": {str(p.relative_to(ROOT)): sha(p) for p in (
                Path(__file__), CAPTURE / "model_decisions.jsonl", CAPTURE / "model_features.npz", OLD / "greedy.json")},
            "models": {}}
    result, outputs = {}, {}
    for name, checkpoint in MODELS.items():
        digest = sha(checkpoint)
        model = load_frozen_actor_critic_checkpoint(checkpoint, device="cpu")
        evaluator = FrozenSearchInference(model, device="cpu", config=InferenceConfig(actor_projection=False))
        predictions = evaluator.policy_value(requests, encoded=encoded)
        records = []
        for row, (logits, value) in zip(rows, predictions, strict=True):
            logits = np.asarray(logits, dtype=np.float64)
            logp = logits - logits.max()
            logp -= np.log(np.exp(logp).sum())
            prior = np.exp(logp)
            order = sorted(range(len(prior)), key=lambda i: (-prior[i], i))
            actual = row["candidate_index"]
            records.append({"decision_index": row["decision_index"], "bga_move": row["bga_move"],
                "actor": row["actor"], "phase": row["request"]["observation"]["phase"],
                "candidates": len(prior), "actual": actual, "choice": order[0],
                "actual_rank": order.index(actual)+1, "actual_probability": float(prior[actual]),
                "actual_nll": float(-logp[actual]), "prior": prior.tolist(),
                "value_actor_raw": value})
        if name == "old_mixed_R20":
            for now, old in zip(records, original, strict=True):
                assert now["choice"] == old["choice"]
                np.testing.assert_allclose(now["prior"], old["prior"], rtol=0, atol=2e-6)
        main_rows = [r for r in records if r["phase"] == "main_action"]
        payment_rows = [r for r in records if r["phase"] == "payment" and r["candidates"] > 1]
        assert len(main_rows) == 56 and len(payment_rows) == 20
        result[name] = {"main": summarize(main_rows),
            "first_player": summarize([r for r in main_rows if r["actor"] == 0]),
            "second_player": summarize([r for r in main_rows if r["actor"] == 1]),
            "choice_payment": summarize(payment_rows)}
        outputs[name] = records
        plan["models"][name] = {"path": str(checkpoint.relative_to(ROOT)), "sha256": digest,
                                 "config": asdict(model.config)}
        assert sha(checkpoint) == digest
        del evaluator, model
        gc.collect()
    previous = outputs["old_mixed_R20"]
    for name in ("current_local12", "current_local13"):
        pairs = [(a, b) for a, b in zip(previous, outputs[name], strict=True) if a["phase"] == "main_action"]
        result[name]["versus_old_main"] = {
            "changed_choices": sum(a["choice"] != b["choice"] for a, b in pairs),
            "old_wrong_new_agrees": [b["bga_move"] for a, b in pairs if a["actual_rank"] != 1 and b["actual_rank"] == 1],
            "old_agrees_new_wrong": [b["bga_move"] for a, b in pairs if a["actual_rank"] == 1 and b["actual_rank"] != 1]}
    plan["seconds"] = time.perf_counter()-started
    (OUT / "plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    (OUT / "greedy.json").write_text(json.dumps(outputs, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    (OUT / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
