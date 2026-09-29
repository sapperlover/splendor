"""Pure-forward comparison of the frozen inference execution modes.

Prepares fixed real positions, measures eager / projected / compiled forwards
without any engine or collection work, and checks every mode against the eager
reference (logits, value, argmax).  This is the "pure forward" layer of the
three-layer measurement; the collection layer lives in
``scripts/benchmark_collect_throughput.py``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from splendor.benchmark_search import fixed_requests
from splendor.features import encode_decision
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.search_inference import FrozenSearchInference, InferenceConfig

MODES = {
    "eager": InferenceConfig(mode="eager", actor_projection=False),
    "projected": InferenceConfig(mode="eager", actor_projection=True),
    "compiled": InferenceConfig(mode="compiled", actor_projection=False),
    "compiled_projected": InferenceConfig(mode="compiled", actor_projection=True),
}


def build(checkpoint: Path, device: str, config: InferenceConfig) -> FrozenSearchInference:
    model = load_frozen_actor_critic_checkpoint(checkpoint, device=device)
    return FrozenSearchInference(model, device=device, config=config)


def measure(evaluator: FrozenSearchInference, encoded, repeats: int, warmup: int):
    for _ in range(warmup):
        evaluator.policy_value_encoded(encoded)
    durations = []
    for _ in range(repeats):
        started = time.perf_counter()
        rows = evaluator.policy_value_encoded(encoded)
        durations.append(time.perf_counter() - started)
    return statistics.median(durations), rows


def compare(reference, rows):
    logits = max(abs(a - b) for lhs, rhs in zip(reference, rows) for a, b in zip(lhs[0], rhs[0]))
    value = max(abs(lhs[1] - rhs[1]) for lhs, rhs in zip(reference, rows))
    argmax = all(
        max(range(len(lhs[0])), key=lambda i: lhs[0][i]) == max(range(len(rhs[0])), key=lambda i: rhs[0][i])
        for lhs, rhs in zip(reference, rows)
    )
    return {"logits_max_abs_diff": logits, "value_max_abs_diff": value, "argmax_identical": argmax}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batches", default="1,4,8,16,32,64")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    requests = [encode_decision(request) for _, request, _ in fixed_requests()]
    result: dict[str, object] = {
        "checkpoint": str(args.checkpoint.resolve()), "device": args.device,
        "torch_threads": args.threads, "repeats": args.repeats, "warmup": args.warmup,
        "per_request": {}, "batches": {},
    }
    evaluators = {name: build(args.checkpoint, args.device, config) for name, config in MODES.items()}

    for index, request in enumerate(requests):
        reference = evaluators["eager"].policy_value_encoded((request,))
        row = {"candidates": len(request.candidate_actions)}
        for name in ("projected", "compiled_projected"):
            row[name] = compare(reference, evaluators[name].policy_value_encoded((request,)))
        result["per_request"][f"request_{index}"] = row

    for batch in (int(part) for part in args.batches.split(",")):
        encoded = (requests * ((batch + len(requests) - 1) // len(requests)))[:batch]
        reference_seconds, reference_rows = measure(evaluators["eager"], encoded, args.repeats, args.warmup)
        row = {"candidates": int(encoded[0].candidate_features.shape[0]), "eager_seconds": reference_seconds}
        for name in ("projected", "compiled", "compiled_projected"):
            seconds, rows = measure(evaluators[name], encoded, args.repeats, args.warmup)
            row[name] = {
                "seconds": seconds,
                "speedup": reference_seconds / seconds,
                **compare(reference_rows, rows),
                "compile": evaluators[name].stats.to_dict()["compile"],
            }
        result["batches"][str(batch)] = row
        print(f"batch={batch:3d} " + " ".join(
            f"{name}={row[name]['seconds']*1000:6.2f}ms({row[name]['speedup']:.2f}x)"
            for name in ("projected", "compiled", "compiled_projected")))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
