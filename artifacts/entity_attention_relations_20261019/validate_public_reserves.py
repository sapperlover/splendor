"""Finite public-reservation inference/search checks; no training or strength test."""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from splendor.actions import ReserveDeck, ReserveMarket
from splendor.actor_critic import build_actor_critic
from splendor.alphazero_recording import environment_identity, read_json, sha256, source_identity, write_json
from splendor.benchmark_search import fixed_requests
from splendor.catalog import load_default_catalog
from splendor.contracts import DecisionRequest
from splendor.engine import GameEngine
from splendor.features import collate_encoded_decisions, encode_decision, encode_decision_state
from splendor.mcts_policy import MCTSConfig
from splendor.native_policy import NativeBatchRunner, NativePUCTSearchPolicy
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.replay import to_primitive
from splendor.search_inference import FrozenSearchInference, InferenceConfig

ART = Path(__file__).resolve().parent
ROOT = ART.parents[1]
OUT = ART / "public_reserves"
SEED = 39240000


def public_reservation_requests():
    engine = GameEngine.new(SEED, first_player=0, mode="checked")
    cases, prefix = [], []
    for index, kind in enumerate((ReserveMarket, ReserveMarket, ReserveDeck, ReserveDeck)):
        request = engine.decision()
        assert isinstance(request, DecisionRequest)
        action = next(action for action in request.legal_actions if isinstance(action, kind))
        prefix.append(to_primitive(action))
        engine.apply(action)
        if index >= 1:
            cases.append((engine.decision(), list(prefix)))
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--preset", choices=("128x2", "256x4"), default="128x2")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    started = time.perf_counter()
    runtime = ROOT / "runtime" / ART.name / "public_reserves" / args.preset / args.device
    runtime.mkdir(parents=True, exist_ok=True)
    preset = ROOT / f"configs/entity_attention_v1/model_relations_{args.preset}.json"
    model = build_actor_critic(read_json(preset)).to(args.device).eval()
    parameters = model.trainable_parameter_count()
    cases = public_reservation_requests()
    encoded = [encode_decision(request) for request, _ in cases]
    states = [encode_decision_state(request) for request, _ in cases]
    eager = FrozenSearchInference(model, device=args.device, config=InferenceConfig(actor_projection=False))
    evaluator = eager if args.device == "cpu" else FrozenSearchInference(model, device=args.device,
        config=InferenceConfig(mode="compiled", actor_projection=False, candidate_buckets=(16, 32, 48, 64),
                               native_transport="reused", compile_failure_policy="fail_fast"))
    expected = eager.policy_value_encoded(encoded)
    for (lp, lv), (rp, rv) in zip(expected, evaluator.policy_value_encoded(encoded)):
        torch.testing.assert_close(torch.tensor(lp), torch.tensor(rp), atol=2e-5, rtol=2e-4)
        torch.testing.assert_close(torch.tensor(lv), torch.tensor(rv), atol=2e-5, rtol=2e-4)
    torch.testing.assert_close(torch.tensor([value for _, value in expected]),
                               torch.tensor(evaluator.value_only_encoded(states)), atol=2e-5, rtol=2e-4)
    policies = [NativePUCTSearchPolicy(model, device=args.device, evaluator=evaluator,
        config=MCTSConfig(simulations=8, determinizations=1, root_noise_fraction=0., seed=SEED+100+i))
        for i in range(len(cases))]
    jobs = [policy.start_job(request) for policy, (request, _) in zip(policies, cases)]
    runner = NativeBatchRunner(evaluator, load_default_catalog(), capacity=4, buckets=(16, 32, 48, 64))
    while not all(job.done for job in jobs):
        assert runner.run([job for job in jobs if not job.done]) > 0
    assert all(job.result.simulations == 8 for job in jobs)
    stats = None if evaluator.compiled is None else evaluator.compiled.stats.to_dict()
    if stats is not None:
        assert sum(stats["compiled_calls"].values()) > 0 and stats["compile_failure_fallbacks"] == 0
        torch.cuda.synchronize()

    legacy_path = ROOT / read_json(ART / "legacy_reference_identity.json")["checkpoint"]
    legacy = load_frozen_actor_critic_checkpoint(legacy_path).eval()
    legacy_batch = collate_encoded_decisions([encode_decision(request) for _, request, _ in fixed_requests()])
    with np.load(ART / "legacy_reference.npz") as golden, torch.inference_mode():
        for name, actual in zip(("logits", "value"), legacy(legacy_batch)):
            torch.testing.assert_close(actual, torch.from_numpy(golden[name]), atol=0, rtol=0)
    result = {"status": "passed", "scope": __doc__, "command": [sys.executable, *sys.argv],
        "device": args.device, "seed": SEED, "model_config": asdict(model.config), "parameters": parameters,
        "preset_sha256": sha256(preset), "validator_sha256": sha256(Path(__file__)),
        "source": source_identity(ROOT, runtime / "source"), "environment": environment_identity(),
        "r43_output_bitwise_unchanged": True, "r43_checkpoint_sha256": sha256(legacy_path),
        "eager_compiled_match": True, "state_only_match": True, "compile": stats,
        "native_search_roots": len(jobs), "simulations_per_root": 8,
        "cases": [{"action_prefix": prefix,
                   "self_public_reservations": sum(card.card is not None for card in request.observation.self_public.reserved_cards),
                   "opponent_public_reservations": sum(card.card is not None for card in request.observation.opponent_public.reserved_cards),
                   "chosen_candidate_index": job.result.chosen_candidate_index}
                  for (request, prefix), job in zip(cases, jobs)],
        "seconds": time.perf_counter() - started}
    write_json(OUT / f"{args.preset}_{args.device}.json", result)
    print({k: result[k] for k in ("status", "device", "parameters", "native_search_roots", "seconds")}, flush=True)


if __name__ == "__main__":
    main()
