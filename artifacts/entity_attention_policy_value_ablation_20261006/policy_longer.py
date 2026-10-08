"""Exploratory same-data continuation of policy-only500 to2000, no strength panel."""
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch

from splendor.alphazero_checkpoint import (load_alphazero_checkpoint, move_optimizer_state,
                                         save_alphazero_checkpoint)
from splendor.alphazero_recording import (append_json, environment_identity, sha256,
                                        source_identity, write_json)
from splendor.offline_train import make_index
from splendor.rng import DeterministicRng
from splendor.selfplay_data import collate_selfplay_samples
from core import actual_learner_config, diagnostic_step, restored_learner
from train import check_inputs, extras

ROOT = Path(__file__).resolve().parents[2]
ART = Path(__file__).resolve().parent
spec = json.loads((ART / "specification.json").read_text())
check_inputs(spec)
torch.set_num_threads(1)
assert torch.cuda.is_available()
parent = ROOT / spec["output"] / "policy_only/step-000500.pt"
parent_identity = {"path": str(parent.relative_to(ROOT)), "sha256": sha256(parent)}
stage = ROOT / spec["output"] / "policy_only_longer"
stage.mkdir(exist_ok=False)
index = make_index({"games": spec["games"]}, 1024**3)
samples = {split: [index.cache.sample(g, i) for g in index.split_games(split)
                  for i in range(g.decision_count)] for split in ("train", "validation")}
learner, _ = restored_learner(ROOT / spec["initial"]["path"], "cuda")
baseline = {split: extras(learner, rows)[1] for split, rows in samples.items()}
payload = load_alphazero_checkpoint(parent, device="cuda")
assert payload.total_train_step == 14900 and payload.round_train_step == 500
learner.model.load_state_dict(payload.model_state_dict)
learner.optimizer.load_state_dict(payload.optimizer_state_dict)
move_optimizer_state(learner.optimizer, learner.device)
learner.sampling_rng = DeterministicRng.restore(payload.sampling_rng_state)
learner.round_train_step = 500
learner.round_steps_target = 2000
learner.total_train_step = payload.total_train_step
learner.model.critic.requires_grad_(False)
torch.set_rng_state(payload.torch_cpu_rng_state.cpu())
torch.cuda.set_rng_state_all([s.cpu() for s in payload.torch_cuda_rng_states])
before = {k: v.detach().clone() for k, v in learner.model.critic.state_dict().items()}
pool = index.position_pool(split="train", round_index=spec["data_round"])
start_rng = learner.sampling_rng.snapshot()
assert asdict(start_rng) == spec["sampling_rng_end"]
source = source_identity(ROOT, ROOT / "runtime/entity_attention_policy_value_ablation_20261006"
                         / "policy_longer_source" / str(time.time_ns()))
started = time.perf_counter()
curve, checkpoints, references = [], [], []
ledger = hashlib.sha256()
for step in range(500, 2001):
    if step in (500, 1000, 2000):
        point = {"step": step, "total_train_step": learner.total_train_step}
        for split, rows in samples.items():
            point[split] = learner.evaluate_samples(rows) | extras(learner, rows, baseline[split])[0]
        curve.append(point)
        append_json(stage / "curve.jsonl", point)
        if step == 500:
            original = json.loads((ART / "fitting.json").read_text())["arms"]["policy_only"]["curve"][-1]
            for split in samples:
                for key in ("policy_kl", "value_loss"):
                    assert abs(point[split][key] - original[split][key]) < 1e-6
        else:
            path = stage / f"step-{step:06d}.pt"
            save_alphazero_checkpoint(path, model=learner.model, optimizer=learner.optimizer,
                learner_config=actual_learner_config(learner, "policy_only"), round_index=payload.round_index,
                round_train_step=step, total_train_step=learner.total_train_step, round_steps_target=2000,
                parent_checkpoint=parent_identity,
                replay_identity={"protocol": "exploratory-same-fixed-round-v1",
                                 "digest": sha256(ART / "specification.json")},
                sampling_rng_state=learner.sampling_rng.snapshot(),
                metadata={"diagnostic_only": True, "arm": "policy_only", "exploratory_continuation": True})
            checkpoints.append({"path": str(path.relative_to(ROOT)), "sha256": sha256(path)})
        print(json.dumps({"event": "longer_fit", "step": step,
                          "train_kl": point["train"]["policy_kl"],
                          "validation_kl": point["validation"]["policy_kl"]}), flush=True)
    if step == 2000:
        break
    drawn = [pool.sample(learner.sampling_rng) for _ in range(256)]
    refs = np.asarray([[r.game_id, r.decision_index] for _, r in drawn], dtype="<i8")
    references.append(refs)
    ledger.update(refs.tobytes())
    batch = collate_selfplay_samples([s for s, _ in drawn])
    append_json(stage / "training_metrics.jsonl", diagnostic_step(learner, batch, "policy_only"))
    if (step + 1) % 250 == 0:
        print(json.dumps({"event": "longer_progress", "step": step + 1}), flush=True)
assert all(torch.equal(before[k], v) for k, v in learner.model.critic.state_dict().items())
refs_path = stage / "actual_batch_references.npy"
np.save(refs_path, np.asarray(references))
check_inputs(spec)
assert sha256(parent) == parent_identity["sha256"]
write_json(ART / "policy_only_longer.json", {
    "protocol": "exploratory-fixed-data-policy-only-longer-v1", "passed": True,
    "command": [sys.executable, *sys.argv], "source": source, "environment": environment_identity(),
    "script_sha256": sha256(Path(__file__)), "core_sha256": sha256(ART / "core.py"),
    "train_helpers_sha256": sha256(ART / "train.py"), "parent": parent_identity,
    "additional_updates": 1500, "batch_size": 256, "curve": curve, "checkpoints": checkpoints,
    "actual_batch_references": {"path": str(refs_path.relative_to(ROOT)), "sha256": sha256(refs_path)},
    "actual_batch_ledger_sha256": ledger.hexdigest(), "sampling_rng_start": asdict(start_rng),
    "sampling_rng_end": asdict(learner.sampling_rng.snapshot()), "critic_parameters_unchanged": True,
    "inputs_unchanged": True, "elapsed_seconds": time.perf_counter() - started,
    "scope": "Added after observing500-step fits. Same fixed175 training games; one arm only, no equal-budget comparison or strength evaluation of1000/2000.",
})
