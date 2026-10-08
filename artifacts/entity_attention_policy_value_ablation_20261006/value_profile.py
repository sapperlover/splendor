"""Describe fixed holdout value predictions with the native search clipping rule."""
import json
from pathlib import Path
import numpy as np
import torch

from splendor.alphazero_checkpoint import load_alphazero_checkpoint
from splendor.alphazero_recording import sha256, write_json
from splendor.offline_train import make_index
from splendor.selfplay_data import collate_selfplay_samples
from train import check_inputs

ROOT = Path(__file__).resolve().parents[2]
ART = Path(__file__).resolve().parent
spec = json.loads((ART / "specification.json").read_text())
check_inputs(spec)
torch.set_num_threads(1)
index = make_index({"games": spec["games"]}, 256 * 1024**2)
samples = [index.cache.sample(game, i) for game in index.split_games("validation")
           for i in range(game.decision_count)]
paths = {"initial": ROOT / spec["initial"]["path"]}
paths.update({f"{arm}_500": ROOT / spec["output"] / arm / "step-000500.pt"
              for arm in spec["arms"]})
profiles = {}
for name, path in paths.items():
    payload = load_alphazero_checkpoint(path)
    model = payload.build_model(device="cpu").eval().requires_grad_(False)
    predictions, targets, weights = [], [], []
    with torch.inference_mode():
        for start in range(0, len(samples), 256):
            batch = collate_selfplay_samples(samples[start:start + 256])
            _, values = model(batch.decisions)
            predictions.extend(values.tolist())
            targets.extend(batch.value_targets.tolist())
            weights.extend(batch.value_weights.tolist())
    values, targets, weights = map(np.asarray, (predictions, targets, weights))
    assert np.all(weights == 1) and np.all(np.abs(targets) == 1)
    clipped = np.clip(values, -1, 1)
    profiles[name] = {
        "checkpoint_sha256": sha256(path), "positions": len(values),
        "raw_mse": float(np.mean((values - targets)**2)),
        "search_clipped_mse": float(np.mean((clipped - targets)**2)),
        "outside_unit_interval_fraction": float(np.mean(np.abs(values) > 1)),
        "raw_prediction_mean": float(values.mean()),
        "raw_prediction_std": float(values.std()),
        "positive_outcome_prediction_mean": float(values[targets == 1].mean()),
        "negative_outcome_prediction_mean": float(values[targets == -1].mean()),
        "clipped_prediction_target_pearson": float(np.corrcoef(clipped, targets)[0, 1]),
        "clipped_sign_agreement": float(np.mean(np.sign(clipped) == targets)),
    }
    del model, payload
check_inputs(spec)
write_json(ART / "value_profile.json", {
    "protocol": "search-clipped-fixed-holdout-value-profile-v1", "profiles": profiles,
    "validation_games": 25, "inputs_unchanged": True,
    "script_sha256": sha256(Path(__file__)),
    "native_search_source_sha256": sha256(ROOT / "src/splendor/native_search.cpp"),
    "scope": "Descriptive CPU FP32 statistics on the same holdout; not an independent strength panel or causal test.",
})
print(json.dumps(profiles), flush=True)
