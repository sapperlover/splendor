"""Finite real-archive validation of score diagnostics; no parameter updates."""
from __future__ import annotations

import ast
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from splendor.alphazero_train import AlphazeroLearner
from splendor.compact_game_cache import CompactGameCache
from splendor.continuation_auxiliary import ContinuationAuxiliaryConfig
from splendor.offline_continuation import ContinuationOfflineConfig
from splendor.offline_objectives import evaluate
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.rng import DeterministicRng
from splendor.score_ownership import game_score_ownership_targets
from splendor.selfplay_data import load_game


ART = Path(__file__).resolve().parent
ROOT = ART.parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    baseline = json.loads((ART / "baseline_identity.json").read_text())
    source = (ROOT / baseline["source_path"]).read_text()
    lines = source.splitlines(keepends=True)
    functions = {node.name: hashlib.sha256("".join(lines[node.lineno-1:node.end_lineno]).encode()).hexdigest()
                 for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)
                 and node.name in baseline["unchanged_function_sha256"]}
    assert functions == baseline["unchanged_function_sha256"]
    config_path = ROOT / "configs/entity_attention_v1/offline_score_ownership_r21_r37_20261026.json"
    config = ContinuationOfflineConfig.from_dict(json.loads(config_path.read_text()))
    auxiliary = ContinuationAuxiliaryConfig(**config.continuation_auxiliary)
    assert digest(Path(config.manifest)) == config.manifest_sha256
    grouped = {i: [] for i in range(config.first_round-1, config.last_round)}
    for line in Path(config.manifest).open():
        row = json.loads(line)
        if row["round_index"] in grouped:
            grouped[row["round_index"]].append(row)
    spec = importlib.util.spec_from_file_location("score_diagnostic_oracle", ROOT / "scripts/validate_continuation_auxiliary.py")
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    rng = np.random.default_rng(20261026)
    games, records = [], []
    for index, rows in grouped.items():
        row = rows[int(rng.integers(len(rows)))]
        path = Path(row["path"])
        assert (path.stat().st_size, path.stat().st_mtime_ns) == (row["size"], row["mtime_ns"])
        arrays, metadata = load_game(path)
        target = game_score_ownership_targets(arrays, metadata)["score_ownership_target"]
        np.testing.assert_array_equal(target, oracle.checked_score_ownership_oracle(arrays, metadata))
        games.append(SimpleNamespace(key=(index, row["game_id"]), round_index=index, game_id=row["game_id"],
                                     decision_count=len(target), path=path))
        records.append({"path": str(path), "sha256": digest(path), "round_index": index,
                        "game_id": row["game_id"], "original_split": row["split"], "positions": len(target)})
    # Reuse the old one-update engineering checkpoint; this is not a trained
    # candidate or a new experiment. No optimizer step or search is performed.
    checkpoint = ROOT / "runtime/score_ownership_20261026/smoke/smoke_checkpoint.pt"
    model = load_frozen_actor_critic_checkpoint(checkpoint, device="cpu")
    learner = AlphazeroLearner(model, replace(config.learner, device="cpu", validation_batch_size=64),
                              sampling_rng=DeterministicRng(20261026))
    index = SimpleNamespace(games=games, split_games=lambda split: games,
                            cache=CompactGameCache(64*1024**2, continuation_auxiliary=auxiliary))
    state = {name: value.clone() for name, value in model.state_dict().items()}
    torch_rng = torch.get_rng_state().clone()
    metrics = evaluate(learner, index, None, None, continuation_auxiliary=auxiliary)
    assert learner.optimizer.state_dict()["state"] == {}
    assert torch.equal(torch_rng, torch.get_rng_state())
    for name, value in state.items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
    count = sum(game.decision_count for game in games)
    assert metrics["score_ownership_positions"] == metrics["value_weight_sum"] == count
    assert metrics["score_ownership_phase_payment_positions"] > 0
    assert metrics["score_ownership_non_payment_positions"] + metrics["score_ownership_phase_payment_positions"] == count
    assert sum(metrics[f"score_ownership_phase_{p}_positions"] for p in
               ("main_action", "payment", "return_tokens", "choose_noble")) == count
    assert sum(metrics[f"score_ownership_stage_{p}_positions"] for p in ("early", "middle", "late")) == count
    result = {"scope": "Engineering validation only: 17 stratified real games, all phases retained; frozen old smoke model, no updates/search/strength conclusion.",
              "passed": True, "seed": 20261026, "games": records, "positions": count, "labels_match_checked_oracle": True,
              "original_loss_functions_unchanged": functions, "model_optimizer_and_rng_unchanged": True,
              "config_sha256": digest(config_path), "checkpoint": str(checkpoint.relative_to(ROOT)),
              "checkpoint_sha256": digest(checkpoint), "metrics": metrics}
    (ART / "real_validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"passed": True, "games": len(games), "positions": count,
                      "payment_positions": metrics["score_ownership_phase_payment_positions"],
                      "non_payment_positions": metrics["score_ownership_non_payment_positions"],
                      "zero_mse_points": metrics["score_ownership_zero_mse_points"],
                      "constant_mse_points": metrics["score_ownership_constant_mse_points"]}, indent=2))


if __name__ == "__main__":
    main()
