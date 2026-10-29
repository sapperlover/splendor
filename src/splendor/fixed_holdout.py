"""Complete frozen whole-game holdouts for online diagnostics and backfills."""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch

from .actor_critic import TorchDecisionBatch, TorchStateBatch
from .alphazero_recording import read_json, sha256, write_json
from .alphazero_train import AlphazeroLearner, LearnerConfig
from .features import FEATURE_SCHEMA_VERSION
from .offline_train import _digest
from .opponent_pool import load_frozen_actor_critic_checkpoint
from .pool_fitting import ratios
from .selfplay_data import STATE_FIELDS, SelfPlaySample

PROTOCOL = "complete-fixed-holdout-v1"
COLUMNS = ("positions", "policy_weight", "ce", "kl", "target_entropy", "entropy", "top1",
           "sq_error", "pred", "pred_sq", "target", "target_sq", "product")


def game_key(record):
    return record["run_id"], record["round_index"], record["game_id"]


def read_manifest(settings):
    path = Path(settings["manifest"]["path"])
    if sha256(path) != settings["manifest"]["sha256"]:
        raise ValueError("fixed holdout manifest changed")
    manifest = read_json(path)
    content = {k: v for k, v in manifest.items() if k != "digest"}
    if manifest["protocol"] != PROTOCOL or manifest["digest"] != _digest(content):
        raise ValueError("fixed holdout protocol/digest mismatch")
    games = manifest["games"]
    if (not games or any(g["split"] != "validation" for g in games)
            or len({game_key(g) for g in games}) != len(games)):
        raise ValueError("fixed holdout requires unique original validation games")
    if type(settings["batch_size"]) is not int or settings["batch_size"] <= 0:
        raise ValueError("fixed holdout batch_size must be a positive integer")
    return manifest


def check_disjoint(manifest, records):
    keys = {game_key(g) for g in manifest["games"]}
    paths = {str(Path(g["path"]).resolve()) for g in manifest["games"]}
    if any(g["split"] == "train" and
           (game_key(g) in keys or str(Path(g["path"]).resolve()) in paths) for g in records):
        raise ValueError("fixed holdout leaked into training")


class FixedHoldout:
    """CPU-resident states/positive-policy rows; only a minibatch enters the GPU."""

    def __init__(self, settings, output, device):
        self.manifest = read_manifest(settings)
        self.device, self.batch_size = torch.device(device), settings["batch_size"]
        self.output = Path(output)
        frozen = self.output / "fixed_holdout/manifest.json"
        if frozen.exists() and read_json(frozen) != self.manifest:
            raise ValueError("run fixed holdout cannot change on resume")
        if not frozen.exists():
            write_json(frozen, self.manifest)
        self._loaded = False
        self._verified = False

    def _load(self):
        fields = {name: [] for name in STATE_FIELDS}
        features, targets, lengths, policy_ids = [], [], [], []
        values, weights, game_ids = [], [], []
        self.verification_samples, self.verification_ids = [], []
        offset = 0
        needed = (*STATE_FIELDS, "candidate_features", "candidate_offsets", "policy_target",
                  "policy_weight", "value_target", "value_weight", "metadata_json",
                  "behavior_policy", "decision_index", "player_id")
        for game_index, record in enumerate(self.manifest["games"]):
            path = Path(record["path"])
            if sha256(path) != record["archive_sha256"]:
                raise ValueError(f"fixed holdout archive changed: {path}")
            with np.load(path, allow_pickle=False) as archive:
                arrays = {name: archive[name] for name in needed}
            metadata = json.loads(arrays.pop("metadata_json").tobytes().decode())
            if (any(metadata[k] != record[k] for k in ("run_id", "round_index", "game_id", "split"))
                    or metadata["feature_schema_version"] != FEATURE_SCHEMA_VERSION):
                raise ValueError("fixed holdout archive metadata/schema changed")
            if not np.all(arrays["value_weight"] == 1):
                raise ValueError("complete fixed holdout requires original unit value weights")
            n = len(arrays["value_target"])
            positive = arrays["policy_weight"] > 0
            if n != record["decision_count"] or np.count_nonzero(positive) != record["policy_samples"]:
                raise ValueError("fixed holdout population changed")
            for name in STATE_FIELDS:
                fields[name].append(arrays[name])
            lens = np.diff(arrays["candidate_offsets"])
            candidates = np.repeat(positive, lens)
            features.append(arrays["candidate_features"][candidates])
            targets.append(arrays["policy_target"][candidates])
            lengths.append(lens[positive])
            policy_ids.append(offset + np.flatnonzero(positive))
            weights.append(arrays["policy_weight"][positive])
            values.append(arrays["value_target"])
            game_ids.append(np.full(n, game_index, dtype=np.int32))
            if game_index < 4:
                indices = np.unique(np.r_[np.linspace(0, n-1, min(n, 8), dtype=int),
                                          np.flatnonzero(positive)[:2]])
                self.verification_samples.extend(SelfPlaySample(arrays, metadata, int(i)) for i in indices)
                self.verification_ids.extend(offset + indices)
            offset += n
        self.fields = {name: np.concatenate(parts) for name, parts in fields.items()}
        self.y = np.concatenate(values).astype(np.float64)
        self.game_ids = np.concatenate(game_ids)
        self.policy_ids = np.concatenate(policy_ids)
        self.pw = np.concatenate(weights).astype(np.float64)
        self.policy_game_ids = self.game_ids[self.policy_ids]
        self.lengths = np.concatenate(lengths)
        self.offsets = np.r_[0, np.cumsum(self.lengths)]
        self.features, self.targets = np.concatenate(features), np.concatenate(targets)
        logs = np.zeros(self.targets.shape, dtype=np.float64)
        np.log(self.targets, out=logs, where=self.targets > 0)
        self.target_entropy = np.add.reduceat(-self.targets * logs, self.offsets[:-1])
        self.policy_order = np.argsort(self.lengths, kind="stable")
        self._loaded = True

    def _states(self, selected):
        return TorchStateBatch(feature_schema_version=FEATURE_SCHEMA_VERSION,
            **{name: torch.as_tensor(value[selected], device=self.device) for name, value in self.fields.items()})

    def evaluate(self, model):
        if not self._loaded:
            self._load()
        started = time.perf_counter()
        was_training = model.training
        matmul_tf32, cudnn_tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
        model.eval()
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        predictions = np.empty(len(self.y), dtype=np.float64)
        policies = np.empty((len(self.policy_ids), 3), dtype=np.float64)
        try:
            with torch.inference_mode():
                for start in range(0, len(self.y), self.batch_size):
                    selected = slice(start, start+self.batch_size)
                    predictions[selected] = model.value_from_state(self._states(selected)).cpu().numpy()
                for start in range(0, len(self.policy_order), self.batch_size):
                    ids = self.policy_order[start:start+self.batch_size]
                    lens = self.lengths[ids]
                    width = int(lens.max())
                    mask = np.arange(width)[None, :] < lens[:, None]
                    gather = np.where(mask, self.offsets[ids, None] + np.arange(width)[None, :], 0)
                    states = self._states(self.policy_ids[ids])
                    batch = TorchDecisionBatch(feature_schema_version=FEATURE_SCHEMA_VERSION,
                        **{name: getattr(states, name) for name in STATE_FIELDS},
                        candidate_features=torch.as_tensor(np.where(mask[..., None], self.features[gather], 0), device=self.device),
                        candidate_mask=torch.as_tensor(mask, device=self.device))
                    target = torch.as_tensor(np.where(mask, self.targets[gather], 0), device=self.device)
                    logits = model.policy_logits(batch).masked_fill(~batch.candidate_mask, -torch.inf)
                    lp = torch.log_softmax(logits, dim=-1)
                    lp = torch.where(batch.candidate_mask, lp, 0.)
                    ce = -(target*lp).sum(-1)
                    entropy = -(torch.where(batch.candidate_mask, lp.exp(), 0.)*lp).sum(-1)
                    best = logits.argmax(-1)
                    correct = target.gather(1, best[:, None]).squeeze(1) >= target.max(-1).values-1e-7
                    policies[ids] = torch.stack((ce, entropy, correct.float()), dim=-1).cpu().numpy()
            if not np.isfinite(predictions).all() or not np.isfinite(policies).all():
                raise ValueError("nonfinite fixed holdout predictions")
            per_game = np.zeros((len(self.manifest["games"]), len(COLUMNS)), dtype=np.float64)
            value_rows = {"positions": np.ones(len(self.y)), "sq_error": (predictions-self.y)**2,
                "pred": predictions, "pred_sq": predictions**2, "target": self.y,
                "target_sq": self.y**2, "product": predictions*self.y}
            policy_rows = {"policy_weight": self.pw, "ce": policies[:, 0]*self.pw,
                "kl": (policies[:, 0]-self.target_entropy)*self.pw, "target_entropy": self.target_entropy*self.pw,
                "entropy": policies[:, 1]*self.pw, "top1": policies[:, 2]*self.pw}
            for name, values in value_rows.items():
                per_game[:, COLUMNS.index(name)] = np.bincount(self.game_ids, values, minlength=len(per_game))
            for name, values in policy_rows.items():
                per_game[:, COLUMNS.index(name)] = np.bincount(self.policy_game_ids, values, minlength=len(per_game))
            result = {"metrics": ratios(dict(zip(COLUMNS, map(float, per_game.sum(axis=0)))))}
            if not self._verified:
                learner = SimpleNamespace(model=model, device=self.device,
                    config=LearnerConfig(device=str(self.device), validation_batch_size=self.batch_size))
                reference = AlphazeroLearner.evaluate_samples(learner, self.verification_samples)
                ids = np.asarray(self.verification_ids)
                pids = np.flatnonzero(np.isin(self.policy_ids, ids))
                differences = {
                    "value_mse": float(np.mean((predictions[ids]-self.y[ids])**2))-reference["value_loss"],
                    "policy_kl": float(np.sum((policies[pids, 0]-self.target_entropy[pids])*self.pw[pids])/self.pw[pids].sum())-reference["policy_kl"]}
                if max(map(abs, differences.values())) > 5e-5:
                    raise ValueError(f"fixed holdout differs from learner joint evaluation: {differences}")
                result["reference_verification"] = {"differences": differences, "positions": len(ids)}
                self._verified = True
            result["seconds"] = time.perf_counter()-started
            return result, per_game
        finally:
            model.train(was_training)
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = matmul_tf32, cudnn_tf32

    def observe(self, model, checkpoint, iteration, step):
        checkpoint = Path(checkpoint)
        path = self.output / "rounds" / f"round-{iteration-1:03d}" / "fixed_holdout" / f"step-{step:09d}.json"
        identity = {"protocol": PROTOCOL, "panel_digest": self.manifest["digest"],
            "checkpoint": {"path": str(checkpoint.resolve()), "sha256": sha256(checkpoint)},
            "iteration": iteration, "step": step, "evaluator_sha256": sha256(Path(__file__))}
        if path.exists():
            result = read_json(path)
            if any(result[k] != v for k, v in identity.items()):
                raise ValueError("fixed holdout result identity changed")
            if sha256(Path(result["per_game"]["path"])) != result["per_game"]["sha256"]:
                raise ValueError("fixed holdout per-game statistics changed")
            return result
        result, per_game = self.evaluate(model)
        path.parent.mkdir(parents=True, exist_ok=True)
        stats = path.with_suffix(".npz")
        temporary = stats.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, per_game=per_game)
        temporary.replace(stats)
        result.update(identity, columns=COLUMNS, batch_size=self.batch_size, device=str(self.device),
            scope="complete fixed original validation games; original positive policy weights; all value positions",
            per_game={"path": str(stats.resolve()), "sha256": sha256(stats)})
        write_json(path, result)
        print(json.dumps({"event": "fixed_holdout_evaluated", "iteration": iteration, "step": step,
            "kl": result["metrics"]["policy_kl"], "mse": result["metrics"]["value_mse"],
            "seconds": result["seconds"]}), flush=True)
        return result

    def backfill(self, completed, steps):
        devices = [self.device.index or 0] if self.device.type == "cuda" else []
        for row in completed:
            stage = self.output / "rounds" / f"round-{row['iteration']-1:03d}"
            snapshot = Path(row["snapshot"]["path"])
            if sha256(snapshot) != row["snapshot"]["sha256"]:
                raise ValueError("backfill replay snapshot changed")
            check_disjoint(self.manifest, read_json(snapshot)["games"])
            for step in steps:
                checkpoint = stage / "checkpoints" / ("initial.pt" if step == 0 else f"step-{step:09d}.pt")
                # Loading a model initializes parameters; isolate that RNG use from online training.
                with torch.random.fork_rng(devices=devices):
                    model = load_frozen_actor_critic_checkpoint(checkpoint, device=self.device)
                    self.observe(model, checkpoint, row["iteration"], step)
                del model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="existing online run directory")
    args = parser.parse_args()
    from .random_budget_iteration import Config
    config = Config.from_dict(read_json(args.config))
    if config.fixed_holdout is None:
        parser.error("config must enable fixed_holdout")
    torch.set_num_threads(1)
    output = args.output.resolve()
    with (output / ".iteration.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        observer = FixedHoldout(config.fixed_holdout, output, config.learner.device)
        observer.backfill(read_json(output / "state.json")["completed_rounds"], config.validation_steps)


if __name__ == "__main__":
    main()
