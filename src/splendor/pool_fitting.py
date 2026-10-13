"""Streaming fitting diagnostics, using the established recent-pool metric definitions."""
from __future__ import annotations
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch
from .alphazero_recording import read_json, sha256, write_json
from .features import GLOBAL_FEATURE_NAMES, SCALES
from .offline_train import _digest as digest
from .rng import DeterministicRng
from .selfplay_data import collate_selfplay_samples
from .alphazero_replay import ReplayReference
from .policy_surprise import PolicySurpriseConfig, PolicySurpriseWeighting

def freeze_subsets(records, seed, per_source):
    rng = DeterministicRng(seed)
    buckets = defaultdict(list)
    for game in records:
        buckets[(game["source_id"], game["split"])].append(game)
    references = []
    for (source, split), games in sorted(buckets.items()):
        starts = np.cumsum([0] + [g["decision_count"] for g in games])
        selected = set()
        while len(selected) < min(per_source, int(starts[-1])):
            selected.add(rng.randbelow(int(starts[-1])))
        for position in sorted(selected):
            i = int(np.searchsorted(starts, position, side="right") - 1)
            references.append({"round_index": games[i]["round_index"], "game_id": games[i]["game_id"],
                "decision_index": int(position - starts[i]), "source_id": source, "split": split})
    return {"references": references, "seed": seed, "per_source_split": per_source, "digest": digest(references)}


def ratios(sums):
    p, v = sums["policy_weight"], sums["positions"]
    result = {**sums, "policy_kl": sums["kl"] / p if p else None,
        "policy_ce": sums["ce"] / p if p else None,
        "target_entropy": sums["target_entropy"] / p if p else None,
        "policy_entropy": sums["entropy"] / p if p else None,
        "teacher_top1_agreement": sums["top1"] / p if p else None,
        "value_mse": sums["sq_error"] / v if v else None,
        "value_zero_mse": sums["target_sq"] / v if v else None,
        "value_bias": (sums["pred"] - sums["target"]) / v if v else None}
    if v:
        x = max(0., sums["pred_sq"] - sums["pred"] ** 2 / v)
        y = max(0., sums["target_sq"] - sums["target"] ** 2 / v)
        result["value_correlation"] = (sums["product"] - sums["pred"] * sums["target"] / v) / math.sqrt(x*y) if x*y > 0 else None
    if "surprise_weight" in sums:
        weight = sums["surprise_weight"]
        result["surprise_policy_ce"] = sums["surprise_ce"] / weight if weight else None
        result["surprise_policy_kl"] = sums["surprise_kl"] / weight if weight else None
    return result


class RoundFitting:
    """Observe fixed training subsets and complete online holdouts; never select weights."""

    def __init__(self, config, output, stage, offline_index, online_index, dataset, *, inherited_references=None):
        self.config, self.stage = config, stage
        self.offline_index, self.online_index = offline_index, online_index
        self.offline_records = dataset["games"] if dataset is not None else None
        self.online_records = [{**g.to_dict(), "source_id": f"g{g.round_index + 1:03d}"}
                               for g in online_index.games]
        old_path = output / "offline/fitting_references.json"
        self.offline_fixed = None
        if self.offline_records is not None:
            if not old_path.exists():
                write_json(old_path, freeze_subsets(self.offline_records, config["seed"],
                                                   config["offline_positions_per_source_split"]))
            self.offline_fixed = read_json(old_path)
        train_path = stage / "fitting/train_references.json"
        if not train_path.exists():
            refs = []
            for generation in sorted({g["round_index"] for g in self.online_records}):
                inherited = [r for r in inherited_references or () if r["round_index"] == generation]
                if inherited:
                    refs.extend(inherited)
                    continue
                subset = freeze_subsets([g for g in self.online_records
                    if g["split"] == "train" and g["round_index"] == generation],
                    config["seed"] + generation, config["train_positions_per_generation"])
                refs.extend(subset["references"])
            write_json(train_path, {"references": refs, "seed": config["seed"],
                "seed_rule": "seed + generation; historical subsets survive window changes",
                "digest": digest(refs)})
        self.train_fixed = read_json(train_path)

    def __call__(self, learner, checkpoint):
        step = learner.round_train_step
        if step not in self.config["steps"]:
            return
        path = self.stage / "fitting" / f"step-{step:09d}.json"
        if path.exists():
            if read_json(path)["checkpoint_sha256"] != sha256(checkpoint):
                raise ValueError("fitting result differs from its committed checkpoint")
            return
        tick = time.perf_counter()
        surprise = self.config.get("policy_surprise_weighting")
        weighting = PolicySurpriseConfig(**surprise) if surprise is not None else None
        result = {"round_step": step, "total_train_step": learner.total_train_step,
            "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256(checkpoint),
            "offline_fixed": evaluate(learner, self.offline_index, self.offline_records, fixed=self.offline_fixed)
                             if self.offline_index is not None else None,
            "online_train_fixed": evaluate(learner, self.online_index, self.online_records, fixed=self.train_fixed,
                                            policy_surprise_weighting=weighting),
            "online_validation_full": evaluate(learner, self.online_index, self.online_records,
                                                policy_surprise_weighting=weighting),
            "stage_definition": "late=max(self_score,opponent_score)>=10; otherwise early_mid",
            "scope": "stored single behavior-search labels; fixed Train subsets and whole-game validation; DEV"}
        result["seconds"] = time.perf_counter() - tick
        write_json(path, result)
        print(f"拟合诊断 round_step={step} global_step={learner.total_train_step} "
              f"seconds={result['seconds']:.1f} path={path}", flush=True)


def evaluate(learner, index, records, *, fixed=None, policy_surprise_weighting=None):
    """Stream <= one minibatch of sample references, including full holdout."""
    by_key = {(r["round_index"], r["game_id"]): r for r in records}
    games = {g.key: g for g in index.games}
    if fixed is None:
        refs = ({"round_index": g.round_index, "game_id": g.game_id, "decision_index": i}
                for g in index.split_games("validation") for i in range(g.decision_count))
    else:
        refs = iter(fixed["references"])
    totals = defaultdict(Counter)
    game_totals = defaultdict(Counter)
    score_columns = [GLOBAL_FEATURE_NAMES.index(n) for n in ("self_score_scaled", "opponent_score_scaled")]
    was_training = learner.model.training
    learner.model.eval()
    weighter = PolicySurpriseWeighting(policy_surprise_weighting) if policy_surprise_weighting is not None else None

    def flush(samples, identities):
        batch = collate_selfplay_samples(samples)
        surprise_weights = None
        if weighter is not None:
            refs = [ReplayReference(i["round_index"], i["run_id"], i["game_id"], s.index)
                    for s, i in zip(samples, identities, strict=True)]
            weighted_batch, _ = weighter.apply(batch, samples, refs)
            surprise_weights = weighted_batch.policy_weights.astype(np.float64)
        with torch.inference_mode():
            logits, values = learner.model(batch.decisions)
            mask = torch.as_tensor(batch.decisions.candidate_mask, device=learner.device)
            lp = torch.log_softmax(logits.masked_fill(~mask, -torch.inf), -1)
            lp = torch.where(mask, lp, torch.zeros_like(lp)).cpu().numpy().astype(np.float64)
            pred = values.cpu().numpy().astype(np.float64)
        target = batch.policy_targets.astype(np.float64)
        logs = np.zeros_like(target)
        np.log(target, out=logs, where=target > 0)
        pw = batch.policy_weights.astype(np.float64)
        y = batch.value_targets.astype(np.float64)
        probs = np.where(batch.decisions.candidate_mask, np.exp(lp), 0)
        ce = -(target * lp).sum(-1)
        th = -(target * logs).sum(-1)
        best = np.argmax(np.where(batch.decisions.candidate_mask, lp, -np.inf), -1)
        correct = target[np.arange(len(samples)), best] >= target.max(-1) - 1e-7
        late = batch.decisions.global_features[:, score_columns].max(-1) * SCALES.player_score >= 10 - 1e-5
        for i, identity in enumerate(identities):
            row = {"positions": 1, "policy_weight": float(pw[i]), "ce": ce[i]*pw[i],
                "kl": (ce[i]-th[i])*pw[i], "target_entropy": th[i]*pw[i],
                "entropy": float(-(probs[i]*lp[i]).sum())*pw[i], "top1": float(correct[i])*pw[i],
                "sq_error": (pred[i]-y[i])**2, "pred": pred[i], "pred_sq": pred[i]**2,
                "target": y[i], "target_sq": y[i]**2, "product": pred[i]*y[i]}
            if surprise_weights is not None:
                row.update(surprise_weight=float(surprise_weights[i]),
                           surprise_ce=ce[i]*surprise_weights[i],
                           surprise_kl=(ce[i]-th[i])*surprise_weights[i])
            for source in (identity["source_id"], "__all_observed__"):
                for stage in ("all", "late" if late[i] else "early_mid"):
                    totals[f"{source}/{identity['split']}/{stage}"].update(row)
            if fixed is None:
                key = f"{identity['round_index']}:{identity['game_id']}"
                for stage in ("all", "late" if late[i] else "early_mid"):
                    game_totals[f"{key}/{stage}"].update(row)

    samples, identities = [], []
    for ref in refs:
        key = (ref["round_index"], ref["game_id"])
        samples.append(index.cache.sample(games[key], ref["decision_index"]))
        identities.append(by_key[key])
        if len(samples) == learner.config.validation_batch_size:
            flush(samples, identities)
            samples, identities = [], []
    if samples:
        flush(samples, identities)
    learner.model.train(was_training)
    metrics = {k: ratios(v) for k, v in sorted(totals.items())}
    populations = defaultdict(Counter)
    for r in records:
        populations[r["split"]][r["source_id"]] += r["policy_samples"]
    # The fixed subsets balance sources. Rank by original pool policy-position
    # weights, never by the unweighted/balanced subset mean.
    weighted = {}
    for split, weights in populations.items():
        if not any(f"{source}/{split}/all" in metrics for source in weights):
            continue
        weighted[split] = {"policy_kl": sum(metrics[f"{source}/{split}/all"]["policy_kl"] * n for source,n in weights.items()) / sum(weights.values())}
        counts = Counter()
        for r in records:
            if r["split"] == split:
                counts[r["source_id"]] += r["decision_count"]
        weighted[split]["value_mse"] = sum(metrics[f"{source}/{split}/all"]["value_mse"] * n for source,n in counts.items()) / sum(counts.values())
    result = {"scope": "complete original validation games" if fixed is None else "fixed per-source train/validation subsets; observed aggregate is source-balanced",
        "set_digest": None if fixed is None else fixed["digest"], "metrics": metrics, "pool_weighted": weighted}
    if fixed is None:
        result["per_game"] = {k: ratios(v) for k,v in game_totals.items()}
    return result

