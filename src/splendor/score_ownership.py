"""Future acquired points by player and source, supervised at every decision."""
from __future__ import annotations

import numpy as np
import torch

from .catalog import load_default_catalog
from .features import CARD_FEATURE_NAMES as C, GLOBAL_FEATURE_NAMES as G, SCALES

PROTOCOL = "remaining-score-by-player-source-v1"
SCORE_SCALE = 15.
OUTPUT_NAMES = tuple(f"{role}_{source}" for role in ("self", "opponent")
                     for source in ("tier1", "tier2", "tier3", "noble"))
NOBLE_POINT_VALUES = tuple(sorted({noble.points for noble in load_default_catalog().nobles}))
DIAGNOSTIC_GROUPS = ("all", "phase_main_action", "phase_payment", "phase_return_tokens",
                     "phase_choose_noble", "non_payment", "stage_early", "stage_middle", "stage_late")


def game_score_ownership_targets(arrays, metadata):
    """Sum future score increments, including the action at the current row.

    Pay applies card points; a sole eligible noble can be auto-claimed in the
    same transition as Pay, a main action or a token return. Remaining score
    increments are noble points, checked against public noble-count changes.
    Terminal_result covers the last transition. No trace/search values needed.
    """
    if metadata.get("status") != "natural_terminal":
        raise ValueError("score ownership labels require a natural terminal")
    g, players = arrays["global_features"], arrays["player_id"]
    n = len(players)
    scores = np.empty((n, 2), dtype=np.int64)
    own = np.rint(g[:, G.index("self_score_scaled")] * SCALES.player_score).astype(np.int64)
    other = np.rint(g[:, G.index("opponent_score_scaled")] * SCALES.player_score).astype(np.int64)
    for player in (0, 1):
        scores[:, player] = np.where(players == player, own, other)
    terminal = np.asarray(metadata["terminal_result"]["scores"], dtype=np.int64)
    increments = np.diff(np.concatenate((scores, terminal[None]), axis=0), axis=0)
    rows = np.arange(n)
    if np.any(increments < 0) or np.any(increments[rows, 1-players] != 0):
        raise ValueError("score ownership requires nondecreasing scores and actor-only increments")
    delta = increments[rows, players]
    pay = g[:, G.index("phase_payment")] > .5
    noble = g[:, G.index("phase_choose_noble")] > .5
    pending = arrays["pending_card"][pay]
    tier = pending[:, C.index("tier_1"):C.index("tier_3")+1] > .5
    if (not np.all(arrays["pending_card_mask"][pay]) or not np.all(pending[:, C.index("known")] > .5)
            or np.any(tier.sum(1) != 1)):
        raise ValueError("payment score ownership requires a known pending card and its tier")
    points = np.rint(pending[:, C.index("points_scaled")] * SCALES.card_points).astype(np.int64)
    card_delta = np.zeros(n, dtype=np.int64)
    card_delta[pay] = points
    noble_delta = delta - card_delta
    if (np.any(noble_delta < 0) or np.any(~np.isin(noble_delta, (0, *NOBLE_POINT_VALUES)))
            or np.any(noble_delta[noble] <= 0)):
        raise ValueError("score increments disagree with purchased cards or claimed nobles")
    counts = np.empty_like(scores)
    own_count = np.rint(g[:, G.index("self_nobles_scaled")] * SCALES.noble_count).astype(np.int64)
    other_count = np.rint(g[:, G.index("opponent_nobles_scaled")] * SCALES.noble_count).astype(np.int64)
    for player in (0, 1):
        counts[:, player] = np.where(players == player, own_count, other_count)
    count_delta = np.diff(counts, axis=0)
    if (np.any(count_delta[rows[:-1], 1-players[:-1]] != 0)
            or not np.array_equal(count_delta[rows[:-1], players[:-1]], (noble_delta[:-1] > 0).astype(np.int64))):
        raise ValueError("noble score increments disagree with public noble counts")
    gains = np.zeros((n, 2, 4), dtype=np.int64)
    gains[rows[pay], players[pay], tier.argmax(1)] = points
    gains[rows, players, 3] = noble_delta
    future = np.cumsum(gains[::-1], axis=0)[::-1]
    if not np.array_equal(future.sum(2), terminal[None] - scores):
        raise ValueError("future score components do not sum to remaining terminal points")
    target = np.concatenate((future[rows, players], future[rows, 1-players]), axis=1)
    # Integer labels retain exact points and are cheap to store/transfer.
    if np.any(target > np.iinfo(np.int16).max):
        raise ValueError("score ownership label exceeds int16 storage")
    return {"score_ownership_target": target.astype(np.int16)}


def score_ownership_loss_sums(predictions, labels, value_weights):
    prediction = predictions["score_ownership"]
    target = torch.as_tensor(labels["score_ownership_target"], device=prediction.device,
                             dtype=prediction.dtype) / SCORE_SCALE
    if prediction.shape != (len(value_weights), len(OUTPUT_NAMES)) or target.shape != prediction.shape:
        raise ValueError("score ownership targets differ from the value batch")
    error = prediction - target
    sums = {"score_ownership_weight_sum": value_weights.sum()}
    for j, name in enumerate(OUTPUT_NAMES):
        sums[f"score_ownership_{name}_sq_sum"] = (error[:, j].square() * value_weights).sum()
        sums[f"score_ownership_{name}_abs_sum"] = (error[:, j].abs() * value_weights).sum()
    return sums


def score_ownership_loss(sums):
    return sum(sums[f"score_ownership_{name}_sq_sum"] for name in OUTPUT_NAMES) / (
        len(OUTPUT_NAMES) * sums["score_ownership_weight_sum"].clamp_min(1e-12))


def score_ownership_gradient_tasks(sums, coefficient):
    n = sums["score_ownership_weight_sum"].clamp_min(1e-12)
    return {**{f"score_ownership_{name}": (sums[f"score_ownership_{name}_sq_sum"] / n,
                                        coefficient / len(OUTPUT_NAMES)) for name in OUTPUT_NAMES},
            "score_ownership": (score_ownership_loss(sums), coefficient)}


@torch.no_grad()
def score_ownership_diagnostic_sums(predictions, labels, value_weights, global_features):
    """Validation-only sufficient statistics; never affect loss or sampling.

    Each baseline is the best constant per output on the complete evaluated
    population (or group), not a mean fitted separately in each minibatch.
    Natural-point moments use float64 to preserve between-batch variation.
    """
    prediction = predictions["score_ownership"].detach().to(torch.float64) * SCORE_SCALE
    target = torch.as_tensor(labels["score_ownership_target"], device=prediction.device, dtype=torch.float64)
    features = np.asarray(global_features)
    phases = features[:, [G.index("phase_" + name) for name in
                         ("main_action", "payment", "return_tokens", "choose_noble")]].argmax(1)
    # These are current public scores, not future outcome or game-length bins.
    leading_score = np.rint(np.maximum(features[:, G.index("self_score_scaled")],
                                      features[:, G.index("opponent_score_scaled")]) * SCALES.player_score)
    masks = np.stack((np.ones(len(features), dtype=bool), *(phases == i for i in range(4)),
                      phases != 1, leading_score < 5, (leading_score >= 5) & (leading_score < 10),
                      leading_score >= 10), axis=1)
    weights = torch.as_tensor(masks, device=prediction.device, dtype=torch.float64)
    weights = (weights * value_weights[:, None]).T
    error = prediction - target
    counts = weights.sum(1)
    sq = weights @ error.square().mean(1)
    absolute = weights @ error.abs().mean(1)
    target_sums, target_sq_sums = weights @ target, weights @ target.square()
    sums = {}
    for i, group in enumerate(DIAGNOSTIC_GROUPS):
        prefix = f"score_ownership_diagnostic_{group}"
        sums.update({f"{prefix}_weight_sum": counts[i], f"{prefix}_sq_sum": sq[i],
                     f"{prefix}_abs_sum": absolute[i]})
        for j, name in enumerate(OUTPUT_NAMES):
            sums[f"{prefix}_{name}_target_sum"] = target_sums[i, j]
            sums[f"{prefix}_{name}_target_sq_sum"] = target_sq_sums[i, j]
    prediction_sums = weights[0] @ prediction
    zero_sums = weights[0] @ (target == 0).to(torch.float64)
    for j, name in enumerate(OUTPUT_NAMES):
        sums[f"score_ownership_diagnostic_{name}_prediction_sum"] = prediction_sums[j]
        sums[f"score_ownership_diagnostic_{name}_zero_sum"] = zero_sums[j]
    return sums


def _diagnostic_metrics(sums):
    if "score_ownership_diagnostic_all_weight_sum" not in sums:
        return {}
    result = {}
    for group in DIAGNOSTIC_GROUPS:
        raw = f"score_ownership_diagnostic_{group}"
        prefix = "score_ownership" if group == "all" else f"score_ownership_{group}"
        n = float(sums[f"{raw}_weight_sum"])
        # Empty groups are absent evidence, not perfect zero-error predictions.
        mse = float(sums[f"{raw}_sq_sum"]) / n if n else None
        mae = float(sums[f"{raw}_abs_sum"]) / n if n else None
        zeros, constants = [], []
        for name in OUTPUT_NAMES:
            mean = float(sums[f"{raw}_{name}_target_sum"]) / n if n else None
            zero = float(sums[f"{raw}_{name}_target_sq_sum"]) / n if n else None
            constant = max(0., zero - mean**2) if n else None
            zeros.append(zero)
            constants.append(constant)
            if group == "all":
                output = f"score_ownership_{name}"
                result[f"{output}_target_mean_points"] = mean
                result[f"{output}_prediction_mean_points"] = (
                    float(sums[f"score_ownership_diagnostic_{name}_prediction_sum"]) / n if n else None)
                result[f"{output}_target_zero_fraction"] = (
                    float(sums[f"score_ownership_diagnostic_{name}_zero_sum"]) / n if n else None)
                result[f"{output}_zero_mse_points"] = zero
                result[f"{output}_constant_mse_points"] = constant
        zero = sum(zeros) / len(OUTPUT_NAMES) if n else None
        constant = sum(constants) / len(OUTPUT_NAMES) if n else None
        result.update({f"{prefix}_zero_mse_points": zero, f"{prefix}_constant_mse_points": constant,
                       f"{prefix}_constant_skill": 1 - mse / constant if constant else None})
        if group != "all":
            result.update({f"{prefix}_positions": n, f"{prefix}_mse_points": mse, f"{prefix}_mae_points": mae})
    return result


def score_ownership_metrics(sums):
    n = max(float(sums["score_ownership_weight_sum"]), 1e-12)
    result = {}
    for name in OUTPUT_NAMES:
        result[f"score_ownership_{name}_mse_points"] = float(sums[f"score_ownership_{name}_sq_sum"]) / n * SCORE_SCALE**2
        result[f"score_ownership_{name}_mae_points"] = float(sums[f"score_ownership_{name}_abs_sum"]) / n * SCORE_SCALE
    mse = sum(result[f"score_ownership_{name}_mse_points"] for name in OUTPUT_NAMES) / len(OUTPUT_NAMES)
    mae = sum(result[f"score_ownership_{name}_mae_points"] for name in OUTPUT_NAMES) / len(OUTPUT_NAMES)
    return result | _diagnostic_metrics(sums) | {"score_ownership_loss": mse / SCORE_SCALE**2, "score_ownership_mse_points": mse,
                     "score_ownership_mae_points": mae, "score_ownership_positions": float(sums["score_ownership_weight_sum"])}
