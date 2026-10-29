"""Opponent's next MAIN_ACTION search distribution, aligned to current visible cards."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .entity_attention import _value_head
from .features import ACTION_FEATURE_NAMES as A, CARD_FEATURE_NAMES as C, GLOBAL_FEATURE_NAMES as G

ACTION_CLASSES = ("buy", "take", "reserve", "other")
CARD_FIELDS = ("market_t1", "market_t2", "market_t3", "self_reserved", "opponent_reserved")
# Intrinsic card identity, excluding owner/source and actor-relative affordability.
CARD_IDENTITY = tuple(C.index(name) for name in (
    "tier_1", "tier_2", "tier_3", "bonus_white", "bonus_blue", "bonus_green", "bonus_red", "bonus_black",
    "points_scaled", "cost_white_scaled", "cost_blue_scaled", "cost_green_scaled", "cost_red_scaled", "cost_black_scaled"))
VISIBLE_SLOTS = 19  # market12, self-reserved3, opponent-reserved3, pending1
UNKNOWN_SLOT = VISIBLE_SLOTS
CARD_CLASSES = VISIBLE_SLOTS + 1
ENTITY_INDICES = (*range(3, 21), 24)
TYPE_GROUPS = (("buy_market", "buy_reserved"), ("take_distinct", "take_pair"),
               ("reserve_market", "reserve_deck"), ("pay", "return_tokens", "claim_noble"))


def visible_card_arrays(arrays):
    cards = np.concatenate([arrays[name] for name in CARD_FIELDS] + [arrays["pending_card"][:, None]], axis=1)
    mask = np.concatenate([arrays[name + "_mask"] for name in CARD_FIELDS]
                          + [arrays["pending_card_mask"][:, None]], axis=1).copy()
    mask &= cards[..., C.index("known")] > .5
    identity = cards[..., CARD_IDENTITY]
    for slot in range(VISIBLE_SLOTS):
        # A pending purchase can still be present in market/reserved slots.
        duplicate = np.any(np.all(identity[:, :slot] == identity[:, slot:slot+1], axis=-1) & mask[:, :slot], axis=1)
        mask[:, slot] &= ~duplicate
    return cards, mask


class OpponentPolicyHead(nn.Module):
    def __init__(self, hidden, readout):
        super().__init__()
        self.action = _value_head(hidden, readout, len(ACTION_CLASSES))
        # Independent buy/reserve logits share only the card representation.
        self.card = nn.Sequential(nn.LayerNorm(4*hidden), nn.Linear(4*hidden, hidden),
                                  nn.GELU(), nn.Linear(hidden, 2))
        self.unknown = _value_head(hidden, readout, 2)

    def forward(self, prepared, state):
        entities, _ = state
        card_tokens = entities[:, ENTITY_INDICES]
        context = entities[:, :3].flatten(1).unsqueeze(1).expand(-1, VISIBLE_SLOTS, -1)
        logits = self.card(torch.cat((card_tokens, context), dim=-1))
        cards = torch.cat([getattr(prepared, name) for name in CARD_FIELDS] + [prepared.pending_card[:, None]], dim=1)
        mask = torch.cat([getattr(prepared, name + "_mask") for name in CARD_FIELDS]
                         + [prepared.pending_card_mask[:, None]], dim=1) & (cards[..., C.index("known")] > .5)
        identity = cards[..., CARD_IDENTITY]
        # Mask duplicate visible identities using the same first-slot convention as labels.
        equal = (identity[:, :, None] == identity[:, None, :]).all(-1)
        earlier = torch.ones((VISIBLE_SLOTS, VISIBLE_SLOTS), dtype=torch.bool, device=cards.device).tril(-1)
        mask = mask & ~(equal & earlier & mask[:, None, :]).any(-1)
        logits = logits.masked_fill(~mask[..., None], -1e9)
        logits = torch.cat((logits, self.unknown(state)[:, None]), dim=1)
        return {"opponent_action": self.action(state), "opponent_buy": logits[..., 0], "opponent_reserve": logits[..., 1]}


def game_opponent_policy_targets(arrays, metadata):
    if metadata.get("status") != "natural_terminal":
        raise ValueError("opponent policy labels require a natural terminal")
    players = arrays["player_id"]
    n = len(players)
    main = np.flatnonzero(arrays["global_features"][:, G.index("phase_main_action")] > .5)
    cards, visible = visible_card_arrays(arrays)
    identities = cards[..., CARD_IDENTITY]
    action = np.zeros((n, len(ACTION_CLASSES)), dtype=np.float32)
    buy = np.zeros((n, CARD_CLASSES), dtype=np.float32)
    reserve = np.zeros_like(buy)
    weight = np.zeros(n, dtype=np.float32)
    main_by_player = [main[players[main] == p] for p in (0, 1)]
    for i, player in enumerate(players):
        future = main_by_player[1 - player]
        j = np.searchsorted(future, i, side="right")
        if j == len(future):
            continue  # Terminal before the opponent acts: no target, not OTHER.
        nxt = future[j]
        # Preserve the source pool's policy eligibility (e.g. low-budget targets off).
        if arrays["policy_weight"][nxt] <= 0:
            continue
        start, end = arrays["candidate_offsets"][nxt:nxt+2]
        candidates = arrays["candidate_features"][start:end]
        policy = arrays["policy_target"][start:end]
        if np.any(policy < 0) or not np.isclose(policy.sum(), 1, atol=1e-6):
            raise ValueError("opponent source policy must be a probability distribution")
        by_type = np.stack([candidates[:, [A.index("type_" + name) for name in group]].sum(axis=1)
                            for group in TYPE_GROUPS], axis=1)
        if not np.all(by_type.sum(axis=1) == 1):
            raise ValueError("opponent candidate has an unknown action type")
        action[i] = policy @ by_type
        first_card = A.index("target_card.valid")
        target_cards = candidates[:, first_card:first_card + len(C)]
        for c, probability in enumerate(policy):
            if probability == 0 or (by_type[c, 0] == 0 and by_type[c, 2] == 0):
                continue
            matches = np.flatnonzero(visible[i] & np.all(identities[i] == target_cards[c, CARD_IDENTITY], axis=1))
            slot = int(matches[0]) if target_cards[c, C.index("known")] > .5 and len(matches) else UNKNOWN_SLOT
            (buy if by_type[c, 0] else reserve)[i, slot] += probability
        weight[i] = arrays["policy_weight"][nxt]
    return {"opponent_action_target": action, "opponent_buy_target": buy,
            "opponent_reserve_target": reserve, "opponent_policy_weight": weight}


def opponent_loss_sums(predictions, labels, value_weights):
    source_weight = torch.as_tensor(labels["opponent_policy_weight"], device=value_weights.device)
    weight = value_weights * source_weight
    sums = {"opponent_weight_sum": weight.sum(),
            "opponent_label_weight_sum": (value_weights * (source_weight > 0)).sum(),
            "opponent_eligible_weight_sum": value_weights.sum()}
    for name, size in (("action", len(ACTION_CLASSES)), ("buy", CARD_CLASSES), ("reserve", CARD_CLASSES)):
        logits = predictions["opponent_" + name]
        target = torch.as_tensor(labels["opponent_" + name + "_target"], device=logits.device)
        if target.shape != logits.shape or target.shape != (len(weight), size):
            raise ValueError("opponent targets differ from the value batch")
        # Card targets retain joint mass: CE is weighted by P(buy)/P(reserve).
        sums[f"opponent_{name}_ce_sum"] = (-(target * F.log_softmax(logits, -1)).sum(-1) * weight).sum()
        if name != "action":
            sums[f"opponent_{name}_weight_sum"] = (target.sum(-1) * weight).sum()
            sums[f"opponent_{name}_unknown_mass_sum"] = (target[:, UNKNOWN_SLOT] * weight).sum()
    return sums


def opponent_metrics(sums):
    n = max(float(sums["opponent_weight_sum"]), 1e-12)
    result = {f"opponent_{name}_ce": float(sums[f"opponent_{name}_ce_sum"]) / n
              for name in ("action", "buy", "reserve")}
    result.update(opponent_policy_loss=sum(result.values()),
                  opponent_positions=float(sums["opponent_weight_sum"]),
                  opponent_label_positions=float(sums["opponent_label_weight_sum"]),
                  opponent_eligible_positions=float(sums["opponent_eligible_weight_sum"]))
    eligible = result["opponent_eligible_positions"]
    result["opponent_label_coverage"] = result["opponent_label_positions"] / eligible if eligible > 0 else None
    for name in ("buy", "reserve"):
        mass = float(sums[f"opponent_{name}_weight_sum"])
        unknown = float(sums[f"opponent_{name}_unknown_mass_sum"])
        result[f"opponent_{name}_mass"] = mass
        result[f"opponent_{name}_unknown_mass"] = unknown
        result[f"opponent_{name}_conditional_ce"] = float(sums[f"opponent_{name}_ce_sum"]) / mass if mass > 0 else None
        result[f"opponent_{name}_unknown_ratio"] = unknown / mass if mass > 0 else None
    mass = result["opponent_buy_mass"] + result["opponent_reserve_mass"]
    unknown = result["opponent_buy_unknown_mass"] + result["opponent_reserve_unknown_mass"]
    result["opponent_unknown_ratio"] = unknown / mass if mass > 0 else None
    return result
