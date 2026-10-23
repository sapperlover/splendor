"""Role-aware set attention over the unchanged observation/action schema."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .actor_critic import ActorCritic, TorchDecisionBatch, TorchStateBatch
from .features import (CARD_FEATURE_NAMES, FEATURE_SCHEMA, FEATURE_SCHEMA_VERSION,
                       GLOBAL_FEATURE_NAMES, SCALES)


MARKET_RELATION_FEATURE_NAMES = (
    *(f"opponent_ordinary_deficit_{color}_scaled" for color in ("white", "blue", "green", "red", "black")),
    "opponent_gold_uncovered_deficit_scaled", "self_can_afford", "opponent_can_afford",
)
_CARD_COST = CARD_FEATURE_NAMES.index("cost_white_scaled")
_CARD_GOLD = CARD_FEATURE_NAMES.index("gold_uncovered_deficit_scaled")
_CARD_MARKET_SOURCE = CARD_FEATURE_NAMES.index("source_market")


def market_purchase_relations(cards: Tensor, mask: Tensor, opponent: Tensor) -> Tensor:
    """Eight current-actor-relative signals for market and public reserved cards.

    Existing card features already contain self deficits and post-gold gap. Totals
    and self/opponent differences are not stacked again. Affordability ignores
    the current decision phase: it tests resources, not legal-action membership.
    """
    known = mask & (cards[..., 1] > 0.5)
    cards = torch.where(known.unsqueeze(-1), cards, 0)
    # Recover public integer counts before thresholding: normalized float32
    # arithmetic can otherwise turn exact affordability into a tiny positive gap.
    costs = (cards[..., _CARD_COST:_CARD_COST + 5] * SCALES.card_cost).round()
    tokens = (opponent[:, :5] * opponent.new_tensor(SCALES.token_by_color[:5])).round()
    bonuses = (opponent[:, 6:11] * SCALES.player_bonus).round()
    gold = (opponent[:, 5] * SCALES.token_by_color[5]).round()
    deficit = (costs - bonuses[:, None, :] - tokens[:, None, :]).clamp_min(0)
    uncovered = (deficit.sum(-1) - gold[:, None]).clamp_min(0)
    relations = torch.cat((deficit / SCALES.card_cost,
                           (uncovered / SCALES.purchase_gap_total).unsqueeze(-1),
                           (cards[..., _CARD_GOLD] == 0).to(cards.dtype).unsqueeze(-1),
                           (uncovered == 0).to(cards.dtype).unsqueeze(-1)), dim=-1)
    return torch.where(known.unsqueeze(-1), relations, 0)


@dataclass(frozen=True, slots=True)
class EntityAttentionConfig:
    architecture: str = "entity_attention_v1"
    feature_schema_version: str = FEATURE_SCHEMA_VERSION
    hidden: int = 128
    heads: int = 4
    layers: int = 2
    ffn_hidden: int = 512
    dropout: float = 0.0
    market_relations: bool = False
    value_readout: str = "global"

    def __post_init__(self) -> None:
        if self.architecture != "entity_attention_v1":
            raise ValueError("EntityAttentionConfig requires entity_attention_v1")
        if self.feature_schema_version != FEATURE_SCHEMA_VERSION:
            raise ValueError("model feature schema mismatch")
        for name in ("hidden", "heads", "layers", "ffn_hidden"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"entity_attention_v1 {name} must be a positive integer")
        if self.hidden % self.heads:
            raise ValueError("entity_attention_v1 hidden must be divisible by heads")
        if type(self.market_relations) is not bool:
            raise ValueError("market_relations must be boolean")
        if self.value_readout not in ("global", "global_players_market"):
            raise ValueError("invalid value_readout")
        if self.dropout != 0:
            raise ValueError("entity_attention_v1 requires dropout0")


class EntityAttentionLayer(nn.Module):
    def __init__(self, hidden: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, hidden)
        self.output = nn.Linear(hidden, hidden)

    def forward(self, query: Tensor, memory: Tensor, valid: Tensor) -> Tensor:
        def split(x: Tensor) -> Tensor:
            return x.reshape(x.shape[0], x.shape[1], self.heads, -1).transpose(1, 2)
        attended = F.scaled_dot_product_attention(
            split(self.query(query)), split(self.key(memory)), split(self.value(memory)),
            attn_mask=valid[:, None, None, :], dropout_p=0.0,
        )
        joined = attended.transpose(1, 2).reshape(query.shape[0], query.shape[1], -1)
        return self.output(joined)


class EntityBlock(nn.Module):
    def __init__(self, config: EntityAttentionConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.hidden)
        self.attention = EntityAttentionLayer(config.hidden, config.heads)
        self.ffn_norm = nn.LayerNorm(config.hidden)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden, config.ffn_hidden), nn.GELU(), nn.Dropout(0.0),
            nn.Linear(config.ffn_hidden, config.hidden), nn.Dropout(0.0),
        )

    def forward(self, tokens: Tensor, valid: Tensor) -> Tensor:
        normalized = self.attention_norm(tokens)
        tokens = tokens + self.attention(normalized, normalized, valid)
        tokens = tokens + self.ffn(self.ffn_norm(tokens))
        return torch.where(valid.unsqueeze(-1), tokens, 0)


class GlobalValueHead(nn.Module):
    def __init__(self, hidden: int, outputs: int = 1) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden),
                                     nn.GELU(), nn.Linear(hidden, outputs))

    def forward(self, state: tuple[Tensor, Tensor]) -> Tensor:
        return self.network(state[0][:, 0])


class PlayersMarketValueHead(nn.Module):
    """Direct readout of global, self, opponent and masked mean market tokens."""
    def __init__(self, hidden: int, outputs: int = 1) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.LayerNorm(4 * hidden), nn.Linear(4 * hidden, hidden),
                                     nn.GELU(), nn.Linear(hidden, outputs))

    def forward(self, state: tuple[Tensor, Tensor]) -> Tensor:
        entities, valid = state
        market_valid = valid[:, 3:15].unsqueeze(-1)
        market = torch.where(market_valid, entities[:, 3:15], 0).sum(1)
        market = market / market_valid.sum(1).clamp_min(1)
        return self.network(torch.cat((entities[:, 0], entities[:, 1], entities[:, 2], market), dim=-1))


def _value_head(hidden: int, readout: str, outputs: int = 1) -> nn.Module:
    return GlobalValueHead(hidden, outputs) if readout == "global" else PlayersMarketValueHead(hidden, outputs)


class EntityAttention(ActorCritic):
    """Same public model API, with one shared state encode and independent action queries."""
    def __init__(self, config: EntityAttentionConfig | None = None) -> None:
        nn.Module.__init__(self)
        self.config = config or EntityAttentionConfig()
        hidden = self.config.hidden
        # Player numeric blocks have identical ordering; every remaining global field
        # (including all public_purchased bits and phase/pending context) is retained.
        own = tuple(i for i, name in enumerate(GLOBAL_FEATURE_NAMES)
                    if name.startswith("self_") and name != "self_is_first_player")
        other = tuple(i for i, name in enumerate(GLOBAL_FEATURE_NAMES) if name.startswith("opponent_")
                      and name != "opponent_completed_turns_scaled")
        self.self_indices = own
        self.opponent_indices = other
        self.context_indices = tuple(i for i in range(FEATURE_SCHEMA.global_dim) if i not in own + other)
        assert len(own) == len(other) == 14
        self.context_encoder = nn.Linear(len(self.context_indices), hidden)
        self.player_encoder = nn.Linear(14, hidden)
        self.card_encoder = nn.Linear(FEATURE_SCHEMA.card_dim, hidden)
        self.noble_encoder = nn.Linear(FEATURE_SCHEMA.noble_dim, hidden)
        self.type_embedding = nn.Embedding(6, hidden)  # global/player/market/reserve/noble/pending
        self.owner_embedding = nn.Embedding(3, hidden)  # none/self/opponent
        self.tier_embedding = nn.Embedding(4, hidden)  # none/market T1/T2/T3
        self.blocks = nn.ModuleList(EntityBlock(self.config) for _ in range(self.config.layers))
        self.action_encoder = nn.Sequential(nn.Linear(FEATURE_SCHEMA.action_dim, hidden), nn.GELU())
        self.query_norm = nn.LayerNorm(hidden)
        self.memory_norm = nn.LayerNorm(hidden)
        self.cross_attention = EntityAttentionLayer(hidden, self.config.heads)
        self.actor = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.critic = _value_head(hidden, self.config.value_readout)
        if self.config.market_relations:
            self.market_relation_encoder = nn.Linear(len(MARKET_RELATION_FEATURE_NAMES), hidden, bias=False)

    def _role(self, kind: int, owner: int = 0, tier: int = 0) -> Tensor:
        return self.type_embedding.weight[kind] + self.owner_embedding.weight[owner] + self.tier_embedding.weight[tier]

    def encode_state(self, batch: TorchDecisionBatch | TorchStateBatch) -> tuple[Tensor, Tensor]:
        global_features = batch.global_features
        tokens = [
            (self.context_encoder(global_features[:, self.context_indices]) + self._role(0)).unsqueeze(1),
            (self.player_encoder(global_features[:, self.self_indices]) + self._role(1, 1)).unsqueeze(1),
            (self.player_encoder(global_features[:, self.opponent_indices]) + self._role(1, 2)).unsqueeze(1),
        ]
        masks = [torch.ones((global_features.shape[0], 3), dtype=torch.bool, device=global_features.device)]
        for features, mask, kind, owner, tier in (
            (batch.market_t1, batch.market_t1_mask, 2, 0, 1),
            (batch.market_t2, batch.market_t2_mask, 2, 0, 2),
            (batch.market_t3, batch.market_t3_mask, 2, 0, 3),
            (batch.self_reserved, batch.self_reserved_mask, 3, 1, 0),
            (batch.opponent_reserved, batch.opponent_reserved_mask, 3, 2, 0),
            (batch.nobles, batch.nobles_mask, 4, 0, 0),
            (batch.pending_card.unsqueeze(1), batch.pending_card_mask.unsqueeze(1), 5, 1, 0),
        ):
            encoder = self.noble_encoder if kind == 4 else self.card_encoder
            encoded = encoder(torch.where(mask.unsqueeze(-1), features, 0)) + self._role(kind, owner, tier)
            if kind in (2, 3) and self.config.market_relations:
                # Public market reservations use the same relation projection.
                # Self-known deck reservations retain their original private
                # features, without this additional public-card channel.
                public_mask = mask & (features[..., _CARD_MARKET_SOURCE] > 0.5)
                relations = market_purchase_relations(features, public_mask, global_features[:, self.opponent_indices])
                encoded = encoded + self.market_relation_encoder(relations)
            tokens.append(torch.where(mask.unsqueeze(-1), encoded, 0))
            masks.append(mask)
        valid = torch.cat(masks, dim=1)
        entities = torch.cat(tokens, dim=1)
        for block in self.blocks:
            entities = block(entities, valid)
        return entities, valid

    def _actor_logits(self, prepared: TorchDecisionBatch, state: tuple[Tensor, Tensor]) -> Tensor:
        raw = torch.where(prepared.candidate_mask.unsqueeze(-1), prepared.candidate_features, 0)
        query = self.action_encoder(raw)
        memory, valid = state
        query = query + self.cross_attention(self.query_norm(query), self.memory_norm(memory), valid)
        return self.actor(query).squeeze(-1)

    def build_actor_projection(self):
        raise ValueError("entity_attention_v1 requires actor_projection=false")


@dataclass(frozen=True, slots=True)
class EntityAttentionAuxConfig:
    architecture: str = "entity_attention_aux_v1"
    feature_schema_version: str = FEATURE_SCHEMA_VERSION
    hidden: int = 128
    heads: int = 4
    layers: int = 2
    ffn_hidden: int = 512
    dropout: float = 0.0
    market_relations: bool = False
    value_readout: str = "global"
    predict_score: bool = True

    def __post_init__(self):
        if self.architecture != "entity_attention_aux_v1":
            raise ValueError("invalid auxiliary attention architecture")
        if type(self.predict_score) is not bool:
            raise ValueError("predict_score must be boolean")
        EntityAttentionConfig(feature_schema_version=self.feature_schema_version,
                              hidden=self.hidden, heads=self.heads, layers=self.layers,
                              ffn_hidden=self.ffn_hidden, dropout=self.dropout,
                              market_relations=self.market_relations, value_readout=self.value_readout)


class EntityAttentionAux(EntityAttention):
    """Duration and optional score heads; search consumes policy and win/loss value."""

    def __init__(self, config: EntityAttentionAuxConfig | None = None):
        config = config or EntityAttentionAuxConfig()
        super().__init__(EntityAttentionConfig(feature_schema_version=config.feature_schema_version,
                                              hidden=config.hidden, heads=config.heads, layers=config.layers,
                                              ffn_hidden=config.ffn_hidden, dropout=config.dropout,
                                              market_relations=config.market_relations, value_readout=config.value_readout))
        self.config = config
        self.score_head = _value_head(config.hidden, config.value_readout)
        self.remaining_turns_head = _value_head(config.hidden, config.value_readout)
        if not config.predict_score:
            # Consume the legacy score-head initialization draws so the remaining
            # head starts identically in matched dual-head/turns-only experiments.
            del self.score_head

    def forward_with_auxiliary(self, batch):
        prepared = self._prepare(batch)
        state = self.encode_state(prepared)
        return (self._actor_logits(prepared, state), self.critic(state).squeeze(-1),
                self.score_head(state).squeeze(-1) if self.config.predict_score else None,
                F.softplus(self.remaining_turns_head(state).squeeze(-1)))


@dataclass(frozen=True, slots=True)
class EntityAttentionPolicyValueConfig:
    """Inference-only composition of two unchanged entity_attention_v1 models."""

    architecture: str = "entity_attention_policy_value_v1"
    feature_schema_version: str = FEATURE_SCHEMA_VERSION

    def __post_init__(self):
        if self.architecture != "entity_attention_policy_value_v1":
            raise ValueError("invalid policy/value composition architecture")
        if self.feature_schema_version != FEATURE_SCHEMA_VERSION:
            raise ValueError("model feature schema mismatch")


class _SelectValue(nn.Module):
    def forward(self, state):
        return state[1]


class EntityAttentionPolicyValue(ActorCritic):
    """Keep each frozen branch's encoder and return policy(A), value(B).

    Compiled search calls encode_state/_actor_logits/critic directly, so the
    opaque state also carries B's value output. This preserves the same routing
    for eager, compiled, policy-only and state-only search entry points.
    """

    def __init__(self, config: EntityAttentionPolicyValueConfig | None = None):
        nn.Module.__init__(self)
        self.config = config or EntityAttentionPolicyValueConfig()
        self.policy_model = EntityAttention()
        self.value_model = EntityAttention()
        self.critic = _SelectValue()

    def encode_state(self, batch: TorchDecisionBatch | TorchStateBatch):
        return (self.policy_model.encode_state(batch),
                self.value_model.critic(self.value_model.encode_state(batch)))

    def _actor_logits(self, prepared: TorchDecisionBatch, state):
        return self.policy_model._actor_logits(prepared, state[0])

    def _policy_logits_prepared(self, prepared, *, projection=None):
        if projection is not None:
            raise ValueError("policy/value composition requires actor_projection=false")
        return self.policy_model._policy_logits_prepared(prepared)

    def build_actor_projection(self):
        raise ValueError("policy/value composition requires actor_projection=false")
