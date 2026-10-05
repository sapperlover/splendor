"""Role-aware set attention over the unchanged observation/action schema."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .actor_critic import ActorCritic, TorchDecisionBatch, TorchStateBatch
from .features import FEATURE_SCHEMA, FEATURE_SCHEMA_VERSION, GLOBAL_FEATURE_NAMES


@dataclass(frozen=True, slots=True)
class EntityAttentionConfig:
    architecture: str = "entity_attention_v1"
    feature_schema_version: str = FEATURE_SCHEMA_VERSION
    hidden: int = 128
    heads: int = 4
    layers: int = 2
    ffn_hidden: int = 512
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.architecture != "entity_attention_v1":
            raise ValueError("EntityAttentionConfig requires entity_attention_v1")
        if self.feature_schema_version != FEATURE_SCHEMA_VERSION:
            raise ValueError("model feature schema mismatch")
        if (self.hidden, self.heads, self.layers, self.ffn_hidden, self.dropout) != (128, 4, 2, 512, 0):
            raise ValueError("entity_attention_v1 fixes hidden128/heads4/layers2/FFN512/dropout0")


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
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden),
                                     nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, state: tuple[Tensor, Tensor]) -> Tensor:
        return self.network(state[0][:, 0])


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
        self.critic = GlobalValueHead(hidden)

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
