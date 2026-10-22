"""Set summaries and a residual MLP, with unpooled target attributes in the actor."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .actor_critic import ActorCritic, ActorProjection, TorchDecisionBatch, TorchStateBatch
from .entity_attention import MARKET_RELATION_FEATURE_NAMES, market_purchase_relations
from .features import CARD_FEATURE_NAMES, FEATURE_SCHEMA, FEATURE_SCHEMA_VERSION, GLOBAL_FEATURE_NAMES


@dataclass(frozen=True, slots=True)
class EntitySetMLPConfig:
    architecture: str = "entity_set_mlp_v1"
    feature_schema_version: str = FEATURE_SCHEMA_VERSION
    card_hidden: int = 64
    noble_hidden: int = 32
    state_hidden: int = 256
    residual_blocks: int = 4
    actor_hidden: int = 128
    critic_hidden: int = 128
    market_relations: bool = True
    action_bottleneck: int = 0
    multiplicative_interaction: bool = False
    action_cross_hidden: int = 0
    action_cross_heads: int = 2

    def __post_init__(self):
        if self.architecture != "entity_set_mlp_v1":
            raise ValueError("EntitySetMLPConfig requires entity_set_mlp_v1")
        if self.feature_schema_version != FEATURE_SCHEMA_VERSION:
            raise ValueError("model feature schema mismatch")
        for name in ("card_hidden", "noble_hidden", "state_hidden", "residual_blocks", "actor_hidden", "critic_hidden"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"entity_set_mlp_v1 {name} must be a positive integer")
        if type(self.action_bottleneck) is not int or self.action_bottleneck < 0:
            raise ValueError("action_bottleneck must be a nonnegative integer (0 disables the branch)")
        for name in ("market_relations", "multiplicative_interaction"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        if type(self.action_cross_hidden) is not int or self.action_cross_hidden < 0:
            raise ValueError("action_cross_hidden must be a nonnegative integer (0 disables attention)")
        if type(self.action_cross_heads) is not int or self.action_cross_heads <= 0:
            raise ValueError("action_cross_heads must be a positive integer")
        if self.action_cross_hidden % self.action_cross_heads:
            raise ValueError("action_cross_hidden must be divisible by action_cross_heads")


class ResidualMLPBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.first = nn.Linear(width, width)
        self.second = nn.Linear(width, width)

    def forward(self, x: Tensor) -> Tensor:
        return F.relu(x + self.second(F.relu(self.first(x))))


class EntitySetMLPState(NamedTuple):
    shared: Tensor
    cards: Tensor
    nobles: Tensor


class _StateHead(nn.Sequential):
    def forward(self, state: Tensor | EntitySetMLPState) -> Tensor:
        return super().forward(state.shared if isinstance(state, EntitySetMLPState) else state)


def _head(width: int, hidden: int) -> nn.Sequential:
    return _StateHead(nn.Linear(width, hidden), nn.ReLU(), nn.Linear(hidden, 1))


class ActionCrossAttention(nn.Module):
    """One action-to-entity read; K/V and the state half of Q are shared."""

    def __init__(self, config: EntitySetMLPConfig, action_dim: int):
        super().__init__()
        h = config.action_cross_hidden
        self.heads = config.action_cross_heads
        self.self_indices = tuple(i for i, name in enumerate(GLOBAL_FEATURE_NAMES)
                                  if name.startswith("self_") and name != "self_is_first_player")
        self.opponent_indices = tuple(i for i, name in enumerate(GLOBAL_FEATURE_NAMES)
                                      if name.startswith("opponent_") and name != "opponent_completed_turns_scaled")
        self.context_indices = tuple(i for i in range(FEATURE_SCHEMA.global_dim)
                                     if i not in self.self_indices + self.opponent_indices)
        self.context_encoder = nn.Linear(len(self.context_indices), h)
        self.player_encoder = nn.Linear(len(self.self_indices), h)
        self.card_alignment = nn.Identity() if config.card_hidden == h else nn.Linear(config.card_hidden, h)
        self.noble_alignment = nn.Linear(config.noble_hidden, h)
        self.type_embedding = nn.Embedding(6, h)
        self.owner_embedding = nn.Embedding(3, h)
        self.tier_embedding = nn.Embedding(4, h)
        self.query_state = nn.Linear(config.state_hidden, h, bias=False)
        self.query_action = nn.Linear(action_dim, h)
        self.query_norm = nn.LayerNorm(h)
        self.memory_norm = nn.LayerNorm(h)
        self.key = nn.Linear(h, h)
        self.value = nn.Linear(h, h)
        self.output = nn.Linear(h, config.actor_hidden)
        # Three role vectors are added to each content embedding. Their summed
        # initial RMS should match the roughly 0.09 card-encoder RMS, rather
        # than dominate it. Scaling preserves the existing RNG draw sequence.
        with torch.no_grad():
            for embedding in (self.type_embedding, self.owner_embedding, self.tier_embedding):
                embedding.weight.mul_(0.05)

    def _role(self, kind: int, owner: int = 0, tier: int = 0) -> Tensor:
        return self.type_embedding.weight[kind] + self.owner_embedding.weight[owner] + self.tier_embedding.weight[tier]

    def _memory(self, batch: TorchDecisionBatch, state: EntitySetMLPState) -> tuple[Tensor, Tensor]:
        g = batch.global_features
        tokens = [
            (self.context_encoder(g[:, self.context_indices]) + self._role(0)).unsqueeze(1),
            (self.player_encoder(g[:, self.self_indices]) + self._role(1, 1)).unsqueeze(1),
            (self.player_encoder(g[:, self.opponent_indices]) + self._role(1, 2)).unsqueeze(1),
        ]
        m, r = FEATURE_SCHEMA.market_capacity, FEATURE_SCHEMA.reserved_capacity
        cards = self.card_alignment(state.cards).split((m, m, m, r, r, 1), dim=1)
        tokens.extend(card + self._role(kind, owner, tier) for card, (kind, owner, tier) in zip(
            cards[:-1], ((2, 0, 1), (2, 0, 2), (2, 0, 3), (3, 1, 0), (3, 2, 0)), strict=True))
        tokens.append(self.noble_alignment(state.nobles) + self._role(4))
        tokens.append(cards[-1] + self._role(5, 1))
        valid = torch.cat((torch.ones((g.shape[0], 3), dtype=torch.bool, device=g.device),
                           batch.market_t1_mask, batch.market_t2_mask, batch.market_t3_mask,
                           batch.self_reserved_mask, batch.opponent_reserved_mask, batch.nobles_mask,
                           batch.pending_card_mask.unsqueeze(1)), dim=1)
        memory = torch.where(valid.unsqueeze(-1), torch.cat(tokens, dim=1), 0)
        return self.memory_norm(memory), valid

    def forward(self, batch: TorchDecisionBatch, state: EntitySetMLPState, raw: Tensor) -> Tensor:
        memory, valid = self._memory(batch, state)
        query = self.query_norm(F.gelu(self.query_state(state.shared).unsqueeze(1) + self.query_action(raw)))

        def split(x: Tensor) -> Tensor:
            return x.reshape(x.shape[0], x.shape[1], self.heads, -1).transpose(1, 2)

        attended = F.scaled_dot_product_attention(split(query), split(self.key(memory)), split(self.value(memory)),
                                                  attn_mask=valid[:, None, None, :], dropout_p=0.0)
        joined = attended.transpose(1, 2).reshape(query.shape)
        return self.output(joined)


class EntitySetMLP(ActorCritic):
    """Set-summary state trunk with an optional action-to-entity attention read.

    The shared state is computed once. Each action also sees its original 82
    attributes (including the target card/noble) and, when enabled, the target
    public card's eight purchase relations. These never pass through pooling.
    An optional nonlinear action branch augments the direct projection; an
    optional per-channel product modulates it using the shared state projection.
    """

    def __init__(self, config: EntitySetMLPConfig | None = None):
        nn.Module.__init__(self)
        self.config = config or EntitySetMLPConfig()
        c = self.config
        self.opponent_indices = tuple(i for i, name in enumerate(GLOBAL_FEATURE_NAMES)
                                      if name.startswith("opponent_") and name != "opponent_completed_turns_scaled")
        self.relation_dim = len(MARKET_RELATION_FEATURE_NAMES) if c.market_relations else 0
        self.card_encoder = nn.Sequential(nn.Linear(FEATURE_SCHEMA.card_dim + self.relation_dim, c.card_hidden),
                                          nn.ReLU(), nn.Linear(c.card_hidden, c.card_hidden), nn.ReLU())
        self.noble_encoder = nn.Sequential(nn.Linear(FEATURE_SCHEMA.noble_dim, c.noble_hidden),
                                           nn.ReLU(), nn.Linear(c.noble_hidden, c.noble_hidden), nn.ReLU())
        self.state_input_dim = (FEATURE_SCHEMA.global_dim + 5 * (2 * c.card_hidden + 1)
                                + 2 * c.noble_hidden + 1 + c.card_hidden)
        self.state_encoder = nn.Sequential(nn.Linear(self.state_input_dim, c.state_hidden), nn.ReLU(),
                                           *(ResidualMLPBlock(c.state_hidden) for _ in range(c.residual_blocks)))
        self.action_representation_dim = FEATURE_SCHEMA.action_dim + self.relation_dim
        self.actor_input_dim = c.state_hidden + self.action_representation_dim
        self.actor = _head(self.actor_input_dim, c.actor_hidden)
        self.critic = _head(c.state_hidden, c.critic_hidden)
        self.action_nonlinear = (
            nn.Sequential(nn.Linear(self.action_representation_dim, c.action_bottleneck), nn.ReLU(),
                          nn.Linear(c.action_bottleneck, c.actor_hidden))
            if c.action_bottleneck else None
        )
        self.interaction_scale = (
            nn.Parameter(torch.ones(c.actor_hidden)) if c.multiplicative_interaction else None
        )
        self.action_cross = ActionCrossAttention(c, self.action_representation_dim) if c.action_cross_hidden else None
        self.action_meta_dim = FEATURE_SCHEMA.action_dim - FEATURE_SCHEMA.card_dim - FEATURE_SCHEMA.noble_dim
        self._card_market_source = CARD_FEATURE_NAMES.index("source_market")

    def _card_inputs(self, cards: Tensor, mask: Tensor, opponent: Tensor) -> Tensor:
        clean = torch.where(mask.unsqueeze(-1), cards, 0)
        if not self.config.market_relations:
            return clean
        public = mask & (clean[..., self._card_market_source] > 0.5)
        return torch.cat((clean, market_purchase_relations(clean, public, opponent)), dim=-1)

    @staticmethod
    def _pool(encoded: Tensor, mask: Tensor) -> Tensor:
        valid = mask.unsqueeze(-1)
        count = mask.sum(1, keepdim=True).to(encoded.dtype)
        mean = torch.where(valid, encoded, 0).sum(1) / count.clamp_min(1)
        maximum = encoded.masked_fill(~valid, -torch.inf).amax(1)
        maximum = torch.where(count > 0, maximum, 0)
        return torch.cat((mean, maximum, count / mask.shape[1]), dim=-1)

    def encode_state(self, batch: TorchDecisionBatch | TorchStateBatch) -> Tensor | EntitySetMLPState:
        opponent = batch.global_features[:, self.opponent_indices]
        fields = ((batch.market_t1, batch.market_t1_mask), (batch.market_t2, batch.market_t2_mask),
                  (batch.market_t3, batch.market_t3_mask), (batch.self_reserved, batch.self_reserved_mask),
                  (batch.opponent_reserved, batch.opponent_reserved_mask))
        inputs = [self._card_inputs(cards, mask, opponent) for cards, mask in fields]
        pending = torch.where(batch.pending_card_mask.unsqueeze(-1), batch.pending_card, 0)
        if self.config.market_relations:
            # The existing relation branch is for market/public reserves only.
            pending = torch.cat((pending, pending.new_zeros((pending.shape[0], self.relation_dim))), dim=-1)
        inputs.append(pending.unsqueeze(1))
        widths = [x.shape[1] for x in inputs]
        card_embeddings = self.card_encoder(torch.cat(inputs, dim=1))
        encoded = card_embeddings.split(widths, dim=1)
        summaries = [self._pool(e, mask) for e, (_, mask) in zip(encoded[:-1], fields, strict=True)]
        noble_raw = torch.where(batch.nobles_mask.unsqueeze(-1), batch.nobles, 0)
        noble_embeddings = self.noble_encoder(noble_raw)
        nobles = self._pool(noble_embeddings, batch.nobles_mask)
        pending = torch.where(batch.pending_card_mask.unsqueeze(-1), encoded[-1].squeeze(1), 0)
        shared = self.state_encoder(torch.cat((batch.global_features, *summaries, nobles, pending), dim=-1))
        if self.action_cross is not None:
            return EntitySetMLPState(shared, card_embeddings, noble_embeddings)
        return shared

    def _action_inputs(self, batch: TorchDecisionBatch) -> Tensor:
        raw = torch.where(batch.candidate_mask.unsqueeze(-1), batch.candidate_features, 0)
        if not self.config.market_relations:
            return raw
        target = raw[..., self.action_meta_dim:self.action_meta_dim + FEATURE_SCHEMA.card_dim]
        public = batch.candidate_mask & (target[..., self._card_market_source] > 0.5)
        opponent = batch.global_features[:, self.opponent_indices]
        return torch.cat((raw, market_purchase_relations(target, public, opponent)), dim=-1)

    def _actor_logits(self, prepared: TorchDecisionBatch, state: Tensor | EntitySetMLPState) -> Tensor:
        # Algebraically the same concatenated linear layer; its state half is
        # shared across candidates in training as well as frozen inference.
        first, _, last = self.actor
        width = self.config.state_hidden
        z = state.shared if isinstance(state, EntitySetMLPState) else state
        shared = F.linear(z, first.weight[:, :width], first.bias)
        raw = self._action_inputs(prepared)
        actions = F.linear(raw, first.weight[:, width:])
        context = self.action_cross(prepared, state, raw) if self.action_cross is not None else None
        return last(self._actor_hidden(shared, actions, raw, context)).squeeze(-1)

    def _actor_hidden(self, shared: Tensor, actions: Tensor, raw: Tensor, context: Tensor | None = None) -> Tensor:
        if self.action_nonlinear is not None:
            actions = actions + self.action_nonlinear(raw)
        shared = shared.unsqueeze(1)
        fused = shared + actions
        if self.interaction_scale is not None:
            fused = fused + self.interaction_scale * shared * actions
        if context is not None:
            fused = fused + context
        return F.relu(fused)

    def _actor_logits_projected(self, prepared: TorchDecisionBatch, state: Tensor | EntitySetMLPState, projection: ActorProjection) -> Tensor:
        z = state.shared if isinstance(state, EntitySetMLPState) else state
        shared = F.linear(z, projection.state_weight, projection.bias)
        raw = self._action_inputs(prepared)
        actions = F.linear(raw, projection.action_weight)
        context = self.action_cross(prepared, state, raw) if self.action_cross is not None else None
        return F.linear(self._actor_hidden(shared, actions, raw, context), projection.output_weight,
                        projection.output_bias).squeeze(-1)


@dataclass(frozen=True, slots=True)
class EntitySetMLPAuxConfig(EntitySetMLPConfig):
    architecture: str = "entity_set_mlp_aux_v1"
    predict_score: bool = True

    def __post_init__(self):
        if self.architecture != "entity_set_mlp_aux_v1":
            raise ValueError("invalid auxiliary set MLP architecture")
        if type(self.predict_score) is not bool:
            raise ValueError("predict_score must be boolean")
        values = asdict(self)
        values.pop("predict_score")
        values["architecture"] = "entity_set_mlp_v1"
        EntitySetMLPConfig(**values)


class EntitySetMLPAux(EntitySetMLP):
    def __init__(self, config: EntitySetMLPAuxConfig | None = None):
        config = config or EntitySetMLPAuxConfig()
        values = asdict(config)
        values.pop("predict_score")
        values["architecture"] = "entity_set_mlp_v1"
        super().__init__(EntitySetMLPConfig(**values))
        self.config = config
        self.score_head = _head(config.state_hidden, config.critic_hidden)
        self.remaining_turns_head = _head(config.state_hidden, config.critic_hidden)
        if not config.predict_score:
            del self.score_head

    def forward_with_auxiliary(self, batch):
        prepared = self._prepare(batch)
        state = self.encode_state(prepared)
        return (self._actor_logits(prepared, state), self.critic(state).squeeze(-1),
                self.score_head(state).squeeze(-1) if self.config.predict_score else None,
                F.softplus(self.remaining_turns_head(state).squeeze(-1)))
