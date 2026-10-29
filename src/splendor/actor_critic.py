"""Small Deep-Sets actor--critic over the public numeric feature contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from os import PathLike
from typing import Any

import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from .features import (
    CARD_FEATURE_NAMES,
    FEATURE_SCHEMA,
    FEATURE_SCHEMA_VERSION,
    NOBLE_FEATURE_NAMES,
    EncodedDecisionBatch,
    EncodedStateBatch,
)


MODEL_WEIGHTS_FORMAT_VERSION = 1


@dataclass(frozen=True, slots=True)
class ActorCriticConfig:
    """The deliberately small first-version network widths."""

    feature_schema_version: str = FEATURE_SCHEMA_VERSION
    card_hidden: int = 64
    noble_hidden: int = 32
    state_hidden: int = 256
    actor_hidden: int = 128
    critic_hidden: int = 128


@dataclass(frozen=True, slots=True)
class TorchDecisionBatch:
    """Tensor-only counterpart of :class:`EncodedDecisionBatch`.

    This type intentionally has no candidate action objects or execution handles.
    """

    feature_schema_version: str
    global_features: Tensor
    market_t1: Tensor
    market_t1_mask: Tensor
    market_t2: Tensor
    market_t2_mask: Tensor
    market_t3: Tensor
    market_t3_mask: Tensor
    self_reserved: Tensor
    self_reserved_mask: Tensor
    opponent_reserved: Tensor
    opponent_reserved_mask: Tensor
    nobles: Tensor
    nobles_mask: Tensor
    pending_card: Tensor
    pending_card_mask: Tensor
    candidate_features: Tensor
    candidate_mask: Tensor

    def to(self, device: torch.device | str) -> TorchDecisionBatch:
        return TorchDecisionBatch(
            feature_schema_version=self.feature_schema_version,
            **{
                name: getattr(self, name).to(device)
                for name in self.__dataclass_fields__
                if name != "feature_schema_version"
            },
        )


@dataclass(frozen=True, slots=True)
class ActorProjection:
    """Frozen actor first layer split into state and action projections.

    ``h_i = ReLU(W @ concat(state, action_i) + b)`` becomes
    ``h_i = ReLU((W_state @ state + b) + W_action @ action_i)`` so the state
    half is computed once per decision instead of once per candidate.  The bias
    and the ReLU apply to the combined sum exactly as before.  Copies are
    contiguous so no per-forward slicing is needed.
    """

    state_weight: Tensor
    action_weight: Tensor
    bias: Tensor
    output_weight: Tensor
    output_bias: Tensor

    def to(self, device: torch.device | str) -> ActorProjection:
        return ActorProjection(
            state_weight=self.state_weight.to(device),
            action_weight=self.action_weight.to(device),
            bias=self.bias.to(device),
            output_weight=self.output_weight.to(device),
            output_bias=self.output_bias.to(device),
        )


@dataclass(frozen=True, slots=True)
class TorchStateBatch:
    """Tensor-only counterpart of :class:`EncodedStateBatch` (no candidates)."""

    feature_schema_version: str
    global_features: Tensor
    market_t1: Tensor
    market_t1_mask: Tensor
    market_t2: Tensor
    market_t2_mask: Tensor
    market_t3: Tensor
    market_t3_mask: Tensor
    self_reserved: Tensor
    self_reserved_mask: Tensor
    opponent_reserved: Tensor
    opponent_reserved_mask: Tensor
    nobles: Tensor
    nobles_mask: Tensor
    pending_card: Tensor
    pending_card_mask: Tensor

    def to(self, device: torch.device | str) -> TorchStateBatch:
        return TorchStateBatch(
            feature_schema_version=self.feature_schema_version,
            **{
                name: getattr(self, name).to(device)
                for name in self.__dataclass_fields__
                if name != "feature_schema_version"
            },
        )


def tensorize_state(
    batch: EncodedStateBatch,
    *,
    device: torch.device | str | None = None,
) -> TorchStateBatch:
    """Convert the frozen NumPy state contract to FP32/bool tensors."""

    tensor_device = torch.device("cpu") if device is None else torch.device(device)

    def tensor(name: str) -> Tensor:
        return torch.as_tensor(getattr(batch, name), device=tensor_device)

    return TorchStateBatch(
        feature_schema_version=batch.feature_schema_version,
        global_features=tensor("global_features"),
        market_t1=tensor("market_t1"),
        market_t1_mask=tensor("market_t1_mask"),
        market_t2=tensor("market_t2"),
        market_t2_mask=tensor("market_t2_mask"),
        market_t3=tensor("market_t3"),
        market_t3_mask=tensor("market_t3_mask"),
        self_reserved=tensor("self_reserved"),
        self_reserved_mask=tensor("self_reserved_mask"),
        opponent_reserved=tensor("opponent_reserved"),
        opponent_reserved_mask=tensor("opponent_reserved_mask"),
        nobles=tensor("nobles"),
        nobles_mask=tensor("nobles_mask"),
        pending_card=tensor("pending_card"),
        pending_card_mask=tensor("pending_card_mask"),
    )


def tensorize_batch(
    batch: EncodedDecisionBatch,
    *,
    device: torch.device | str | None = None,
) -> TorchDecisionBatch:
    """Convert the frozen NumPy contract to FP32/bool tensors."""

    tensor_device = torch.device("cpu") if device is None else torch.device(device)

    def tensor(name: str) -> Tensor:
        return torch.as_tensor(getattr(batch, name), device=tensor_device)

    return TorchDecisionBatch(
        feature_schema_version=batch.feature_schema_version,
        global_features=tensor("global_features"),
        market_t1=tensor("market_t1"),
        market_t1_mask=tensor("market_t1_mask"),
        market_t2=tensor("market_t2"),
        market_t2_mask=tensor("market_t2_mask"),
        market_t3=tensor("market_t3"),
        market_t3_mask=tensor("market_t3_mask"),
        self_reserved=tensor("self_reserved"),
        self_reserved_mask=tensor("self_reserved_mask"),
        opponent_reserved=tensor("opponent_reserved"),
        opponent_reserved_mask=tensor("opponent_reserved_mask"),
        nobles=tensor("nobles"),
        nobles_mask=tensor("nobles_mask"),
        pending_card=tensor("pending_card"),
        pending_card_mask=tensor("pending_card_mask"),
        candidate_features=tensor("candidate_features"),
        candidate_mask=tensor("candidate_mask"),
    )


class CardEncoder(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(FEATURE_SCHEMA.card_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features)


class NobleEncoder(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(FEATURE_SCHEMA.noble_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features)


class ActorCritic(nn.Module):
    """Permutation-invariant state encoder with per-candidate actor scores."""

    def __init__(self, config: ActorCriticConfig | None = None) -> None:
        super().__init__()
        self.config = config or ActorCriticConfig()
        if self.config.feature_schema_version != FEATURE_SCHEMA.version:
            raise ValueError(
                "model feature schema version does not match the runtime schema: "
                f"{self.config.feature_schema_version!r} != {FEATURE_SCHEMA.version!r}"
            )

        self.card_encoder = CardEncoder(self.config.card_hidden)
        self.noble_encoder = NobleEncoder(self.config.noble_hidden)

        self.card_set_summary_dim = self.config.card_hidden + 1
        self.noble_set_summary_dim = self.config.noble_hidden + 1
        self.state_input_dim = (
            FEATURE_SCHEMA.global_dim
            + 5 * self.card_set_summary_dim
            + self.noble_set_summary_dim
            + self.config.card_hidden
        )
        self.action_meta_dim = (
            FEATURE_SCHEMA.action_dim
            - FEATURE_SCHEMA.card_dim
            - FEATURE_SCHEMA.noble_dim
        )
        self.action_representation_dim = (
            self.action_meta_dim
            + self.config.card_hidden
            + self.config.noble_hidden
        )
        self.actor_input_dim = (
            self.config.state_hidden + self.action_representation_dim
        )

        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_input_dim, self.config.state_hidden),
            nn.ReLU(),
            nn.Linear(self.config.state_hidden, self.config.state_hidden),
            nn.ReLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(self.actor_input_dim, self.config.actor_hidden),
            nn.ReLU(),
            nn.Linear(self.config.actor_hidden, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(self.config.state_hidden, self.config.critic_hidden),
            nn.ReLU(),
            nn.Linear(self.config.critic_hidden, 1),
        )

        self._card_valid_index = CARD_FEATURE_NAMES.index("valid")
        self._noble_valid_index = NOBLE_FEATURE_NAMES.index("valid")

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _prepare(
        self, batch: EncodedDecisionBatch | TorchDecisionBatch
    ) -> TorchDecisionBatch:
        prepared = (
            tensorize_batch(batch, device=self.device)
            if isinstance(batch, EncodedDecisionBatch)
            else batch.to(self.device)
        )
        if prepared.feature_schema_version != self.config.feature_schema_version:
            raise ValueError(
                "batch feature schema version does not match model configuration"
            )
        if prepared.candidate_mask.ndim != 2 or not torch.all(
            prepared.candidate_mask.any(dim=1)
        ):
            raise ValueError("every sample must contain at least one legal candidate")
        return prepared

    def _pool_cards(self, features: Tensor, mask: Tensor, capacity: int) -> Tensor:
        expanded_mask = mask.unsqueeze(-1)
        masked_features = torch.where(expanded_mask, features, torch.zeros_like(features))
        encoded = self.card_encoder(masked_features)
        encoded = torch.where(expanded_mask, encoded, torch.zeros_like(encoded))
        pooled = encoded.sum(dim=1) / float(capacity)
        count = mask.sum(dim=1, dtype=features.dtype).unsqueeze(-1) / float(capacity)
        return torch.cat((pooled, count), dim=-1)

    def _pool_nobles(self, features: Tensor, mask: Tensor) -> Tensor:
        expanded_mask = mask.unsqueeze(-1)
        masked_features = torch.where(expanded_mask, features, torch.zeros_like(features))
        encoded = self.noble_encoder(masked_features)
        encoded = torch.where(expanded_mask, encoded, torch.zeros_like(encoded))
        pooled = encoded.sum(dim=1) / float(FEATURE_SCHEMA.noble_capacity)
        count = (
            mask.sum(dim=1, dtype=features.dtype).unsqueeze(-1)
            / float(FEATURE_SCHEMA.noble_capacity)
        )
        return torch.cat((pooled, count), dim=-1)

    def encode_state(self, batch: TorchDecisionBatch | TorchStateBatch) -> Tensor:
        card_fields = (
            (batch.market_t1, batch.market_t1_mask, FEATURE_SCHEMA.market_capacity),
            (batch.market_t2, batch.market_t2_mask, FEATURE_SCHEMA.market_capacity),
            (batch.market_t3, batch.market_t3_mask, FEATURE_SCHEMA.market_capacity),
            (batch.self_reserved, batch.self_reserved_mask, FEATURE_SCHEMA.reserved_capacity),
            (batch.opponent_reserved, batch.opponent_reserved_mask, FEATURE_SCHEMA.reserved_capacity),
        )
        raw_parts = [torch.where(mask.unsqueeze(-1), values, 0) for values, mask, _ in card_fields]
        raw_parts.append(torch.where(batch.pending_card_mask.unsqueeze(-1), batch.pending_card, 0).unsqueeze(1))
        widths = [part.shape[1] for part in raw_parts]
        encoded_parts = self.card_encoder(torch.cat(raw_parts, dim=1)).split(widths, dim=1)
        card_sets = tuple(
            torch.cat((
                torch.where(mask.unsqueeze(-1), encoded, 0).sum(dim=1) / float(capacity),
                mask.sum(dim=1, dtype=values.dtype).unsqueeze(-1) / float(capacity),
            ), dim=-1)
            for (values, mask, capacity), encoded in zip(card_fields, encoded_parts[:-1])
        )
        pending = torch.where(batch.pending_card_mask.unsqueeze(-1), encoded_parts[-1].squeeze(1), 0)
        state_input = torch.cat(
            (
                batch.global_features,
                *card_sets,
                self._pool_nobles(batch.nobles, batch.nobles_mask),
                pending,
            ),
            dim=-1,
        )
        return self.state_encoder(state_input)

    def encode_actions(self, candidate_features: Tensor) -> Tensor:
        card_start = self.action_meta_dim
        noble_start = card_start + FEATURE_SCHEMA.card_dim
        meta = candidate_features[..., :card_start]
        card_raw = candidate_features[..., card_start:noble_start]
        noble_raw = candidate_features[..., noble_start:]

        card_present = card_raw[..., self._card_valid_index : self._card_valid_index + 1] > 0
        noble_present = (
            noble_raw[..., self._noble_valid_index : self._noble_valid_index + 1] > 0
        )
        card_raw = torch.where(card_present, card_raw, torch.zeros_like(card_raw))
        noble_raw = torch.where(noble_present, noble_raw, torch.zeros_like(noble_raw))
        card_embedding = self.card_encoder(card_raw)
        noble_embedding = self.noble_encoder(noble_raw)
        card_embedding = torch.where(
            card_present, card_embedding, torch.zeros_like(card_embedding)
        )
        noble_embedding = torch.where(
            noble_present, noble_embedding, torch.zeros_like(noble_embedding)
        )
        return torch.cat((meta, card_embedding, noble_embedding), dim=-1)

    def build_actor_projection(self) -> ActorProjection:
        """Contiguous state/action views of the frozen actor's first layer."""

        first, _activation, last = self.actor
        if not isinstance(first, nn.Linear) or not isinstance(last, nn.Linear):
            raise TypeError("actor projection requires Linear, ReLU, Linear")
        weight = first.weight.detach()
        if weight.shape[1] != self.actor_input_dim:
            raise ValueError("actor first layer does not match the configured input width")
        state_hidden = self.config.state_hidden
        return ActorProjection(
            state_weight=weight[:, :state_hidden].contiguous(),
            action_weight=weight[:, state_hidden:].contiguous(),
            bias=first.bias.detach().contiguous(),
            output_weight=last.weight.detach().contiguous(),
            output_bias=last.bias.detach().contiguous(),
        )

    def _actor_logits(self, prepared: TorchDecisionBatch, state: Tensor) -> Tensor:
        candidate_raw = torch.where(
            prepared.candidate_mask.unsqueeze(-1),
            prepared.candidate_features,
            torch.zeros_like(prepared.candidate_features),
        )
        actions = self.encode_actions(candidate_raw)
        expanded_state = state.unsqueeze(1).expand(-1, actions.shape[1], -1)
        return self.actor(torch.cat((expanded_state, actions), dim=-1)).squeeze(-1)

    def _actor_logits_projected(
        self, prepared: TorchDecisionBatch, state: Tensor, projection: ActorProjection
    ) -> Tensor:
        candidate_raw = torch.where(
            prepared.candidate_mask.unsqueeze(-1),
            prepared.candidate_features,
            torch.zeros_like(prepared.candidate_features),
        )
        actions = self.encode_actions(candidate_raw)
        hidden = state @ projection.state_weight.transpose(0, 1) + projection.bias
        hidden = hidden.unsqueeze(1) + actions @ projection.action_weight.transpose(0, 1)
        hidden = torch.relu(hidden)
        return (hidden @ projection.output_weight.transpose(0, 1) + projection.output_bias).squeeze(-1)

    def _forward_prepared(
        self, prepared: TorchDecisionBatch, *, projection: ActorProjection | None = None
    ) -> tuple[Tensor, Tensor]:
        """Trusted tensor core used by frozen search after encoding checks."""
        state = self.encode_state(prepared)
        logits = (
            self._actor_logits(prepared, state) if projection is None
            else self._actor_logits_projected(prepared, state, projection)
        )
        return logits, self.critic(state).squeeze(-1)

    def forward(
        self, batch: EncodedDecisionBatch | TorchDecisionBatch
    ) -> tuple[Tensor, Tensor]:
        return self._forward_prepared(self._prepare(batch))

    def _policy_logits_prepared(
        self, prepared: TorchDecisionBatch, *, projection: ActorProjection | None = None
    ) -> Tensor:
        state = self.encode_state(prepared)
        return (
            self._actor_logits(prepared, state) if projection is None
            else self._actor_logits_projected(prepared, state, projection)
        )

    def policy_logits(
        self, batch: EncodedDecisionBatch | TorchDecisionBatch
    ) -> Tensor:
        """Frozen-search actor path without evaluating the critic head."""
        return self._policy_logits_prepared(self._prepare(batch))

    def value_only(
        self, batch: EncodedDecisionBatch | TorchDecisionBatch
    ) -> Tensor:
        """Evaluate the critic without encoding candidates or running the actor."""

        prepared = self._prepare(batch)
        return self.critic(self.encode_state(prepared)).squeeze(-1)

    def value_from_state(
        self, batch: EncodedStateBatch | TorchStateBatch
    ) -> Tensor:
        """Evaluate the critic from a state-only encoding.

        Equivalent to :meth:`value_only` for the same state, but the caller
        never builds candidate features.
        """

        prepared = (
            tensorize_state(batch, device=self.device)
            if isinstance(batch, EncodedStateBatch)
            else batch.to(self.device)
        )
        if prepared.feature_schema_version != self.config.feature_schema_version:
            raise ValueError(
                "batch feature schema version does not match model configuration"
            )
        return self.critic(self.encode_state(prepared)).squeeze(-1)

    @staticmethod
    def masked_logits(logits: Tensor, candidate_mask: Tensor) -> Tensor:
        if logits.shape != candidate_mask.shape:
            raise ValueError("logits and candidate mask shapes must match")
        if candidate_mask.ndim != 2 or not torch.all(candidate_mask.any(dim=1)):
            raise ValueError("every sample must contain at least one legal candidate")
        return logits.masked_fill(~candidate_mask, -torch.inf)

    def distribution(
        self, batch: EncodedDecisionBatch | TorchDecisionBatch
    ) -> tuple[Categorical, Tensor]:
        prepared = self._prepare(batch)
        logits, values = self(prepared)
        return Categorical(logits=self.masked_logits(logits, prepared.candidate_mask)), values

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def save_weights(self, path: str | PathLike[str]) -> None:
        torch.save(
            {
                "format_version": MODEL_WEIGHTS_FORMAT_VERSION,
                "feature_schema_version": self.config.feature_schema_version,
                "config": asdict(self.config),
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load_weights(
        cls,
        path: str | PathLike[str],
        *,
        map_location: Any = None,
    ) -> ActorCritic:
        payload = torch.load(path, map_location=map_location, weights_only=True)
        if payload.get("format_version") != MODEL_WEIGHTS_FORMAT_VERSION:
            raise ValueError("unsupported model weights format version")
        if payload.get("feature_schema_version") != FEATURE_SCHEMA.version:
            raise ValueError("model weights feature schema version mismatch")
        model = build_actor_critic(payload["config"])
        model.load_state_dict(payload["state_dict"], strict=True)
        return model


def model_config_from_dict(config: dict[str, Any] | ActorCriticConfig):
    """Parse identity without initializing a model or consuming its RNG."""
    values = asdict(config) if isinstance(config, ActorCriticConfig) else dict(config)
    architecture = values.pop("architecture", "deep_sets_v1")
    if architecture == "deep_sets_v1":
        return ActorCriticConfig(**values)
    if architecture in ("entity_set_mlp_v1", "entity_set_mlp_aux_v1"):
        from .entity_set_mlp import EntitySetMLPConfig, EntitySetMLPAuxConfig
        return (EntitySetMLPConfig if architecture == "entity_set_mlp_v1" else EntitySetMLPAuxConfig)(**values)
    if architecture == "entity_attention_v1":
        from .entity_attention import EntityAttentionConfig
        return EntityAttentionConfig(**values)
    if architecture == "entity_attention_aux_v1":
        from .entity_attention import EntityAttentionAuxConfig
        return EntityAttentionAuxConfig(**values)
    if architecture == "entity_attention_progress_v1":
        from .progress_auxiliary import EntityAttentionProgressConfig
        return EntityAttentionProgressConfig(**values)
    if architecture == "entity_attention_continuation_v1":
        from .continuation_auxiliary import EntityAttentionContinuationConfig
        return EntityAttentionContinuationConfig(**values)
    if architecture == "entity_attention_policy_value_v1":
        from .entity_attention import EntityAttentionPolicyValueConfig
        return EntityAttentionPolicyValueConfig(**values)
    raise ValueError(f"unknown model architecture: {architecture!r}")


def build_actor_critic(config: dict[str, Any] | ActorCriticConfig) -> ActorCritic:
    """Dispatch explicitly; absent architecture identifies the legacy Deep-Sets format."""
    parsed = model_config_from_dict(config)
    if isinstance(parsed, ActorCriticConfig):
        return ActorCritic(parsed)
    from .continuation_auxiliary import EntityAttentionContinuation, EntityAttentionContinuationConfig
    if isinstance(parsed, EntityAttentionContinuationConfig):
        return EntityAttentionContinuation(parsed)
    from .progress_auxiliary import EntityAttentionProgress, EntityAttentionProgressConfig
    if isinstance(parsed, EntityAttentionProgressConfig):
        return EntityAttentionProgress(parsed)
    from .entity_set_mlp import EntitySetMLP, EntitySetMLPAux, EntitySetMLPConfig, EntitySetMLPAuxConfig
    if isinstance(parsed, EntitySetMLPAuxConfig):
        return EntitySetMLPAux(parsed)
    if isinstance(parsed, EntitySetMLPConfig):
        return EntitySetMLP(parsed)
    from .entity_attention import (EntityAttention, EntityAttentionAux, EntityAttentionAuxConfig,
                                   EntityAttentionPolicyValue, EntityAttentionPolicyValueConfig)
    if isinstance(parsed, EntityAttentionAuxConfig):
        return EntityAttentionAux(parsed)
    if isinstance(parsed, EntityAttentionPolicyValueConfig):
        return EntityAttentionPolicyValue(parsed)
    return EntityAttention(parsed)
