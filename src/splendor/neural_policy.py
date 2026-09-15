"""Policy-safe ActorCritic inference adapters."""

from __future__ import annotations

import torch

from .actor_critic import ActorCritic
from .actions import Action
from .contracts import DecisionRequest
from .features import collate_encoded_decisions, encode_decision


class GreedyActorCriticPolicy:
    """Deterministic evaluation policy over engine-provided legal candidates."""

    def __init__(
        self,
        model: ActorCritic,
        *,
        device: torch.device | str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.device = torch.device(device)

    def choose(self, request: DecisionRequest) -> Action:
        encoded = encode_decision(request)
        batch = collate_encoded_decisions((encoded,))
        was_training = self.model.training
        self.model.eval()
        with torch.inference_mode():
            logits, _ = self.model(batch)
            chosen = int(torch.argmax(logits[0]).item())
        self.model.train(was_training)
        return request.legal_actions[chosen]
