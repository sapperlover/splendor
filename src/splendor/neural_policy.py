"""Policy-safe ActorCritic inference adapters."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass

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


@dataclass(slots=True)
class NeuralInferenceTiming:
    """Shared timing sink for frozen neural opponents in one rollout."""

    seconds: float = 0.0
    decisions: int = 0

    def reset(self) -> None:
        self.seconds = 0.0
        self.decisions = 0


class SamplingActorCriticPolicy:
    """Sample from a frozen policy using one episode-local RNG stream."""

    def __init__(
        self,
        model: ActorCritic,
        *,
        seed: int,
        device: torch.device | str = "cpu",
        timing: NeuralInferenceTiming | None = None,
    ) -> None:
        self.model = model.to(device)
        self.device = torch.device(device)
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(seed)
        self.timing = timing

    def choose(self, request: DecisionRequest) -> Action:
        started = time.perf_counter()
        encoded = encode_decision(request)
        batch = collate_encoded_decisions((encoded,))
        was_training = self.model.training
        self.model.eval()
        with torch.inference_mode():
            distribution, _ = self.model.distribution(batch)
            probabilities = distribution.probs[0].detach().cpu()
            chosen = int(
                torch.multinomial(
                    probabilities,
                    num_samples=1,
                    generator=self._generator,
                ).item()
            )
        self.model.train(was_training)
        if self.timing is not None:
            self.timing.seconds += time.perf_counter() - started
            self.timing.decisions += 1
        return request.legal_actions[chosen]


def frozen_actor_critic_copy(
    model: ActorCritic,
    *,
    device: torch.device | str = "cpu",
) -> ActorCritic:
    """Create a parameter-independent inference snapshot of a learning model."""

    snapshot = copy.deepcopy(model).to(device)
    snapshot.eval()
    snapshot.requires_grad_(False)
    return snapshot
