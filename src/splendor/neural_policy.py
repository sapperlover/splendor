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
    calls: int = 0

    def reset(self) -> None:
        self.seconds = 0.0
        self.decisions = 0
        self.calls = 0

    @property
    def mean_batch_size(self) -> float:
        return self.decisions / self.calls if self.calls else 0.0


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
            self.timing.calls += 1
        return request.legal_actions[chosen]


def choose_sampling_actor_critic_batch(
    policies: tuple[SamplingActorCriticPolicy, ...],
    requests: tuple[DecisionRequest, ...],
) -> tuple[Action, ...]:
    """Choose for several episode-local samplers with one shared-model forward."""

    if not policies or len(policies) != len(requests):
        raise ValueError("one non-empty policy is required per opponent request")
    model = policies[0].model
    device = policies[0].device
    if any(policy.model is not model or policy.device != device for policy in policies):
        raise ValueError("batched opponent policies must share one model and device")

    started = time.perf_counter()
    encoded = tuple(encode_decision(request) for request in requests)
    batch = collate_encoded_decisions(encoded)
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        distribution, _ = model.distribution(batch)
        probabilities = distribution.probs.detach().cpu()
        chosen = tuple(
            int(
                torch.multinomial(
                    probabilities[index],
                    num_samples=1,
                    generator=policy._generator,
                ).item()
            )
            for index, policy in enumerate(policies)
        )
    model.train(was_training)
    elapsed = time.perf_counter() - started
    timing_groups: dict[int, tuple[NeuralInferenceTiming, int]] = {}
    for policy in policies:
        if policy.timing is None:
            continue
        key = id(policy.timing)
        timing, count = timing_groups.get(key, (policy.timing, 0))
        timing_groups[key] = (timing, count + 1)
    for timing, count in timing_groups.values():
        timing.seconds += elapsed
        timing.decisions += count
        timing.calls += 1
    return tuple(
        request.legal_actions[index]
        for request, index in zip(requests, chosen, strict=True)
    )


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
