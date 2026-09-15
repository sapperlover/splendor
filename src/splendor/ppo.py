"""Minimal on-policy rollout, GAE, PPO update, and training checkpoints."""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from torch import Tensor

from .actor_critic import ActorCritic, ActorCriticConfig
from .features import (
    FEATURE_SCHEMA,
    EncodedDecision,
    collate_encoded_decisions,
)
from .training_env import LearningEnvironment


TRAINING_CHECKPOINT_FORMAT_VERSION = 1


@dataclass(frozen=True, slots=True)
class PPOConfig:
    """All first-version PPO hyperparameters in one place."""

    gamma: float = 1.0
    gae_lambda: float = 0.95
    learning_rate: float = 3e-4
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    minibatch_size: int = 512
    target_kl: float | None = 0.02
    rollout_steps: int = 8192
    advantage_epsilon: float = 1e-8

    def __post_init__(self) -> None:
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be in [0, 1]")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 <= self.clip_coef < 1.0:
            raise ValueError("clip_coef must be in [0, 1)")
        if self.value_coef < 0.0 or self.entropy_coef < 0.0:
            raise ValueError("loss coefficients must be non-negative")
        if self.max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive")
        if self.ppo_epochs <= 0 or self.minibatch_size <= 0 or self.rollout_steps <= 0:
            raise ValueError("PPO sizes and epoch count must be positive")
        if self.target_kl is not None and self.target_kl <= 0.0:
            raise ValueError("target_kl must be positive or None")
        if self.advantage_epsilon <= 0.0:
            raise ValueError("advantage_epsilon must be positive")


@dataclass(frozen=True, slots=True)
class PolicySample:
    chosen_candidate_index: int
    old_logprob: float
    old_value: float


class PolicySampler:
    """Sample legal candidate indices from one ActorCritic policy version.

    The dedicated CPU generator keeps policy sampling independent of engine and
    opponent randomness.  Only the small probability vector crosses back to CPU.
    """

    def __init__(
        self,
        model: ActorCritic,
        *,
        seed: int,
        device: torch.device | str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.device = torch.device(device)
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(seed)

    def sample(self, encoded: EncodedDecision) -> PolicySample:
        batch = collate_encoded_decisions((encoded,))
        was_training = self.model.training
        self.model.eval()
        with torch.inference_mode():
            distribution, values = self.model.distribution(batch)
            probabilities = distribution.probs[0].detach().cpu()
            chosen = int(
                torch.multinomial(
                    probabilities,
                    num_samples=1,
                    generator=self._generator,
                ).item()
            )
            chosen_tensor = torch.tensor(chosen, device=self.device)
            logprob = float(distribution.log_prob(chosen_tensor.unsqueeze(0))[0].item())
            value = float(values[0].item())
        self.model.train(was_training)
        return PolicySample(chosen, logprob, value)

    def value(self, encoded: EncodedDecision) -> float:
        batch = collate_encoded_decisions((encoded,))
        was_training = self.model.training
        self.model.eval()
        with torch.inference_mode():
            _, values = self.model(batch)
            value = float(values[0].item())
        self.model.train(was_training)
        return value

    def rng_state(self) -> Tensor:
        return self._generator.get_state().clone()

    def set_rng_state(self, state: Tensor) -> None:
        self._generator.set_state(state.cpu())


@dataclass(frozen=True, slots=True)
class RolloutSample:
    """One same-learning-player timestep stored entirely as CPU data."""

    encoded: EncodedDecision
    chosen_candidate_index: int
    old_logprob: float
    old_value: float
    reward: float
    terminated: bool
    truncated: bool
    episode_id: str
    learning_player: int
    bootstrap_value: float


@dataclass(frozen=True, slots=True)
class RolloutBatch:
    samples: tuple[RolloutSample, ...]

    def __len__(self) -> int:
        return len(self.samples)


@dataclass(frozen=True, slots=True)
class PreparedRollout:
    samples: tuple[RolloutSample, ...]
    advantages: np.ndarray
    returns: np.ndarray

    def __len__(self) -> int:
        return len(self.samples)


EnvironmentFactory = Callable[[int], LearningEnvironment]


class RolloutCollector:
    """Collect an on-policy batch while retaining a mid-game environment."""

    def __init__(
        self,
        sampler: PolicySampler,
        environment_factory: EnvironmentFactory,
        config: PPOConfig | None = None,
    ) -> None:
        self.sampler = sampler
        self.environment_factory = environment_factory
        self.config = config or PPOConfig()
        self._environment: LearningEnvironment | None = None
        self._next_episode_index = 0

    @property
    def active_environment(self) -> LearningEnvironment | None:
        return self._environment

    @property
    def next_episode_index(self) -> int:
        return self._next_episode_index

    def collect(self, steps: int | None = None) -> RolloutBatch:
        target = steps if steps is not None else self.config.rollout_steps
        if target <= 0:
            raise ValueError("rollout steps must be positive")

        samples: list[RolloutSample] = []
        while len(samples) < target:
            if self._environment is None:
                self._environment = self.environment_factory(self._next_episode_index)
                self._next_episode_index += 1
            environment = self._environment
            if environment.done:
                self._environment = None
                continue

            decision = environment.current_decision()
            policy_sample = self.sampler.sample(decision.encoded)
            transition = environment.step_learning_action(
                policy_sample.chosen_candidate_index,
                decision_id=decision.decision_id,
            )
            if transition.terminated:
                bootstrap_value = 0.0
            else:
                if transition.next_decision is None:
                    raise AssertionError("non-terminal transition has no real successor")
                bootstrap_value = self.sampler.value(transition.next_decision.encoded)

            samples.append(
                RolloutSample(
                    encoded=decision.encoded,
                    chosen_candidate_index=policy_sample.chosen_candidate_index,
                    old_logprob=policy_sample.old_logprob,
                    old_value=policy_sample.old_value,
                    reward=transition.reward,
                    terminated=transition.terminated,
                    truncated=transition.truncated,
                    episode_id=transition.episode_id,
                    learning_player=transition.learning_player,
                    bootstrap_value=bootstrap_value,
                )
            )
            if transition.terminated or transition.truncated:
                self._environment = None

        return RolloutBatch(tuple(samples))


def compute_gae(
    rollout: RolloutBatch | Sequence[RolloutSample],
    *,
    gamma: float,
    gae_lambda: float,
) -> PreparedRollout:
    """Compute GAE without crossing terminal, truncation, episode, or batch edges."""

    samples = rollout.samples if isinstance(rollout, RolloutBatch) else tuple(rollout)
    if not samples:
        raise ValueError("cannot compute GAE for an empty rollout")

    advantages = np.zeros(len(samples), dtype=np.float64)
    next_advantage = 0.0
    for index in range(len(samples) - 1, -1, -1):
        sample = samples[index]
        if sample.terminated and sample.truncated:
            raise ValueError("a rollout sample cannot be both terminated and truncated")

        bootstrap = 0.0 if sample.terminated else sample.bootstrap_value
        delta = sample.reward + gamma * bootstrap - sample.old_value
        continues_in_batch = (
            not sample.terminated
            and not sample.truncated
            and index + 1 < len(samples)
            and samples[index + 1].episode_id == sample.episode_id
            and samples[index + 1].learning_player == sample.learning_player
        )
        advantages[index] = delta + (
            gamma * gae_lambda * next_advantage if continues_in_batch else 0.0
        )
        next_advantage = advantages[index]

    old_values = np.asarray([sample.old_value for sample in samples], dtype=np.float64)
    returns = advantages + old_values
    return PreparedRollout(
        samples=tuple(samples),
        advantages=advantages.astype(np.float32),
        returns=returns.astype(np.float32),
    )


def normalize_advantages(
    advantages: np.ndarray,
    *,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Population-variance normalization, stable for a one-sample batch."""

    values = np.asarray(advantages, dtype=np.float64)
    normalized = (values - values.mean()) / math.sqrt(values.var() + epsilon)
    return normalized.astype(np.float32)


def clipped_policy_loss(
    new_logprob: Tensor,
    old_logprob: Tensor,
    advantages: Tensor,
    clip_coef: float,
) -> Tensor:
    """Negative PPO clipped surrogate objective."""

    ratio = torch.exp(new_logprob - old_logprob)
    unclipped = advantages * ratio
    clipped = advantages * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
    return -torch.minimum(unclipped, clipped).mean()


def explained_variance(returns: Tensor, predictions: Tensor) -> float:
    """Return the critic diagnostic without producing NaN for constant targets."""

    target_variance = torch.var(returns, unbiased=False)
    if float(target_variance.item()) <= 1e-12:
        return 0.0
    residual_variance = torch.var(returns - predictions, unbiased=False)
    return float((1.0 - residual_variance / target_variance).item())


@dataclass(frozen=True, slots=True)
class PPOUpdateStats:
    policy_loss: float
    value_loss: float
    entropy: float
    total_loss: float
    approx_kl: float
    clip_fraction: float
    grad_norm: float
    explained_variance: float
    epochs_completed: int
    minibatches_processed: int
    valid_sample_count: int
    entropy_active_sample_count: int
    gradient_clipped_minibatches: int
    early_stopped: bool


class PPOTrainer:
    """One-process first-version PPO optimizer over ragged legal candidates."""

    def __init__(
        self,
        model: ActorCritic,
        config: PPOConfig | None = None,
        *,
        device: torch.device | str = "cpu",
        shuffle_seed: int = 0,
    ) -> None:
        self.config = config or PPOConfig()
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate
        )
        self.global_learning_decision_step = 0
        self.update_count = 0
        self._shuffle_generator = torch.Generator(device="cpu")
        self._shuffle_generator.manual_seed(shuffle_seed)

    def update(self, rollout: PreparedRollout) -> PPOUpdateStats:
        if not rollout.samples:
            raise ValueError("cannot update from an empty rollout")
        sample_count = len(rollout.samples)
        normalized = normalize_advantages(
            rollout.advantages, epsilon=self.config.advantage_epsilon
        )
        active_count = sum(
            int(np.count_nonzero(sample.encoded.candidate_mask) > 1)
            for sample in rollout.samples
        )

        metric_sums = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "total_loss": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
        }
        processed_weight = 0
        entropy_weight = 0
        grad_norm_sum = 0.0
        clipped_minibatches = 0
        minibatches = 0
        epochs_completed = 0
        early_stopped = False

        self.model.train()
        for epoch in range(self.config.ppo_epochs):
            shuffled = torch.randperm(
                sample_count, generator=self._shuffle_generator
            ).tolist()
            for start in range(0, sample_count, self.config.minibatch_size):
                indices = shuffled[start : start + self.config.minibatch_size]
                encoded = collate_encoded_decisions(
                    tuple(rollout.samples[index].encoded for index in indices)
                )
                distribution, values = self.model.distribution(encoded)
                actions = torch.as_tensor(
                    [rollout.samples[index].chosen_candidate_index for index in indices],
                    dtype=torch.long,
                    device=self.device,
                )
                old_logprob = torch.as_tensor(
                    [rollout.samples[index].old_logprob for index in indices],
                    dtype=torch.float32,
                    device=self.device,
                )
                advantages = torch.as_tensor(
                    normalized[indices], dtype=torch.float32, device=self.device
                )
                returns = torch.as_tensor(
                    rollout.returns[indices], dtype=torch.float32, device=self.device
                )
                new_logprob = distribution.log_prob(actions)
                logratio = new_logprob - old_logprob
                ratio = torch.exp(logratio)

                policy_loss = clipped_policy_loss(
                    new_logprob,
                    old_logprob,
                    advantages,
                    self.config.clip_coef,
                )
                value_loss = torch.mean((values - returns) ** 2)
                entropy_values = distribution.entropy()
                valid_candidates = torch.as_tensor(
                    encoded.candidate_mask.sum(axis=1),
                    dtype=torch.long,
                    device=self.device,
                )
                entropy_active = valid_candidates > 1
                if torch.any(entropy_active):
                    mean_entropy = entropy_values[entropy_active].mean()
                    minibatch_entropy_count = int(entropy_active.sum().item())
                else:
                    mean_entropy = entropy_values.sum() * 0.0
                    minibatch_entropy_count = 0
                total_loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * mean_entropy
                )
                approx_kl = torch.mean((ratio - 1.0) - logratio)
                clip_fraction = torch.mean(
                    (torch.abs(ratio - 1.0) > self.config.clip_coef).float()
                )

                for name, value in (
                    ("new_logprob", new_logprob),
                    ("value", values),
                    ("entropy", entropy_values),
                    ("loss", total_loss),
                    ("approx_kl", approx_kl),
                ):
                    if not torch.all(torch.isfinite(value)):
                        raise FloatingPointError(f"non-finite PPO {name}")

                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                for parameter in self.model.parameters():
                    if parameter.grad is not None and not torch.all(
                        torch.isfinite(parameter.grad)
                    ):
                        raise FloatingPointError("non-finite PPO gradient")
                grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                grad_norm = float(grad_norm_tensor.item())
                if not math.isfinite(grad_norm):
                    raise FloatingPointError("non-finite PPO gradient norm")
                if grad_norm > self.config.max_grad_norm:
                    clipped_minibatches += 1
                self.optimizer.step()

                weight = len(indices)
                metric_sums["policy_loss"] += float(policy_loss.item()) * weight
                metric_sums["value_loss"] += float(value_loss.item()) * weight
                metric_sums["total_loss"] += float(total_loss.item()) * weight
                metric_sums["approx_kl"] += float(approx_kl.item()) * weight
                metric_sums["clip_fraction"] += float(clip_fraction.item()) * weight
                metric_sums["entropy"] += (
                    float(mean_entropy.item()) * minibatch_entropy_count
                )
                processed_weight += weight
                entropy_weight += minibatch_entropy_count
                grad_norm_sum += grad_norm
                minibatches += 1

                if (
                    self.config.target_kl is not None
                    and float(approx_kl.item()) > self.config.target_kl
                ):
                    early_stopped = True
                    break
            epochs_completed = epoch + 1
            if early_stopped:
                break

        predictions = self._predict_values(rollout.samples)
        returns_tensor = torch.as_tensor(rollout.returns, dtype=torch.float32)
        variance = explained_variance(returns_tensor, predictions)
        self.global_learning_decision_step += sample_count
        self.update_count += 1

        return PPOUpdateStats(
            policy_loss=metric_sums["policy_loss"] / processed_weight,
            value_loss=metric_sums["value_loss"] / processed_weight,
            entropy=(
                metric_sums["entropy"] / entropy_weight if entropy_weight else 0.0
            ),
            total_loss=metric_sums["total_loss"] / processed_weight,
            approx_kl=metric_sums["approx_kl"] / processed_weight,
            clip_fraction=metric_sums["clip_fraction"] / processed_weight,
            grad_norm=grad_norm_sum / minibatches,
            explained_variance=variance,
            epochs_completed=epochs_completed,
            minibatches_processed=minibatches,
            valid_sample_count=sample_count,
            entropy_active_sample_count=active_count,
            gradient_clipped_minibatches=clipped_minibatches,
            early_stopped=early_stopped,
        )

    def _predict_values(self, samples: Sequence[RolloutSample]) -> Tensor:
        predictions: list[Tensor] = []
        was_training = self.model.training
        self.model.eval()
        with torch.inference_mode():
            for start in range(0, len(samples), self.config.minibatch_size):
                batch_samples = samples[start : start + self.config.minibatch_size]
                batch = collate_encoded_decisions(
                    tuple(sample.encoded for sample in batch_samples)
                )
                _, values = self.model(batch)
                predictions.append(values.detach().cpu())
        self.model.train(was_training)
        return torch.cat(predictions)

    def save_checkpoint(
        self,
        path: str | PathLike[str],
        *,
        sampler: PolicySampler,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        numpy_state = np.random.get_state()
        payload = {
            "format_version": TRAINING_CHECKPOINT_FORMAT_VERSION,
            "feature_schema_version": FEATURE_SCHEMA.version,
            "model_config": asdict(self.model.config),
            "ppo_config": asdict(self.config),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_learning_decision_step": self.global_learning_decision_step,
            "update_count": self.update_count,
            "python_rng_state": random.getstate(),
            "numpy_rng_state": {
                "bit_generator": numpy_state[0],
                "keys": torch.from_numpy(numpy_state[1].copy()),
                "position": numpy_state[2],
                "has_gauss": numpy_state[3],
                "cached_gaussian": numpy_state[4],
            },
            "torch_cpu_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_states": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
            ),
            "shuffle_rng_state": self._shuffle_generator.get_state(),
            "sampler_rng_state": sampler.rng_state(),
            "metadata": {} if metadata is None else metadata,
        }
        temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(checkpoint_path)

    @classmethod
    def load_checkpoint(
        cls,
        path: str | PathLike[str],
        *,
        device: torch.device | str = "cpu",
    ) -> tuple[PPOTrainer, PolicySampler, dict[str, Any]]:
        payload = torch.load(path, map_location=device, weights_only=True)
        if payload.get("format_version") != TRAINING_CHECKPOINT_FORMAT_VERSION:
            raise ValueError("unsupported training checkpoint format version")
        if payload.get("feature_schema_version") != FEATURE_SCHEMA.version:
            raise ValueError("training checkpoint feature schema version mismatch")

        model = ActorCritic(ActorCriticConfig(**payload["model_config"]))
        model.load_state_dict(payload["model_state_dict"], strict=True)
        trainer = cls(
            model,
            PPOConfig(**payload["ppo_config"]),
            device=device,
        )
        trainer.optimizer.load_state_dict(payload["optimizer_state_dict"])
        trainer.global_learning_decision_step = int(
            payload["global_learning_decision_step"]
        )
        trainer.update_count = int(payload["update_count"])
        trainer._shuffle_generator.set_state(payload["shuffle_rng_state"].cpu())

        random.setstate(payload["python_rng_state"])
        numpy_state = payload["numpy_rng_state"]
        np.random.set_state(
            (
                numpy_state["bit_generator"],
                numpy_state["keys"].cpu().numpy().astype(np.uint32, copy=False),
                int(numpy_state["position"]),
                int(numpy_state["has_gauss"]),
                float(numpy_state["cached_gaussian"]),
            )
        )
        torch.set_rng_state(payload["torch_cpu_rng_state"].cpu())
        cuda_states = payload["torch_cuda_rng_states"]
        if cuda_states and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([state.cpu() for state in cuda_states])

        sampler = PolicySampler(trainer.model, seed=0, device=device)
        sampler.set_rng_state(payload["sampler_rng_state"])
        return trainer, sampler, dict(payload["metadata"])
