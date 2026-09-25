from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions import Categorical
from torch.optim import Optimizer

from .behavior_cloning import mask_illegal_logits
from .card_play_encoding import ACTION_COUNT, OBSERVATION_FEATURE_SIZE


@dataclass(frozen=True, slots=True)
class PPOStep:
    """One card-play decision collected from the behavior policy."""

    features: Tensor
    legal_mask: Tensor
    action: int
    log_probability: float
    value: float


@dataclass(frozen=True, slots=True)
class PPOBatch:
    features: Tensor
    legal_masks: Tensor
    actions: Tensor
    old_log_probabilities: Tensor
    old_values: Tensor
    returns: Tensor


@dataclass(frozen=True, slots=True)
class PPOUpdateMetrics:
    policy_loss: float
    value_loss: float
    entropy: float
    approximate_kl: float
    clip_fraction: float
    samples: int


class ActorCriticCardPlayNetwork(nn.Module):
    """Shared card-play encoder with separate policy and value heads."""

    def __init__(
        self,
        *,
        input_size: int = OBSERVATION_FEATURE_SIZE,
        hidden_sizes: tuple[int, int] = (512, 256),
        action_count: int = ACTION_COUNT,
    ) -> None:
        super().__init__()

        first_hidden, second_hidden = hidden_sizes
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.action_count = action_count

        self.shared = nn.Sequential(
            nn.Linear(input_size, first_hidden),
            nn.ReLU(),
            nn.Linear(first_hidden, second_hidden),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(second_hidden, action_count)
        self.value_head = nn.Linear(second_hidden, 1)

        # The policy is normally initialized from BC/DAgger. A zero value head
        # starts PPO with a neutral estimate before settlement returns are seen.
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.shared(features)
        logits = self.policy_head(hidden)
        values = self.value_head(hidden).squeeze(-1)
        return logits, values


class PPORolloutBuffer:
    """On-policy rollout storage for complete focal-player trajectories."""

    def __init__(self) -> None:
        self._steps: list[PPOStep] = []
        self._returns: list[float] = []

    @property
    def sample_count(self) -> int:
        return len(self._steps)

    def add_episode(self, steps: list[PPOStep], terminal_return: float) -> None:
        if not steps:
            raise ValueError("A PPO episode must contain at least one focal-player step.")

        self._steps.extend(steps)
        self._returns.extend([float(terminal_return)] * len(steps))

    def as_batch(self, device: torch.device) -> PPOBatch:
        if not self._steps:
            raise ValueError("Cannot create a PPO batch from an empty rollout buffer.")

        return PPOBatch(
            features=torch.stack([step.features for step in self._steps]).to(device),
            legal_masks=torch.stack([step.legal_mask for step in self._steps]).to(device),
            actions=torch.tensor(
                [step.action for step in self._steps],
                dtype=torch.long,
                device=device,
            ),
            old_log_probabilities=torch.tensor(
                [step.log_probability for step in self._steps],
                dtype=torch.float32,
                device=device,
            ),
            old_values=torch.tensor(
                [step.value for step in self._steps],
                dtype=torch.float32,
                device=device,
            ),
            returns=torch.tensor(
                self._returns,
                dtype=torch.float32,
                device=device,
            ),
        )


def _validate_checkpoint_shape(
    *,
    checkpoint: dict[str, Any],
    model: ActorCriticCardPlayNetwork,
    source_name: str,
) -> None:
    input_size = int(checkpoint.get("input_size", OBSERVATION_FEATURE_SIZE))
    action_count = int(checkpoint.get("action_count", ACTION_COUNT))
    raw_hidden_sizes = checkpoint.get("hidden_sizes", model.hidden_sizes)

    if not isinstance(raw_hidden_sizes, (tuple, list)) or len(raw_hidden_sizes) != 2:
        raise ValueError(f"{source_name} hidden_sizes must contain two values.")

    hidden_sizes = (int(raw_hidden_sizes[0]), int(raw_hidden_sizes[1]))

    if input_size != model.input_size:
        raise ValueError(
            f"{source_name} observation size does not match PPO model: "
            f"{input_size} != {model.input_size}."
        )
    if action_count != model.action_count:
        raise ValueError(
            f"{source_name} action count does not match PPO model: "
            f"{action_count} != {model.action_count}."
        )
    if hidden_sizes != model.hidden_sizes:
        raise ValueError(
            f"{source_name} hidden sizes do not match PPO model: "
            f"{hidden_sizes} != {model.hidden_sizes}."
        )


def initialize_from_behavior_checkpoint(
    *,
    model: ActorCriticCardPlayNetwork,
    checkpoint_path: str | Path,
    device: torch.device,
) -> None:
    """Copy the BC/DAgger MLP trunk and policy head into an actor-critic model."""
    checkpoint: Any = torch.load(
        Path(checkpoint_path),
        map_location=device,
        weights_only=True,
    )
    if not isinstance(checkpoint, dict):
        raise TypeError("Behavior checkpoint must be a dictionary.")

    _validate_checkpoint_shape(
        checkpoint=checkpoint,
        model=model,
        source_name="Behavior checkpoint",
    )

    source = checkpoint.get("model_state_dict")
    if not isinstance(source, dict):
        raise TypeError("Behavior checkpoint does not contain model_state_dict.")

    destination = model.state_dict()
    mapping = {
        "shared.0.weight": "network.0.weight",
        "shared.0.bias": "network.0.bias",
        "shared.2.weight": "network.2.weight",
        "shared.2.bias": "network.2.bias",
        "policy_head.weight": "network.4.weight",
        "policy_head.bias": "network.4.bias",
    }

    for destination_key, source_key in mapping.items():
        source_tensor = source.get(source_key)
        if not isinstance(source_tensor, Tensor):
            raise KeyError(f"Behavior checkpoint is missing tensor {source_key!r}.")
        if destination[destination_key].shape != source_tensor.shape:
            raise ValueError(
                f"Checkpoint tensor shape mismatch for {source_key}: "
                f"{tuple(source_tensor.shape)} != "
                f"{tuple(destination[destination_key].shape)}."
            )
        destination[destination_key] = source_tensor.to(
            device=destination[destination_key].device,
            dtype=destination[destination_key].dtype,
        )

    model.load_state_dict(destination)


def initialize_from_checkpoint(
    *,
    model: ActorCriticCardPlayNetwork,
    checkpoint_path: str | Path,
    device: torch.device,
) -> str:
    """
    Initialize PPO from either a BC/DAgger checkpoint or an existing PPO model.

    Returns a short source label for logging.
    """
    checkpoint: Any = torch.load(
        Path(checkpoint_path),
        map_location=device,
        weights_only=True,
    )
    if not isinstance(checkpoint, dict):
        raise TypeError("Initial checkpoint must be a dictionary.")

    if checkpoint.get("format") == "ppo_actor_critic_v1":
        _validate_checkpoint_shape(
            checkpoint=checkpoint,
            model=model,
            source_name="PPO checkpoint",
        )
        state_dict = checkpoint.get("model_state_dict")
        if not isinstance(state_dict, dict):
            raise TypeError("PPO checkpoint does not contain model_state_dict.")
        model.load_state_dict(state_dict)
        return "PPO"

    initialize_from_behavior_checkpoint(
        model=model,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    return "BC/DAgger"


def save_ppo_checkpoint(
    *,
    path: str | Path,
    model: ActorCriticCardPlayNetwork,
    iteration: int,
    mean_payment: float,
    seed: int,
    initialized_from: str | Path | None,
    validation_delta: float | None = None,
) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "format": "ppo_actor_critic_v1",
            "model_state_dict": model.state_dict(),
            "input_size": model.input_size,
            "hidden_sizes": model.hidden_sizes,
            "action_count": model.action_count,
            "iteration": iteration,
            "mean_payment": mean_payment,
            "validation_delta": validation_delta,
            "seed": seed,
            "initialized_from": (
                str(initialized_from) if initialized_from is not None else None
            ),
        },
        checkpoint_path,
    )


def load_ppo_model(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> ActorCriticCardPlayNetwork:
    resolved_device = torch.device(device)
    checkpoint: Any = torch.load(
        Path(checkpoint_path),
        map_location=resolved_device,
        weights_only=True,
    )
    if not isinstance(checkpoint, dict):
        raise TypeError("PPO checkpoint must be a dictionary.")
    if checkpoint.get("format") != "ppo_actor_critic_v1":
        raise ValueError("Checkpoint is not a supported PPO actor-critic checkpoint.")

    input_size = int(checkpoint.get("input_size", OBSERVATION_FEATURE_SIZE))
    action_count = int(checkpoint.get("action_count", ACTION_COUNT))
    raw_hidden_sizes = checkpoint.get("hidden_sizes", (512, 256))
    if not isinstance(raw_hidden_sizes, (tuple, list)) or len(raw_hidden_sizes) != 2:
        raise ValueError("PPO checkpoint hidden_sizes must contain two values.")

    model = ActorCriticCardPlayNetwork(
        input_size=input_size,
        hidden_sizes=(int(raw_hidden_sizes[0]), int(raw_hidden_sizes[1])),
        action_count=action_count,
    ).to(resolved_device)

    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise TypeError("PPO checkpoint does not contain model_state_dict.")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def ppo_update(
    *,
    model: ActorCriticCardPlayNetwork,
    optimizer: Optimizer,
    rollout: PPORolloutBuffer,
    device: torch.device,
    epochs: int,
    minibatch_size: int,
    clip_epsilon: float,
    value_coefficient: float,
    entropy_coefficient: float,
    max_grad_norm: float,
    generator: torch.Generator,
) -> PPOUpdateMetrics:
    if epochs <= 0:
        raise ValueError("PPO epochs must be greater than zero.")
    if minibatch_size <= 0:
        raise ValueError("PPO minibatch size must be greater than zero.")
    if not 0.0 < clip_epsilon < 1.0:
        raise ValueError("PPO clip epsilon must be between zero and one.")

    batch = rollout.as_batch(device)
    sample_count = batch.actions.shape[0]

    advantages = batch.returns - batch.old_values
    if sample_count > 1:
        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8
        )

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    total_kl = 0.0
    total_clip_fraction = 0.0
    total_seen = 0

    model.train()

    for _ in range(epochs):
        permutation = torch.randperm(sample_count, generator=generator)

        for start in range(0, sample_count, minibatch_size):
            cpu_indices = permutation[start : start + minibatch_size]
            indices = cpu_indices.to(device)

            logits, values = model(batch.features[indices])
            masked_logits = mask_illegal_logits(logits, batch.legal_masks[indices])
            distribution = Categorical(logits=masked_logits)
            new_log_probabilities = distribution.log_prob(batch.actions[indices])
            entropy = distribution.entropy().mean()

            old_log_probabilities = batch.old_log_probabilities[indices]
            ratios = torch.exp(new_log_probabilities - old_log_probabilities)
            minibatch_advantages = advantages[indices]

            unclipped = ratios * minibatch_advantages
            clipped = torch.clamp(
                ratios,
                1.0 - clip_epsilon,
                1.0 + clip_epsilon,
            ) * minibatch_advantages
            policy_loss = -torch.minimum(unclipped, clipped).mean()

            value_loss = F.mse_loss(values, batch.returns[indices])
            loss = (
                policy_loss
                + value_coefficient * value_loss
                - entropy_coefficient * entropy
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approximate_kl = (
                    old_log_probabilities - new_log_probabilities
                ).mean()
                clip_fraction = (
                    (ratios - 1.0).abs() > clip_epsilon
                ).to(torch.float32).mean()

            count = int(indices.shape[0])
            total_policy_loss += float(policy_loss.item()) * count
            total_value_loss += float(value_loss.item()) * count
            total_entropy += float(entropy.item()) * count
            total_kl += float(approximate_kl.item()) * count
            total_clip_fraction += float(clip_fraction.item()) * count
            total_seen += count

    model.eval()

    if total_seen == 0:
        raise RuntimeError("PPO update processed no samples.")

    return PPOUpdateMetrics(
        policy_loss=total_policy_loss / total_seen,
        value_loss=total_value_loss / total_seen,
        entropy=total_entropy / total_seen,
        approximate_kl=total_kl / total_seen,
        clip_fraction=total_clip_fraction / total_seen,
        samples=sample_count,
    )
