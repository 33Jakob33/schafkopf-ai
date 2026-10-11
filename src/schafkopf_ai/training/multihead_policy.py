from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn
from torch.utils.data import Dataset

from .behavior_cloning import BehaviorCloningArrays
from .card_play_encoding import ACTION_COUNT
from .card_play_encoding_v2 import OBSERVATION_V2_FEATURE_SIZE
from .card_play_roles import CardPlayRole


class RoleBehaviorCloningDataset(Dataset[tuple[Tensor, Tensor, Tensor, Tensor]]):
    """Behavior-cloning dataset that additionally exposes each example's role."""

    def __init__(self, arrays: BehaviorCloningArrays) -> None:
        role_ids = arrays.role_ids
        if role_ids is None:
            raise ValueError("Role-aware training requires role_ids in the dataset.")
        self.arrays = arrays
        self.role_ids = role_ids

    def __len__(self) -> int:
        return self.arrays.example_count

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        features = torch.from_numpy(self.arrays.features[index]).to(torch.float32)
        legal_mask = torch.from_numpy(self.arrays.legal_masks[index]).to(torch.bool)
        target = torch.tensor(int(self.arrays.targets[index]), dtype=torch.long)
        role = torch.tensor(int(self.role_ids[index]), dtype=torch.long)
        return features, legal_mask, target, role


class MultiHeadCardPlayPolicyNetwork(nn.Module):
    """
    Shared V2 representation with one policy head per strategic role.

    All examples update the shared backbone. Only the head selected by role_ids
    contributes logits for a sample, preserving role-specific tactics without
    giving up cross-contract representation learning.
    """

    architecture = "multihead_mlp_v1"

    def __init__(
        self,
        *,
        input_size: int = OBSERVATION_V2_FEATURE_SIZE,
        shared_hidden_sizes: tuple[int, int] = (512, 512),
        head_hidden_size: int = 256,
        action_count: int = ACTION_COUNT,
    ) -> None:
        super().__init__()

        if input_size != OBSERVATION_V2_FEATURE_SIZE:
            raise ValueError(
                "Multi-head policy currently requires V2 observations: "
                f"{input_size} != {OBSERVATION_V2_FEATURE_SIZE}."
            )
        if any(size <= 0 for size in shared_hidden_sizes):
            raise ValueError("Shared hidden sizes must be positive.")
        if head_hidden_size <= 0:
            raise ValueError("Head hidden size must be positive.")

        self.input_size = input_size
        self.shared_hidden_sizes = shared_hidden_sizes
        self.head_hidden_size = head_hidden_size
        self.action_count = action_count
        self.role_count = len(CardPlayRole)
        self.observation_version = "v2"

        first_hidden, second_hidden = shared_hidden_sizes
        self.backbone = nn.Sequential(
            nn.Linear(input_size, first_hidden),
            nn.ReLU(),
            nn.Linear(first_hidden, second_hidden),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList(
            nn.Sequential(
                nn.Linear(second_hidden, head_hidden_size),
                nn.ReLU(),
                nn.Linear(head_hidden_size, action_count),
            )
            for _ in CardPlayRole
        )

    def forward(self, features: Tensor, role_ids: Tensor) -> Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_size:
            raise ValueError(
                "Multi-head policy expects a two-dimensional V2 feature batch "
                f"with width {self.input_size}; got {tuple(features.shape)}."
            )
        if role_ids.ndim != 1 or role_ids.shape[0] != features.shape[0]:
            raise ValueError(
                "role_ids must be a one-dimensional tensor matching batch size."
            )
        if bool(((role_ids < 0) | (role_ids >= self.role_count)).any()):
            raise ValueError("role_ids contains an unsupported strategic role.")

        shared = self.backbone(features)
        all_logits = torch.stack(
            [head(shared) for head in self.heads],
            dim=1,
        )
        batch_indices = torch.arange(features.shape[0], device=features.device)
        selected = all_logits[batch_indices, role_ids]
        return cast(Tensor, selected)
