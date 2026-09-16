from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset

from .card_play_encoding import ACTION_COUNT, OBSERVATION_FEATURE_SIZE


@dataclass(frozen=True, slots=True)
class BehaviorCloningArrays:
    """Dense arrays stored in a compressed behavioral-cloning dataset."""

    features: np.ndarray
    legal_masks: np.ndarray
    targets: np.ndarray
    game_ids: np.ndarray
    legal_counts: np.ndarray

    @property
    def example_count(self) -> int:
        return int(self.targets.shape[0])


class BehaviorCloningDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """PyTorch view over pre-encoded card-play demonstrations."""

    def __init__(self, arrays: BehaviorCloningArrays) -> None:
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays.example_count

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        features = torch.from_numpy(self.arrays.features[index]).to(torch.float32)
        legal_mask = torch.from_numpy(self.arrays.legal_masks[index]).to(torch.bool)
        target = torch.tensor(int(self.arrays.targets[index]), dtype=torch.long)
        return features, legal_mask, target


class CardPlayPolicyNetwork(nn.Module):
    """Simple MLP policy over the 32 physical card actions."""

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

        self.network = nn.Sequential(
            nn.Linear(input_size, first_hidden),
            nn.ReLU(),
            nn.Linear(first_hidden, second_hidden),
            nn.ReLU(),
            nn.Linear(second_hidden, action_count),
        )

    def forward(self, features: Tensor) -> Tensor:
        return cast(Tensor, self.network(features))


def mask_illegal_logits(logits: Tensor, legal_mask: Tensor) -> Tensor:
    """Replace logits of illegal card actions by the minimum finite value."""
    if logits.shape != legal_mask.shape:
        raise ValueError(
            "Logits and legal-action mask must have identical shapes: "
            f"{tuple(logits.shape)} != {tuple(legal_mask.shape)}."
        )

    legal_mask = legal_mask.to(dtype=torch.bool, device=logits.device)

    if logits.ndim == 1:
        if not bool(legal_mask.any()):
            raise ValueError("At least one legal action is required.")
    elif logits.ndim == 2:
        if bool((~legal_mask.any(dim=1)).any()):
            raise ValueError("Every batch item requires at least one legal action.")
    else:
        raise ValueError("Card-play logits must be one- or two-dimensional.")

    minimum = torch.finfo(logits.dtype).min
    return logits.masked_fill(~legal_mask, minimum)


def load_behavior_cloning_arrays(path: str | Path) -> BehaviorCloningArrays:
    """Load and validate a compressed demonstration dataset."""
    dataset_path = Path(path)

    with np.load(dataset_path, allow_pickle=False) as data:
        arrays = BehaviorCloningArrays(
            features=data["features"],
            legal_masks=data["legal_masks"],
            targets=data["targets"],
            game_ids=data["game_ids"],
            legal_counts=data["legal_counts"],
        )

    _validate_arrays(arrays)
    return arrays


def save_behavior_cloning_arrays(
    path: str | Path,
    arrays: BehaviorCloningArrays,
) -> None:
    """Validate and save a compressed demonstration dataset."""
    _validate_arrays(arrays)

    dataset_path = Path(path)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        dataset_path,
        features=arrays.features,
        legal_masks=arrays.legal_masks,
        targets=arrays.targets,
        game_ids=arrays.game_ids,
        legal_counts=arrays.legal_counts,
    )


def split_indices_by_game(
    game_ids: np.ndarray,
    *,
    validation_fraction: float = 0.10,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """Split examples by complete games to prevent within-game leakage."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one.")

    unique_games = sorted({int(game_id) for game_id in game_ids.tolist()})
    if len(unique_games) < 2:
        raise ValueError(
            "At least two games are required for a train/validation split."
        )

    rng = random.Random(seed)
    rng.shuffle(unique_games)

    validation_games = max(1, round(len(unique_games) * validation_fraction))
    validation_games = min(validation_games, len(unique_games) - 1)

    validation_ids = frozenset(unique_games[:validation_games])

    train_indices: list[int] = []
    validation_indices: list[int] = []

    for index, game_id in enumerate(game_ids.tolist()):
        if int(game_id) in validation_ids:
            validation_indices.append(index)
        else:
            train_indices.append(index)

    return train_indices, validation_indices


def _validate_arrays(arrays: BehaviorCloningArrays) -> None:
    example_count = arrays.example_count

    if arrays.features.shape != (example_count, OBSERVATION_FEATURE_SIZE):
        raise ValueError(
            "Unexpected feature shape: "
            f"{arrays.features.shape}; expected "
            f"({example_count}, {OBSERVATION_FEATURE_SIZE})."
        )

    if arrays.legal_masks.shape != (example_count, ACTION_COUNT):
        raise ValueError(
            "Unexpected legal-mask shape: "
            f"{arrays.legal_masks.shape}; expected ({example_count}, {ACTION_COUNT})."
        )

    for name, values in (
        ("game_ids", arrays.game_ids),
        ("legal_counts", arrays.legal_counts),
    ):
        if values.shape != (example_count,):
            raise ValueError(f"Unexpected {name} shape: {values.shape}.")

    if arrays.targets.shape != (example_count,):
        raise ValueError(f"Unexpected target shape: {arrays.targets.shape}.")

    if example_count == 0:
        raise ValueError("Behavioral-cloning dataset cannot be empty.")

    if np.any(arrays.legal_counts < 1):
        raise ValueError("Every example must contain at least one legal card.")

    if np.any(arrays.targets >= ACTION_COUNT):
        raise ValueError("Dataset contains an invalid target action index.")

    rows = np.arange(example_count)
    if not np.all(arrays.legal_masks[rows, arrays.targets.astype(np.int64)]):
        raise ValueError("Every target action must be legal in its example.")
