from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

from schafkopf_ai.training.behavior_cloning import (
    BehaviorCloningDataset,
    CardPlayPolicyModel,
    architecture_for_observation_version,
    create_card_play_policy,
    load_behavior_cloning_arrays,
    mask_illegal_logits,
    split_indices_by_game,
)
from schafkopf_ai.training.card_play_encoding import ACTION_COUNT
from schafkopf_ai.training.observation_encoding import (
    infer_observation_version,
    resolve_checkpoint_observation_version,
)


@dataclass(frozen=True, slots=True)
class EpochMetrics:
    loss: float
    accuracy: float
    nontrivial_accuracy: float
    examples: int
    nontrivial_examples: int


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def run_epoch(
    *,
    model: CardPlayPolicyModel,
    loader: DataLoader[tuple[Tensor, Tensor, Tensor]],
    device: torch.device,
    optimizer: AdamW | None,
) -> EpochMetrics:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    nontrivial_correct = 0
    nontrivial_examples = 0

    for features, legal_masks, targets in loader:
        features = features.to(device)
        legal_masks = legal_masks.to(device)
        targets = targets.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(features)
            masked_logits = mask_illegal_logits(logits, legal_masks)
            loss = F.cross_entropy(masked_logits, targets)

            if training:
                loss.backward()
                optimizer.step()

        batch_size = targets.shape[0]
        predictions = masked_logits.argmax(dim=1)
        correct = predictions.eq(targets)
        nontrivial = legal_masks.sum(dim=1) > 1

        total_loss += float(loss.item()) * batch_size
        total_correct += int(correct.sum().item())
        total_examples += batch_size
        nontrivial_correct += int((correct & nontrivial).sum().item())
        nontrivial_examples += int(nontrivial.sum().item())

    if total_examples == 0:
        raise RuntimeError("Training or validation loader produced no examples.")

    return EpochMetrics(
        loss=total_loss / total_examples,
        accuracy=total_correct / total_examples,
        nontrivial_accuracy=(
            nontrivial_correct / nontrivial_examples if nontrivial_examples else 0.0
        ),
        examples=total_examples,
        nontrivial_examples=nontrivial_examples,
    )


def save_checkpoint(
    *,
    path: Path,
    model: CardPlayPolicyModel,
    epoch: int,
    validation_metrics: EpochMetrics,
    dataset_path: Path,
    seed: int,
    initialized_from: Path | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "architecture": model.architecture,
            "input_size": model.input_size,
            "hidden_sizes": model.hidden_sizes,
            "gru_hidden_size": getattr(model, "gru_hidden_size", None),
            "action_count": model.action_count,
            "observation_version": model.observation_version,
            "epoch": epoch,
            "validation_loss": validation_metrics.loss,
            "validation_accuracy": validation_metrics.accuracy,
            "validation_nontrivial_accuracy": validation_metrics.nontrivial_accuracy,
            "dataset_path": str(dataset_path),
            "seed": seed,
            "initialized_from": (
                str(initialized_from) if initialized_from is not None else None
            ),
        },
        path,
    )


def load_initial_weights(
    *,
    model: CardPlayPolicyModel,
    checkpoint_path: Path,
    device: torch.device,
) -> None:
    """Warm-start a policy from a compatible behavioral-cloning checkpoint."""
    checkpoint: Any = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    if not isinstance(checkpoint, dict):
        raise TypeError("Initial checkpoint must be a dictionary.")

    input_size = int(checkpoint.get("input_size", model.input_size))
    action_count = int(checkpoint.get("action_count", ACTION_COUNT))
    raw_hidden_sizes = checkpoint.get("hidden_sizes", model.hidden_sizes)

    if not isinstance(raw_hidden_sizes, (tuple, list)) or len(raw_hidden_sizes) != 2:
        raise ValueError("Initial checkpoint hidden_sizes must contain two values.")

    hidden_sizes = (int(raw_hidden_sizes[0]), int(raw_hidden_sizes[1]))

    checkpoint_version = resolve_checkpoint_observation_version(
        input_size=input_size,
        explicit_version=checkpoint.get("observation_version"),
    )
    checkpoint_architecture = checkpoint.get(
        "architecture",
        architecture_for_observation_version(checkpoint_version),
    )
    if not isinstance(checkpoint_architecture, str):
        raise TypeError("Initial checkpoint architecture must be a string.")

    if input_size != model.input_size:
        raise ValueError(
            "Initial checkpoint observation size does not match the model: "
            f"{input_size} != {model.input_size}."
        )
    if checkpoint_version != model.observation_version:
        raise ValueError(
            "Initial checkpoint observation version does not match the model: "
            f"{checkpoint_version} != {model.observation_version}."
        )
    if checkpoint_architecture != model.architecture:
        raise ValueError(
            "Initial checkpoint architecture does not match the model: "
            f"{checkpoint_architecture} != {model.architecture}."
        )
    if action_count != model.action_count:
        raise ValueError(
            "Initial checkpoint action count does not match the model: "
            f"{action_count} != {model.action_count}."
        )
    if hidden_sizes != model.hidden_sizes:
        raise ValueError(
            "Initial checkpoint hidden sizes do not match the model: "
            f"{hidden_sizes} != {model.hidden_sizes}."
        )

    if model.architecture == "gru_v1":
        checkpoint_gru_hidden = int(
            checkpoint.get("gru_hidden_size", model.gru_hidden_size)
        )
        if checkpoint_gru_hidden != model.gru_hidden_size:
            raise ValueError(
                "Initial checkpoint GRU hidden size does not match the model: "
                f"{checkpoint_gru_hidden} != {model.gru_hidden_size}."
            )

    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise TypeError("Initial checkpoint does not contain model_state_dict.")

    model.load_state_dict(state_dict)


def train(
    *,
    dataset_path: Path,
    output: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    validation_fraction: float,
    seed: int,
    device_name: str,
    hidden_sizes: tuple[int, int],
    gru_hidden_size: int,
    init_checkpoint: Path | None = None,
) -> None:
    if epochs <= 0:
        raise ValueError("epochs must be greater than zero.")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = resolve_device(device_name)
    arrays = load_behavior_cloning_arrays(dataset_path)
    dataset = BehaviorCloningDataset(arrays)
    train_indices, validation_indices = split_indices_by_game(
        arrays.game_ids,
        validation_fraction=validation_fraction,
        seed=seed,
    )

    train_dataset = Subset(dataset, train_indices)
    validation_dataset = Subset(dataset, validation_indices)

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    input_size = int(arrays.features.shape[1])
    observation_version = infer_observation_version(input_size)
    model = create_card_play_policy(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        gru_hidden_size=gru_hidden_size,
        action_count=ACTION_COUNT,
    ).to(device)

    if init_checkpoint is not None:
        load_initial_weights(
            model=model,
            checkpoint_path=init_checkpoint,
            device=device,
        )

    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    best_nontrivial_accuracy = -1.0
    start = time.perf_counter()

    print("Behavioral-cloning / DAgger training")
    print("=" * 88)
    print(f"Dataset:                   {dataset_path}")
    print(f"Examples:                  {arrays.example_count:,}")
    print(f"Training examples:         {len(train_indices):,}")
    print(f"Validation examples:       {len(validation_indices):,}")
    print(f"Device:                    {device}")
    print(f"Observation version:       {observation_version}")
    print(f"Observation features:      {input_size}")
    print(f"Architecture:              {model.architecture}")
    print(f"Hidden sizes:              {hidden_sizes}")
    if model.architecture == "gru_v1":
        print(f"GRU hidden size:           {model.gru_hidden_size}")
    print(f"Batch size:                {batch_size}")
    print(f"Learning rate:             {learning_rate:g}")
    print(
        "Initialized from:           "
        f"{init_checkpoint if init_checkpoint is not None else 'random weights'}"
    )
    print()

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
        )
        validation_metrics = run_epoch(
            model=model,
            loader=validation_loader,
            device=device,
            optimizer=None,
        )

        elapsed = time.perf_counter() - start
        print(
            f"Epoch {epoch:>2}/{epochs} "
            f"| train loss {train_metrics.loss:.4f} "
            f"acc {train_metrics.accuracy:>7.2%} "
            f"nontrivial {train_metrics.nontrivial_accuracy:>7.2%} "
            f"| val loss {validation_metrics.loss:.4f} "
            f"acc {validation_metrics.accuracy:>7.2%} "
            f"nontrivial {validation_metrics.nontrivial_accuracy:>7.2%} "
            f"| {elapsed:>7.1f}s"
        )

        if validation_metrics.nontrivial_accuracy > best_nontrivial_accuracy:
            best_nontrivial_accuracy = validation_metrics.nontrivial_accuracy
            save_checkpoint(
                path=output,
                model=model,
                epoch=epoch,
                validation_metrics=validation_metrics,
                dataset_path=dataset_path,
                seed=seed,
                initialized_from=init_checkpoint,
            )
            print(f"  saved new best checkpoint -> {output}")

    print("\nTraining complete")
    print("-" * 88)
    print(f"Best validation non-trivial accuracy: {best_nontrivial_accuracy:.2%}")
    print(f"Checkpoint:                            {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a card-play policy on heuristic demonstrations or an "
            "aggregated DAgger dataset. V3 datasets use the hybrid GRU policy."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/behavior_cloning/heuristic_card_play.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/behavior_cloning/card_play_mlp.pt"),
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Optional compatible checkpoint used to warm-start training.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch device such as cpu/cuda, or auto (default).",
    )
    parser.add_argument("--hidden-1", type=int, default=512)
    parser.add_argument("--hidden-2", type=int, default=256)
    parser.add_argument(
        "--gru-hidden-size",
        type=int,
        default=128,
        help="Hidden size of the V3 public-history GRU (default: 128).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(
        dataset_path=args.dataset,
        output=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        device_name=args.device,
        hidden_sizes=(args.hidden_1, args.hidden_2),
        gru_hidden_size=args.gru_hidden_size,
        init_checkpoint=args.init_checkpoint,
    )


if __name__ == "__main__":
    main()
