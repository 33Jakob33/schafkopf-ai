from __future__ import annotations

import argparse
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from schafkopf_ai.training.behavior_cloning import (
    filter_training_indices,
    load_behavior_cloning_arrays,
    mask_illegal_logits,
    split_indices_by_game,
)
from schafkopf_ai.training.card_play_roles import CardPlayRole
from schafkopf_ai.training.multihead_policy import (
    MultiHeadCardPlayPolicyNetwork,
    RoleBehaviorCloningDataset,
)


@dataclass(frozen=True, slots=True)
class MultiHeadMetrics:
    loss: float
    accuracy: float
    macro_role_accuracy: float
    role_accuracy: dict[CardPlayRole, float]


def _resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _role_balance_weights(
    role_ids: np.ndarray,
    indices: list[int],
    *,
    alpha: float,
) -> np.ndarray:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("role_balance_alpha must be between 0 and 1.")
    selected = role_ids[np.asarray(indices, dtype=np.int64)].astype(np.int64)
    counts = np.bincount(selected, minlength=len(CardPlayRole))

    weights = np.empty(len(indices), dtype=np.float64)
    for position, role_id in enumerate(selected.tolist()):
        count = int(counts[role_id])
        if count <= 0:
            raise RuntimeError("Observed role has zero frequency.")
        weights[position] = count ** (-alpha)

    return weights / float(weights.mean())


def _run_epoch(
    *,
    model: MultiHeadCardPlayPolicyNetwork,
    loader: DataLoader[tuple[Tensor, Tensor, Tensor, Tensor]],
    device: torch.device,
    optimizer: AdamW | None,
) -> MultiHeadMetrics:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    role_correct = {role: 0 for role in CardPlayRole}
    role_examples = {role: 0 for role in CardPlayRole}

    for features, legal_masks, targets, role_ids in loader:
        features = features.to(device)
        legal_masks = legal_masks.to(device)
        targets = targets.to(device)
        role_ids = role_ids.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(features, role_ids)
            masked_logits = mask_illegal_logits(logits, legal_masks)
            loss = F.cross_entropy(masked_logits, targets)

            if training:
                loss.backward()
                optimizer.step()

        predictions = masked_logits.argmax(dim=1)
        correct = predictions.eq(targets)
        batch_size = int(targets.shape[0])

        total_loss += float(loss.item()) * batch_size
        total_correct += int(correct.sum().item())
        total_examples += batch_size

        for role in CardPlayRole:
            selected = role_ids == int(role)
            count = int(selected.sum().item())
            if count:
                role_examples[role] += count
                role_correct[role] += int((correct & selected).sum().item())

    if total_examples == 0:
        raise RuntimeError("Training or validation loader produced no examples.")

    role_accuracy = {
        role: role_correct[role] / role_examples[role]
        for role in CardPlayRole
        if role_examples[role] > 0
    }
    macro = statistics.fmean(role_accuracy.values()) if role_accuracy else 0.0

    return MultiHeadMetrics(
        loss=total_loss / total_examples,
        accuracy=total_correct / total_examples,
        macro_role_accuracy=macro,
        role_accuracy=role_accuracy,
    )


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
    shared_hidden_sizes: tuple[int, int],
    head_hidden_size: int,
    nontrivial_only: bool,
    role_balance_alpha: float,
) -> None:
    if epochs <= 0:
        raise ValueError("epochs must be greater than zero.")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")
    if not 0.0 <= role_balance_alpha <= 1.0:
        raise ValueError("role_balance_alpha must be between 0 and 1.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    arrays = load_behavior_cloning_arrays(dataset_path)
    if arrays.role_ids is None:
        raise ValueError(
            "Multi-head training requires role_ids. Regenerate the dataset with "
            "the current demonstration collector."
        )

    dataset = RoleBehaviorCloningDataset(arrays)
    split_train, split_validation = split_indices_by_game(
        arrays.game_ids,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    train_indices = filter_training_indices(
        split_train,
        arrays.legal_counts,
        nontrivial_only=nontrivial_only,
    )
    validation_indices = filter_training_indices(
        split_validation,
        arrays.legal_counts,
        nontrivial_only=nontrivial_only,
    )
    if not train_indices or not validation_indices:
        raise ValueError("Training and validation selections must both be non-empty.")

    train_dataset = Subset(dataset, train_indices)
    validation_dataset = Subset(dataset, validation_indices)

    generator = torch.Generator()
    generator.manual_seed(seed)

    if role_balance_alpha > 0.0:
        weights = _role_balance_weights(
            arrays.role_ids,
            train_indices,
            alpha=role_balance_alpha,
        )
        sampler = WeightedRandomSampler(
            torch.from_numpy(weights),
            num_samples=len(train_indices),
            replacement=True,
            generator=generator,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
        )
    else:
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

    device = _resolve_device(device_name)
    model = MultiHeadCardPlayPolicyNetwork(
        input_size=int(arrays.features.shape[1]),
        shared_hidden_sizes=shared_hidden_sizes,
        head_hidden_size=head_hidden_size,
    ).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    print("Shared-backbone multi-head behavioral cloning")
    print("=" * 92)
    print(f"Dataset:                   {dataset_path}")
    print(f"Examples:                  {arrays.example_count:,}")
    print(f"Training examples:         {len(train_indices):,}")
    print(f"Validation examples:       {len(validation_indices):,}")
    print(f"Device:                    {device}")
    print(f"Architecture:              {model.architecture}")
    print(f"Shared hidden sizes:       {shared_hidden_sizes}")
    print(f"Head hidden size:          {head_hidden_size}")
    print(f"Non-trivial only:          {nontrivial_only}")
    print(f"Role balance alpha:        {role_balance_alpha:g}")
    print()

    best_macro_accuracy = -1.0
    start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
        )
        validation_metrics = _run_epoch(
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
            f"macro-role {train_metrics.macro_role_accuracy:>7.2%} "
            f"| val loss {validation_metrics.loss:.4f} "
            f"acc {validation_metrics.accuracy:>7.2%} "
            f"macro-role {validation_metrics.macro_role_accuracy:>7.2%} "
            f"| {elapsed:>7.1f}s"
        )

        if validation_metrics.macro_role_accuracy > best_macro_accuracy:
            best_macro_accuracy = validation_metrics.macro_role_accuracy
            output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "architecture": model.architecture,
                    "input_size": model.input_size,
                    "shared_hidden_sizes": model.shared_hidden_sizes,
                    "head_hidden_size": model.head_hidden_size,
                    "action_count": model.action_count,
                    "observation_version": model.observation_version,
                    "role_count": model.role_count,
                    "epoch": epoch,
                    "validation_accuracy": validation_metrics.accuracy,
                    "validation_macro_role_accuracy": (
                        validation_metrics.macro_role_accuracy
                    ),
                    "validation_role_accuracy": {
                        role.key: accuracy
                        for role, accuracy in validation_metrics.role_accuracy.items()
                    },
                    "dataset_path": str(dataset_path),
                    "seed": seed,
                    "training_nontrivial_only": nontrivial_only,
                    "role_balance_alpha": role_balance_alpha,
                },
                output,
            )
            print(f"  saved new best checkpoint -> {output}")

    print("\nTraining complete")
    print("-" * 92)
    print(f"Best validation macro-role accuracy: {best_macro_accuracy:.2%}")
    print(f"Checkpoint:                          {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a V2 shared-backbone card-play policy with one policy head "
            "per known strategic role."
        )
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/behavior_cloning/card_play_multihead_v2.pt"),
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--shared-hidden-1", type=int, default=512)
    parser.add_argument("--shared-hidden-2", type=int, default=512)
    parser.add_argument("--head-hidden", type=int, default=256)
    parser.add_argument(
        "--nontrivial-only",
        action="store_true",
        help="Train only on decisions with more than one legal action.",
    )
    parser.add_argument(
        "--role-balance-alpha",
        type=float,
        default=0.0,
        help=(
            "Tempered role rebalancing in [0,1]. 0 preserves the dataset "
            "distribution; 0.5 moderately upweights rare roles."
        ),
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
        shared_hidden_sizes=(args.shared_hidden_1, args.shared_hidden_2),
        head_hidden_size=args.head_hidden,
        nontrivial_only=args.nontrivial_only,
        role_balance_alpha=args.role_balance_alpha,
    )


if __name__ == "__main__":
    main()
