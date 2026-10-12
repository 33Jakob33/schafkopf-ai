from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

from schafkopf_ai.training.action_values import (
    ActionValueDataset,
    CardPlayQNetwork,
    load_action_value_arrays,
)
from schafkopf_ai.training.behavior_cloning import split_indices_by_game


@dataclass(frozen=True, slots=True)
class QMetrics:
    loss: float
    mae_payment: float
    action_accuracy: float


def _resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _masked_q_loss(
    predictions: Tensor,
    targets: Tensor,
    legal_mask: Tensor,
    q_stds: Tensor,
    q_counts: Tensor,
    *,
    target_scale: float,
    uncertainty_weighting: bool,
    huber_beta: float,
) -> Tensor:
    if target_scale <= 0.0:
        raise ValueError("target_scale must be positive.")
    if huber_beta <= 0.0:
        raise ValueError("huber_beta must be positive.")

    legal = legal_mask.to(torch.bool)
    safe_targets = torch.where(legal, targets / target_scale, torch.zeros_like(targets))
    per_action = F.smooth_l1_loss(
        predictions,
        safe_targets,
        reduction="none",
        beta=huber_beta,
    )

    weights = legal.to(predictions.dtype)
    if uncertainty_weighting:
        safe_counts = q_counts.clamp_min(1).to(predictions.dtype)
        stderr = torch.where(
            legal,
            q_stds / safe_counts.sqrt(),
            torch.zeros_like(q_stds),
        )
        reliability = 1.0 / (1.0 + stderr / target_scale)
        weights = weights * reliability

    weighted = per_action * weights
    per_state_denominator = weights.sum(dim=1).clamp_min(1e-8)
    per_state = weighted.sum(dim=1) / per_state_denominator
    return per_state.mean()


def _run_epoch(
    *,
    model: CardPlayQNetwork,
    loader: DataLoader[
        tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
    ],
    device: torch.device,
    optimizer: AdamW | None,
    target_scale: float,
    uncertainty_weighting: bool,
    huber_beta: float,
) -> QMetrics:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_states = 0
    absolute_error_sum = 0.0
    legal_action_count = 0
    correct_actions = 0

    for (
        features,
        legal_masks,
        q_values,
        q_stds,
        q_counts,
        rollout_actions,
    ) in loader:
        features = features.to(device)
        legal_masks = legal_masks.to(device)
        q_values = q_values.to(device)
        q_stds = q_stds.to(device)
        q_counts = q_counts.to(device)
        rollout_actions = rollout_actions.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            predictions = model(features)
            loss = _masked_q_loss(
                predictions,
                q_values,
                legal_masks,
                q_stds,
                q_counts,
                target_scale=target_scale,
                uncertainty_weighting=uncertainty_weighting,
                huber_beta=huber_beta,
            )
            if training:
                loss.backward()
                optimizer.step()

        batch_size = int(features.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_states += batch_size

        legal = legal_masks.to(torch.bool)
        prediction_payments = predictions.detach() * target_scale
        safe_targets = torch.where(
            legal,
            q_values,
            torch.zeros_like(q_values),
        )
        absolute_error_sum += float(
            (torch.abs(prediction_payments - safe_targets) * legal).sum().item()
        )
        legal_action_count += int(legal.sum().item())

        minimum = torch.finfo(predictions.dtype).min
        greedy = predictions.masked_fill(~legal, minimum).argmax(dim=1)
        correct_actions += int(greedy.eq(rollout_actions).sum().item())

    if total_states == 0 or legal_action_count == 0:
        raise RuntimeError("Q-value data loader produced no supervised states.")

    return QMetrics(
        loss=total_loss / total_states,
        mae_payment=absolute_error_sum / legal_action_count,
        action_accuracy=correct_actions / total_states,
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
    hidden_sizes: tuple[int, int],
    target_scale: float,
    uncertainty_weighting: bool,
    huber_beta: float,
) -> None:
    if epochs <= 0:
        raise ValueError("epochs must be greater than zero.")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one.")
    if target_scale <= 0.0:
        raise ValueError("target_scale must be positive.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    arrays = load_action_value_arrays(dataset_path)
    dataset = ActionValueDataset(arrays)
    train_indices, validation_indices = split_indices_by_game(
        arrays.game_ids,
        validation_fraction=validation_fraction,
        seed=seed,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        Subset(dataset, validation_indices),
        batch_size=batch_size,
        shuffle=False,
    )

    device = _resolve_device(device_name)
    model = CardPlayQNetwork(
        hidden_sizes=hidden_sizes,
    ).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    print("Action-value regression")
    print("=" * 92)
    print(f"Dataset:                   {dataset_path}")
    print(f"States:                    {arrays.example_count:,}")
    print(f"Training states:           {len(train_indices):,}")
    print(f"Validation states:         {len(validation_indices):,}")
    print(f"Device:                    {device}")
    print(f"Architecture:              {model.architecture}")
    print(f"Hidden sizes:              {hidden_sizes}")
    print(f"Target scale:              {target_scale:g}")
    print(f"Uncertainty weighting:     {uncertainty_weighting}")
    print(f"Huber beta:                {huber_beta:g}")
    print()

    best_action_accuracy = -1.0
    best_mae = math.inf
    start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            target_scale=target_scale,
            uncertainty_weighting=uncertainty_weighting,
            huber_beta=huber_beta,
        )
        validation_metrics = _run_epoch(
            model=model,
            loader=validation_loader,
            device=device,
            optimizer=None,
            target_scale=target_scale,
            uncertainty_weighting=uncertainty_weighting,
            huber_beta=huber_beta,
        )

        elapsed = time.perf_counter() - start
        print(
            f"Epoch {epoch:>2}/{epochs} "
            f"| train loss {train_metrics.loss:.4f} "
            f"MAE {train_metrics.mae_payment:>7.3f} "
            f"action {train_metrics.action_accuracy:>7.2%} "
            f"| val loss {validation_metrics.loss:.4f} "
            f"MAE {validation_metrics.mae_payment:>7.3f} "
            f"action {validation_metrics.action_accuracy:>7.2%} "
            f"| {elapsed:>7.1f}s"
        )

        improved = (
            validation_metrics.action_accuracy > best_action_accuracy
            or (
                validation_metrics.action_accuracy == best_action_accuracy
                and validation_metrics.mae_payment < best_mae
            )
        )
        if improved:
            best_action_accuracy = validation_metrics.action_accuracy
            best_mae = validation_metrics.mae_payment
            output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "architecture": model.architecture,
                    "input_size": model.input_size,
                    "hidden_sizes": model.hidden_sizes,
                    "action_count": model.action_count,
                    "observation_version": model.observation_version,
                    "target_scale": target_scale,
                    "epoch": epoch,
                    "validation_action_accuracy": best_action_accuracy,
                    "validation_mae_payment": best_mae,
                    "dataset_path": str(dataset_path),
                    "seed": seed,
                    "uncertainty_weighting": uncertainty_weighting,
                    "huber_beta": huber_beta,
                },
                output,
            )
            print(f"  saved new best checkpoint -> {output}")

    print("\nTraining complete")
    print("-" * 92)
    print(f"Best validation action agreement: {best_action_accuracy:.2%}")
    print(f"Best validation Q MAE:            {best_mae:.3f}")
    print(f"Checkpoint:                       {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a V2 network to predict Monte-Carlo expected payment for "
            "all 32 physical card actions."
        )
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/action_values/card_play_q_v2.pt"),
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hidden-1", type=int, default=512)
    parser.add_argument("--hidden-2", type=int, default=256)
    parser.add_argument(
        "--target-scale",
        type=float,
        default=100.0,
        help="Divide raw payment targets by this value during regression.",
    )
    parser.add_argument(
        "--uncertainty-weighting",
        action="store_true",
        help=(
            "Downweight noisy Q targets using their Monte-Carlo standard error. "
            "Off by default for the first clean baseline."
        ),
    )
    parser.add_argument(
        "--huber-beta",
        type=float,
        default=0.10,
        help="Smooth-L1 transition point in scaled Q units.",
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
        target_scale=args.target_scale,
        uncertainty_weighting=args.uncertainty_weighting,
        huber_beta=args.huber_beta,
    )


if __name__ == "__main__":
    main()
