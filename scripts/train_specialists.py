from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from schafkopf_ai.training.card_play_roles import ROLE_KEYS

MANIFEST_FORMAT = "specialist_card_play_v1"


def train_specialists(
    *,
    dataset: Path,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    validation_fraction: float,
    seed: int,
    device: str,
    hidden_1: int,
    hidden_2: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer = Path(__file__).with_name("train_behavior_clone.py")

    roles: dict[str, str] = {}
    for role in ROLE_KEYS:
        checkpoint = output_dir / f"{role}.pt"
        command = [
            sys.executable,
            str(trainer),
            "--dataset",
            str(dataset),
            "--output",
            str(checkpoint),
            "--epochs",
            str(epochs),
            "--batch-size",
            str(batch_size),
            "--learning-rate",
            str(learning_rate),
            "--weight-decay",
            str(weight_decay),
            "--validation-fraction",
            str(validation_fraction),
            "--seed",
            str(seed),
            "--device",
            device,
            "--hidden-1",
            str(hidden_1),
            "--hidden-2",
            str(hidden_2),
            "--nontrivial-only",
            "--specialist-role",
            role,
        ]

        print(f"\nTraining specialist: {role}")
        print("=" * 88)
        subprocess.run(command, check=True)
        roles[role] = checkpoint.name

    manifest = {
        "format": MANIFEST_FORMAT,
        "dataset": str(dataset),
        "roles": roles,
    }
    manifest_path = output_dir / "specialists.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nSpecialist training complete")
    print("=" * 88)
    print(f"Manifest: {manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train one V2 card-play model per contract/role and create a "
            "specialist manifest for routed inference."
        )
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/behavior_cloning/specialists_v2"),
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hidden-1", type=int, default=512)
    parser.add_argument("--hidden-2", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_specialists(
        dataset=args.dataset,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        device=args.device,
        hidden_1=args.hidden_1,
        hidden_2=args.hidden_2,
    )


if __name__ == "__main__":
    main()
