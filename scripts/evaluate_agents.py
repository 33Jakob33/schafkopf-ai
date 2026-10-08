from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

from schafkopf_ai.evaluation.paired import (
    EvaluationAgentSpec,
    PairedStats,
    available_agent_kinds,
    confidence_interval_95,
    evaluate_paired_agents,
)
from schafkopf_ai.game.game_type import GameType


def _default_name(kind: str) -> str:
    names = {
        "heuristic": "Heuristic",
        "bc": "BC",
        "dagger": "DAgger",
        "neural": "Neural",
        "ppo": "PPO",
    }
    return names.get(kind.lower(), kind)


def run_evaluation(
    *,
    reference_kind: str,
    reference_checkpoint: Path | None,
    reference_name: str | None,
    candidate_kind: str,
    candidate_checkpoint: Path | None,
    candidate_name: str | None,
    games: int,
    seed: int,
    progress_every: int,
    device: str,
) -> None:
    reference = EvaluationAgentSpec(
        kind=reference_kind,
        name=reference_name or _default_name(reference_kind),
        checkpoint=reference_checkpoint,
    )
    candidate = EvaluationAgentSpec(
        kind=candidate_kind,
        name=candidate_name or _default_name(candidate_kind),
        checkpoint=candidate_checkpoint,
    )

    start = time.perf_counter()

    def print_progress(completed: int, stats: PairedStats) -> None:
        elapsed = time.perf_counter() - start
        delta = statistics.fmean(stats.deltas)
        print(
            f"{completed:>7,}/{games:,} paired deals "
            f"| {elapsed:>8.2f} s "
            f"| {completed / elapsed:>7.1f} pairs/s "
            f"| {candidate.name} - {reference.name} {delta:+.3f}"
        )

    result = evaluate_paired_agents(
        reference_spec=reference,
        candidate_spec=candidate,
        games=games,
        seed=seed,
        device=device,
        progress_every=progress_every,
        progress_callback=print_progress,
    )

    elapsed = time.perf_counter() - start
    ci_low, ci_high = confidence_interval_95(result.overall.deltas)

    print(
        f"\n{result.candidate_name} vs {result.reference_name} "
        "— generic paired card-play evaluation"
    )
    print("=" * 96)
    print(f"Paired deals:               {games:,}")
    print(f"Rounds executed:            {games * 2:,}")
    print(
        f"Reference:                  {result.reference_name} ({result.reference_kind})"
    )
    print(f"Reference checkpoint:       {result.reference_checkpoint or '-'}")
    print(
        f"Candidate:                  {result.candidate_name} ({result.candidate_kind})"
    )
    print(f"Candidate checkpoint:       {result.candidate_checkpoint or '-'}")
    print(f"Seed:                       {seed}")
    print(f"Elapsed:                    {elapsed:.3f} s")

    print("\nPrimary paired result")
    print("-" * 96)
    print(
        f"{result.reference_name:<26}{result.mean_reference_payment:+.3f} mean payment"
    )
    print(
        f"{result.candidate_name:<26}{result.mean_candidate_payment:+.3f} mean payment"
    )
    print(
        f"Mean {result.candidate_name} - {result.reference_name}: "
        f"{result.mean_delta:+.3f}"
    )
    print(f"95% CI paired difference:   [{ci_low:+.3f}, {ci_high:+.3f}]")
    print(f"{result.reference_name} win rate: {result.reference_win_rate:.2%}")
    print(f"{result.candidate_name} win rate: {result.candidate_win_rate:.2%}")

    print("\nPer-deal payment comparison")
    print("-" * 96)
    print(f"{result.candidate_name} better: {result.candidate_better:>10,}")
    print(f"Equal payment:              {result.equal:>10,}")
    print(f"{result.reference_name} better: {result.reference_better:>10,}")

    print("\nPaired result by contract")
    print("-" * 96)
    for game_type in GameType:
        stats = result.by_contract.get(game_type)
        if stats is None or not stats.deltas:
            print(f"{game_type.value:<16} no games")
            continue

        contract_ci_low, contract_ci_high = confidence_interval_95(stats.deltas)
        print(
            f"{game_type.value:<16} "
            f"n={len(stats.deltas):>7,}  "
            f"ref={statistics.fmean(stats.reference_payments):>+8.3f}  "
            f"cand={statistics.fmean(stats.candidate_payments):>+8.3f}  "
            f"delta={statistics.fmean(stats.deltas):>+8.3f}  "
            f"CI=[{contract_ci_low:+.3f}, {contract_ci_high:+.3f}]"
        )


def parse_args() -> argparse.Namespace:
    supported = ", ".join(available_agent_kinds())
    parser = argparse.ArgumentParser(
        description=(
            "Compare any two registered card-play agents on identical paired deals. "
            "The reported delta is always candidate payment minus reference payment."
        )
    )
    parser.add_argument(
        "--reference-kind",
        required=True,
        help=f"Reference agent kind. Registered kinds: {supported}.",
    )
    parser.add_argument(
        "--reference-checkpoint",
        type=Path,
        default=None,
        help="Checkpoint for the reference agent, when its loader requires one.",
    )
    parser.add_argument(
        "--reference-name",
        default=None,
        help="Display name for the reference agent.",
    )
    parser.add_argument(
        "--candidate-kind",
        required=True,
        help=f"Candidate agent kind. Registered kinds: {supported}.",
    )
    parser.add_argument(
        "--candidate-checkpoint",
        type=Path,
        default=None,
        help="Checkpoint for the candidate agent, when its loader requires one.",
    )
    parser.add_argument(
        "--candidate-name",
        default=None,
        help="Display name for the candidate agent.",
    )
    parser.add_argument("--games", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--progress-every", type=int, default=1_000)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_evaluation(
        reference_kind=args.reference_kind,
        reference_checkpoint=args.reference_checkpoint,
        reference_name=args.reference_name,
        candidate_kind=args.candidate_kind,
        candidate_checkpoint=args.candidate_checkpoint,
        candidate_name=args.candidate_name,
        games=args.games,
        seed=args.seed,
        progress_every=args.progress_every,
        device=args.device,
    )


if __name__ == "__main__":
    main()
