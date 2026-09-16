from __future__ import annotations

import argparse
import random
import time
from collections import Counter
from pathlib import Path

from schafkopf_ai.game.game_type import GameType
from schafkopf_ai.game.rules import RAMSCH_RULES
from schafkopf_ai.runner.round_runner import RoundRunner
from schafkopf_ai.training.behavior_cloning import save_behavior_cloning_arrays
from schafkopf_ai.training.demonstrations import (
    BehaviorCloningCollector,
    RecordingHeuristicAgent,
)

CARD_PLAYS_PER_GAME = 32


def run_generation(
    *,
    games: int,
    seed: int,
    output: Path,
    progress_every: int,
) -> None:
    if games <= 0:
        raise ValueError("games must be greater than zero.")
    if progress_every < 0:
        raise ValueError("progress_every cannot be negative.")

    collector = BehaviorCloningCollector(games * CARD_PLAYS_PER_GAME)
    rng = random.Random(seed)
    contract_counts: Counter[GameType] = Counter()

    start = time.perf_counter()

    for game_index in range(games):
        agents = tuple(
            RecordingHeuristicAgent(
                collector=collector,
                game_id=game_index,
            )
            for _ in range(4)
        )

        result = RoundRunner(
            agents=agents,
            rules=RAMSCH_RULES,
            starting_player=game_index % 4,
            rng=rng,
        ).run()

        contract_counts[result.contract.game_type] += 1

        expected_count = (game_index + 1) * CARD_PLAYS_PER_GAME
        if collector.count != expected_count:
            raise RuntimeError(
                "Unexpected number of recorded card decisions: "
                f"{collector.count} != {expected_count}."
            )

        completed = game_index + 1
        if progress_every and completed % progress_every == 0:
            elapsed = time.perf_counter() - start
            rate = completed / elapsed
            print(
                f"{completed:>7,}/{games:,} games "
                f"| {elapsed:>8.2f} s "
                f"| {rate:>7.1f} games/s "
                f"| {collector.count:>9,} examples"
            )

    arrays = collector.arrays()
    save_behavior_cloning_arrays(output, arrays)

    elapsed = time.perf_counter() - start
    nontrivial = int((arrays.legal_counts > 1).sum())

    print("\nBehavioral-cloning dataset generated")
    print("=" * 72)
    print(f"Games:                     {games:,}")
    print(f"Examples:                  {arrays.example_count:,}")
    print(f"Non-trivial decisions:     {nontrivial:,}")
    print(f"Non-trivial share:         {nontrivial / arrays.example_count:.2%}")
    print(f"Seed:                      {seed}")
    print(f"Elapsed:                   {elapsed:.3f} s")
    print(f"Output:                    {output}")

    print("\nContract distribution")
    print("-" * 72)
    for game_type in GameType:
        count = contract_counts[game_type]
        print(f"{game_type.value:<12} {count:>7,}  ({count / games:>7.2%})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate card-play demonstrations by letting four HeuristicAgents "
            "play complete rounds."
        )
    )
    parser.add_argument(
        "--games",
        type=int,
        default=2_000,
        help="Number of complete rounds to record (default: 2000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Master random seed (default: 42).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/behavior_cloning/heuristic_card_play.npz"),
        help="Compressed output dataset path.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=250,
        help="Print progress every N games; 0 disables progress output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_generation(
        games=args.games,
        seed=args.seed,
        output=args.output,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
