from __future__ import annotations

import argparse
import random
import time
from collections import Counter

from schafkopf_ai.agents.agent import Agent
from schafkopf_ai.agents.random_agent import RandomAgent
from schafkopf_ai.game.game_type import GameType
from schafkopf_ai.game.rules import RAMSCH_RULES
from schafkopf_ai.runner.round_runner import RoundResult, RoundRunner


def validate_round(result: RoundResult) -> None:
    """Validate invariants that must hold after every completed round."""
    if not result.game_state.is_complete:
        raise RuntimeError("Round finished with an incomplete GameState.")

    if len(result.game_state.completed_tricks) != 8:
        raise RuntimeError("Completed round does not contain exactly eight tricks.")

    played_cards = [
        play.card
        for trick in result.game_state.completed_tricks
        for play in trick.plays
    ]

    if len(played_cards) != 32:
        raise RuntimeError("Completed round did not play exactly 32 cards.")

    if len(set(played_cards)) != 32:
        raise RuntimeError("Completed round contains duplicate played cards.")

    if any(len(player) != 0 for player in result.game_state.players):
        raise RuntimeError("At least one player still holds cards after the round.")

    if sum(result.player_points) != 120:
        raise RuntimeError(
            f"Augen do not sum to 120: {result.player_points}."
        )

    if sum(result.payments) != 0:
        raise RuntimeError(
            f"Payments do not sum to zero: {result.payments}."
        )

    if result.redeals != 0:
        raise RuntimeError(
            "RAMSCH_RULES should finish all-pass bidding as Ramsch without redealing."
        )


def create_agents(seed: int) -> tuple[Agent, Agent, Agent, Agent]:
    """Create four independently seeded random agents."""
    return (
        RandomAgent(rng=random.Random(seed + 1)),
        RandomAgent(rng=random.Random(seed + 2)),
        RandomAgent(rng=random.Random(seed + 3)),
        RandomAgent(rng=random.Random(seed + 4)),
    )


def run_simulation(*, games: int, seed: int, progress_every: int) -> None:
    if games <= 0:
        raise ValueError("games must be greater than zero.")

    if progress_every < 0:
        raise ValueError("progress_every cannot be negative.")

    deal_rng = random.Random(seed)
    agents = create_agents(seed + 10_000)

    contract_counts: Counter[GameType] = Counter()
    winner_counts: Counter[int] = Counter()
    schneider_count = 0
    schwarz_count = 0
    total_game_value = 0

    start = time.perf_counter()

    for game_index in range(games):
        runner = RoundRunner(
            agents=agents,
            rules=RAMSCH_RULES,
            starting_player=game_index % 4,
            rng=deal_rng,
        )

        result = runner.run()
        validate_round(result)

        contract_counts[result.contract.game_type] += 1

        for winner in result.game_result.winner_players:
            winner_counts[winner] += 1

        if result.game_result.schneider:
            schneider_count += 1

        if result.game_result.schwarz:
            schwarz_count += 1

        total_game_value += result.settlement.game_value

        completed = game_index + 1
        if progress_every and completed % progress_every == 0:
            elapsed = time.perf_counter() - start
            rate = completed / elapsed
            print(
                f"{completed:>7,}/{games:,} games "
                f"| {elapsed:>8.2f} s "
                f"| {rate:>8.1f} games/s"
            )

    elapsed = time.perf_counter() - start
    games_per_second = games / elapsed

    print("\nSimulation complete")
    print("=" * 60)
    print(f"Games:              {games:,}")
    print(f"Seed:               {seed}")
    print(f"Elapsed:            {elapsed:.3f} s")
    print(f"Throughput:         {games_per_second:.1f} games/s")
    print(f"Time per game:      {(elapsed / games) * 1_000:.3f} ms")
    print(f"Invariant failures: 0")

    print("\nContracts")
    print("-" * 60)
    for game_type in GameType:
        count = contract_counts[game_type]
        percentage = (count / games) * 100
        print(f"{game_type.value:<12} {count:>7,}  ({percentage:>6.2f}%)")

    print("\nAdditional statistics")
    print("-" * 60)
    print(f"Schneider games:    {schneider_count:,}")
    print(f"Schwarz games:      {schwarz_count:,}")
    print(f"Mean game value:    {total_game_value / games:.2f}")

    print("\nWinner appearances")
    print("-" * 60)
    for player in range(4):
        print(f"Player {player}:          {winner_counts[player]:>7,}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run complete Schafkopf rounds with four RandomAgents."
    )
    parser.add_argument(
        "--games",
        type=int,
        default=10_000,
        help="Number of complete rounds to simulate (default: 10000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Master random seed (default: 42).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1_000,
        help="Print progress every N games; 0 disables progress output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_simulation(
        games=args.games,
        seed=args.seed,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
