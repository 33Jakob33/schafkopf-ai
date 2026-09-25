from __future__ import annotations

import argparse
import math
import random
import statistics
import time
from collections import Counter
from dataclasses import dataclass

from schafkopf_ai.agents.agent import Agent
from schafkopf_ai.agents.heuristic_agent import HeuristicAgent
from schafkopf_ai.agents.random_agent import RandomAgent
from schafkopf_ai.game.game_type import GameType
from schafkopf_ai.game.rules import RAMSCH_RULES
from schafkopf_ai.runner.round_runner import RoundResult, RoundRunner


@dataclass
class SeatStats:
    games: int = 0
    wins: int = 0
    net_payment: int = 0


@dataclass
class RoleStats:
    games: int = 0
    wins: int = 0
    net_payment: int = 0


def validate_round(result: RoundResult) -> None:
    """Check core invariants on every evaluated round."""
    if not result.game_state.is_complete:
        raise RuntimeError("Round finished with an incomplete GameState.")

    if len(result.game_state.completed_tricks) != 8:
        raise RuntimeError("Completed round does not contain exactly eight tricks.")

    played_cards = [
        play.card
        for trick in result.game_state.completed_tricks
        for play in trick.plays
    ]

    if len(played_cards) != 32 or len(set(played_cards)) != 32:
        raise RuntimeError("Completed round must contain 32 unique played cards.")

    if any(len(player) != 0 for player in result.game_state.players):
        raise RuntimeError("At least one player still holds cards after the round.")

    if sum(result.player_points) != 120:
        raise RuntimeError(f"Augen do not sum to 120: {result.player_points}.")

    if sum(result.payments) != 0:
        raise RuntimeError(f"Payments do not sum to zero: {result.payments}.")

    if result.redeals != 0:
        raise RuntimeError("RAMSCH_RULES should never require a redeal.")


def create_agents(
    *,
    heuristic_seat: int,
    random_rngs: tuple[random.Random, random.Random, random.Random, random.Random],
) -> tuple[Agent, Agent, Agent, Agent]:
    agents: list[Agent] = []

    for seat in range(4):
        if seat == heuristic_seat:
            agents.append(HeuristicAgent())
        else:
            agents.append(RandomAgent(rng=random_rngs[seat]))

    return agents[0], agents[1], agents[2], agents[3]


def confidence_interval_95(values: list[int]) -> tuple[float, float]:
    """Normal-approximation 95% CI for the mean payment per game."""
    if len(values) < 2:
        mean = float(values[0]) if values else 0.0
        return mean, mean

    mean = statistics.fmean(values)
    standard_error = statistics.stdev(values) / math.sqrt(len(values))
    margin = 1.96 * standard_error
    return mean - margin, mean + margin


def run_evaluation(*, games: int, seed: int, progress_every: int) -> None:
    if games <= 0:
        raise ValueError("games must be greater than zero.")
    if progress_every < 0:
        raise ValueError("progress_every cannot be negative.")

    deal_rng = random.Random(seed)
    random_rngs = (
        random.Random(seed + 10_001),
        random.Random(seed + 10_002),
        random.Random(seed + 10_003),
        random.Random(seed + 10_004),
    )

    seat_stats = [SeatStats() for _ in range(4)]
    declarer_stats = RoleStats()
    non_declarer_stats = RoleStats()
    ramsch_stats = RoleStats()

    contract_counts: Counter[GameType] = Counter()
    heuristic_declared_contracts: Counter[GameType] = Counter()
    heuristic_payments: list[int] = []

    heuristic_wins = 0
    heuristic_schneider_wins = 0
    heuristic_schwarz_wins = 0

    start = time.perf_counter()

    for game_index in range(games):
        heuristic_seat = game_index % 4
        agents = create_agents(
            heuristic_seat=heuristic_seat,
            random_rngs=random_rngs,
        )

        result = RoundRunner(
            agents=agents,
            rules=RAMSCH_RULES,
            starting_player=game_index % 4,
            rng=deal_rng,
        ).run()
        validate_round(result)

        won = heuristic_seat in result.game_result.winner_players
        payment = result.payments[heuristic_seat]
        heuristic_payments.append(payment)

        if won:
            heuristic_wins += 1

        contract_counts[result.contract.game_type] += 1

        seat = seat_stats[heuristic_seat]
        seat.games += 1
        seat.wins += int(won)
        seat.net_payment += payment

        if result.contract.game_type is GameType.RAMSCH:
            role = ramsch_stats
        elif result.contract.declarer == heuristic_seat:
            role = declarer_stats
            heuristic_declared_contracts[result.contract.game_type] += 1
        else:
            role = non_declarer_stats

        role.games += 1
        role.wins += int(won)
        role.net_payment += payment

        if won and result.game_result.schneider:
            heuristic_schneider_wins += 1
        if won and result.game_result.schwarz:
            heuristic_schwarz_wins += 1

        completed = game_index + 1
        if progress_every and completed % progress_every == 0:
            elapsed = time.perf_counter() - start
            rate = completed / elapsed
            mean_payment = statistics.fmean(heuristic_payments)
            print(
                f"{completed:>7,}/{games:,} games "
                f"| {elapsed:>8.2f} s "
                f"| {rate:>7.1f} games/s "
                f"| heuristic mean payment {mean_payment:+.3f}"
            )

    elapsed = time.perf_counter() - start
    mean_payment = statistics.fmean(heuristic_payments)
    median_payment = statistics.median(heuristic_payments)
    ci_low, ci_high = confidence_interval_95(heuristic_payments)

    print("\nHeuristicAgent vs 3 RandomAgents")
    print("=" * 72)
    print(f"Games:                     {games:,}")
    print(f"Seed:                      {seed}")
    print(f"Elapsed:                   {elapsed:.3f} s")
    print(f"Throughput:                {games / elapsed:.1f} games/s")
    print("Invariant failures:        0")

    print("\nPrimary performance")
    print("-" * 72)
    print(
        f"Heuristic winner rate:     {heuristic_wins / games:>8.2%} "
        f"({heuristic_wins:,}/{games:,})"
    )
    print(f"Heuristic net payment:     {sum(heuristic_payments):>+10,}")
    print(f"Mean payment / game:       {mean_payment:>+10.3f}")
    print(f"Median payment / game:     {median_payment:>+10.3f}")
    print(f"95% CI mean payment:       [{ci_low:+.3f}, {ci_high:+.3f}]")
    print(f"Schneider wins:            {heuristic_schneider_wins:>10,}")
    print(f"Schwarz wins:              {heuristic_schwarz_wins:>10,}")

    print("\nPerformance by role")
    print("-" * 72)
    for name, stats in (
        ("Declarer", declarer_stats),
        ("Non-declarer", non_declarer_stats),
        ("Ramsch", ramsch_stats),
    ):
        if stats.games == 0:
            print(f"{name:<16} no games")
            continue
        print(
            f"{name:<16} games={stats.games:>7,}  "
            f"win={stats.wins / stats.games:>7.2%}  "
            f"net={stats.net_payment:>+9,}  "
            f"mean={stats.net_payment / stats.games:>+8.3f}"
        )

    print("\nPerformance by heuristic seat")
    print("-" * 72)
    for seat_index, stats in enumerate(seat_stats):
        print(
            f"Seat {seat_index}:          games={stats.games:>7,}  "
            f"win={stats.wins / stats.games:>7.2%}  "
            f"net={stats.net_payment:>+9,}  "
            f"mean={stats.net_payment / stats.games:>+8.3f}"
        )

    print("\nOverall contract distribution")
    print("-" * 72)
    for game_type in GameType:
        count = contract_counts[game_type]
        print(f"{game_type.value:<12} {count:>7,}  ({count / games:>7.2%})")

    print("\nContracts declared by HeuristicAgent")
    print("-" * 72)
    total_heuristic_declarations = sum(heuristic_declared_contracts.values())
    for game_type in GameType:
        count = heuristic_declared_contracts[game_type]
        percentage = (
            count / total_heuristic_declarations
            if total_heuristic_declarations
            else 0.0
        )
        print(f"{game_type.value:<12} {count:>7,}  ({percentage:>7.2%})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one HeuristicAgent against three RandomAgents while "
            "rotating the heuristic through all four seats."
        )
    )
    parser.add_argument(
        "--games",
        type=int,
        default=10_000,
        help="Number of complete rounds to evaluate (default: 10000).",
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
    run_evaluation(
        games=args.games,
        seed=args.seed,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
