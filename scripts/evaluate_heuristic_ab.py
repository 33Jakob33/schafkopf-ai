from __future__ import annotations

import argparse
import math
import random
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field

from schafkopf_ai.agents.agent import Agent
from schafkopf_ai.agents.heuristic_agent import HeuristicAgent
from schafkopf_ai.agents.random_agent import RandomAgent
from schafkopf_ai.agents.strategic_heuristic_agent import StrategicHeuristicAgent
from schafkopf_ai.game.game_type import GameType
from schafkopf_ai.game.rules import RAMSCH_RULES
from schafkopf_ai.runner.round_runner import RoundResult, RoundRunner


@dataclass
class ComparisonStats:
    games: int = 0
    original_wins: int = 0
    strategic_wins: int = 0
    original_payment: int = 0
    strategic_payment: int = 0
    differences: list[int] = field(default_factory=list)

    def add(
        self,
        *,
        original_won: bool,
        strategic_won: bool,
        original_payment: int,
        strategic_payment: int,
    ) -> None:
        self.games += 1
        self.original_wins += int(original_won)
        self.strategic_wins += int(strategic_won)
        self.original_payment += original_payment
        self.strategic_payment += strategic_payment
        self.differences.append(strategic_payment - original_payment)

    @property
    def mean_original(self) -> float:
        return self.original_payment / self.games if self.games else 0.0

    @property
    def mean_strategic(self) -> float:
        return self.strategic_payment / self.games if self.games else 0.0

    @property
    def mean_difference(self) -> float:
        return statistics.fmean(self.differences) if self.differences else 0.0


@dataclass
class PairOutcomeCounts:
    strategic_better: int = 0
    equal_payment: int = 0
    original_better: int = 0
    both_win: int = 0
    strategic_only_win: int = 0
    original_only_win: int = 0
    both_lose: int = 0

    def add(
        self,
        *,
        original_won: bool,
        strategic_won: bool,
        original_payment: int,
        strategic_payment: int,
    ) -> None:
        if strategic_payment > original_payment:
            self.strategic_better += 1
        elif strategic_payment < original_payment:
            self.original_better += 1
        else:
            self.equal_payment += 1

        if original_won and strategic_won:
            self.both_win += 1
        elif strategic_won:
            self.strategic_only_win += 1
        elif original_won:
            self.original_only_win += 1
        else:
            self.both_lose += 1


def validate_round(result: RoundResult) -> None:
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


def confidence_interval_95(values: list[int]) -> tuple[float, float]:
    """Normal-approximation 95% CI for a paired mean difference."""
    if len(values) < 2:
        mean = float(values[0]) if values else 0.0
        return mean, mean

    mean = statistics.fmean(values)
    standard_error = statistics.stdev(values) / math.sqrt(len(values))
    margin = 1.96 * standard_error
    return mean - margin, mean + margin


def create_agents(
    *,
    focal_agent: Agent,
    focal_seat: int,
    opponent_seeds: tuple[int, int, int, int],
) -> tuple[Agent, Agent, Agent, Agent]:
    agents: list[Agent] = []
    for seat in range(4):
        if seat == focal_seat:
            agents.append(focal_agent)
        else:
            agents.append(RandomAgent(rng=random.Random(opponent_seeds[seat])))

    return agents[0], agents[1], agents[2], agents[3]


def run_one(
    *,
    focal_agent: Agent,
    focal_seat: int,
    starting_player: int,
    deal_seed: int,
    opponent_seeds: tuple[int, int, int, int],
) -> RoundResult:
    agents = create_agents(
        focal_agent=focal_agent,
        focal_seat=focal_seat,
        opponent_seeds=opponent_seeds,
    )
    result = RoundRunner(
        agents=agents,
        rules=RAMSCH_RULES,
        starting_player=starting_player,
        rng=random.Random(deal_seed),
    ).run()
    validate_round(result)
    return result


def role_name(result: RoundResult, focal_seat: int) -> str:
    if result.contract.game_type is GameType.RAMSCH:
        return "Ramsch"
    if result.contract.declarer == focal_seat:
        return "Declarer"
    return "Non-declarer"


def add_pair(
    stats: ComparisonStats,
    outcomes: PairOutcomeCounts,
    *,
    original: RoundResult,
    strategic: RoundResult,
    focal_seat: int,
) -> None:
    original_won = focal_seat in original.game_result.winner_players
    strategic_won = focal_seat in strategic.game_result.winner_players
    original_payment = original.payments[focal_seat]
    strategic_payment = strategic.payments[focal_seat]

    stats.add(
        original_won=original_won,
        strategic_won=strategic_won,
        original_payment=original_payment,
        strategic_payment=strategic_payment,
    )
    outcomes.add(
        original_won=original_won,
        strategic_won=strategic_won,
        original_payment=original_payment,
        strategic_payment=strategic_payment,
    )


def print_stats(name: str, stats: ComparisonStats) -> None:
    if stats.games == 0:
        print(f"{name:<16} no games")
        return

    ci_low, ci_high = confidence_interval_95(stats.differences)
    print(
        f"{name:<16} n={stats.games:>7,}  "
        f"orig={stats.mean_original:>+8.3f}  "
        f"strat={stats.mean_strategic:>+8.3f}  "
        f"delta={stats.mean_difference:>+8.3f}  "
        f"CI=[{ci_low:+.3f}, {ci_high:+.3f}]"
    )


def run_evaluation(*, games: int, seed: int, progress_every: int) -> None:
    if games <= 0:
        raise ValueError("games must be greater than zero.")
    if progress_every < 0:
        raise ValueError("progress_every cannot be negative.")

    overall = ComparisonStats()
    outcomes = PairOutcomeCounts()
    by_contract: dict[GameType, ComparisonStats] = {
        game_type: ComparisonStats() for game_type in GameType
    }
    by_role: dict[str, ComparisonStats] = defaultdict(ComparisonStats)
    by_seat: dict[int, ComparisonStats] = {
        seat: ComparisonStats() for seat in range(4)
    }
    by_start_offset: dict[int, ComparisonStats] = {
        offset: ComparisonStats() for offset in range(4)
    }

    start = time.perf_counter()

    for game_index in range(games):
        # Rotate focal seat and forehand independently so all 16 combinations
        # occur evenly over each 16-game block.
        focal_seat = game_index % 4
        starting_player = (game_index // 4) % 4
        start_offset = (focal_seat - starting_player) % 4

        deal_seed = seed * 1_000_003 + game_index
        opponent_seed_base = seed * 10_000_019 + game_index * 10
        opponent_seeds = (
            opponent_seed_base + 1,
            opponent_seed_base + 2,
            opponent_seed_base + 3,
            opponent_seed_base + 4,
        )

        original = run_one(
            focal_agent=HeuristicAgent(),
            focal_seat=focal_seat,
            starting_player=starting_player,
            deal_seed=deal_seed,
            opponent_seeds=opponent_seeds,
        )
        strategic = run_one(
            focal_agent=StrategicHeuristicAgent(),
            focal_seat=focal_seat,
            starting_player=starting_player,
            deal_seed=deal_seed,
            opponent_seeds=opponent_seeds,
        )

        if original.game_state.initial_hands != strategic.game_state.initial_hands:
            raise RuntimeError("A/B pair did not receive identical dealt hands.")

        # StrategicHeuristicAgent inherits the exact same bidding policy. A
        # mismatch would mean the paired card-play comparison is contaminated.
        if original.contract != strategic.contract:
            raise RuntimeError(
                "A/B pair produced different contracts despite identical bidding policy: "
                f"{original.contract} != {strategic.contract}."
            )

        pair_outcomes = PairOutcomeCounts()
        add_pair(
            overall,
            outcomes,
            original=original,
            strategic=strategic,
            focal_seat=focal_seat,
        )

        contract_stats = by_contract[original.contract.game_type]
        add_pair(
            contract_stats,
            pair_outcomes,
            original=original,
            strategic=strategic,
            focal_seat=focal_seat,
        )

        role = role_name(original, focal_seat)
        add_pair(
            by_role[role],
            PairOutcomeCounts(),
            original=original,
            strategic=strategic,
            focal_seat=focal_seat,
        )
        add_pair(
            by_seat[focal_seat],
            PairOutcomeCounts(),
            original=original,
            strategic=strategic,
            focal_seat=focal_seat,
        )
        add_pair(
            by_start_offset[start_offset],
            PairOutcomeCounts(),
            original=original,
            strategic=strategic,
            focal_seat=focal_seat,
        )

        completed = game_index + 1
        if progress_every and completed % progress_every == 0:
            elapsed = time.perf_counter() - start
            print(
                f"{completed:>7,}/{games:,} paired deals "
                f"| {elapsed:>8.2f} s "
                f"| {completed / elapsed:>7.1f} pairs/s "
                f"| delta {overall.mean_difference:+.3f}"
            )

    elapsed = time.perf_counter() - start
    ci_low, ci_high = confidence_interval_95(overall.differences)

    print("\nHeuristicAgent vs StrategicHeuristicAgent — paired A/B")
    print("=" * 88)
    print(f"Paired deals:               {games:,}")
    print(f"Rounds executed:            {games * 2:,}")
    print(f"Seed:                       {seed}")
    print(f"Elapsed:                    {elapsed:.3f} s")
    print(f"Throughput:                 {games / elapsed:.1f} paired deals/s")
    print("Deal mismatches:            0")
    print("Contract mismatches:        0")
    print("Invariant failures:         0")

    print("\nPrimary paired result")
    print("-" * 88)
    print(f"Original mean payment:      {overall.mean_original:+.3f}")
    print(f"Strategic mean payment:     {overall.mean_strategic:+.3f}")
    print(f"Mean paired improvement:    {overall.mean_difference:+.3f}")
    print(f"95% CI paired improvement:  [{ci_low:+.3f}, {ci_high:+.3f}]")
    print(
        f"Original win rate:          {overall.original_wins / games:>8.2%}"
    )
    print(
        f"Strategic win rate:         {overall.strategic_wins / games:>8.2%}"
    )
    print(
        f"Total payment improvement:  "
        f"{overall.strategic_payment - overall.original_payment:>+10,}"
    )

    print("\nPer-deal payment comparison")
    print("-" * 88)
    print(f"Strategic better:           {outcomes.strategic_better:>8,}")
    print(f"Equal payment:              {outcomes.equal_payment:>8,}")
    print(f"Original better:            {outcomes.original_better:>8,}")

    print("\nPaired win outcomes")
    print("-" * 88)
    print(f"Both win:                   {outcomes.both_win:>8,}")
    print(f"Strategic only wins:        {outcomes.strategic_only_win:>8,}")
    print(f"Original only wins:         {outcomes.original_only_win:>8,}")
    print(f"Both lose:                  {outcomes.both_lose:>8,}")

    print("\nPaired result by role")
    print("-" * 88)
    for role in ("Declarer", "Non-declarer", "Ramsch"):
        print_stats(role, by_role[role])

    print("\nPaired result by contract")
    print("-" * 88)
    for game_type in GameType:
        print_stats(game_type.value, by_contract[game_type])

    print("\nPaired result by focal seat")
    print("-" * 88)
    for seat in range(4):
        print_stats(f"Seat {seat}", by_seat[seat])

    print("\nPaired result by position relative to forehand")
    print("-" * 88)
    for offset in range(4):
        label = {
            0: "Forehand",
            1: "Second",
            2: "Third",
            3: "Fourth",
        }[offset]
        print_stats(label, by_start_offset[offset])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Paired A/B evaluation of HeuristicAgent and StrategicHeuristicAgent "
            "against identically seeded RandomAgents on identical deals."
        )
    )
    parser.add_argument(
        "--games",
        type=int,
        default=10_000,
        help="Number of paired deals to evaluate (default: 10000).",
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
        help="Print progress every N paired deals; 0 disables progress output.",
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
