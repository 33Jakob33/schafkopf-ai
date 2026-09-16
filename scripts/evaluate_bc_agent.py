from __future__ import annotations

import argparse
import math
import random
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from schafkopf_ai.agents.agent import Agent
from schafkopf_ai.agents.heuristic_agent import HeuristicAgent
from schafkopf_ai.agents.neural_card_play_agent import NeuralCardPlayAgent
from schafkopf_ai.agents.random_agent import RandomAgent
from schafkopf_ai.game.game_type import GameType
from schafkopf_ai.game.rules import RAMSCH_RULES
from schafkopf_ai.runner.round_runner import RoundResult, RoundRunner


@dataclass(slots=True)
class PairedStats:
    heuristic_payments: list[int] = field(default_factory=list)
    neural_payments: list[int] = field(default_factory=list)
    deltas: list[int] = field(default_factory=list)

    def add(self, heuristic_payment: int, neural_payment: int) -> None:
        self.heuristic_payments.append(heuristic_payment)
        self.neural_payments.append(neural_payment)
        self.deltas.append(neural_payment - heuristic_payment)


def confidence_interval_95(values: list[int]) -> tuple[float, float]:
    if len(values) < 2:
        mean = float(values[0]) if values else 0.0
        return mean, mean

    mean = statistics.fmean(values)
    standard_error = statistics.stdev(values) / math.sqrt(len(values))
    margin = 1.96 * standard_error
    return mean - margin, mean + margin


def validate_round(result: RoundResult) -> None:
    if not result.game_state.is_complete:
        raise RuntimeError("Round finished with an incomplete GameState.")
    if len(result.game_state.completed_tricks) != 8:
        raise RuntimeError("Completed round does not contain eight tricks.")
    if sum(result.player_points) != 120:
        raise RuntimeError(f"Augen do not sum to 120: {result.player_points}.")
    if sum(result.payments) != 0:
        raise RuntimeError(f"Payments do not sum to zero: {result.payments}.")


def run_variant(
    *,
    focal_agent: Agent,
    game_index: int,
    master_seed: int,
) -> RoundResult:
    focal_seat = game_index % 4
    starting_player = (game_index // 4) % 4

    deal_seed = master_seed + game_index * 1009 + 1
    agents: list[Agent] = []

    for seat in range(4):
        if seat == focal_seat:
            agents.append(focal_agent)
        else:
            random_seed = master_seed + game_index * 10_007 + seat * 101 + 50_000
            agents.append(RandomAgent(rng=random.Random(random_seed)))

    result = RoundRunner(
        agents=agents,
        rules=RAMSCH_RULES,
        starting_player=starting_player,
        rng=random.Random(deal_seed),
    ).run()
    validate_round(result)
    return result


def run_evaluation(
    *,
    checkpoint: Path,
    games: int,
    seed: int,
    progress_every: int,
    device: str,
) -> None:
    if games <= 0:
        raise ValueError("games must be greater than zero.")

    heuristic = HeuristicAgent()
    neural = NeuralCardPlayAgent.from_checkpoint(checkpoint, device=device)

    overall = PairedStats()
    by_contract: dict[GameType, PairedStats] = defaultdict(PairedStats)

    heuristic_wins = 0
    neural_wins = 0
    neural_better = 0
    equal = 0
    heuristic_better = 0

    start = time.perf_counter()

    for game_index in range(games):
        focal_seat = game_index % 4

        heuristic_result = run_variant(
            focal_agent=heuristic,
            game_index=game_index,
            master_seed=seed,
        )
        neural_result = run_variant(
            focal_agent=neural,
            game_index=game_index,
            master_seed=seed,
        )

        if heuristic_result.contract != neural_result.contract:
            raise RuntimeError(
                "Paired runs produced different contracts despite identical bidding: "
                f"{heuristic_result.contract} != {neural_result.contract}."
            )

        heuristic_payment = heuristic_result.payments[focal_seat]
        neural_payment = neural_result.payments[focal_seat]
        overall.add(heuristic_payment, neural_payment)
        by_contract[heuristic_result.contract.game_type].add(
            heuristic_payment,
            neural_payment,
        )

        heuristic_wins += int(
            focal_seat in heuristic_result.game_result.winner_players
        )
        neural_wins += int(focal_seat in neural_result.game_result.winner_players)

        if neural_payment > heuristic_payment:
            neural_better += 1
        elif neural_payment == heuristic_payment:
            equal += 1
        else:
            heuristic_better += 1

        completed = game_index + 1
        if progress_every and completed % progress_every == 0:
            elapsed = time.perf_counter() - start
            delta = statistics.fmean(overall.deltas)
            print(
                f"{completed:>7,}/{games:,} paired deals "
                f"| {elapsed:>8.2f} s "
                f"| {completed / elapsed:>7.1f} pairs/s "
                f"| neural delta {delta:+.3f}"
            )

    elapsed = time.perf_counter() - start
    heuristic_mean = statistics.fmean(overall.heuristic_payments)
    neural_mean = statistics.fmean(overall.neural_payments)
    delta = statistics.fmean(overall.deltas)
    ci_low, ci_high = confidence_interval_95(overall.deltas)

    print("\nBehavioralCloningAgent vs HeuristicAgent — paired card-play evaluation")
    print("=" * 88)
    print(f"Paired deals:               {games:,}")
    print(f"Rounds executed:            {games * 2:,}")
    print(f"Checkpoint:                 {checkpoint}")
    print(f"Seed:                       {seed}")
    print(f"Elapsed:                    {elapsed:.3f} s")

    print("\nPrimary paired result")
    print("-" * 88)
    print(f"Heuristic mean payment:     {heuristic_mean:+.3f}")
    print(f"Neural mean payment:        {neural_mean:+.3f}")
    print(f"Mean neural improvement:    {delta:+.3f}")
    print(f"95% CI paired improvement:  [{ci_low:+.3f}, {ci_high:+.3f}]")
    print(f"Heuristic win rate:         {heuristic_wins / games:.2%}")
    print(f"Neural win rate:            {neural_wins / games:.2%}")

    print("\nPer-deal payment comparison")
    print("-" * 88)
    print(f"Neural better:              {neural_better:>8,}")
    print(f"Equal payment:              {equal:>8,}")
    print(f"Heuristic better:           {heuristic_better:>8,}")

    print("\nPaired result by contract")
    print("-" * 88)
    for game_type in GameType:
        stats = by_contract.get(game_type)
        if stats is None or not stats.deltas:
            print(f"{game_type.value:<16} no games")
            continue

        contract_ci_low, contract_ci_high = confidence_interval_95(stats.deltas)
        print(
            f"{game_type.value:<16} "
            f"n={len(stats.deltas):>7,}  "
            f"heur={statistics.fmean(stats.heuristic_payments):>+8.3f}  "
            f"neural={statistics.fmean(stats.neural_payments):>+8.3f}  "
            f"delta={statistics.fmean(stats.deltas):>+8.3f}  "
            f"CI=[{contract_ci_low:+.3f}, {contract_ci_high:+.3f}]"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a behavior-cloned card policy with HeuristicAgent on paired "
            "deals. Both use the same heuristic bidding policy."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/behavior_cloning/card_play_mlp.pt"),
    )
    parser.add_argument("--games", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=1_000)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_evaluation(
        checkpoint=args.checkpoint,
        games=args.games,
        seed=args.seed,
        progress_every=args.progress_every,
        device=args.device,
    )


if __name__ == "__main__":
    main()
