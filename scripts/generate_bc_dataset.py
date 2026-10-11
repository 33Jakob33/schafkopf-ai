from __future__ import annotations

import argparse
import random
import time
from collections import Counter
from pathlib import Path
from typing import Literal

from schafkopf_ai.agents.agent import Agent
from schafkopf_ai.agents.heuristic_agent import HeuristicConfig
from schafkopf_ai.agents.heuristic_bidding import (
    BIDDING_PROFILES,
    bidding_config_for_profile,
)
from schafkopf_ai.agents.heuristic_bidding_random_card_agent import (
    HeuristicBiddingRandomCardAgent,
)
from schafkopf_ai.agents.random_agent import RandomAgent
from schafkopf_ai.game.game_type import GameType
from schafkopf_ai.game.rules import RAMSCH_RULES
from schafkopf_ai.runner.round_runner import RoundRunner
from schafkopf_ai.training.behavior_cloning import save_behavior_cloning_arrays
from schafkopf_ai.training.card_play_roles import CardPlayRole
from schafkopf_ai.training.demonstrations import (
    BehaviorCloningCollector,
    RecordingHeuristicAgent,
)
from schafkopf_ai.training.observation_encoding import OBSERVATION_VERSIONS

GenerationMode = Literal[
    "all-heuristic",
    "focal-random",
    "heuristic-bid-random-card",
]
GENERATION_MODES: tuple[GenerationMode, ...] = (
    "all-heuristic",
    "focal-random",
    "heuristic-bid-random-card",
)

CARD_PLAYS_PER_GAME = 32
CARD_PLAYS_PER_FOCAL_GAME = 8


def _run_all_heuristic_game(
    *,
    collector: BehaviorCloningCollector,
    game_index: int,
    rng: random.Random,
) -> GameType:
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
    return result.contract.game_type


def _run_focal_random_game(
    *,
    collector: BehaviorCloningCollector,
    game_index: int,
    seed: int,
) -> GameType:
    """
    Record one heuristic focal player against three RandomAgents.

    Seat rotation, starting-player rotation, deal seeding, and opponent seeding
    intentionally mirror the paired evaluator. This produces demonstrations
    from the state distribution on which learned focal agents are evaluated,
    while still using the heuristic as the supervised card-play target.
    """
    focal_seat = game_index % 4
    starting_player = (game_index // 4) % 4
    deal_seed = seed + game_index * 1_009 + 1

    agents: list[Agent] = []
    for seat in range(4):
        if seat == focal_seat:
            agents.append(
                RecordingHeuristicAgent(
                    collector=collector,
                    game_id=game_index,
                )
            )
        else:
            opponent_seed = seed + game_index * 10_007 + seat * 101 + 50_000
            agents.append(RandomAgent(rng=random.Random(opponent_seed)))

    result = RoundRunner(
        agents=agents,
        rules=RAMSCH_RULES,
        starting_player=starting_player,
        rng=random.Random(deal_seed),
    ).run()
    return result.contract.game_type


def _run_heuristic_bid_random_card_game(
    *,
    collector: BehaviorCloningCollector,
    game_index: int,
    seed: int,
    bidding_profile: str,
) -> GameType:
    """
    Record one heuristic focal player against random card-play opponents.

    All four seats use heuristic bidding, so the resulting contract
    distribution reflects the bidding baseline instead of RandomAgent bidding.
    Only the three opponents randomize card play.
    """
    focal_seat = game_index % 4
    starting_player = (game_index // 4) % 4
    deal_seed = seed + game_index * 1_009 + 1

    bidding_config = bidding_config_for_profile(bidding_profile)

    agents: list[Agent] = []
    for seat in range(4):
        if seat == focal_seat:
            agents.append(
                RecordingHeuristicAgent(
                    collector=collector,
                    game_id=game_index,
                    config=HeuristicConfig(bidding=bidding_config),
                )
            )
        else:
            opponent_seed = seed + game_index * 10_007 + seat * 101 + 50_000
            agents.append(
                HeuristicBiddingRandomCardAgent(
                    rng=random.Random(opponent_seed),
                    bidding_config=bidding_config,
                )
            )

    result = RoundRunner(
        agents=agents,
        rules=RAMSCH_RULES,
        starting_player=starting_player,
        rng=random.Random(deal_seed),
    ).run()
    return result.contract.game_type


def run_generation(
    *,
    games: int,
    seed: int,
    output: Path,
    progress_every: int,
    observation_version: str,
    generation_mode: GenerationMode = "all-heuristic",
    bidding_profile: str = "baseline",
) -> None:
    if games <= 0:
        raise ValueError("games must be greater than zero.")
    if progress_every < 0:
        raise ValueError("progress_every cannot be negative.")
    if generation_mode not in GENERATION_MODES:
        raise ValueError(
            f"Unsupported generation mode {generation_mode!r}; "
            f"expected one of {GENERATION_MODES}."
        )

    examples_per_game = (
        CARD_PLAYS_PER_GAME
        if generation_mode == "all-heuristic"
        else CARD_PLAYS_PER_FOCAL_GAME
    )
    collector = BehaviorCloningCollector(
        games * examples_per_game,
        observation_version=observation_version,
    )
    all_heuristic_rng = random.Random(seed)
    contract_counts: Counter[GameType] = Counter()

    start = time.perf_counter()

    for game_index in range(games):
        if generation_mode == "all-heuristic":
            game_type = _run_all_heuristic_game(
                collector=collector,
                game_index=game_index,
                rng=all_heuristic_rng,
            )
        elif generation_mode == "focal-random":
            game_type = _run_focal_random_game(
                collector=collector,
                game_index=game_index,
                seed=seed,
            )
        else:
            game_type = _run_heuristic_bid_random_card_game(
                collector=collector,
                game_index=game_index,
                seed=seed,
                bidding_profile=bidding_profile,
            )

        contract_counts[game_type] += 1

        expected_count = (game_index + 1) * examples_per_game
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
    print(f"Generation mode:            {generation_mode}")
    print(f"Bidding profile:            {bidding_profile}")
    print(f"Games:                      {games:,}")
    print(f"Examples:                   {arrays.example_count:,}")
    print(f"Non-trivial decisions:      {nontrivial:,}")
    print(f"Non-trivial share:          {nontrivial / arrays.example_count:.2%}")
    print(f"Observation version:        {collector.observation_version}")
    print(f"Feature count:              {arrays.features.shape[1]}")
    print(f"Contract metadata:          {arrays.contract_ids is not None}")
    print(f"Role metadata:              {arrays.role_ids is not None}")
    print(f"Seed:                       {seed}")
    print(f"Elapsed:                    {elapsed:.3f} s")
    print(f"Output:                     {output}")

    print("\nContract distribution")
    print("-" * 72)
    for game_type in GameType:
        count = contract_counts[game_type]
        print(f"{game_type.value:<12} {count:>7,}  ({count / games:>7.2%})")

    if arrays.role_ids is not None:
        print("\nRecorded decision distribution by specialist role")
        print("-" * 72)
        for role in CardPlayRole:
            count = int((arrays.role_ids == int(role)).sum())
            print(f"{role.key:<24} {count:>9,}  ({count / arrays.example_count:>7.2%})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate heuristic card-play demonstrations. "
            "heuristic-bid-random-card is the realistic card-play mode: all "
            "players bid heuristically, while only opponent card play is random."
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
    parser.add_argument(
        "--observation-version",
        choices=OBSERVATION_VERSIONS,
        default="v1",
        help="Card-play observation encoding to store (default: v1).",
    )
    parser.add_argument(
        "--bidding-profile",
        choices=BIDDING_PROFILES,
        default="conservative",
        help=(
            "Heuristic bidding threshold preset used by "
            "heuristic-bid-random-card (default: conservative)."
        ),
    )
    parser.add_argument(
        "--generation-mode",
        choices=GENERATION_MODES,
        default="all-heuristic",
        help=(
            "all-heuristic records all four heuristic players (32 examples/game); "
            "focal-random keeps the legacy fully-random opponents; "
            "heuristic-bid-random-card records one rotating heuristic focal "
            "player while all seats bid heuristically and opponent card play "
            "remains random (8 examples/game)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_generation(
        games=args.games,
        seed=args.seed,
        output=args.output,
        progress_every=args.progress_every,
        observation_version=args.observation_version,
        generation_mode=args.generation_mode,
        bidding_profile=args.bidding_profile,
    )


if __name__ == "__main__":
    main()
