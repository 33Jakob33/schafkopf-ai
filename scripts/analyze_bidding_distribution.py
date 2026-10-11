from __future__ import annotations

import argparse
import random
import time
from collections import Counter

from schafkopf_ai.agents.heuristic_agent import HeuristicAgent, HeuristicConfig
from schafkopf_ai.agents.heuristic_bidding import (
    BIDDING_PROFILES,
    BiddingHeuristicConfig,
    bidding_config_for_profile,
)
from schafkopf_ai.game.bidding import BiddingState
from schafkopf_ai.game.deck import Deck
from schafkopf_ai.game.game_type import GameType
from schafkopf_ai.game.rules import RAMSCH_RULES
from schafkopf_ai.runner.bidding_runner import BiddingRunner


def _config_from_args(args: argparse.Namespace) -> BiddingHeuristicConfig:
    base = bidding_config_for_profile(args.profile)
    return BiddingHeuristicConfig(
        sauspiel_min_confidence=(
            base.sauspiel_min_confidence
            if args.sauspiel_min is None
            else args.sauspiel_min
        ),
        wenz_min_confidence=(
            base.wenz_min_confidence if args.wenz_min is None else args.wenz_min
        ),
        geier_min_confidence=base.geier_min_confidence,
        solo_min_confidence=(
            base.solo_min_confidence if args.solo_min is None else args.solo_min
        ),
        solo_margin=base.solo_margin if args.solo_margin is None else args.solo_margin,
    )


def analyze(
    *,
    games: int,
    seed: int,
    config: BiddingHeuristicConfig,
) -> None:
    if games <= 0:
        raise ValueError("games must be greater than zero.")

    rng = random.Random(seed)
    agents = tuple(
        HeuristicAgent(config=HeuristicConfig(bidding=config)) for _ in range(4)
    )
    counts: Counter[GameType] = Counter()
    declarer_counts = [0, 0, 0, 0]
    start = time.perf_counter()

    for game_index in range(games):
        deck = Deck()
        deck.shuffle(rng)
        hands = deck.deal()
        state = BiddingState(
            hands=hands,
            rules=RAMSCH_RULES,
            starting_player=game_index % 4,
        )
        result = BiddingRunner(state=state, agents=agents).run()
        if result.contract is None:
            raise RuntimeError("RAMSCH_RULES bidding unexpectedly requested a redeal.")

        counts[result.contract.game_type] += 1
        if result.contract.declarer is not None:
            declarer_counts[result.contract.declarer] += 1

    elapsed = time.perf_counter() - start

    print("Bidding distribution analysis")
    print("=" * 72)
    print(f"Games:                      {games:,}")
    print(f"Seed:                       {seed}")
    print(f"Elapsed:                    {elapsed:.3f} s")
    print(f"Sauspiel min confidence:    {config.sauspiel_min_confidence:.3f}")
    print(f"Wenz min confidence:        {config.wenz_min_confidence:.3f}")
    print(f"Solo min confidence:        {config.solo_min_confidence:.3f}")
    print(f"Solo margin:                {config.solo_margin:.3f}")

    print("\nContract distribution")
    print("-" * 72)
    for game_type in GameType:
        count = counts[game_type]
        print(f"{game_type.value:<12} {count:>9,}  ({count / games:>7.2%})")

    print("\nDeclarer seat distribution")
    print("-" * 72)
    bid_games = sum(declarer_counts)
    for seat, count in enumerate(declarer_counts):
        share = count / bid_games if bid_games else 0.0
        print(f"seat {seat:<7} {count:>9,}  ({share:>7.2%})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Quickly inspect contract frequencies from heuristic bidding only. "
            "No card-play simulation or neural inference is performed."
        )
    )
    parser.add_argument("--games", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--profile",
        choices=BIDDING_PROFILES,
        default="conservative",
    )
    parser.add_argument("--sauspiel-min", type=float, default=None)
    parser.add_argument("--wenz-min", type=float, default=None)
    parser.add_argument("--solo-min", type=float, default=None)
    parser.add_argument("--solo-margin", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyze(
        games=args.games,
        seed=args.seed,
        config=_config_from_args(args),
    )


if __name__ == "__main__":
    main()
