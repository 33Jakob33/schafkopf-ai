from __future__ import annotations

import argparse
import random
import time
from collections import Counter
from pathlib import Path
from typing import Literal, cast

from schafkopf_ai.agents.agent import Agent
from schafkopf_ai.agents.heuristic_agent import HeuristicAgent, HeuristicConfig
from schafkopf_ai.agents.heuristic_bidding import (
    BIDDING_PROFILES,
    bidding_config_for_profile,
)
from schafkopf_ai.agents.heuristic_bidding_random_card_agent import (
    HeuristicBiddingRandomCardAgent,
)
from schafkopf_ai.game.game_type import GameType
from schafkopf_ai.game.rules import RAMSCH_RULES
from schafkopf_ai.runner.round_runner import RoundRunner
from schafkopf_ai.search.rollout import (
    CONTINUATION_PROFILES,
    MonteCarloActionEvaluator,
)
from schafkopf_ai.training.action_values import (
    ActionValueCollector,
    RecordingActionValueAgent,
    save_action_value_arrays,
)
from schafkopf_ai.training.card_play_roles import CardPlayRole

StateOpponentMode = Literal["heuristic", "heuristic-bid-random-card"]
STATE_OPPONENT_MODES: tuple[StateOpponentMode, ...] = (
    "heuristic",
    "heuristic-bid-random-card",
)


def _agents_for_game(
    *,
    collector: ActionValueCollector,
    game_index: int,
    seed: int,
    bidding_profile: str,
    state_opponent_mode: StateOpponentMode,
    samples_per_action: int,
    continuation_profile: str,
    epsilon: float,
) -> tuple[Agent, Agent, Agent, Agent]:
    focal_seat = game_index % 4
    bidding_config = bidding_config_for_profile(bidding_profile)

    evaluator = MonteCarloActionEvaluator(
        samples_per_action=samples_per_action,
        seed=seed + game_index * 1_000_003 + 70_001,
        continuation_profile=continuation_profile,
        epsilon=epsilon,
    )

    agents: list[Agent] = []
    for seat in range(4):
        if seat == focal_seat:
            agents.append(
                RecordingActionValueAgent(
                    collector=collector,
                    evaluator=evaluator,
                    game_id=game_index,
                    config=HeuristicConfig(bidding=bidding_config),
                )
            )
            continue

        if state_opponent_mode == "heuristic":
            agents.append(
                HeuristicAgent(config=HeuristicConfig(bidding=bidding_config))
            )
            continue

        opponent_seed = seed + game_index * 10_007 + seat * 101 + 50_000
        agents.append(
            HeuristicBiddingRandomCardAgent(
                rng=random.Random(opponent_seed),
                bidding_config=bidding_config,
            )
        )

    return cast(tuple[Agent, Agent, Agent, Agent], tuple(agents))


def run_generation(
    *,
    states: int,
    seed: int,
    output: Path,
    samples_per_action: int,
    continuation_profile: str,
    epsilon: float,
    bidding_profile: str,
    state_opponent_mode: StateOpponentMode,
    progress_every: int,
    max_games: int,
) -> None:
    if states <= 0:
        raise ValueError("states must be greater than zero.")
    if samples_per_action <= 0:
        raise ValueError("samples_per_action must be greater than zero.")
    if progress_every < 0:
        raise ValueError("progress_every cannot be negative.")
    if max_games <= 0:
        raise ValueError("max_games must be greater than zero.")
    if state_opponent_mode not in STATE_OPPONENT_MODES:
        raise ValueError(
            f"Unsupported state opponent mode {state_opponent_mode!r}; "
            f"expected one of {STATE_OPPONENT_MODES}."
        )

    collector = ActionValueCollector(states)
    contract_counts: Counter[GameType] = Counter()
    start = time.perf_counter()
    next_progress = progress_every if progress_every else states + 1

    game_index = 0
    while not collector.is_full and game_index < max_games:
        focal_seat = game_index % 4
        starting_player = (game_index // 4) % 4
        deal_seed = seed + game_index * 1_009 + 1

        agents = _agents_for_game(
            collector=collector,
            game_index=game_index,
            seed=seed,
            bidding_profile=bidding_profile,
            state_opponent_mode=state_opponent_mode,
            samples_per_action=samples_per_action,
            continuation_profile=continuation_profile,
            epsilon=epsilon,
        )
        result = RoundRunner(
            agents=agents,
            rules=RAMSCH_RULES,
            starting_player=starting_player,
            rng=random.Random(deal_seed),
        ).run()
        contract_counts[result.contract.game_type] += 1
        game_index += 1

        if collector.count >= next_progress:
            elapsed = time.perf_counter() - start
            state_rate = collector.count / elapsed
            print(
                f"{collector.count:>7,}/{states:,} states "
                f"| {game_index:>6,} games "
                f"| focal seat {focal_seat} "
                f"| {elapsed:>9.2f} s "
                f"| {state_rate:>6.2f} states/s"
            )
            while next_progress <= collector.count:
                next_progress += progress_every

    if not collector.is_full:
        raise RuntimeError(
            f"Stopped after {game_index:,} games with only "
            f"{collector.count:,}/{states:,} states. Increase --max-games."
        )

    arrays = collector.arrays()
    save_action_value_arrays(output, arrays)

    elapsed = time.perf_counter() - start
    disagreement = int((arrays.heuristic_actions != arrays.rollout_actions).sum())
    legal_q = arrays.q_values[arrays.legal_masks]
    legal_std = arrays.q_stds[arrays.legal_masks]

    print("\nAction-value dataset generated")
    print("=" * 88)
    print(f"States:                     {arrays.example_count:,}")
    print(f"Games simulated:            {game_index:,}")
    print(f"State opponent mode:        {state_opponent_mode}")
    print(f"Bidding profile:            {bidding_profile}")
    print(f"Continuation profile:       {continuation_profile}")
    print(f"Epsilon:                    {epsilon:.3f}")
    print(f"Samples per legal action:   {samples_per_action}")
    print(f"Feature count:              {arrays.features.shape[1]}")
    print(f"Mean legal actions/state:   {arrays.legal_counts.mean():.3f}")
    print(f"Mean Q target:              {float(legal_q.mean()):+.3f}")
    print(f"Mean Q target std:          {float(legal_std.mean()):.3f}")
    print(f"Teacher disagreements:      {disagreement:,}")
    print(f"Disagreement share:        {disagreement / arrays.example_count:.2%}")
    print(f"Seed:                       {seed}")
    print(f"Elapsed:                    {elapsed:.3f} s")
    print(f"Output:                     {output}")

    print("\nPlayed-game contract distribution")
    print("-" * 88)
    for game_type in GameType:
        count = contract_counts[game_type]
        share = count / game_index if game_index else 0.0
        print(f"{game_type.value:<12} {count:>7,}  ({share:>7.2%})")

    print("\nRecorded state distribution by role")
    print("-" * 88)
    for role in CardPlayRole:
        count = int((arrays.role_ids == int(role)).sum())
        print(f"{role.key:<24} {count:>9,}  ({count / arrays.example_count:>7.2%})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate V2 action-value supervision from Monte-Carlo rollouts. "
            "Only non-trivial focal-player states are stored."
        )
    )
    parser.add_argument("--states", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/action_values/action_values_v2_mixed_20k.npz"),
    )
    parser.add_argument("--samples-per-action", type=int, default=8)
    parser.add_argument(
        "--continuation-profile",
        choices=CONTINUATION_PROFILES,
        default="mixed",
        help=(
            "mixed uses 50% heuristic, 25% epsilon-heuristic and 25% random "
            "continuations for sample counts divisible by four."
        ),
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.10,
        help="Random-action probability inside epsilon-heuristic continuations.",
    )
    parser.add_argument(
        "--state-opponent-mode",
        choices=STATE_OPPONENT_MODES,
        default="heuristic-bid-random-card",
        help=(
            "Policy used by non-focal players while collecting states. "
            "All modes still use the configured heuristic bidding profile."
        ),
    )
    parser.add_argument(
        "--bidding-profile",
        choices=BIDDING_PROFILES,
        default="conservative",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=250,
        help="Print progress after roughly this many collected states; 0 disables.",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=100_000,
        help="Safety cap on complete games used to collect the requested states.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_generation(
        states=args.states,
        seed=args.seed,
        output=args.output,
        samples_per_action=args.samples_per_action,
        continuation_profile=args.continuation_profile,
        epsilon=args.epsilon,
        bidding_profile=args.bidding_profile,
        state_opponent_mode=args.state_opponent_mode,
        progress_every=args.progress_every,
        max_games=args.max_games,
    )


if __name__ == "__main__":
    main()
