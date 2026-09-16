from __future__ import annotations

import argparse
import random
import time
from collections import Counter
from pathlib import Path

import torch

from schafkopf_ai.agents.agent import Agent
from schafkopf_ai.agents.heuristic_agent import HeuristicAgent
from schafkopf_ai.agents.neural_card_play_agent import NeuralCardPlayAgent
from schafkopf_ai.agents.random_agent import RandomAgent
from schafkopf_ai.game.game_type import GameType
from schafkopf_ai.game.rules import RAMSCH_RULES
from schafkopf_ai.runner.round_runner import RoundRunner
from schafkopf_ai.training.behavior_cloning import (
    load_behavior_cloning_arrays,
    save_behavior_cloning_arrays,
)
from schafkopf_ai.training.dagger import (
    DAggerRecordingAgent,
    aggregate_dagger_arrays,
)
from schafkopf_ai.training.demonstrations import BehaviorCloningCollector

CARD_PLAYS_PER_FOCAL_GAME = 8


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_generation(
    *,
    base_dataset: Path,
    checkpoint: Path,
    output: Path,
    games: int,
    seed: int,
    progress_every: int,
    device_name: str,
) -> None:
    if games <= 0:
        raise ValueError("games must be greater than zero.")
    if progress_every < 0:
        raise ValueError("progress_every cannot be negative.")

    device = resolve_device(device_name)
    base_arrays = load_behavior_cloning_arrays(base_dataset)
    policy_agent = NeuralCardPlayAgent.from_checkpoint(
        checkpoint,
        device=device,
    )
    collector = BehaviorCloningCollector(games * CARD_PLAYS_PER_FOCAL_GAME)

    contract_counts: Counter[GameType] = Counter()
    total_disagreements = 0
    total_nontrivial = 0
    total_nontrivial_disagreements = 0

    start = time.perf_counter()

    for game_index in range(games):
        focal_seat = game_index % 4
        starting_player = (game_index // 4) % 4

        dagger_agent = DAggerRecordingAgent(
            policy_agent=policy_agent,
            expert_agent=HeuristicAgent(),
            collector=collector,
            game_id=game_index,
        )

        agents: list[Agent] = []
        for seat in range(4):
            if seat == focal_seat:
                agents.append(dagger_agent)
            else:
                opponent_seed = seed + 100_000 + game_index * 10 + seat
                agents.append(RandomAgent(rng=random.Random(opponent_seed)))

        deal_rng = random.Random(seed + game_index * 1_009)
        result = RoundRunner(
            agents=agents,
            rules=RAMSCH_RULES,
            starting_player=starting_player,
            rng=deal_rng,
        ).run()

        if dagger_agent.decisions != CARD_PLAYS_PER_FOCAL_GAME:
            raise RuntimeError(
                "DAgger focal agent did not make exactly eight card decisions: "
                f"{dagger_agent.decisions}."
            )

        contract_counts[result.contract.game_type] += 1
        total_disagreements += dagger_agent.disagreements
        total_nontrivial += dagger_agent.nontrivial_decisions
        total_nontrivial_disagreements += dagger_agent.nontrivial_disagreements

        completed = game_index + 1
        if progress_every and completed % progress_every == 0:
            elapsed = time.perf_counter() - start
            disagreement_rate = total_disagreements / collector.count
            nontrivial_rate = (
                total_nontrivial_disagreements / total_nontrivial
                if total_nontrivial
                else 0.0
            )
            print(
                f"{completed:>7,}/{games:,} games "
                f"| {elapsed:>8.2f} s "
                f"| {completed / elapsed:>7.1f} games/s "
                f"| expert disagreement {disagreement_rate:>7.2%} "
                f"| nontrivial {nontrivial_rate:>7.2%}"
            )

    additions = collector.arrays()
    aggregated = aggregate_dagger_arrays(base_arrays, additions)
    save_behavior_cloning_arrays(output, aggregated)

    elapsed = time.perf_counter() - start
    disagreement_rate = total_disagreements / additions.example_count
    nontrivial_rate = (
        total_nontrivial_disagreements / total_nontrivial if total_nontrivial else 0.0
    )

    print("\nDAgger dataset generated")
    print("=" * 80)
    print(f"Base dataset:               {base_dataset}")
    print(f"Policy checkpoint:          {checkpoint}")
    print(f"Device:                     {device}")
    print(f"Games:                      {games:,}")
    print(f"New expert labels:          {additions.example_count:,}")
    print(f"Base examples:              {base_arrays.example_count:,}")
    print(f"Aggregated examples:        {aggregated.example_count:,}")
    print(f"Policy/expert disagreement: {disagreement_rate:.2%}")
    print(f"Non-trivial decisions:      {total_nontrivial:,}")
    print(f"Non-trivial disagreement:   {nontrivial_rate:.2%}")
    print(f"Seed:                       {seed}")
    print(f"Elapsed:                    {elapsed:.3f} s")
    print(f"Output:                     {output}")

    print("\nContract distribution during learner rollouts")
    print("-" * 80)
    for game_type in GameType:
        count = contract_counts[game_type]
        print(f"{game_type.value:<12} {count:>7,}  ({count / games:>7.2%})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect DAgger labels on states reached by the learned card-play "
            "policy against three RandomAgents and append them to the dataset."
        )
    )
    parser.add_argument(
        "--base-dataset",
        type=Path,
        default=Path("data/behavior_cloning/heuristic_card_play.npz"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/behavior_cloning/card_play_mlp.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/behavior_cloning/dagger_card_play.npz"),
    )
    parser.add_argument(
        "--games",
        type=int,
        default=2_000,
        help="Learner rollouts to label (default: 2000).",
    )
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=250,
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch device such as cpu/cuda, or auto (default).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_generation(
        base_dataset=args.base_dataset,
        checkpoint=args.checkpoint,
        output=args.output,
        games=args.games,
        seed=args.seed,
        progress_every=args.progress_every,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
