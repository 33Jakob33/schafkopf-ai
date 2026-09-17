from __future__ import annotations

import argparse
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.optim import Adam

from schafkopf_ai.agents.agent import Agent
from schafkopf_ai.agents.heuristic_agent import HeuristicAgent
from schafkopf_ai.agents.ppo_card_play_agent import PPOCardPlayAgent
from schafkopf_ai.agents.random_agent import RandomAgent
from schafkopf_ai.game.rules import RAMSCH_RULES
from schafkopf_ai.runner.round_runner import RoundRunner
from schafkopf_ai.training.card_play_encoding import (
    ACTION_COUNT,
    OBSERVATION_FEATURE_SIZE,
)
from schafkopf_ai.training.ppo import (
    ActorCriticCardPlayNetwork,
    PPORolloutBuffer,
    initialize_from_behavior_checkpoint,
    ppo_update,
    save_ppo_checkpoint,
)


@dataclass(frozen=True, slots=True)
class RolloutMetrics:
    games: int
    wins: int
    payments: tuple[int, ...]
    samples: int
    random_opponents: int
    heuristic_opponents: int

    @property
    def mean_payment(self) -> float:
        return statistics.fmean(self.payments)

    @property
    def win_rate(self) -> float:
        return self.wins / self.games


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def collect_rollout(
    *,
    model: ActorCriticCardPlayNetwork,
    device: torch.device,
    games: int,
    global_game_offset: int,
    seed: int,
    random_opponent_probability: float,
    reward_scale: float,
) -> tuple[PPORolloutBuffer, RolloutMetrics]:
    if games <= 0:
        raise ValueError("rollout games must be greater than zero.")
    if not 0.0 <= random_opponent_probability <= 1.0:
        raise ValueError("random opponent probability must be between zero and one.")
    if reward_scale <= 0.0:
        raise ValueError("reward scale must be positive.")

    rollout = PPORolloutBuffer()
    learner = PPOCardPlayAgent(
        model=model,
        device=device,
        stochastic=True,
        record_trajectory=True,
    )

    payments: list[int] = []
    wins = 0
    random_opponents = 0
    heuristic_opponents = 0

    for local_game_index in range(games):
        game_index = global_game_offset + local_game_index
        focal_seat = game_index % 4
        starting_player = (game_index // 4) % 4
        selection_rng = random.Random(seed + game_index * 7_919 + 17)

        agents: list[Agent] = []
        for seat in range(4):
            if seat == focal_seat:
                agents.append(learner)
                continue

            if selection_rng.random() < random_opponent_probability:
                opponent_seed = seed + game_index * 10_007 + seat * 101 + 90_000
                agents.append(RandomAgent(rng=random.Random(opponent_seed)))
                random_opponents += 1
            else:
                agents.append(HeuristicAgent())
                heuristic_opponents += 1

        deal_seed = seed + game_index * 1_009 + 300_000
        result = RoundRunner(
            agents=agents,
            rules=RAMSCH_RULES,
            starting_player=starting_player,
            rng=random.Random(deal_seed),
        ).run()

        trajectory = learner.drain_trajectory()
        if len(trajectory) != 8:
            raise RuntimeError(
                "A focal player must make exactly eight card-play decisions: "
                f"got {len(trajectory)}."
            )

        payment = result.payments[focal_seat]
        payments.append(payment)
        wins += int(focal_seat in result.game_result.winner_players)
        rollout.add_episode(trajectory, payment * reward_scale)

    return rollout, RolloutMetrics(
        games=games,
        wins=wins,
        payments=tuple(payments),
        samples=rollout.sample_count,
        random_opponents=random_opponents,
        heuristic_opponents=heuristic_opponents,
    )


def train(
    *,
    init_checkpoint: Path,
    output: Path,
    best_output: Path,
    iterations: int,
    rollout_games: int,
    ppo_epochs: int,
    minibatch_size: int,
    learning_rate: float,
    clip_epsilon: float,
    value_coefficient: float,
    entropy_coefficient: float,
    max_grad_norm: float,
    random_opponent_probability: float,
    reward_scale: float,
    seed: int,
    device_name: str,
    hidden_sizes: tuple[int, int],
) -> None:
    if iterations <= 0:
        raise ValueError("iterations must be greater than zero.")
    if learning_rate <= 0.0:
        raise ValueError("learning rate must be positive.")

    random.seed(seed)
    torch.manual_seed(seed)
    device = resolve_device(device_name)

    model = ActorCriticCardPlayNetwork(
        input_size=OBSERVATION_FEATURE_SIZE,
        hidden_sizes=hidden_sizes,
        action_count=ACTION_COUNT,
    ).to(device)
    initialize_from_behavior_checkpoint(
        model=model,
        checkpoint_path=init_checkpoint,
        device=device,
    )

    optimizer = Adam(model.parameters(), lr=learning_rate)
    permutation_generator = torch.Generator()
    permutation_generator.manual_seed(seed + 500_000)

    best_mean_payment = float("-inf")
    total_games = 0
    start = time.perf_counter()

    print("PPO card-play training")
    print("=" * 96)
    print(f"Initialized from:           {init_checkpoint}")
    print(f"Device:                     {device}")
    print(f"Iterations:                 {iterations}")
    print(f"Rollout games / iteration:  {rollout_games:,}")
    print(f"PPO epochs:                 {ppo_epochs}")
    print(f"Minibatch size:             {minibatch_size}")
    print(f"Learning rate:              {learning_rate:g}")
    print(f"Clip epsilon:               {clip_epsilon:g}")
    print(f"Entropy coefficient:        {entropy_coefficient:g}")
    print(f"Reward scale:               {reward_scale:g}")
    print(
        "Opponent pool:              "
        f"{random_opponent_probability:.0%} RandomAgent / "
        f"{1.0 - random_opponent_probability:.0%} HeuristicAgent"
    )
    print()

    for iteration in range(1, iterations + 1):
        rollout, rollout_metrics = collect_rollout(
            model=model,
            device=device,
            games=rollout_games,
            global_game_offset=total_games,
            seed=seed,
            random_opponent_probability=random_opponent_probability,
            reward_scale=reward_scale,
        )
        total_games += rollout_games

        update_metrics = ppo_update(
            model=model,
            optimizer=optimizer,
            rollout=rollout,
            device=device,
            epochs=ppo_epochs,
            minibatch_size=minibatch_size,
            clip_epsilon=clip_epsilon,
            value_coefficient=value_coefficient,
            entropy_coefficient=entropy_coefficient,
            max_grad_norm=max_grad_norm,
            generator=permutation_generator,
        )

        mean_payment = rollout_metrics.mean_payment
        save_ppo_checkpoint(
            path=output,
            model=model,
            iteration=iteration,
            mean_payment=mean_payment,
            seed=seed,
            initialized_from=init_checkpoint,
        )

        marker = ""
        if mean_payment > best_mean_payment:
            best_mean_payment = mean_payment
            save_ppo_checkpoint(
                path=best_output,
                model=model,
                iteration=iteration,
                mean_payment=mean_payment,
                seed=seed,
                initialized_from=init_checkpoint,
            )
            marker = "  [best]"

        elapsed = time.perf_counter() - start
        print(
            f"Iter {iteration:>3}/{iterations} "
            f"| games {total_games:>7,} "
            f"| samples {update_metrics.samples:>6,} "
            f"| pay {mean_payment:>+7.3f} "
            f"| win {rollout_metrics.win_rate:>6.2%} "
            f"| pi {update_metrics.policy_loss:>+8.4f} "
            f"| v {update_metrics.value_loss:>8.4f} "
            f"| ent {update_metrics.entropy:>6.3f} "
            f"| KL {update_metrics.approximate_kl:>+7.4f} "
            f"| clip {update_metrics.clip_fraction:>6.2%} "
            f"| {elapsed:>7.1f}s{marker}"
        )

    print("\nPPO training complete")
    print("-" * 96)
    print(f"Total games:                {total_games:,}")
    print(f"Best rollout mean payment:  {best_mean_payment:+.3f}")
    print(f"Latest checkpoint:          {output}")
    print(f"Best checkpoint:            {best_output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a PPO card-play policy initialized from a BC/DAgger checkpoint. "
            "Bidding remains heuristic-controlled."
        )
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=Path("models/behavior_cloning/card_play_dagger.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/ppo/card_play_ppo.pt"),
    )
    parser.add_argument(
        "--best-output",
        type=Path,
        default=Path("models/ppo/card_play_ppo_best.pt"),
    )
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--rollout-games", type=int, default=1_000)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--random-opponent-probability", type=float, default=0.25)
    parser.add_argument("--reward-scale", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hidden-1", type=int, default=512)
    parser.add_argument("--hidden-2", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(
        init_checkpoint=args.init_checkpoint,
        output=args.output,
        best_output=args.best_output,
        iterations=args.iterations,
        rollout_games=args.rollout_games,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        learning_rate=args.learning_rate,
        clip_epsilon=args.clip_epsilon,
        value_coefficient=args.value_coefficient,
        entropy_coefficient=args.entropy_coefficient,
        max_grad_norm=args.max_grad_norm,
        random_opponent_probability=args.random_opponent_probability,
        reward_scale=args.reward_scale,
        seed=args.seed,
        device_name=args.device,
        hidden_sizes=(args.hidden_1, args.hidden_2),
    )


if __name__ == "__main__":
    main()
