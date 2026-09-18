from __future__ import annotations

import argparse
import copy
import random
import statistics
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.optim import Adam

from schafkopf_ai.agents.agent import Agent
from schafkopf_ai.agents.heuristic_agent import HeuristicAgent
from schafkopf_ai.agents.ppo_card_play_agent import PPOCardPlayAgent
from schafkopf_ai.agents.random_agent import RandomAgent
from schafkopf_ai.game.game_type import GameType
from schafkopf_ai.game.rules import RAMSCH_RULES
from schafkopf_ai.runner.round_runner import RoundResult, RoundRunner
from schafkopf_ai.training.card_play_encoding import (
    ACTION_COUNT,
    OBSERVATION_FEATURE_SIZE,
)
from schafkopf_ai.training.ppo import (
    ActorCriticCardPlayNetwork,
    PPORolloutBuffer,
    initialize_from_checkpoint,
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
    self_play_opponents: int

    @property
    def mean_payment(self) -> float:
        return statistics.fmean(self.payments)

    @property
    def win_rate(self) -> float:
        return self.wins / self.games


@dataclass(frozen=True, slots=True)
class ValidationMetrics:
    games: int
    policy_payments: tuple[int, ...]
    heuristic_payments: tuple[int, ...]
    deltas: tuple[int, ...]
    policy_wins: int
    heuristic_wins: int

    @property
    def mean_policy_payment(self) -> float:
        return statistics.fmean(self.policy_payments)

    @property
    def mean_heuristic_payment(self) -> float:
        return statistics.fmean(self.heuristic_payments)

    @property
    def mean_delta(self) -> float:
        return statistics.fmean(self.deltas)

    @property
    def policy_win_rate(self) -> float:
        return self.policy_wins / self.games


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _validate_round(result: RoundResult) -> None:
    if not result.game_state.is_complete:
        raise RuntimeError("Validation round finished with an incomplete GameState.")
    if len(result.game_state.completed_tricks) != 8:
        raise RuntimeError("Validation round does not contain eight tricks.")
    if sum(result.player_points) != 120:
        raise RuntimeError(f"Validation Augen do not sum to 120: {result.player_points}.")
    if sum(result.payments) != 0:
        raise RuntimeError(f"Validation payments do not sum to zero: {result.payments}.")


def _run_validation_variant(
    *,
    focal_agent: Agent,
    game_index: int,
    master_seed: int,
) -> RoundResult:
    focal_seat = game_index % 4
    starting_player = (game_index // 4) % 4
    deal_seed = master_seed + game_index * 1_009 + 1

    agents: list[Agent] = []
    for seat in range(4):
        if seat == focal_seat:
            agents.append(focal_agent)
        else:
            opponent_seed = master_seed + game_index * 10_007 + seat * 101 + 50_000
            agents.append(RandomAgent(rng=random.Random(opponent_seed)))

    result = RoundRunner(
        agents=agents,
        rules=RAMSCH_RULES,
        starting_player=starting_player,
        rng=random.Random(deal_seed),
    ).run()
    _validate_round(result)
    return result


def validate_policy(
    *,
    model: ActorCriticCardPlayNetwork,
    device: torch.device,
    games: int,
    seed: int,
) -> ValidationMetrics:
    """
    Evaluate a deterministic PPO policy on a fixed paired hold-out benchmark.

    The candidate policy and HeuristicAgent receive the same deal, starting seat,
    bidding setup, and RandomAgent seeds. These games use a separate validation
    seed and are never used for PPO updates.
    """
    if games <= 0:
        raise ValueError("validation games must be greater than zero.")

    policy = PPOCardPlayAgent(
        model=model,
        device=device,
        stochastic=False,
        record_trajectory=False,
    )
    heuristic = HeuristicAgent()

    policy_payments: list[int] = []
    heuristic_payments: list[int] = []
    deltas: list[int] = []
    policy_wins = 0
    heuristic_wins = 0

    for game_index in range(games):
        focal_seat = game_index % 4

        heuristic_result = _run_validation_variant(
            focal_agent=heuristic,
            game_index=game_index,
            master_seed=seed,
        )
        policy_result = _run_validation_variant(
            focal_agent=policy,
            game_index=game_index,
            master_seed=seed,
        )

        if heuristic_result.contract != policy_result.contract:
            raise RuntimeError(
                "Validation variants produced different contracts despite "
                "identical heuristic bidding."
            )

        heuristic_payment = heuristic_result.payments[focal_seat]
        policy_payment = policy_result.payments[focal_seat]

        heuristic_payments.append(heuristic_payment)
        policy_payments.append(policy_payment)
        deltas.append(policy_payment - heuristic_payment)
        heuristic_wins += int(
            focal_seat in heuristic_result.game_result.winner_players
        )
        policy_wins += int(focal_seat in policy_result.game_result.winner_players)

    return ValidationMetrics(
        games=games,
        policy_payments=tuple(policy_payments),
        heuristic_payments=tuple(heuristic_payments),
        deltas=tuple(deltas),
        policy_wins=policy_wins,
        heuristic_wins=heuristic_wins,
    )


def _snapshot_agent(
    *,
    model: ActorCriticCardPlayNetwork,
    device: torch.device,
) -> PPOCardPlayAgent:
    snapshot_model = copy.deepcopy(model).to(device)
    snapshot_model.eval()
    return PPOCardPlayAgent(
        model=snapshot_model,
        device=device,
        stochastic=False,
        record_trajectory=False,
    )


def collect_rollout(
    *,
    model: ActorCriticCardPlayNetwork,
    device: torch.device,
    games: int,
    global_game_offset: int,
    seed: int,
    random_opponent_probability: float,
    self_play_opponent_probability: float,
    self_play_agents: tuple[PPOCardPlayAgent, ...],
    reward_scale: float,
) -> tuple[PPORolloutBuffer, RolloutMetrics]:
    if games <= 0:
        raise ValueError("rollout games must be greater than zero.")
    if not 0.0 <= random_opponent_probability <= 1.0:
        raise ValueError("random opponent probability must be between zero and one.")
    if not 0.0 <= self_play_opponent_probability <= 1.0:
        raise ValueError("self-play opponent probability must be between zero and one.")
    if random_opponent_probability + self_play_opponent_probability > 1.0:
        raise ValueError(
            "random and self-play opponent probabilities cannot sum above one."
        )
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
    self_play_opponents = 0

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

            draw = selection_rng.random()
            if draw < random_opponent_probability:
                opponent_seed = seed + game_index * 10_007 + seat * 101 + 90_000
                agents.append(RandomAgent(rng=random.Random(opponent_seed)))
                random_opponents += 1
            elif (
                draw
                < random_opponent_probability + self_play_opponent_probability
                and self_play_agents
            ):
                agents.append(selection_rng.choice(self_play_agents))
                self_play_opponents += 1
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
        self_play_opponents=self_play_opponents,
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
    self_play_opponent_probability: float,
    opponent_pool_size: int,
    snapshot_every: int,
    reward_scale: float,
    validation_every: int,
    validation_games: int,
    validation_seed: int,
    seed: int,
    device_name: str,
    hidden_sizes: tuple[int, int],
) -> None:
    if iterations <= 0:
        raise ValueError("iterations must be greater than zero.")
    if learning_rate <= 0.0:
        raise ValueError("learning rate must be positive.")
    if validation_every <= 0:
        raise ValueError("validation_every must be greater than zero.")
    if validation_games <= 0:
        raise ValueError("validation_games must be greater than zero.")
    if snapshot_every <= 0:
        raise ValueError("snapshot_every must be greater than zero.")
    if opponent_pool_size < 0:
        raise ValueError("opponent_pool_size cannot be negative.")
    if random_opponent_probability + self_play_opponent_probability > 1.0:
        raise ValueError(
            "random and self-play opponent probabilities cannot sum above one."
        )

    random.seed(seed)
    torch.manual_seed(seed)
    device = resolve_device(device_name)

    model = ActorCriticCardPlayNetwork(
        input_size=OBSERVATION_FEATURE_SIZE,
        hidden_sizes=hidden_sizes,
        action_count=ACTION_COUNT,
    ).to(device)
    initialization_kind = initialize_from_checkpoint(
        model=model,
        checkpoint_path=init_checkpoint,
        device=device,
    )

    optimizer = Adam(model.parameters(), lr=learning_rate)
    permutation_generator = torch.Generator()
    permutation_generator.manual_seed(seed + 500_000)

    opponent_pool: deque[PPOCardPlayAgent] = deque(maxlen=opponent_pool_size or None)
    if self_play_opponent_probability > 0.0 and opponent_pool_size > 0:
        opponent_pool.append(_snapshot_agent(model=model, device=device))

    total_games = 0
    start = time.perf_counter()

    heuristic_probability = (
        1.0 - random_opponent_probability - self_play_opponent_probability
    )

    print("PPO card-play training with fixed validation and opponent pool")
    print("=" * 104)
    print(f"Initialized from:           {init_checkpoint} ({initialization_kind})")
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
        "Opponent mix:               "
        f"{random_opponent_probability:.0%} Random / "
        f"{heuristic_probability:.0%} Heuristic / "
        f"{self_play_opponent_probability:.0%} snapshot PPO"
    )
    print(f"Snapshot pool size:          {opponent_pool_size}")
    print(f"Snapshot every:             {snapshot_every} iterations")
    print(f"Validation every:           {validation_every} iterations")
    print(f"Validation games:           {validation_games:,}")
    print(f"Validation seed:            {validation_seed}")
    print()

    initial_validation = validate_policy(
        model=model,
        device=device,
        games=validation_games,
        seed=validation_seed,
    )
    best_validation_delta = initial_validation.mean_delta
    best_validation_iteration = 0
    save_ppo_checkpoint(
        path=best_output,
        model=model,
        iteration=0,
        mean_payment=initial_validation.mean_policy_payment,
        validation_delta=initial_validation.mean_delta,
        seed=seed,
        initialized_from=init_checkpoint,
    )

    print(
        "Initial validation"
        f" | PPO {initial_validation.mean_policy_payment:+.3f}"
        f" | heuristic {initial_validation.mean_heuristic_payment:+.3f}"
        f" | delta {initial_validation.mean_delta:+.3f}"
        f" | win {initial_validation.policy_win_rate:.2%}"
        "  [val-best]"
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
            self_play_opponent_probability=self_play_opponent_probability,
            self_play_agents=tuple(opponent_pool),
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
            validation_delta=None,
            seed=seed,
            initialized_from=init_checkpoint,
        )

        if (
            self_play_opponent_probability > 0.0
            and opponent_pool_size > 0
            and iteration % snapshot_every == 0
        ):
            opponent_pool.append(_snapshot_agent(model=model, device=device))

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
            f"| pool {len(opponent_pool):>2} "
            f"| {elapsed:>7.1f}s"
        )

        if iteration % validation_every == 0 or iteration == iterations:
            validation = validate_policy(
                model=model,
                device=device,
                games=validation_games,
                seed=validation_seed,
            )

            marker = ""
            if validation.mean_delta > best_validation_delta:
                best_validation_delta = validation.mean_delta
                best_validation_iteration = iteration
                save_ppo_checkpoint(
                    path=best_output,
                    model=model,
                    iteration=iteration,
                    mean_payment=validation.mean_policy_payment,
                    validation_delta=validation.mean_delta,
                    seed=seed,
                    initialized_from=init_checkpoint,
                )
                marker = "  [val-best]"

            print(
                "  validation"
                f" | PPO {validation.mean_policy_payment:+.3f}"
                f" | heuristic {validation.mean_heuristic_payment:+.3f}"
                f" | delta {validation.mean_delta:+.3f}"
                f" | win {validation.policy_win_rate:.2%}"
                f"{marker}"
            )

    print("\nPPO training complete")
    print("-" * 104)
    print(f"Total training games:       {total_games:,}")
    print(f"Best validation delta:      {best_validation_delta:+.3f}")
    print(f"Best validation iteration:  {best_validation_iteration}")
    print(f"Latest checkpoint:          {output}")
    print(f"Best checkpoint:            {best_output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a PPO card-play policy with fixed paired validation and "
            "snapshot-opponent self-play. Bidding remains heuristic-controlled."
        )
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=Path("models/behavior_cloning/card_play_dagger.pt"),
        help="BC/DAgger or PPO checkpoint used to initialize training.",
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
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--random-opponent-probability", type=float, default=0.15)
    parser.add_argument(
        "--self-play-opponent-probability",
        type=float,
        default=0.35,
    )
    parser.add_argument("--opponent-pool-size", type=int, default=8)
    parser.add_argument("--snapshot-every", type=int, default=5)
    parser.add_argument("--reward-scale", type=float, default=0.01)
    parser.add_argument("--validation-every", type=int, default=5)
    parser.add_argument("--validation-games", type=int, default=2_000)
    parser.add_argument("--validation-seed", type=int, default=20_260_918)
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
        self_play_opponent_probability=args.self_play_opponent_probability,
        opponent_pool_size=args.opponent_pool_size,
        snapshot_every=args.snapshot_every,
        reward_scale=args.reward_scale,
        validation_every=args.validation_every,
        validation_games=args.validation_games,
        validation_seed=args.validation_seed,
        seed=args.seed,
        device_name=args.device,
        hidden_sizes=(args.hidden_1, args.hidden_2),
    )


if __name__ == "__main__":
    main()
