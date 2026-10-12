from __future__ import annotations

import random
import statistics
from dataclasses import dataclass
from typing import Literal, cast

from schafkopf_ai.agents.agent import Agent
from schafkopf_ai.agents.heuristic_agent import HeuristicAgent
from schafkopf_ai.agents.random_agent import RandomAgent
from schafkopf_ai.game.bidding import BiddingAction, BiddingObservation
from schafkopf_ai.game.card import Card
from schafkopf_ai.game.game_result import result_from_game_state
from schafkopf_ai.game.observation import PlayerObservation
from schafkopf_ai.game.settlement import settlement_from_game_state
from schafkopf_ai.runner.game_runner import GameRunner

from .hidden_state import Determinization, HiddenStateSampler

ContinuationKind = Literal["heuristic", "epsilon-heuristic", "random"]
ContinuationProfile = Literal["heuristic", "mixed"]
CONTINUATION_PROFILES: tuple[ContinuationProfile, ...] = ("heuristic", "mixed")

# With x8 this becomes exactly 4 heuristic, 2 epsilon-heuristic, 2 random.
_MIXED_SCHEDULE: tuple[ContinuationKind, ...] = (
    "heuristic",
    "heuristic",
    "epsilon-heuristic",
    "random",
)


@dataclass(frozen=True, slots=True)
class ActionValueEstimate:
    """Monte-Carlo estimate of focal-player payment for one legal card."""

    card: Card
    mean_payment: float
    std_payment: float
    sample_count: int
    payments: tuple[int, ...]


class EpsilonHeuristicAgent(Agent):
    """Heuristic card play with occasional uniformly random legal exploration."""

    def __init__(
        self,
        *,
        rng: random.Random,
        epsilon: float = 0.10,
    ) -> None:
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be between zero and one.")
        self.rng = rng
        self.epsilon = epsilon
        self.heuristic = HeuristicAgent()

    def choose_bidding_action(
        self,
        observation: BiddingObservation,
        legal_actions: tuple[BiddingAction, ...],
    ) -> BiddingAction:
        return self.heuristic.choose_bidding_action(observation, legal_actions)

    def choose_card(
        self,
        observation: PlayerObservation,
        legal_cards: tuple[Card, ...],
    ) -> Card:
        if not legal_cards:
            raise ValueError("No legal cards available.")
        if len(legal_cards) == 1:
            return legal_cards[0]
        if self.rng.random() < self.epsilon:
            return self.rng.choice(legal_cards)
        return self.heuristic.choose_card(observation, legal_cards)


class MonteCarloActionEvaluator:
    """
    Estimate Q(s,a) by hidden-state sampling and complete-game continuations.

    The same sampled determinizations and continuation random seeds are reused
    for every legal candidate action. This common-random-numbers design reduces
    comparison variance.

    Profiles:
    - heuristic: all continuations use the deterministic heuristic baseline.
    - mixed: 50% heuristic, 25% epsilon-heuristic, 25% random continuations
      for sample counts divisible by four (for x8: 4/2/2 exactly).
    """

    def __init__(
        self,
        *,
        samples_per_action: int = 8,
        seed: int = 0,
        max_sampling_attempts: int = 5_000,
        continuation_profile: str = "heuristic",
        epsilon: float = 0.10,
    ) -> None:
        if samples_per_action <= 0:
            raise ValueError("samples_per_action must be positive.")
        if continuation_profile not in CONTINUATION_PROFILES:
            raise ValueError(
                f"Unsupported continuation profile {continuation_profile!r}; "
                f"expected one of {CONTINUATION_PROFILES}."
            )
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be between zero and one.")

        self.samples_per_action = samples_per_action
        self.seed = seed
        self.continuation_profile = cast(ContinuationProfile, continuation_profile)
        self.epsilon = epsilon
        self.sampler = HiddenStateSampler(
            rng=random.Random(seed),
            max_attempts_per_sample=max_sampling_attempts,
        )
        self._evaluation_index = 0

    def evaluate(
        self,
        observation: PlayerObservation,
        legal_cards: tuple[Card, ...],
    ) -> tuple[ActionValueEstimate, ...]:
        if not legal_cards:
            raise ValueError("No legal cards available for rollout evaluation.")

        determinizations = self.sampler.sample_many(
            observation,
            self.samples_per_action,
        )
        continuation_kinds = self._continuation_kinds(self.samples_per_action)
        evaluation_index = self._evaluation_index
        self._evaluation_index += 1

        payments_by_card: dict[Card, list[int]] = {card: [] for card in legal_cards}

        for sample_index, determinization in enumerate(determinizations):
            continuation_seed = (
                self.seed
                + evaluation_index * 1_000_003
                + sample_index * 10_007
                + 97
            )
            continuation_kind = continuation_kinds[sample_index]

            for card in legal_cards:
                payments_by_card[card].append(
                    self._rollout_payment(
                        determinization=determinization,
                        observation=observation,
                        first_card=card,
                        continuation_kind=continuation_kind,
                        continuation_seed=continuation_seed,
                        epsilon=self.epsilon,
                    )
                )

        estimates: list[ActionValueEstimate] = []
        for card in legal_cards:
            payments = tuple(payments_by_card[card])
            estimates.append(
                ActionValueEstimate(
                    card=card,
                    mean_payment=statistics.fmean(payments),
                    std_payment=statistics.stdev(payments) if len(payments) > 1 else 0.0,
                    sample_count=len(payments),
                    payments=payments,
                )
            )

        return tuple(estimates)

    def _continuation_kinds(self, count: int) -> tuple[ContinuationKind, ...]:
        if self.continuation_profile == "heuristic":
            return tuple("heuristic" for _ in range(count))

        return tuple(_MIXED_SCHEDULE[index % len(_MIXED_SCHEDULE)] for index in range(count))

    @staticmethod
    def _rollout_payment(
        *,
        determinization: Determinization,
        observation: PlayerObservation,
        first_card: Card,
        continuation_kind: ContinuationKind,
        continuation_seed: int,
        epsilon: float,
    ) -> int:
        state = determinization.build_state(observation)
        focal_player = observation.player_index

        if first_card not in state.legal_moves(focal_player):
            raise RuntimeError(
                "Candidate rollout card became illegal after reconstruction."
            )

        state.play_card(focal_player, first_card)
        agents = MonteCarloActionEvaluator._continuation_agents(
            kind=continuation_kind,
            seed=continuation_seed,
            epsilon=epsilon,
        )
        GameRunner(state=state, agents=agents).run()

        result = result_from_game_state(state)
        settlement = settlement_from_game_state(state, result)
        return settlement.payments[focal_player]

    @staticmethod
    def _continuation_agents(
        *,
        kind: ContinuationKind,
        seed: int,
        epsilon: float,
    ) -> tuple[Agent, Agent, Agent, Agent]:
        if kind == "heuristic":
            return (
                HeuristicAgent(),
                HeuristicAgent(),
                HeuristicAgent(),
                HeuristicAgent(),
            )

        if kind == "epsilon-heuristic":
            return tuple(
                EpsilonHeuristicAgent(
                    rng=random.Random(seed + seat * 101),
                    epsilon=epsilon,
                )
                for seat in range(4)
            )  # type: ignore[return-value]

        return tuple(
            RandomAgent(rng=random.Random(seed + seat * 101)) for seat in range(4)
        )  # type: ignore[return-value]
