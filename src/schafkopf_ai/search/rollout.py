from __future__ import annotations

import random
import statistics
from dataclasses import dataclass

from schafkopf_ai.agents.heuristic_agent import HeuristicAgent
from schafkopf_ai.game.card import Card
from schafkopf_ai.game.game_result import result_from_game_state
from schafkopf_ai.game.observation import PlayerObservation
from schafkopf_ai.game.settlement import settlement_from_game_state
from schafkopf_ai.runner.game_runner import GameRunner

from .hidden_state import Determinization, HiddenStateSampler


@dataclass(frozen=True, slots=True)
class ActionValueEstimate:
    """Monte-Carlo estimate of focal-player payment for one legal card."""

    card: Card
    mean_payment: float
    payments: tuple[int, ...]


class MonteCarloActionEvaluator:
    """
    Estimate Q(s,a) with hidden-state sampling and heuristic continuations.

    The same sampled determinizations are reused for every legal action in a
    decision. This common-random-numbers design reduces comparison variance.
    After the candidate action, all four seats follow the deterministic
    heuristic policy until settlement.
    """

    def __init__(
        self,
        *,
        samples_per_action: int = 8,
        seed: int = 0,
        max_sampling_attempts: int = 5_000,
    ) -> None:
        if samples_per_action <= 0:
            raise ValueError("samples_per_action must be positive.")
        self.samples_per_action = samples_per_action
        self.sampler = HiddenStateSampler(
            rng=random.Random(seed),
            max_attempts_per_sample=max_sampling_attempts,
        )

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
        payments_by_card: dict[Card, list[int]] = {card: [] for card in legal_cards}

        for determinization in determinizations:
            for card in legal_cards:
                payments_by_card[card].append(
                    self._rollout_payment(
                        determinization=determinization,
                        observation=observation,
                        first_card=card,
                    )
                )

        return tuple(
            ActionValueEstimate(
                card=card,
                mean_payment=statistics.fmean(payments_by_card[card]),
                payments=tuple(payments_by_card[card]),
            )
            for card in legal_cards
        )

    @staticmethod
    def _rollout_payment(
        *,
        determinization: Determinization,
        observation: PlayerObservation,
        first_card: Card,
    ) -> int:
        state = determinization.build_state(observation)
        focal_player = observation.player_index

        if first_card not in state.legal_moves(focal_player):
            raise RuntimeError(
                "Candidate rollout card became illegal after reconstruction."
            )

        state.play_card(focal_player, first_card)
        agents = tuple(HeuristicAgent() for _ in range(4))
        GameRunner(state=state, agents=agents).run()

        result = result_from_game_state(state)
        settlement = settlement_from_game_state(state, result)
        return settlement.payments[focal_player]
