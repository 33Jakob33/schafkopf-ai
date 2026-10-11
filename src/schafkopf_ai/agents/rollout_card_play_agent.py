from __future__ import annotations

from schafkopf_ai.game.bidding import BiddingAction, BiddingObservation
from schafkopf_ai.game.card import Card
from schafkopf_ai.game.observation import PlayerObservation
from schafkopf_ai.search.rollout import ActionValueEstimate, MonteCarloActionEvaluator

from .agent import Agent
from .heuristic_agent import HeuristicAgent


class RolloutCardPlayAgent(Agent):
    """
    One-step Monte-Carlo policy improvement over the heuristic baseline.

    For every non-forced move, the agent samples plausible hidden hands,
    evaluates each legal card by heuristic continuation to final settlement,
    and chooses the card with the highest estimated focal-player payment.
    """

    def __init__(
        self,
        *,
        samples_per_action: int = 8,
        seed: int = 0,
        max_sampling_attempts: int = 5_000,
        bidding_agent: Agent | None = None,
    ) -> None:
        self.evaluator = MonteCarloActionEvaluator(
            samples_per_action=samples_per_action,
            seed=seed,
            max_sampling_attempts=max_sampling_attempts,
        )
        self.bidding_agent = (
            HeuristicAgent() if bidding_agent is None else bidding_agent
        )
        self.tie_breaker = HeuristicAgent()
        self.last_action_values: tuple[ActionValueEstimate, ...] = ()

    def choose_bidding_action(
        self,
        observation: BiddingObservation,
        legal_actions: tuple[BiddingAction, ...],
    ) -> BiddingAction:
        return self.bidding_agent.choose_bidding_action(observation, legal_actions)

    def choose_card(
        self,
        observation: PlayerObservation,
        legal_cards: tuple[Card, ...],
    ) -> Card:
        if not legal_cards:
            raise ValueError("No legal cards available.")
        if len(legal_cards) == 1:
            self.last_action_values = ()
            return legal_cards[0]

        estimates = self.evaluator.evaluate(observation, legal_cards)
        self.last_action_values = estimates

        best_value = max(estimate.mean_payment for estimate in estimates)
        best_cards = tuple(
            estimate.card
            for estimate in estimates
            if estimate.mean_payment == best_value
        )

        heuristic_choice = self.tie_breaker.choose_card(observation, legal_cards)
        if heuristic_choice in best_cards:
            return heuristic_choice

        return next(card for card in legal_cards if card in best_cards)
