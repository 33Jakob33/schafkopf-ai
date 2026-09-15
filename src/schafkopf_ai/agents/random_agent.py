import random

from schafkopf_ai.game.bidding import (
    BiddingAction,
    BiddingObservation,
)
from schafkopf_ai.game.card import Card
from schafkopf_ai.game.observation import PlayerObservation

from .agent import Agent


class RandomAgent(Agent):
    """
    Agent that uniformly selects from the available legal actions.
    """

    def __init__(
        self,
        rng: random.Random | None = None,
    ) -> None:
        self._rng = rng or random.Random()

    def choose_bidding_action(
        self,
        observation: BiddingObservation,
        legal_actions: tuple[BiddingAction, ...],
    ) -> BiddingAction:
        del observation

        if not legal_actions:
            raise ValueError("No legal bidding actions available.")

        return self._rng.choice(legal_actions)

    def choose_card(
        self,
        observation: PlayerObservation,
        legal_cards: tuple[Card, ...],
    ) -> Card:
        del observation

        if not legal_cards:
            raise ValueError("No legal cards available.")

        return self._rng.choice(legal_cards)
