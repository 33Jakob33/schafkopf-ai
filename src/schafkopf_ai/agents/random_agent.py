import random

from schafkopf_ai.game.card import Card
from schafkopf_ai.game.observation import PlayerObservation

from .agent import Agent


class RandomAgent(Agent):
    """
    An agent that selects uniformly at random from the legal cards.
    """

    def __init__(
        self,
        rng: random.Random | None = None,
    ) -> None:
        """
        Create a RandomAgent.

        Args:
            rng:
                Optional random-number generator. Supplying one makes
                agent behavior reproducible in tests and simulations.
        """
        self._rng = rng or random.Random()

    def choose_card(
        self,
        observation: PlayerObservation,
        legal_cards: tuple[Card, ...],
    ) -> Card:
        """
        Choose one card uniformly at random from the legal moves.
        """
        del observation

        if not legal_cards:
            raise ValueError("No legal cards available.")

        return self._rng.choice(legal_cards)