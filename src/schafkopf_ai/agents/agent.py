from abc import ABC, abstractmethod

from schafkopf_ai.game.bidding import (
    BiddingAction,
    BiddingObservation,
)
from schafkopf_ai.game.card import Card
from schafkopf_ai.game.observation import PlayerObservation


class Agent(ABC):
    """
    Base class for all Schafkopf agents.

    An agent receives:
    - the game state visible from its own perspective,
    - only the cards it is legally allowed to play.

    The agent must return exactly one of those legal cards.
    """

    @abstractmethod
    def choose_card(
        self,
        observation: PlayerObservation,
        legal_cards: tuple[Card, ...],
    ) -> Card:
        """
        Choose one card from the legal actions.

        Args:
            observation:
                Information visible to the current player.

            legal_cards:
                Cards the player is currently allowed to play.

        Returns:
            The selected card.

        Raises:
            ValueError:
                Agents should raise an error if no legal actions
                are available.
        """
        raise NotImplementedError

    @abstractmethod
    def choose_bidding_action(
        self,
        observation: BiddingObservation,
        legal_actions: tuple[BiddingAction, ...],
    ) -> BiddingAction:
        """Choose one legal bidding action."""
        raise NotImplementedError
