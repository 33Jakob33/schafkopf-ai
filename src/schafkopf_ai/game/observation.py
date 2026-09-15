from dataclasses import dataclass

from schafkopf_ai.game.scoring import points_by_player

from .card import Card
from .game_contract import GameContract
from .trick import TrickPlay


@dataclass(frozen=True, slots=True)
class PlayerObservation:
    """
    Public game information from the perspective of one player.

    The observation deliberately contains only information that the
    player is allowed to know. In particular, it never exposes the
    remaining cards in other players' hands.
    """

    player_index: int
    hand: tuple[Card, ...]
    contract: GameContract

    current_player: int | None

    current_trick: tuple[TrickPlay, ...]
    completed_tricks: tuple[tuple[TrickPlay, ...], ...]
    points_by_player: tuple[int, int, int, int]

    called_ace_released: bool

    @property
    def cards_played(self) -> tuple[Card, ...]:
        """
        Return all cards that have been publicly played so far.

        Cards from completed tricks are returned first, followed by
        cards from the current trick.
        """
        completed_cards = tuple(
            play.card for trick in self.completed_tricks for play in trick
        )

        current_cards = tuple(play.card for play in self.current_trick)

        return completed_cards + current_cards

    @property
    def cards_remaining(self) -> int:
        """Return the number of cards remaining in this player's hand."""
        return len(self.hand)

    @property
    def trick_number(self) -> int:
        """
        Return the current trick number using human-friendly numbering.

        The first trick is 1 and the final trick is 8.

        Once the game is complete, this returns 8.
        """
        return min(
            len(self.completed_tricks) + 1,
            8,
        )
