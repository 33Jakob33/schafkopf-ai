from collections.abc import Sequence
from dataclasses import dataclass, field

from .card import Card


@dataclass(slots=True)
class Player:
    """Represents one of the four players in a Schafkopf round."""

    index: int
    hand: list[Card] = field(default_factory=list)

    PLAYER_COUNT = 4

    def __post_init__(self) -> None:
        if not 0 <= self.index < self.PLAYER_COUNT:
            raise ValueError(
                f"Player index must be between 0 and {self.PLAYER_COUNT - 1}."
            )

        if len(self.hand) != len(set(self.hand)):
            raise ValueError(f"Player {self.index} has duplicate cards.")

    @property
    def cards(self) -> tuple[Card, ...]:
        """Return the player's hand as an immutable tuple."""
        return tuple(self.hand)

    def set_hand(self, cards: Sequence[Card]) -> None:
        """Replace the player's current hand."""
        if len(cards) != len(set(cards)):
            raise ValueError("A hand cannot contain duplicate cards.")

        self.hand = list(cards)

    def has_card(self, card: Card) -> bool:
        """Return whether the player currently holds a card."""
        return card in self.hand

    def remove_card(self, card: Card) -> None:
        """Remove a card from the player's hand."""
        try:
            self.hand.remove(card)
        except ValueError as exc:
            raise ValueError(f"Player {self.index} does not hold {card}.") from exc

    def __len__(self) -> int:
        return len(self.hand)
