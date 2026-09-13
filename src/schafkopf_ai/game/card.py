from dataclasses import dataclass
from enum import Enum


class Suit(Enum):
    """The four suits used in a Bavarian Schafkopf deck."""

    EICHEL = "Eichel"
    GRAS = "Gras"
    HERZ = "Herz"
    SCHELLEN = "Schellen"

    def __str__(self) -> str:
        return self.value


class Rank(Enum):
    """The eight ranks used in a Bavarian Schafkopf deck."""

    SEVEN = "7"
    EIGHT = "8"
    NINE = "9"
    UNTER = "Unter"
    OBER = "Ober"
    KING = "König"
    TEN = "10"
    ACE = "Ass"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class Card:
    """
    A single Schafkopf card.

    A Card deliberately does not know whether it is trump.
    Trump status depends on the currently played game type.
    """

    suit: Suit
    rank: Rank

    def __str__(self) -> str:
        return f"{self.suit.value} {self.rank.value}"

    def __repr__(self) -> str:
        return f"Card(suit={self.suit.name}, rank={self.rank.name})"
