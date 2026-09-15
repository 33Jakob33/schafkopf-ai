from dataclasses import dataclass

from .card import Suit
from .game_type import GameType


@dataclass(frozen=True, slots=True)
class GameContract:
    """
    Describes the contract currently being played.

    Solo:
        Requires a trump suit.

    Sauspiel:
        Requires a called suit.

    declarer:
        Player who announced the game. Required when determining
        the result of Sauspiel, Solo, Wenz and Geier.

        Ramsch has no declarer.
    """

    game_type: GameType
    trump_suit: Suit | None = None
    called_suit: Suit | None = None
    declarer: int | None = None

    def __post_init__(self) -> None:
        if self.declarer is not None and not 0 <= self.declarer < 4:
            raise ValueError("Declarer must be a player index between 0 and 3.")

        if self.game_type is GameType.RAMSCH and self.declarer is not None:
            raise ValueError("Ramsch does not have a declarer.")

        if self.game_type is GameType.SOLO:
            if self.trump_suit is None:
                raise ValueError("A Solo requires a trump suit.")

            if self.called_suit is not None:
                raise ValueError("A Solo cannot have a called suit.")

            return

        if self.trump_suit is not None:
            raise ValueError(
                f"{self.game_type.value} does not use a configurable trump suit."
            )

        if self.game_type is GameType.SAUSPIEL:
            if self.called_suit is None:
                raise ValueError("A Sauspiel requires a called suit.")

            if self.called_suit is Suit.HERZ:
                raise ValueError(
                    "Herz cannot be called in a Sauspiel because Herz is trump."
                )

            return

        if self.called_suit is not None:
            raise ValueError(f"{self.game_type.value} does not use a called suit.")
