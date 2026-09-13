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
        Requires a called suit. Herz cannot be called because Herz
        is trump in a Sauspiel.

    Other currently supported games:
        Require neither.
    """

    game_type: GameType
    trump_suit: Suit | None = None
    called_suit: Suit | None = None

    def __post_init__(self) -> None:
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
