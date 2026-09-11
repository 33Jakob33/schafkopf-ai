from dataclasses import dataclass
from enum import Enum

from .game_type import GameType


class AllPassAction(Enum):
    """
    Defines what happens when all four players pass during bidding.
    """

    REDEAL = "redeal"
    RAMSCH = "ramsch"


@dataclass(frozen=True, slots=True)
class GameRules:
    """
    Configurable rules for a Schafkopf table.

    The rules determine which game types are enabled and what
    happens when no player announces a game.
    """

    enabled_game_types: frozenset[GameType]
    all_pass_action: AllPassAction = AllPassAction.REDEAL

    def __post_init__(self) -> None:
        if (
            self.all_pass_action is AllPassAction.RAMSCH
            and GameType.RAMSCH not in self.enabled_game_types
        ):
            raise ValueError("All-pass action is RAMSCH, but Ramsch is not enabled.")

    def is_enabled(self, game_type: GameType) -> bool:
        """Return whether the given game type is enabled."""
        return game_type in self.enabled_game_types

    @property
    def bid_game_types(self) -> frozenset[GameType]:
        """
        Return all enabled game types that players may actively bid.

        Ramsch is excluded because it is triggered when everybody passes.
        """
        return frozenset(
            game_type for game_type in self.enabled_game_types if game_type.can_be_bid
        )


STANDARD_RULES = GameRules(
    enabled_game_types=frozenset(
        {
            GameType.SAUSPIEL,
            GameType.SOLO,
            GameType.WENZ,
        }
    ),
    all_pass_action=AllPassAction.REDEAL,
)


RAMSCH_RULES = GameRules(
    enabled_game_types=frozenset(
        {
            GameType.SAUSPIEL,
            GameType.SOLO,
            GameType.WENZ,
            GameType.RAMSCH,
        }
    ),
    all_pass_action=AllPassAction.RAMSCH,
)
