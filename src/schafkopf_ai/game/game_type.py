from enum import Enum


class GameType(Enum):
    """
    Supported Schafkopf game types.


    """

    SAUSPIEL = "Sauspiel"
    SOLO = "Solo"
    WENZ = "Wenz"
    GEIER = "Geier"
    RAMSCH = "Ramsch"

    @property
    def can_be_bid(self) -> bool:
        """
        Return whether players can actively announce this game type.

        Ramsch is a fallback when all players pass rather than
        a normal bid.
        """
        return self is not GameType.RAMSCH

    def __str__(self) -> str:
        return self.value
