from functools import cache

from .card import Card, Rank, Suit
from .game_contract import GameContract
from .game_type import GameType


# Suit order for Ober and Unter:
#
# strongest → weakest
#
# Eichel > Gras > Herz > Schellen
TRUMP_SUIT_ORDER: tuple[Suit, ...] = (
    Suit.EICHEL,
    Suit.GRAS,
    Suit.HERZ,
    Suit.SCHELLEN,
)


# Rank order for ordinary cards of a trump suit:
#
# strongest → weakest
#
# Ober and Unter are deliberately excluded because they are handled
# separately.
TRUMP_PLAIN_RANK_ORDER: tuple[Rank, ...] = (
    Rank.ACE,
    Rank.TEN,
    Rank.KING,
    Rank.NINE,
    Rank.EIGHT,
    Rank.SEVEN,
)


OBER_TRUMPS: tuple[Card, ...] = tuple(
    Card(suit=suit, rank=Rank.OBER) for suit in TRUMP_SUIT_ORDER
)


UNTER_TRUMPS: tuple[Card, ...] = tuple(
    Card(suit=suit, rank=Rank.UNTER) for suit in TRUMP_SUIT_ORDER
)


def _plain_suit_trumps(suit: Suit) -> tuple[Card, ...]:
    """
    Return the six ordinary trump cards of a suit.

    Ober and Unter are not included because they are represented
    separately in the trump hierarchy.
    """
    return tuple(Card(suit=suit, rank=rank) for rank in TRUMP_PLAIN_RANK_ORDER)


@cache
def trump_order(contract: GameContract) -> tuple[Card, ...]:
    """
    Return all trumps for a game contract from strongest to weakest.

    Sauspiel:
        Ober > Unter > Herz

    Solo:
        Ober > Unter > selected trump suit

    Wenz:
        Unter only

    Geier:
        Ober only

    Ramsch:
        Ober > Unter > Herz
    """

    game_type = contract.game_type

    if game_type in {GameType.SAUSPIEL, GameType.RAMSCH}:
        return OBER_TRUMPS + UNTER_TRUMPS + _plain_suit_trumps(Suit.HERZ)

    if game_type is GameType.SOLO:
        if contract.trump_suit is None:
            # This should already be prevented by GameContract,
            # but keeps the function safe for static type checking.
            raise ValueError("A Solo requires a trump suit.")

        return OBER_TRUMPS + UNTER_TRUMPS + _plain_suit_trumps(contract.trump_suit)

    if game_type is GameType.WENZ:
        return UNTER_TRUMPS

    if game_type is GameType.GEIER:
        return OBER_TRUMPS

    raise ValueError(f"Unsupported game type: {game_type}")


def is_trump(card: Card, contract: GameContract) -> bool:
    """Return whether a card is trump in the given game contract."""
    return card in trump_order(contract)


def trump_strength(card: Card, contract: GameContract) -> int:
    """
    Return the strength of a trump card.

    Higher values represent stronger trumps.

    Raises:
        ValueError:
            If the supplied card is not trump in the given contract.
    """
    order = trump_order(contract)

    try:
        index = order.index(card)
    except ValueError as exc:
        raise ValueError(f"{card} is not trump in {contract.game_type.value}.") from exc

    return len(order) - index
