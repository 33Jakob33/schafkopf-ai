from collections.abc import Sequence

from .card import Card, Rank
from .game_contract import GameContract
from .trick import Trick


PLAYER_COUNT = 4
TRICKS_PER_GAME = 8
TOTAL_GAME_POINTS = 120


CARD_POINTS: dict[Rank, int] = {
    Rank.ACE: 11,
    Rank.TEN: 10,
    Rank.KING: 4,
    Rank.OBER: 3,
    Rank.UNTER: 2,
    Rank.NINE: 0,
    Rank.EIGHT: 0,
    Rank.SEVEN: 0,
}


def card_points(card: Card) -> int:
    """Return the Augen value of a single card."""
    return CARD_POINTS[card.rank]


def trick_points(trick: Trick) -> int:
    """
    Return the total Augen contained in a completed trick.

    Raises:
        ValueError:
            If the trick is not complete.
    """
    if not trick.is_complete:
        raise ValueError("Cannot calculate points for an incomplete trick.")

    return sum(card_points(play.card) for play in trick.plays)


def points_by_player(
    completed_tricks: Sequence[Trick],
    contract: GameContract,
) -> tuple[int, int, int, int]:
    """
    Calculate the Augen collected by each player.

    The points of each trick are awarded to the player who won
    that trick.

    Returns:
        Tuple in player order:

        (player_0_points,
         player_1_points,
         player_2_points,
         player_3_points)
    """
    points = [0] * PLAYER_COUNT

    for trick in completed_tricks:
        if not trick.is_complete:
            raise ValueError("Only completed tricks can be scored.")

        winner = trick.winner(contract)

        points[winner.player] += trick_points(trick)

    return (
        points[0],
        points[1],
        points[2],
        points[3],
    )


def total_points(
    completed_tricks: Sequence[Trick],
) -> int:
    """Return the total Augen contained in completed tricks."""
    return sum(trick_points(trick) for trick in completed_tricks)


def validate_complete_game_points(
    completed_tricks: Sequence[Trick],
) -> None:
    """
    Verify the 120-Augen invariant for a completed game.

    A complete Schafkopf round must contain exactly eight tricks
    worth exactly 120 Augen in total.

    Raises:
        ValueError:
            If there are not exactly eight completed tricks or
            their point total is not 120.
    """
    if len(completed_tricks) != TRICKS_PER_GAME:
        raise ValueError(
            f"A completed Schafkopf game must contain exactly {TRICKS_PER_GAME} tricks."
        )

    points = total_points(completed_tricks)

    if points != TOTAL_GAME_POINTS:
        raise ValueError(
            f"A completed Schafkopf game must contain exactly "
            f"{TOTAL_GAME_POINTS} Augen, found {points}."
        )
