from collections.abc import Sequence
from dataclasses import dataclass, field

from .card import Card, Rank
from .game_contract import GameContract
from .game_type import GameType
from .trump import is_trump, trump_strength

STANDARD_PLAIN_RANK_ORDER: tuple[Rank, ...] = (
    Rank.ACE,
    Rank.TEN,
    Rank.KING,
    Rank.NINE,
    Rank.EIGHT,
    Rank.SEVEN,
)

WENZ_PLAIN_RANK_ORDER: tuple[Rank, ...] = (
    Rank.ACE,
    Rank.TEN,
    Rank.KING,
    Rank.OBER,
    Rank.NINE,
    Rank.EIGHT,
    Rank.SEVEN,
)

GEIER_PLAIN_RANK_ORDER: tuple[Rank, ...] = (
    Rank.ACE,
    Rank.TEN,
    Rank.KING,
    Rank.UNTER,
    Rank.NINE,
    Rank.EIGHT,
    Rank.SEVEN,
)


@dataclass(frozen=True, slots=True)
class TrickPlay:
    """A card played by one player during a trick."""

    player: int
    card: Card


def plain_rank_order(contract: GameContract) -> tuple[Rank, ...]:
    """
    Return the rank order for non-trump cards from strongest to weakest.

    Sauspiel / Solo / Ramsch:
        Ober and Unter are trumps and therefore do not appear here.

    Wenz:
        Unter are trumps; Ober return to their normal suits.

    Geier:
        Ober are trumps; Unter return to their normal suits.
    """
    if contract.game_type in {
        GameType.SAUSPIEL,
        GameType.SOLO,
        GameType.RAMSCH,
    }:
        return STANDARD_PLAIN_RANK_ORDER

    if contract.game_type is GameType.WENZ:
        return WENZ_PLAIN_RANK_ORDER

    if contract.game_type is GameType.GEIER:
        return GEIER_PLAIN_RANK_ORDER

    raise ValueError(f"Unsupported game type: {contract.game_type.value}")


def plain_card_strength(
    card: Card,
    contract: GameContract,
) -> int:
    """
    Return the strength of a non-trump card.

    Higher values mean stronger cards.

    Raises:
        ValueError:
            If the card is trump or its rank cannot occur as a
            non-trump card in this contract.
    """
    if is_trump(card, contract):
        raise ValueError(f"{card} is trump and has no plain-card strength.")

    order = plain_rank_order(contract)

    try:
        index = order.index(card.rank)
    except ValueError as exc:
        raise ValueError(
            f"{card} cannot occur as a non-trump card in {contract.game_type.value}."
        ) from exc

    return len(order) - index


def card_beats(
    challenger: Card,
    current_winner: Card,
    lead_card: Card,
    contract: GameContract,
) -> bool:
    """
    Return whether challenger beats the current winning card.

    Rules:
    1. Trump beats every non-trump.
    2. Between two trumps, the higher trump wins.
    3. Without trump, only cards matching the led suit can win.
    4. Within the led suit, the higher-ranked card wins.
    """
    challenger_is_trump = is_trump(challenger, contract)
    winner_is_trump = is_trump(current_winner, contract)

    if challenger_is_trump and not winner_is_trump:
        return True

    if winner_is_trump and not challenger_is_trump:
        return False

    if challenger_is_trump and winner_is_trump:
        return trump_strength(challenger, contract) > trump_strength(
            current_winner, contract
        )

    # Neither card is trump.
    #
    # If the trick was opened with a trump, neither of these cards can
    # beat that original trump. In normal use current_winner would
    # therefore already be trump, but this keeps the comparison robust.
    if is_trump(lead_card, contract):
        return False

    lead_suit = lead_card.suit

    challenger_follows_suit = challenger.suit is lead_suit
    winner_follows_suit = current_winner.suit is lead_suit

    if challenger_follows_suit and not winner_follows_suit:
        return True

    if winner_follows_suit and not challenger_follows_suit:
        return False

    if not challenger_follows_suit:
        return False

    return plain_card_strength(challenger, contract) > plain_card_strength(
        current_winner, contract
    )


def winning_play(
    plays: Sequence[TrickPlay],
    contract: GameContract,
) -> TrickPlay:
    """
    Determine the currently winning play.

    The first played card determines the led suit or trump.
    """
    if not plays:
        raise ValueError("Cannot determine winner of an empty trick.")

    lead_card = plays[0].card
    winner = plays[0]

    for play in plays[1:]:
        if card_beats(
            challenger=play.card,
            current_winner=winner.card,
            lead_card=lead_card,
            contract=contract,
        ):
            winner = play

    return winner


@dataclass(slots=True)
class Trick:
    """
    Represents one complete Schafkopf trick.

    Players are numbered 0 through 3.
    """

    starting_player: int
    plays: list[TrickPlay] = field(default_factory=list)

    PLAYER_COUNT = 4

    def __post_init__(self) -> None:
        if not 0 <= self.starting_player < self.PLAYER_COUNT:
            raise ValueError(
                f"Player index must be between 0 and {self.PLAYER_COUNT - 1}."
            )

    @property
    def is_complete(self) -> bool:
        """Return whether all four players have played."""
        return len(self.plays) == self.PLAYER_COUNT

    @property
    def next_player(self) -> int | None:
        """Return the player whose turn it is."""
        if self.is_complete:
            return None

        return (self.starting_player + len(self.plays)) % self.PLAYER_COUNT

    def play_card(
        self,
        player: int,
        card: Card,
    ) -> None:
        """
        Add a card to the trick.

        This checks turn order and duplicate cards, but deliberately
        does not check suit-following rules yet.
        """
        if self.is_complete:
            raise ValueError("The trick is already complete.")

        if player != self.next_player:
            raise ValueError(
                f"It is player {self.next_player}'s turn, not player {player}'s."
            )

        if any(play.card == card for play in self.plays):
            raise ValueError(f"{card} has already been played in this trick.")

        self.plays.append(
            TrickPlay(
                player=player,
                card=card,
            )
        )

    def current_winner(
        self,
        contract: GameContract,
    ) -> TrickPlay:
        """Return the currently winning player and card."""
        return winning_play(self.plays, contract)

    def winner(
        self,
        contract: GameContract,
    ) -> TrickPlay:
        """Return the winner of a completed trick."""
        if not self.is_complete:
            raise ValueError(
                "Cannot determine final winner before the trick is complete."
            )

        return winning_play(self.plays, contract)
