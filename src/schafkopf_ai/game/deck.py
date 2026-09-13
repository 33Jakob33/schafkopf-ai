import random
from collections.abc import Iterator

from .card import Card, Rank, Suit


class Deck:
    """A standard 32-card Bavarian Schafkopf deck."""

    CARDS_PER_PLAYER = 8
    PLAYER_COUNT = 4

    def __init__(self) -> None:
        self._cards: list[Card] = self._create_cards()

    @staticmethod
    def _create_cards() -> list[Card]:
        """Create all 32 unique Schafkopf cards."""
        return [Card(suit=suit, rank=rank) for suit in Suit for rank in Rank]

    @property
    def cards(self) -> tuple[Card, ...]:
        """
        Return the current cards as an immutable tuple.

        This prevents callers from accidentally modifying the deck.
        """
        return tuple(self._cards)

    def shuffle(self, rng: random.Random | None = None) -> None:
        """
        Shuffle the deck.

        An optional Random instance can be supplied to make
        simulations and tests reproducible.
        """
        if rng is None:
            random.shuffle(self._cards)
        else:
            rng.shuffle(self._cards)

    def reset(self) -> None:
        """Restore the deck to a complete, unshuffled 32-card deck."""
        self._cards = self._create_cards()

    def deal(self) -> tuple[tuple[Card, ...], ...]:
        """
        Deal eight cards to each of four players.

        Traditional Schafkopf dealing is represented as two rounds:
        each player receives four cards, followed by another four.

        The deck is empty after dealing.
        """
        required_cards = self.PLAYER_COUNT * self.CARDS_PER_PLAYER

        if len(self._cards) != required_cards:
            raise ValueError(
                f"Cannot deal Schafkopf game: expected "
                f"{required_cards} cards, found {len(self._cards)}."
            )

        hands: list[list[Card]] = [[] for _ in range(self.PLAYER_COUNT)]

        cards_per_round = 4

        for _ in range(2):
            for player_index in range(self.PLAYER_COUNT):
                for _ in range(cards_per_round):
                    hands[player_index].append(self._cards.pop(0))

        return tuple(tuple(hand) for hand in hands)

    def __len__(self) -> int:
        return len(self._cards)

    def __iter__(self) -> Iterator[Card]:
        return iter(self._cards)
