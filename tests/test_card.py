from dataclasses import FrozenInstanceError

import pytest

from schafkopf_ai.game.card import Card, Rank, Suit


def test_suit_contains_four_suits() -> None:
    assert len(Suit) == 4


def test_rank_contains_eight_ranks() -> None:
    assert len(Rank) == 8


def test_card_creation() -> None:
    card = Card(Suit.EICHEL, Rank.ACE)

    assert card.suit is Suit.EICHEL
    assert card.rank is Rank.ACE


def test_card_string_representation() -> None:
    card = Card(Suit.HERZ, Rank.OBER)

    assert str(card) == "Herz Ober"


def test_card_repr() -> None:
    card = Card(Suit.GRAS, Rank.UNTER)

    assert repr(card) == "Card(suit=GRAS, rank=UNTER)"


def test_equal_cards_are_equal() -> None:
    card_1 = Card(Suit.SCHELLEN, Rank.TEN)
    card_2 = Card(Suit.SCHELLEN, Rank.TEN)

    assert card_1 == card_2


def test_different_cards_are_not_equal() -> None:
    card_1 = Card(Suit.SCHELLEN, Rank.TEN)
    card_2 = Card(Suit.SCHELLEN, Rank.ACE)

    assert card_1 != card_2


def test_card_is_hashable() -> None:
    card = Card(Suit.EICHEL, Rank.ACE)

    cards = {card}

    assert card in cards


def test_card_is_immutable() -> None:
    card = Card(Suit.EICHEL, Rank.ACE)

    with pytest.raises(FrozenInstanceError):
        card.suit = Suit.HERZ  # type: ignore[misc]
