import random

import pytest

from schafkopf_ai.game.card import Rank, Suit
from schafkopf_ai.game.deck import Deck


def test_new_deck_contains_32_cards() -> None:
    deck = Deck()

    assert len(deck) == 32


def test_deck_contains_32_unique_cards() -> None:
    deck = Deck()

    assert len(set(deck.cards)) == 32


def test_each_suit_contains_eight_cards() -> None:
    deck = Deck()

    for suit in Suit:
        cards_of_suit = [card for card in deck.cards if card.suit is suit]

        assert len(cards_of_suit) == 8


def test_each_rank_appears_four_times() -> None:
    deck = Deck()

    for rank in Rank:
        cards_of_rank = [card for card in deck.cards if card.rank is rank]

        assert len(cards_of_rank) == 4


def test_deal_returns_four_hands() -> None:
    deck = Deck()

    hands = deck.deal()

    assert len(hands) == 4


def test_each_player_receives_eight_cards() -> None:
    deck = Deck()

    hands = deck.deal()

    for hand in hands:
        assert len(hand) == 8


def test_all_dealt_cards_are_unique() -> None:
    deck = Deck()

    hands = deck.deal()

    dealt_cards = [card for hand in hands for card in hand]

    assert len(dealt_cards) == 32
    assert len(set(dealt_cards)) == 32


def test_deck_is_empty_after_dealing() -> None:
    deck = Deck()

    deck.deal()

    assert len(deck) == 0


def test_reset_restores_full_deck() -> None:
    deck = Deck()

    deck.deal()
    deck.reset()

    assert len(deck) == 32
    assert len(set(deck.cards)) == 32


def test_seeded_shuffle_is_reproducible() -> None:
    deck_1 = Deck()
    deck_2 = Deck()

    rng_1 = random.Random(42)
    rng_2 = random.Random(42)

    deck_1.shuffle(rng_1)
    deck_2.shuffle(rng_2)

    assert deck_1.cards == deck_2.cards


def test_different_seeds_produce_different_order() -> None:
    deck_1 = Deck()
    deck_2 = Deck()

    deck_1.shuffle(random.Random(42))
    deck_2.shuffle(random.Random(123))

    assert deck_1.cards != deck_2.cards


def test_second_deal_fails_when_deck_is_empty() -> None:
    deck = Deck()

    deck.deal()

    with pytest.raises(ValueError):
        deck.deal()


def test_cards_property_cannot_modify_internal_deck() -> None:
    deck = Deck()

    cards = deck.cards

    assert isinstance(cards, tuple)
    assert len(cards) == 32
    assert len(deck) == 32


def test_deck_is_iterable() -> None:
    deck = Deck()

    cards = list(deck)

    assert len(cards) == 32
    assert tuple(cards) == deck.cards
