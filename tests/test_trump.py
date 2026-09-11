import pytest

from schafkopf_ai.game.card import Card, Rank, Suit
from schafkopf_ai.game.game_contract import GameContract
from schafkopf_ai.game.game_type import GameType
from schafkopf_ai.game.trump import (
    is_trump,
    trump_order,
    trump_strength,
)


def sauspiel(
    called_suit: Suit = Suit.EICHEL,
) -> GameContract:
    """Return a Sauspiel contract for trump tests."""
    return GameContract(
        game_type=GameType.SAUSPIEL,
        called_suit=called_suit,
    )


def test_sauspiel_has_14_trumps() -> None:
    contract = sauspiel()

    assert len(trump_order(contract)) == 14


def test_ramsch_has_14_trumps() -> None:
    contract = GameContract(GameType.RAMSCH)

    assert len(trump_order(contract)) == 14


def test_sauspiel_and_ramsch_have_same_trump_order() -> None:
    sauspiel_contract = sauspiel()
    ramsch_contract = GameContract(GameType.RAMSCH)

    assert trump_order(sauspiel_contract) == trump_order(ramsch_contract)


@pytest.mark.parametrize(
    "called_suit",
    [
        Suit.EICHEL,
        Suit.GRAS,
        Suit.SCHELLEN,
    ],
)
def test_called_suit_does_not_change_sauspiel_trumps(
    called_suit: Suit,
) -> None:
    contract = sauspiel(called_suit)

    assert trump_order(contract) == trump_order(sauspiel(Suit.EICHEL))


def test_sauspiel_ober_order() -> None:
    contract = sauspiel()

    expected = (
        Card(Suit.EICHEL, Rank.OBER),
        Card(Suit.GRAS, Rank.OBER),
        Card(Suit.HERZ, Rank.OBER),
        Card(Suit.SCHELLEN, Rank.OBER),
    )

    assert trump_order(contract)[:4] == expected


def test_sauspiel_unter_order() -> None:
    contract = sauspiel()

    expected = (
        Card(Suit.EICHEL, Rank.UNTER),
        Card(Suit.GRAS, Rank.UNTER),
        Card(Suit.HERZ, Rank.UNTER),
        Card(Suit.SCHELLEN, Rank.UNTER),
    )

    assert trump_order(contract)[4:8] == expected


def test_sauspiel_herz_order() -> None:
    contract = sauspiel()

    expected = (
        Card(Suit.HERZ, Rank.ACE),
        Card(Suit.HERZ, Rank.TEN),
        Card(Suit.HERZ, Rank.KING),
        Card(Suit.HERZ, Rank.NINE),
        Card(Suit.HERZ, Rank.EIGHT),
        Card(Suit.HERZ, Rank.SEVEN),
    )

    assert trump_order(contract)[8:] == expected


def test_non_herz_plain_card_is_not_trump_in_sauspiel() -> None:
    contract = sauspiel()

    card = Card(Suit.EICHEL, Rank.ACE)

    assert not is_trump(card, contract)


def test_herz_plain_card_is_trump_in_sauspiel() -> None:
    contract = sauspiel()

    card = Card(Suit.HERZ, Rank.ACE)

    assert is_trump(card, contract)


def test_all_obers_are_trump_in_sauspiel() -> None:
    contract = sauspiel()

    for suit in Suit:
        assert is_trump(
            Card(suit, Rank.OBER),
            contract,
        )


def test_all_unters_are_trump_in_sauspiel() -> None:
    contract = sauspiel()

    for suit in Suit:
        assert is_trump(
            Card(suit, Rank.UNTER),
            contract,
        )


@pytest.mark.parametrize(
    "trump_suit",
    list(Suit),
)
def test_solo_has_14_trumps(
    trump_suit: Suit,
) -> None:
    contract = GameContract(
        game_type=GameType.SOLO,
        trump_suit=trump_suit,
    )

    assert len(trump_order(contract)) == 14


@pytest.mark.parametrize(
    "trump_suit",
    list(Suit),
)
def test_plain_cards_of_solo_suit_are_trump(
    trump_suit: Suit,
) -> None:
    contract = GameContract(
        game_type=GameType.SOLO,
        trump_suit=trump_suit,
    )

    assert is_trump(
        Card(trump_suit, Rank.ACE),
        contract,
    )

    assert is_trump(
        Card(trump_suit, Rank.SEVEN),
        contract,
    )


def test_plain_card_of_other_suit_is_not_trump_in_solo() -> None:
    contract = GameContract(
        game_type=GameType.SOLO,
        trump_suit=Suit.EICHEL,
    )

    assert not is_trump(
        Card(Suit.GRAS, Rank.ACE),
        contract,
    )


def test_wenz_has_four_trumps() -> None:
    contract = GameContract(GameType.WENZ)

    assert len(trump_order(contract)) == 4


def test_wenz_only_unters_are_trump() -> None:
    contract = GameContract(GameType.WENZ)

    for suit in Suit:
        assert is_trump(
            Card(suit, Rank.UNTER),
            contract,
        )

        assert not is_trump(
            Card(suit, Rank.OBER),
            contract,
        )


def test_wenz_trump_order() -> None:
    contract = GameContract(GameType.WENZ)

    assert trump_order(contract) == (
        Card(Suit.EICHEL, Rank.UNTER),
        Card(Suit.GRAS, Rank.UNTER),
        Card(Suit.HERZ, Rank.UNTER),
        Card(Suit.SCHELLEN, Rank.UNTER),
    )


def test_geier_has_four_trumps() -> None:
    contract = GameContract(GameType.GEIER)

    assert len(trump_order(contract)) == 4


def test_geier_only_obers_are_trump() -> None:
    contract = GameContract(GameType.GEIER)

    for suit in Suit:
        assert is_trump(
            Card(suit, Rank.OBER),
            contract,
        )

        assert not is_trump(
            Card(suit, Rank.UNTER),
            contract,
        )


def test_geier_trump_order() -> None:
    contract = GameContract(GameType.GEIER)

    assert trump_order(contract) == (
        Card(Suit.EICHEL, Rank.OBER),
        Card(Suit.GRAS, Rank.OBER),
        Card(Suit.HERZ, Rank.OBER),
        Card(Suit.SCHELLEN, Rank.OBER),
    )


def test_stronger_trump_has_higher_strength() -> None:
    contract = sauspiel()

    eichel_ober = Card(
        Suit.EICHEL,
        Rank.OBER,
    )
    schellen_ober = Card(
        Suit.SCHELLEN,
        Rank.OBER,
    )
    herz_ace = Card(
        Suit.HERZ,
        Rank.ACE,
    )

    assert (
        trump_strength(eichel_ober, contract)
        > trump_strength(schellen_ober, contract)
        > trump_strength(herz_ace, contract)
    )


def test_trump_strength_rejects_non_trump() -> None:
    contract = sauspiel()

    card = Card(
        Suit.EICHEL,
        Rank.ACE,
    )

    with pytest.raises(
        ValueError,
        match="is not trump",
    ):
        trump_strength(card, contract)


@pytest.mark.parametrize(
    "contract",
    [
        GameContract(
            GameType.SAUSPIEL,
            called_suit=Suit.EICHEL,
        ),
        GameContract(GameType.RAMSCH),
        GameContract(GameType.WENZ),
        GameContract(GameType.GEIER),
        GameContract(
            GameType.SOLO,
            trump_suit=Suit.EICHEL,
        ),
        GameContract(
            GameType.SOLO,
            trump_suit=Suit.GRAS,
        ),
        GameContract(
            GameType.SOLO,
            trump_suit=Suit.HERZ,
        ),
        GameContract(
            GameType.SOLO,
            trump_suit=Suit.SCHELLEN,
        ),
    ],
)
def test_trump_order_contains_no_duplicates(
    contract: GameContract,
) -> None:
    trumps = trump_order(contract)

    assert len(trumps) == len(set(trumps))
