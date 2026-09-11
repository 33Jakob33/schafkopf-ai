import pytest

from schafkopf_ai.game.card import Card, Rank, Suit
from schafkopf_ai.game.game_contract import GameContract
from schafkopf_ai.game.game_type import GameType
from schafkopf_ai.game.legal_moves import (
    LegalMoveContext,
    legal_moves,
)
from schafkopf_ai.game.trick import Trick


def sauspiel() -> GameContract:
    return GameContract(
        GameType.SAUSPIEL,
        called_suit=Suit.EICHEL,
    )


def test_all_cards_legal_when_leading() -> None:
    contract = GameContract(GameType.WENZ)

    hand = [
        Card(Suit.EICHEL, Rank.ACE),
        Card(Suit.GRAS, Rank.TEN),
        Card(Suit.HERZ, Rank.NINE),
    ]

    trick = Trick(starting_player=0)

    assert legal_moves(
        hand,
        trick,
        contract,
    ) == tuple(hand)


def test_player_must_follow_suit() -> None:
    contract = GameContract(GameType.WENZ)

    trick = Trick(starting_player=0)
    trick.play_card(
        0,
        Card(Suit.GRAS, Rank.SEVEN),
    )

    hand = [
        Card(Suit.GRAS, Rank.ACE),
        Card(Suit.GRAS, Rank.NINE),
        Card(Suit.EICHEL, Rank.ACE),
        Card(Suit.SCHELLEN, Rank.TEN),
    ]

    moves = legal_moves(
        hand,
        trick,
        contract,
    )

    assert moves == (
        Card(Suit.GRAS, Rank.ACE),
        Card(Suit.GRAS, Rank.NINE),
    )


def test_player_without_suit_may_play_anything() -> None:
    contract = GameContract(GameType.WENZ)

    trick = Trick(starting_player=0)
    trick.play_card(
        0,
        Card(Suit.GRAS, Rank.SEVEN),
    )

    hand = [
        Card(Suit.EICHEL, Rank.ACE),
        Card(Suit.SCHELLEN, Rank.TEN),
        Card(Suit.HERZ, Rank.NINE),
    ]

    assert legal_moves(
        hand,
        trick,
        contract,
    ) == tuple(hand)


def test_player_must_follow_trump() -> None:
    contract = GameContract(GameType.SAUSPIEL, called_suit=Suit.EICHEL)

    trick = Trick(starting_player=0)
    trick.play_card(
        0,
        Card(Suit.HERZ, Rank.SEVEN),
    )

    hand = [
        Card(Suit.EICHEL, Rank.OBER),
        Card(Suit.HERZ, Rank.ACE),
        Card(Suit.GRAS, Rank.ACE),
        Card(Suit.SCHELLEN, Rank.TEN),
    ]

    moves = legal_moves(
        hand,
        trick,
        contract,
    )

    assert moves == (
        Card(Suit.EICHEL, Rank.OBER),
        Card(Suit.HERZ, Rank.ACE),
    )


def test_player_without_trump_may_play_anything() -> None:
    contract = GameContract(GameType.WENZ)

    trick = Trick(starting_player=0)
    trick.play_card(
        0,
        Card(Suit.EICHEL, Rank.UNTER),
    )

    hand = [
        Card(Suit.EICHEL, Rank.ACE),
        Card(Suit.GRAS, Rank.TEN),
        Card(Suit.HERZ, Rank.NINE),
    ]

    assert legal_moves(
        hand,
        trick,
        contract,
    ) == tuple(hand)


def test_sauspiel_ober_does_not_count_as_its_printed_suit() -> None:
    contract = sauspiel()

    trick = Trick(starting_player=0)
    trick.play_card(
        0,
        Card(Suit.GRAS, Rank.SEVEN),
    )

    hand = [
        Card(Suit.GRAS, Rank.OBER),
        Card(Suit.EICHEL, Rank.TEN),
        Card(Suit.SCHELLEN, Rank.NINE),
    ]

    # Gras Ober is trump, not a Gras suit card.
    assert legal_moves(
        hand,
        trick,
        contract,
    ) == tuple(hand)


def test_wenz_ober_counts_as_normal_suit_card() -> None:
    contract = GameContract(GameType.WENZ)

    trick = Trick(starting_player=0)
    trick.play_card(
        0,
        Card(Suit.GRAS, Rank.SEVEN),
    )

    hand = [
        Card(Suit.GRAS, Rank.OBER),
        Card(Suit.EICHEL, Rank.ACE),
    ]

    assert legal_moves(
        hand,
        trick,
        contract,
    ) == (Card(Suit.GRAS, Rank.OBER),)


def test_geier_unter_counts_as_normal_suit_card() -> None:
    contract = GameContract(GameType.GEIER)

    trick = Trick(starting_player=0)
    trick.play_card(
        0,
        Card(Suit.EICHEL, Rank.SEVEN),
    )

    hand = [
        Card(Suit.EICHEL, Rank.UNTER),
        Card(Suit.GRAS, Rank.ACE),
    ]

    assert legal_moves(
        hand,
        trick,
        contract,
    ) == (Card(Suit.EICHEL, Rank.UNTER),)


def test_called_player_must_play_called_ace_when_suit_led() -> None:
    contract = sauspiel()

    trick = Trick(starting_player=0)
    trick.play_card(
        0,
        Card(Suit.EICHEL, Rank.SEVEN),
    )

    hand = [
        Card(Suit.EICHEL, Rank.ACE),
        Card(Suit.EICHEL, Rank.TEN),
        Card(Suit.GRAS, Rank.ACE),
    ]

    assert legal_moves(
        hand,
        trick,
        contract,
    ) == (Card(Suit.EICHEL, Rank.ACE),)


def test_called_ace_cannot_be_discarded_on_other_suit() -> None:
    contract = sauspiel()

    trick = Trick(starting_player=0)
    trick.play_card(
        0,
        Card(Suit.GRAS, Rank.SEVEN),
    )

    hand = [
        Card(Suit.EICHEL, Rank.ACE),
        Card(Suit.SCHELLEN, Rank.TEN),
        Card(Suit.HERZ, Rank.NINE),
    ]

    moves = legal_moves(
        hand,
        trick,
        contract,
    )

    assert Card(Suit.EICHEL, Rank.ACE) not in moves


def test_called_ace_may_be_played_in_final_trick() -> None:
    contract = sauspiel()

    trick = Trick(starting_player=0)
    trick.play_card(
        0,
        Card(Suit.GRAS, Rank.SEVEN),
    )

    hand = [
        Card(Suit.EICHEL, Rank.ACE),
    ]

    assert legal_moves(
        hand,
        trick,
        contract,
    ) == (Card(Suit.EICHEL, Rank.ACE),)


def test_called_player_can_lead_called_ace() -> None:
    contract = sauspiel()

    trick = Trick(starting_player=1)

    hand = [
        Card(Suit.EICHEL, Rank.ACE),
        Card(Suit.EICHEL, Rank.TEN),
        Card(Suit.GRAS, Rank.SEVEN),
    ]

    moves = legal_moves(
        hand,
        trick,
        contract,
    )

    assert Card(Suit.EICHEL, Rank.ACE) in moves


def test_called_player_cannot_lead_below_ace_with_fewer_than_four() -> None:
    contract = sauspiel()

    trick = Trick(starting_player=0)

    hand = [
        Card(Suit.EICHEL, Rank.ACE),
        Card(Suit.EICHEL, Rank.TEN),
        Card(Suit.GRAS, Rank.SEVEN),
        Card(Suit.HERZ, Rank.NINE),
    ]

    moves = legal_moves(
        hand,
        trick,
        contract,
    )

    assert Card(Suit.EICHEL, Rank.ACE) in moves
    assert Card(Suit.EICHEL, Rank.TEN) not in moves


def test_called_player_can_run_away_with_four_called_suit_cards() -> None:
    contract = sauspiel()

    trick = Trick(starting_player=0)

    hand = [
        Card(Suit.EICHEL, Rank.ACE),
        Card(Suit.EICHEL, Rank.TEN),
        Card(Suit.EICHEL, Rank.KING),
        Card(Suit.EICHEL, Rank.NINE),
        Card(Suit.GRAS, Rank.SEVEN),
    ]

    moves = legal_moves(
        hand,
        trick,
        contract,
    )

    assert Card(Suit.EICHEL, Rank.TEN) in moves
    assert Card(Suit.EICHEL, Rank.KING) in moves
    assert Card(Suit.EICHEL, Rank.NINE) in moves


def test_called_ace_is_normal_after_running_away() -> None:
    contract = sauspiel()

    trick = Trick(starting_player=0)
    trick.play_card(
        0,
        Card(Suit.GRAS, Rank.SEVEN),
    )

    hand = [
        Card(Suit.EICHEL, Rank.ACE),
        Card(Suit.SCHELLEN, Rank.TEN),
    ]

    context = LegalMoveContext(
        called_ace_released=True,
    )

    moves = legal_moves(
        hand,
        trick,
        contract,
        context,
    )

    assert Card(Suit.EICHEL, Rank.ACE) in moves


def test_completed_trick_rejected() -> None:
    contract = GameContract(GameType.WENZ)

    trick = Trick(starting_player=0)

    trick.play_card(
        0,
        Card(Suit.EICHEL, Rank.SEVEN),
    )
    trick.play_card(
        1,
        Card(Suit.EICHEL, Rank.EIGHT),
    )
    trick.play_card(
        2,
        Card(Suit.EICHEL, Rank.NINE),
    )
    trick.play_card(
        3,
        Card(Suit.EICHEL, Rank.TEN),
    )

    with pytest.raises(
        ValueError,
        match="completed trick",
    ):
        legal_moves(
            [
                Card(Suit.GRAS, Rank.ACE),
            ],
            trick,
            contract,
        )
