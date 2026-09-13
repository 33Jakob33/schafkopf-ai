import pytest

from schafkopf_ai.game.card import Card, Rank, Suit
from schafkopf_ai.game.game_contract import GameContract
from schafkopf_ai.game.game_type import GameType
from schafkopf_ai.game.trick import (
    Trick,
    TrickPlay,
    card_beats,
    plain_card_strength,
    winning_play,
)


def sauspiel() -> GameContract:
    """Return a standard Eichel-Sauspiel contract for tests."""
    return GameContract(
        game_type=GameType.SAUSPIEL,
        called_suit=Suit.EICHEL,
    )


def test_ace_beats_ten_of_led_suit() -> None:
    contract = sauspiel()

    lead = Card(Suit.EICHEL, Rank.TEN)
    challenger = Card(Suit.EICHEL, Rank.ACE)

    assert card_beats(
        challenger,
        lead,
        lead,
        contract,
    )


def test_off_suit_ace_does_not_beat_led_card() -> None:
    contract = sauspiel()

    lead = Card(Suit.EICHEL, Rank.SEVEN)
    challenger = Card(Suit.GRAS, Rank.ACE)

    assert not card_beats(
        challenger,
        lead,
        lead,
        contract,
    )


def test_trump_beats_non_trump() -> None:
    contract = sauspiel()

    lead = Card(Suit.EICHEL, Rank.ACE)
    trump = Card(Suit.HERZ, Rank.SEVEN)

    assert card_beats(
        trump,
        lead,
        lead,
        contract,
    )


def test_non_trump_does_not_beat_trump() -> None:
    contract = sauspiel()

    lead = Card(Suit.EICHEL, Rank.ACE)
    trump = Card(Suit.HERZ, Rank.SEVEN)
    challenger = Card(Suit.EICHEL, Rank.TEN)

    assert not card_beats(
        challenger,
        trump,
        lead,
        contract,
    )


def test_stronger_trump_beats_weaker_trump() -> None:
    contract = sauspiel()

    lead = Card(Suit.HERZ, Rank.SEVEN)
    current = Card(Suit.SCHELLEN, Rank.OBER)
    challenger = Card(Suit.EICHEL, Rank.OBER)

    assert card_beats(
        challenger,
        current,
        lead,
        contract,
    )


def test_weaker_trump_does_not_beat_stronger_trump() -> None:
    contract = sauspiel()

    lead = Card(Suit.HERZ, Rank.SEVEN)
    current = Card(Suit.EICHEL, Rank.OBER)
    challenger = Card(Suit.SCHELLEN, Rank.OBER)

    assert not card_beats(
        challenger,
        current,
        lead,
        contract,
    )


def test_wenz_ober_is_normal_suit_card() -> None:
    contract = GameContract(GameType.WENZ)

    ober = Card(Suit.EICHEL, Rank.OBER)
    nine = Card(Suit.EICHEL, Rank.NINE)

    assert plain_card_strength(ober, contract) > plain_card_strength(nine, contract)


def test_wenz_unter_is_trump() -> None:
    contract = GameContract(GameType.WENZ)

    lead = Card(Suit.EICHEL, Rank.ACE)
    unter = Card(Suit.SCHELLEN, Rank.UNTER)

    assert card_beats(
        unter,
        lead,
        lead,
        contract,
    )


def test_geier_unter_is_normal_suit_card() -> None:
    contract = GameContract(GameType.GEIER)

    unter = Card(Suit.GRAS, Rank.UNTER)
    nine = Card(Suit.GRAS, Rank.NINE)

    assert plain_card_strength(unter, contract) > plain_card_strength(nine, contract)


def test_geier_ober_is_trump() -> None:
    contract = GameContract(GameType.GEIER)

    lead = Card(Suit.EICHEL, Rank.ACE)
    ober = Card(Suit.SCHELLEN, Rank.OBER)

    assert card_beats(
        ober,
        lead,
        lead,
        contract,
    )


def test_solo_trump_suit_beats_other_suit() -> None:
    contract = GameContract(
        game_type=GameType.SOLO,
        trump_suit=Suit.GRAS,
    )

    lead = Card(Suit.EICHEL, Rank.ACE)
    trump = Card(Suit.GRAS, Rank.SEVEN)

    assert card_beats(
        trump,
        lead,
        lead,
        contract,
    )


def test_winning_play_returns_trump_player() -> None:
    contract = sauspiel()

    plays = [
        TrickPlay(
            player=0,
            card=Card(Suit.EICHEL, Rank.ACE),
        ),
        TrickPlay(
            player=1,
            card=Card(Suit.EICHEL, Rank.TEN),
        ),
        TrickPlay(
            player=2,
            card=Card(Suit.HERZ, Rank.SEVEN),
        ),
        TrickPlay(
            player=3,
            card=Card(Suit.GRAS, Rank.ACE),
        ),
    ]

    winner = winning_play(plays, contract)

    assert winner.player == 2


def test_off_suit_high_card_does_not_win() -> None:
    contract = GameContract(GameType.WENZ)

    plays = [
        TrickPlay(
            player=0,
            card=Card(Suit.GRAS, Rank.NINE),
        ),
        TrickPlay(
            player=1,
            card=Card(Suit.EICHEL, Rank.ACE),
        ),
        TrickPlay(
            player=2,
            card=Card(Suit.GRAS, Rank.KING),
        ),
        TrickPlay(
            player=3,
            card=Card(Suit.SCHELLEN, Rank.ACE),
        ),
    ]

    winner = winning_play(plays, contract)

    assert winner.player == 2


def test_empty_trick_has_no_winning_play() -> None:
    contract = sauspiel()

    with pytest.raises(
        ValueError,
        match="empty trick",
    ):
        winning_play([], contract)


def test_trick_enforces_turn_order() -> None:
    trick = Trick(starting_player=1)

    with pytest.raises(
        ValueError,
        match="player 1",
    ):
        trick.play_card(
            player=2,
            card=Card(Suit.EICHEL, Rank.ACE),
        )


def test_trick_wraps_player_order() -> None:
    trick = Trick(starting_player=3)

    assert trick.next_player == 3

    trick.play_card(
        3,
        Card(Suit.EICHEL, Rank.SEVEN),
    )

    assert trick.next_player == 0

    trick.play_card(
        0,
        Card(Suit.EICHEL, Rank.EIGHT),
    )

    assert trick.next_player == 1


def test_complete_trick_has_no_next_player() -> None:
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

    assert trick.is_complete
    assert trick.next_player is None


def test_cannot_play_fifth_card() -> None:
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
        match="already complete",
    ):
        trick.play_card(
            0,
            Card(Suit.EICHEL, Rank.ACE),
        )


def test_winner_requires_complete_trick() -> None:
    contract = sauspiel()

    trick = Trick(starting_player=0)

    trick.play_card(
        0,
        Card(Suit.EICHEL, Rank.SEVEN),
    )

    with pytest.raises(
        ValueError,
        match="before the trick is complete",
    ):
        trick.winner(contract)


def test_complete_trick_returns_correct_winner() -> None:
    contract = sauspiel()

    trick = Trick(starting_player=2)

    trick.play_card(
        2,
        Card(Suit.GRAS, Rank.ACE),
    )
    trick.play_card(
        3,
        Card(Suit.GRAS, Rank.TEN),
    )
    trick.play_card(
        0,
        Card(Suit.HERZ, Rank.EIGHT),
    )
    trick.play_card(
        1,
        Card(Suit.GRAS, Rank.KING),
    )

    winner = trick.winner(contract)

    assert winner.player == 0
    assert winner.card == Card(
        Suit.HERZ,
        Rank.EIGHT,
    )


def test_current_winner_can_be_determined_before_trick_complete() -> None:
    contract = sauspiel()

    trick = Trick(starting_player=0)

    trick.play_card(
        0,
        Card(Suit.GRAS, Rank.TEN),
    )
    trick.play_card(
        1,
        Card(Suit.GRAS, Rank.ACE),
    )

    winner = trick.current_winner(contract)

    assert winner.player == 1
    assert winner.card == Card(
        Suit.GRAS,
        Rank.ACE,
    )
