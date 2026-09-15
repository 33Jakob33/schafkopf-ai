import random

import pytest

from schafkopf_ai.agents.random_agent import RandomAgent
from schafkopf_ai.game.card import Card, Rank, Suit
from schafkopf_ai.game.deck import Deck
from schafkopf_ai.game.game_contract import GameContract
from schafkopf_ai.game.game_state import GameState
from schafkopf_ai.game.game_type import GameType
from schafkopf_ai.game.scoring import (
    TOTAL_GAME_POINTS,
    card_points,
    points_by_player,
    total_points,
    trick_points,
    validate_complete_game_points,
)
from schafkopf_ai.game.trick import Trick
from schafkopf_ai.runner.game_runner import GameRunner


@pytest.mark.parametrize(
    ("rank", "expected_points"),
    [
        (Rank.ACE, 11),
        (Rank.TEN, 10),
        (Rank.KING, 4),
        (Rank.OBER, 3),
        (Rank.UNTER, 2),
        (Rank.NINE, 0),
        (Rank.EIGHT, 0),
        (Rank.SEVEN, 0),
    ],
)
def test_card_points(
    rank: Rank,
    expected_points: int,
) -> None:
    card = Card(
        Suit.EICHEL,
        rank,
    )

    assert card_points(card) == expected_points


def test_complete_deck_contains_120_points() -> None:
    deck = Deck()

    points = sum(card_points(card) for card in deck.cards)

    assert points == TOTAL_GAME_POINTS
    assert points == 120


def test_trick_points() -> None:
    trick = Trick(starting_player=0)

    trick.play_card(
        0,
        Card(Suit.EICHEL, Rank.ACE),
    )
    trick.play_card(
        1,
        Card(Suit.EICHEL, Rank.TEN),
    )
    trick.play_card(
        2,
        Card(Suit.SCHELLEN, Rank.UNTER),
    )
    trick.play_card(
        3,
        Card(Suit.EICHEL, Rank.KING),
    )

    # 11 + 10 + 2 + 4
    assert trick_points(trick) == 27


def test_incomplete_trick_cannot_be_scored() -> None:
    trick = Trick(starting_player=0)

    trick.play_card(
        0,
        Card(Suit.EICHEL, Rank.ACE),
    )

    with pytest.raises(
        ValueError,
        match="incomplete trick",
    ):
        trick_points(trick)


def test_points_are_awarded_to_trick_winner() -> None:
    contract = GameContract(GameType.WENZ)

    trick = Trick(starting_player=0)

    trick.play_card(
        0,
        Card(Suit.EICHEL, Rank.ACE),
    )
    trick.play_card(
        1,
        Card(Suit.EICHEL, Rank.TEN),
    )
    trick.play_card(
        2,
        Card(Suit.SCHELLEN, Rank.UNTER),
    )
    trick.play_card(
        3,
        Card(Suit.EICHEL, Rank.KING),
    )

    # In a Wenz, Schellen Unter is trump.
    # Player 2 therefore wins all 27 Augen.
    points = points_by_player(
        [trick],
        contract,
    )

    assert points == (
        0,
        0,
        27,
        0,
    )


def test_points_from_multiple_tricks_are_accumulated() -> None:
    contract = GameContract(GameType.WENZ)

    trick_one = Trick(starting_player=0)

    trick_one.play_card(
        0,
        Card(Suit.EICHEL, Rank.ACE),
    )
    trick_one.play_card(
        1,
        Card(Suit.EICHEL, Rank.TEN),
    )
    trick_one.play_card(
        2,
        Card(Suit.SCHELLEN, Rank.UNTER),
    )
    trick_one.play_card(
        3,
        Card(Suit.EICHEL, Rank.KING),
    )

    # Player 2 wins: 27 Augen.

    trick_two = Trick(starting_player=0)

    trick_two.play_card(
        0,
        Card(Suit.GRAS, Rank.SEVEN),
    )
    trick_two.play_card(
        1,
        Card(Suit.GRAS, Rank.ACE),
    )
    trick_two.play_card(
        2,
        Card(Suit.GRAS, Rank.TEN),
    )
    trick_two.play_card(
        3,
        Card(Suit.EICHEL, Rank.NINE),
    )

    # Player 1 wins: 21 Augen.

    points = points_by_player(
        [trick_one, trick_two],
        contract,
    )

    assert points == (
        0,
        21,
        27,
        0,
    )


def test_total_points_sums_all_tricks() -> None:
    contract = GameContract(GameType.WENZ)

    trick = Trick(starting_player=0)

    trick.play_card(
        0,
        Card(Suit.EICHEL, Rank.ACE),
    )
    trick.play_card(
        1,
        Card(Suit.EICHEL, Rank.TEN),
    )
    trick.play_card(
        2,
        Card(Suit.SCHELLEN, Rank.UNTER),
    )
    trick.play_card(
        3,
        Card(Suit.EICHEL, Rank.KING),
    )

    # Contract isn't needed to count Augen, only to determine
    # who receives them.
    assert trick_points(trick) == 27

    points = points_by_player(
        [trick],
        contract,
    )

    assert sum(points) == 27
    assert total_points([trick]) == 27


def create_random_game(
    seed: int = 42,
) -> GameState:
    deck = Deck()
    deck.shuffle(random.Random(seed))

    hands = deck.deal()

    state = GameState.from_hands(
        hands=hands,
        contract=GameContract(GameType.WENZ),
        starting_player=0,
    )

    agents = [
        RandomAgent(rng=random.Random(seed + player_index + 1))
        for player_index in range(4)
    ]

    runner = GameRunner(
        state=state,
        agents=agents,
    )

    return runner.run()


def test_completed_random_game_contains_120_points() -> None:
    state = create_random_game()

    assert state.is_complete

    assert total_points(state.completed_tricks) == TOTAL_GAME_POINTS


def test_player_points_sum_to_120() -> None:
    state = create_random_game()

    points = points_by_player(
        state.completed_tricks,
        state.contract,
    )

    assert sum(points) == TOTAL_GAME_POINTS


def test_complete_game_passes_point_validation() -> None:
    state = create_random_game()

    validate_complete_game_points(state.completed_tricks)


def test_incomplete_game_fails_point_validation() -> None:
    state = create_random_game()

    with pytest.raises(
        ValueError,
        match="exactly 8 tricks",
    ):
        validate_complete_game_points(state.completed_tricks[:7])
