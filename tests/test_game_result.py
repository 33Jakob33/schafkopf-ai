import pytest

from schafkopf_ai.game.card import Suit
from schafkopf_ai.game.game_contract import GameContract
from schafkopf_ai.game.game_result import determine_game_result
from schafkopf_ai.game.game_type import GameType


def test_sauspiel_combines_declarer_and_called_player() -> None:
    contract = GameContract(
        GameType.SAUSPIEL,
        called_suit=Suit.EICHEL,
        declarer=0,
    )

    result = determine_game_result(
        contract=contract,
        player_points=(30, 20, 37, 33),
        trick_winners=(
            0,
            1,
            2,
            3,
            0,
            2,
            1,
            3,
        ),
        called_ace_player=2,
    )

    assert result.winner_players == (0, 2)
    assert result.loser_players == (1, 3)

    assert result.points == (67, 53)

    assert result.declarer_team == (0, 2)
    assert result.defender_team == (1, 3)

    assert result.declarer_won
    assert not result.schneider
    assert not result.schwarz


def test_declarer_wins_with_61_points() -> None:
    contract = GameContract(
        GameType.WENZ,
        declarer=0,
    )

    result = determine_game_result(
        contract=contract,
        player_points=(61, 20, 20, 19),
        trick_winners=(
            0,
            1,
            0,
            2,
            0,
            3,
            0,
            1,
        ),
    )

    assert result.declarer_won
    assert result.winner_players == (0,)
    assert result.points == (61, 59)


def test_declarer_loses_with_60_points() -> None:
    contract = GameContract(
        GameType.WENZ,
        declarer=0,
    )

    result = determine_game_result(
        contract=contract,
        player_points=(60, 20, 20, 20),
        trick_winners=(
            0,
            1,
            0,
            2,
            0,
            3,
            1,
            2,
        ),
    )

    assert not result.declarer_won
    assert result.winner_players == (
        1,
        2,
        3,
    )

    assert result.points == (
        60,
        60,
    )


def test_declarer_wins_schneider_at_91_points() -> None:
    contract = GameContract(
        GameType.WENZ,
        declarer=0,
    )

    result = determine_game_result(
        contract=contract,
        player_points=(91, 10, 10, 9),
        trick_winners=(
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
        ),
    )

    assert result.declarer_won
    assert result.schneider


def test_defenders_are_not_schneider_with_30_points() -> None:
    contract = GameContract(
        GameType.WENZ,
        declarer=0,
    )

    result = determine_game_result(
        contract=contract,
        player_points=(90, 10, 10, 10),
        trick_winners=(
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
        ),
    )

    assert result.declarer_won
    assert not result.schneider


def test_declarer_loses_schneider_with_30_points() -> None:
    contract = GameContract(
        GameType.WENZ,
        declarer=0,
    )

    result = determine_game_result(
        contract=contract,
        player_points=(30, 30, 30, 30),
        trick_winners=(
            0,
            1,
            2,
            3,
            1,
            2,
            3,
            1,
        ),
    )

    assert not result.declarer_won
    assert result.schneider


def test_declarer_is_not_schneider_with_31_points() -> None:
    contract = GameContract(
        GameType.WENZ,
        declarer=0,
    )

    result = determine_game_result(
        contract=contract,
        player_points=(31, 30, 30, 29),
        trick_winners=(
            0,
            1,
            2,
            3,
            1,
            2,
            3,
            1,
        ),
    )

    assert not result.declarer_won
    assert not result.schneider


def test_schwarz_when_losing_side_takes_no_trick() -> None:
    contract = GameContract(
        GameType.WENZ,
        declarer=0,
    )

    result = determine_game_result(
        contract=contract,
        player_points=(120, 0, 0, 0),
        trick_winners=(
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ),
    )

    assert result.declarer_won
    assert result.schneider
    assert result.schwarz


def test_zero_point_trick_prevents_schwarz() -> None:
    contract = GameContract(
        GameType.WENZ,
        declarer=0,
    )

    result = determine_game_result(
        contract=contract,
        player_points=(120, 0, 0, 0),
        trick_winners=(
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
        ),
    )

    assert result.declarer_won
    assert result.schneider
    assert not result.schwarz


@pytest.mark.parametrize(
    "contract",
    [
        GameContract(
            GameType.WENZ,
            declarer=2,
        ),
        GameContract(
            GameType.GEIER,
            declarer=2,
        ),
        GameContract(
            GameType.SOLO,
            trump_suit=Suit.GRAS,
            declarer=2,
        ),
    ],
)
def test_solo_games_are_one_against_three(
    contract: GameContract,
) -> None:
    result = determine_game_result(
        contract=contract,
        player_points=(15, 15, 70, 20),
        trick_winners=(
            2,
            2,
            0,
            2,
            1,
            2,
            3,
            2,
        ),
    )

    assert result.declarer_team == (2,)
    assert result.defender_team == (
        0,
        1,
        3,
    )

    assert result.winner_players == (2,)
    assert result.declarer_won


def test_ramsch_is_scored_individually() -> None:
    contract = GameContract(GameType.RAMSCH)

    result = determine_game_result(
        contract=contract,
        player_points=(
            20,
            50,
            10,
            40,
        ),
        trick_winners=(
            0,
            1,
            2,
            3,
            1,
            1,
            3,
            2,
        ),
    )

    assert result.points == (
        20,
        50,
        10,
        40,
    )

    assert result.winner_players == (2,)
    assert result.loser_players == (1,)

    assert result.declarer is None
    assert result.declarer_team is None
    assert result.defender_team is None
    assert result.declarer_won is None

    assert not result.schneider
    assert not result.schwarz


def test_result_requires_120_points() -> None:
    contract = GameContract(
        GameType.WENZ,
        declarer=0,
    )

    with pytest.raises(
        ValueError,
        match="120 Augen",
    ):
        determine_game_result(
            contract=contract,
            player_points=(60, 20, 20, 19),
            trick_winners=(
                0,
                1,
                2,
                3,
                0,
                1,
                2,
                3,
            ),
        )


def test_result_requires_eight_tricks() -> None:
    contract = GameContract(
        GameType.WENZ,
        declarer=0,
    )

    with pytest.raises(
        ValueError,
        match="8 trick winners",
    ):
        determine_game_result(
            contract=contract,
            player_points=(60, 20, 20, 20),
            trick_winners=(
                0,
                1,
                2,
            ),
        )


def test_team_game_requires_declarer() -> None:
    contract = GameContract(GameType.WENZ)

    with pytest.raises(
        ValueError,
        match="requires a declarer",
    ):
        determine_game_result(
            contract=contract,
            player_points=(60, 20, 20, 20),
            trick_winners=(
                0,
                1,
                2,
                3,
                0,
                1,
                2,
                3,
            ),
        )
