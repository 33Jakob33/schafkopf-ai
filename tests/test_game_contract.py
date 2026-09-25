import pytest

from schafkopf_ai.game.card import Suit
from schafkopf_ai.game.game_contract import GameContract
from schafkopf_ai.game.game_type import GameType


def test_solo_requires_trump_suit() -> None:
    with pytest.raises(
        ValueError,
        match="A Solo requires a trump suit",
    ):
        GameContract(game_type=GameType.SOLO)


def test_solo_accepts_trump_suit() -> None:
    contract = GameContract(
        game_type=GameType.SOLO,
        trump_suit=Suit.EICHEL,
    )

    assert contract.game_type is GameType.SOLO
    assert contract.trump_suit is Suit.EICHEL


def test_sauspiel_requires_called_suit() -> None:
    with pytest.raises(
        ValueError,
        match="requires a called suit",
    ):
        GameContract(GameType.SAUSPIEL)


def test_sauspiel_accepts_called_suit() -> None:
    contract = GameContract(
        GameType.SAUSPIEL,
        called_suit=Suit.EICHEL,
    )

    assert contract.called_suit is Suit.EICHEL


def test_sauspiel_cannot_call_herz() -> None:
    with pytest.raises(
        ValueError,
        match="Herz cannot be called",
    ):
        GameContract(
            GameType.SAUSPIEL,
            called_suit=Suit.HERZ,
        )


@pytest.mark.parametrize(
    "game_type",
    [
        GameType.SAUSPIEL,
        GameType.WENZ,
        GameType.GEIER,
        GameType.RAMSCH,
    ],
)
def test_non_solo_rejects_trump_suit(
    game_type: GameType,
) -> None:
    with pytest.raises(
        ValueError,
        match="does not use a configurable trump suit",
    ):
        GameContract(
            game_type=game_type,
            trump_suit=Suit.HERZ,
        )


@pytest.mark.parametrize(
    "game_type",
    [
        GameType.WENZ,
        GameType.GEIER,
        GameType.RAMSCH,
    ],
)
def test_simple_contract_requires_no_suit(
    game_type: GameType,
) -> None:
    contract = GameContract(game_type)

    assert contract.trump_suit is None
    assert contract.called_suit is None


@pytest.mark.parametrize(
    "game_type",
    [
        GameType.WENZ,
        GameType.GEIER,
        GameType.RAMSCH,
    ],
)
def test_non_solo_contract_does_not_require_trump_suit(
    game_type: GameType,
) -> None:
    contract = GameContract(game_type=game_type)

    assert contract.trump_suit is None
