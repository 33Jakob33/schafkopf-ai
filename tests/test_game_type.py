from schafkopf_ai.game.game_type import GameType


def test_game_type_values() -> None:
    assert GameType.SAUSPIEL.value == "Sauspiel"
    assert GameType.SOLO.value == "Solo"
    assert GameType.WENZ.value == "Wenz"
    assert GameType.GEIER.value == "Geier"
    assert GameType.RAMSCH.value == "Ramsch"


def test_normal_game_types_can_be_bid() -> None:
    assert GameType.SAUSPIEL.can_be_bid
    assert GameType.SOLO.can_be_bid
    assert GameType.WENZ.can_be_bid
    assert GameType.GEIER.can_be_bid


def test_ramsch_cannot_be_bid() -> None:
    assert not GameType.RAMSCH.can_be_bid


def test_game_type_string_representation() -> None:
    assert str(GameType.SAUSPIEL) == "Sauspiel"
    assert str(GameType.RAMSCH) == "Ramsch"
