import pytest

from schafkopf_ai.game.game_type import GameType
from schafkopf_ai.game.rules import (
    RAMSCH_RULES,
    STANDARD_RULES,
    AllPassAction,
    GameRules,
)


def test_standard_rules_enable_expected_games() -> None:
    assert STANDARD_RULES.is_enabled(GameType.SAUSPIEL)
    assert STANDARD_RULES.is_enabled(GameType.SOLO)
    assert STANDARD_RULES.is_enabled(GameType.WENZ)


def test_standard_rules_disable_geier_and_ramsch() -> None:
    assert not STANDARD_RULES.is_enabled(GameType.GEIER)
    assert not STANDARD_RULES.is_enabled(GameType.RAMSCH)


def test_standard_rules_redeal_if_everyone_passes() -> None:
    assert STANDARD_RULES.all_pass_action is AllPassAction.REDEAL


def test_ramsch_rules_enable_ramsch() -> None:
    assert RAMSCH_RULES.is_enabled(GameType.RAMSCH)


def test_ramsch_rules_use_ramsch_after_all_pass() -> None:
    assert RAMSCH_RULES.all_pass_action is AllPassAction.RAMSCH


def test_ramsch_is_not_in_bid_game_types() -> None:
    assert GameType.RAMSCH not in RAMSCH_RULES.bid_game_types


def test_enabled_bid_games_are_returned() -> None:
    rules = GameRules(
        enabled_game_types=frozenset(
            {
                GameType.SOLO,
                GameType.WENZ,
                GameType.GEIER,
            }
        )
    )

    assert rules.bid_game_types == frozenset(
        {
            GameType.SOLO,
            GameType.WENZ,
            GameType.GEIER,
        }
    )


def test_custom_rules_can_disable_sauspiel() -> None:
    rules = GameRules(
        enabled_game_types=frozenset(
            {
                GameType.SOLO,
                GameType.WENZ,
            }
        )
    )

    assert not rules.is_enabled(GameType.SAUSPIEL)
    assert rules.is_enabled(GameType.SOLO)
    assert rules.is_enabled(GameType.WENZ)


def test_ramsch_all_pass_requires_ramsch_to_be_enabled() -> None:
    with pytest.raises(
        ValueError,
        match="Ramsch is not enabled",
    ):
        GameRules(
            enabled_game_types=frozenset(
                {
                    GameType.SAUSPIEL,
                    GameType.SOLO,
                    GameType.WENZ,
                }
            ),
            all_pass_action=AllPassAction.RAMSCH,
        )


def test_ramsch_can_be_enabled_without_being_all_pass_action() -> None:
    rules = GameRules(
        enabled_game_types=frozenset(
            {
                GameType.SAUSPIEL,
                GameType.SOLO,
                GameType.WENZ,
                GameType.RAMSCH,
            }
        ),
        all_pass_action=AllPassAction.REDEAL,
    )

    assert rules.is_enabled(GameType.RAMSCH)
    assert rules.all_pass_action is AllPassAction.REDEAL
