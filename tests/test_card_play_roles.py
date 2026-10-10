from schafkopf_ai.game.card import Card, Rank, Suit
from schafkopf_ai.game.game_contract import GameContract
from schafkopf_ai.game.game_type import GameType
from schafkopf_ai.game.observation import PlayerObservation
from schafkopf_ai.training.card_play_roles import (
    CardPlayRole,
    infer_card_play_role,
)


def _observation(
    *,
    player: int,
    contract: GameContract,
    known_called_ace_player: int | None = None,
) -> PlayerObservation:
    return PlayerObservation(
        player_index=player,
        hand=(Card(Suit.EICHEL, Rank.SEVEN),),
        contract=contract,
        current_player=player,
        current_trick=(),
        completed_tricks=(),
        points_by_player=(0, 0, 0, 0),
        called_ace_released=False,
        known_called_ace_player=known_called_ace_player,
    )


def test_sauspiel_roles_are_observer_known() -> None:
    contract = GameContract(
        GameType.SAUSPIEL,
        called_suit=Suit.GRAS,
        declarer=0,
    )

    assert (
        infer_card_play_role(_observation(player=0, contract=contract))
        is CardPlayRole.SAUSPIEL_DECLARER
    )
    assert (
        infer_card_play_role(
            _observation(
                player=2,
                contract=contract,
                known_called_ace_player=2,
            )
        )
        is CardPlayRole.SAUSPIEL_PARTNER
    )
    assert (
        infer_card_play_role(_observation(player=1, contract=contract))
        is CardPlayRole.SAUSPIEL_DEFENDER
    )


def test_solo_and_wenz_roles_distinguish_declarer_and_defender() -> None:
    solo = GameContract(GameType.SOLO, trump_suit=Suit.HERZ, declarer=1)
    wenz = GameContract(GameType.WENZ, declarer=3)

    assert (
        infer_card_play_role(_observation(player=1, contract=solo))
        is CardPlayRole.SOLO_DECLARER
    )
    assert (
        infer_card_play_role(_observation(player=0, contract=solo))
        is CardPlayRole.SOLO_DEFENDER
    )
    assert (
        infer_card_play_role(_observation(player=3, contract=wenz))
        is CardPlayRole.WENZ_DECLARER
    )
    assert (
        infer_card_play_role(_observation(player=2, contract=wenz))
        is CardPlayRole.WENZ_DEFENDER
    )


def test_ramsch_has_one_shared_role() -> None:
    contract = GameContract(GameType.RAMSCH)

    for player in range(4):
        assert (
            infer_card_play_role(_observation(player=player, contract=contract))
            is CardPlayRole.RAMSCH
        )
