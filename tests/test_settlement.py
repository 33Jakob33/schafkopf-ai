from collections.abc import Mapping

from schafkopf_ai.game.card import Card, Rank, Suit
from schafkopf_ai.game.deck import Deck

from schafkopf_ai.game.game_contract import GameContract
from schafkopf_ai.game.game_result import determine_game_result
from schafkopf_ai.game.game_type import GameType
from schafkopf_ai.game.settlement import (
    GameValueRules,
    calculate_laufende,
    settle_game,
)


def create_hands_with_forced_cards(
    forced: Mapping[int, tuple[Card, ...]],
) -> tuple[tuple[Card, ...], ...]:
    hands: list[list[Card]] = [list(forced.get(player, ())) for player in range(4)]

    forced_cards = {card for cards in forced.values() for card in cards}

    remaining = [card for card in Deck().cards if card not in forced_cards]

    for player in range(4):
        while len(hands[player]) < 8:
            hands[player].append(remaining.pop(0))

    return tuple(tuple(hand) for hand in hands)


def test_sauspiel_base_value_is_ten() -> None:
    rules = GameValueRules()

    assert rules.base_value(GameType.SAUSPIEL) == 10


def test_solo_wenz_geier_base_value_is_twenty() -> None:
    rules = GameValueRules()

    assert rules.base_value(GameType.SOLO) == 20

    assert rules.base_value(GameType.WENZ) == 20

    assert rules.base_value(GameType.GEIER) == 20


def test_three_laufende_count_in_solo() -> None:
    contract = GameContract(
        GameType.SOLO,
        trump_suit=Suit.HERZ,
        declarer=0,
    )

    hands = create_hands_with_forced_cards(
        {
            0: (
                Card(Suit.EICHEL, Rank.OBER),
                Card(Suit.GRAS, Rank.OBER),
                Card(Suit.HERZ, Rank.OBER),
            ),
            1: (Card(Suit.SCHELLEN, Rank.OBER),),
        }
    )

    count = calculate_laufende(
        contract=contract,
        initial_hands=hands,
        declarer_team=(0,),
    )

    assert count == 3


def test_two_laufende_do_not_count_in_solo() -> None:
    contract = GameContract(
        GameType.SOLO,
        trump_suit=Suit.HERZ,
        declarer=0,
    )

    hands = create_hands_with_forced_cards(
        {
            0: (
                Card(Suit.EICHEL, Rank.OBER),
                Card(Suit.GRAS, Rank.OBER),
            ),
            1: (Card(Suit.HERZ, Rank.OBER),),
        }
    )

    count = calculate_laufende(
        contract=contract,
        initial_hands=hands,
        declarer_team=(0,),
    )

    assert count == 0


def test_two_laufende_count_in_wenz() -> None:
    contract = GameContract(
        GameType.WENZ,
        declarer=0,
    )

    hands = create_hands_with_forced_cards(
        {
            0: (
                Card(Suit.EICHEL, Rank.UNTER),
                Card(Suit.GRAS, Rank.UNTER),
            ),
            1: (Card(Suit.HERZ, Rank.UNTER),),
        }
    )

    count = calculate_laufende(
        contract=contract,
        initial_hands=hands,
        declarer_team=(0,),
    )

    assert count == 2


def test_solo_value_with_laufende_schneider_and_schwarz() -> None:
    contract = GameContract(
        GameType.SOLO,
        trump_suit=Suit.HERZ,
        declarer=0,
    )

    hands = create_hands_with_forced_cards(
        {
            0: (
                Card(Suit.EICHEL, Rank.OBER),
                Card(Suit.GRAS, Rank.OBER),
                Card(Suit.HERZ, Rank.OBER),
            ),
            1: (Card(Suit.SCHELLEN, Rank.OBER),),
        }
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

    settlement = settle_game(
        result=result,
        contract=contract,
        initial_hands=hands,
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

    assert settlement.base_value == 20

    assert settlement.laufende_count == 3
    assert settlement.laufende_bonus == 30

    assert settlement.schneider_bonus == 10
    assert settlement.schwarz_bonus == 10

    assert settlement.game_value == 70


def test_solo_winner_receives_payment_from_all_three_losers() -> None:
    contract = GameContract(
        GameType.WENZ,
        declarer=0,
    )

    result = determine_game_result(
        contract=contract,
        player_points=(70, 20, 20, 10),
        trick_winners=(
            0,
            0,
            1,
            0,
            2,
            0,
            3,
            0,
        ),
    )

    hands = create_hands_with_forced_cards({})

    settlement = settle_game(
        result=result,
        contract=contract,
        initial_hands=hands,
        trick_winners=(
            0,
            0,
            1,
            0,
            2,
            0,
            3,
            0,
        ),
    )

    assert sum(settlement.payments) == 0


def test_ramsch_loser_pays_ten_to_every_player() -> None:
    contract = GameContract(GameType.RAMSCH)

    trick_winners = (
        0,
        1,
        2,
        3,
        0,
        1,
        2,
        3,
    )

    result = determine_game_result(
        contract=contract,
        player_points=(
            20,
            50,
            10,
            40,
        ),
        trick_winners=trick_winners,
    )

    settlement = settle_game(
        result=result,
        contract=contract,
        initial_hands=create_hands_with_forced_cards({}),
        trick_winners=trick_winners,
    )

    assert settlement.jungfrau_players == ()

    assert settlement.game_value == 10

    assert settlement.payments == (
        10,
        -30,
        10,
        10,
    )


def test_one_jungfrau_doubles_ramsch() -> None:
    contract = GameContract(GameType.RAMSCH)

    trick_winners = (
        0,
        1,
        1,
        2,
        0,
        1,
        2,
        0,
    )

    result = determine_game_result(
        contract=contract,
        player_points=(
            20,
            50,
            10,
            40,
        ),
        trick_winners=trick_winners,
    )

    settlement = settle_game(
        result=result,
        contract=contract,
        initial_hands=create_hands_with_forced_cards({}),
        trick_winners=trick_winners,
    )

    assert settlement.jungfrau_players == (3,)

    assert settlement.game_value == 20

    assert settlement.payments == (
        20,
        -60,
        20,
        20,
    )
