from schafkopf_ai.game.bidding import (
    BiddingAction,
    BiddingActionType,
    BiddingPhase,
    BiddingState,
    BidValues,
    callable_sauspiel_suits,
)
from schafkopf_ai.game.card import Card, Rank, Suit
from schafkopf_ai.game.deck import Deck
from schafkopf_ai.game.game_type import GameType
from schafkopf_ai.game.rules import (
    RAMSCH_RULES,
    STANDARD_RULES,
    AllPassAction,
    GameRules,
)


def create_hands() -> tuple[
    tuple[Card, ...],
    ...,
]:
    deck = Deck()
    return deck.deal()


def extended_rules() -> GameRules:
    return GameRules(
        enabled_game_types=frozenset(
            {
                GameType.SAUSPIEL,
                GameType.WENZ,
                GameType.GEIER,
                GameType.SOLO,
            }
        ),
        all_pass_action=AllPassAction.REDEAL,
    )


def test_wenz_and_geier_remain_separate_game_types() -> None:
    values = BidValues()

    assert GameType.WENZ is not GameType.GEIER

    assert values.value(GameType.WENZ) == values.value(GameType.GEIER) == 2


def test_bid_values_can_be_changed_independently() -> None:
    values = BidValues(
        sauspiel=1,
        geier=2,
        wenz=3,
        solo=4,
    )

    assert values.value(GameType.GEIER) == 2
    assert values.value(GameType.WENZ) == 3


def test_cannot_call_ace_you_hold() -> None:
    hand = (
        Card(Suit.EICHEL, Rank.ACE),
        Card(Suit.EICHEL, Rank.TEN),
    )

    assert Suit.EICHEL not in callable_sauspiel_suits(hand)


def test_plain_card_allows_calling_suit() -> None:
    hand = (Card(Suit.EICHEL, Rank.TEN),)

    assert Suit.EICHEL in callable_sauspiel_suits(hand)


def test_ober_does_not_allow_calling_suit() -> None:
    hand = (Card(Suit.EICHEL, Rank.OBER),)

    assert Suit.EICHEL not in callable_sauspiel_suits(hand)


def test_interest_round_follows_seating_order() -> None:
    state = BiddingState(
        hands=create_hands(),
        rules=STANDARD_RULES,
        starting_player=2,
    )

    assert state.current_player == 2

    state.apply_action(
        2,
        BiddingAction.play(),
    )

    assert state.current_player == 3

    state.apply_action(
        3,
        BiddingAction.pass_action(),
    )

    assert state.current_player == 0


def test_single_interested_player_goes_to_announcement() -> None:
    state = BiddingState(
        hands=create_hands(),
        rules=STANDARD_RULES,
    )

    state.apply_action(
        0,
        BiddingAction.play(),
    )

    state.apply_action(
        1,
        BiddingAction.pass_action(),
    )

    state.apply_action(
        2,
        BiddingAction.pass_action(),
    )

    state.apply_action(
        3,
        BiddingAction.pass_action(),
    )

    assert state.phase is BiddingPhase.ANNOUNCEMENT

    assert state.current_player == 0


def test_two_interested_players_enter_negotiation() -> None:
    state = BiddingState(
        hands=create_hands(),
        rules=STANDARD_RULES,
    )

    state.apply_action(
        0,
        BiddingAction.play(),
    )

    state.apply_action(
        1,
        BiddingAction.pass_action(),
    )

    state.apply_action(
        2,
        BiddingAction.play(),
    )

    state.apply_action(
        3,
        BiddingAction.pass_action(),
    )

    assert state.phase is BiddingPhase.NEGOTIATION

    assert state.incumbent == 0
    assert state.challenger == 2

    # Challenger acts first.
    assert state.current_player == 2

    # Sauspiel is the current lowest level.
    assert state.current_bid_value == 1


def test_challenger_raises_and_incumbent_holds() -> None:
    state = BiddingState(
        hands=create_hands(),
        rules=STANDARD_RULES,
    )

    state.apply_action(0, BiddingAction.play())
    state.apply_action(1, BiddingAction.pass_action())
    state.apply_action(2, BiddingAction.play())
    state.apply_action(3, BiddingAction.pass_action())

    # Player 2 must exceed Sauspiel level.
    legal = state.legal_actions(2)

    assert BiddingAction.raise_to(2) in legal

    state.apply_action(
        2,
        BiddingAction.raise_to(2),
    )

    # Now player 0 must answer.
    assert state.current_player == 0

    legal = state.legal_actions(0)

    assert BiddingAction.hold() in legal

    state.apply_action(
        0,
        BiddingAction.hold(),
    )

    # Player 2 must now either raise further or pass.
    assert state.current_player == 2

    legal = state.legal_actions(2)

    assert BiddingAction.raise_to(2) not in legal
    assert BiddingAction.raise_to(3) in legal
    assert BiddingAction.pass_action() in legal


def test_wenz_and_geier_are_both_valid_at_value_two() -> None:
    state = BiddingState(
        hands=create_hands(),
        rules=extended_rules(),
    )

    state.apply_action(0, BiddingAction.play())
    state.apply_action(1, BiddingAction.pass_action())
    state.apply_action(2, BiddingAction.play())
    state.apply_action(3, BiddingAction.pass_action())

    state.apply_action(
        2,
        BiddingAction.raise_to(2),
    )

    state.apply_action(
        0,
        BiddingAction.hold(),
    )

    state.apply_action(
        2,
        BiddingAction.pass_action(),
    )

    assert state.phase is BiddingPhase.ANNOUNCEMENT

    actions = state.legal_actions(0)

    announced_types = {
        action.contract.game_type
        for action in actions
        if (
            action.action_type is BiddingActionType.ANNOUNCE
            and action.contract is not None
        )
    }

    assert GameType.WENZ in announced_types
    assert GameType.GEIER in announced_types

    # Sauspiel is no longer possible because player 0
    # committed to value 2.
    assert GameType.SAUSPIEL not in announced_types


def test_winner_can_announce_wenz() -> None:
    state = BiddingState(
        hands=create_hands(),
        rules=STANDARD_RULES,
    )

    state.apply_action(0, BiddingAction.play())
    state.apply_action(1, BiddingAction.pass_action())
    state.apply_action(2, BiddingAction.pass_action())
    state.apply_action(3, BiddingAction.pass_action())

    actions = state.legal_actions(0)

    wenz_action = next(
        action
        for action in actions
        if (action.contract is not None and action.contract.game_type is GameType.WENZ)
    )

    state.apply_action(
        0,
        wenz_action,
    )

    assert state.is_complete

    result = state.result()

    assert result.contract is not None
    assert result.contract.game_type is GameType.WENZ
    assert result.contract.declarer == 0


def test_all_pass_causes_redeal() -> None:
    state = BiddingState(
        hands=create_hands(),
        rules=STANDARD_RULES,
    )

    for player in range(4):
        state.apply_action(
            player,
            BiddingAction.pass_action(),
        )

    assert state.is_complete

    result = state.result()

    assert result.requires_redeal
    assert result.contract is None


def test_all_pass_starts_ramsch() -> None:
    state = BiddingState(
        hands=create_hands(),
        rules=RAMSCH_RULES,
    )

    for player in range(4):
        state.apply_action(
            player,
            BiddingAction.pass_action(),
        )

    result = state.result()

    assert not result.requires_redeal
    assert result.contract is not None

    assert result.contract.game_type is GameType.RAMSCH

    assert result.contract.declarer is None
