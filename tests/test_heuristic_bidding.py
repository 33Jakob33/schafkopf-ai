from schafkopf_ai.agents.heuristic_bidding import (
    BiddingHeuristicConfig,
    evaluate_contract,
    preferred_contract,
)
from schafkopf_ai.game.card import Card, Rank, Suit
from schafkopf_ai.game.game_contract import GameContract
from schafkopf_ai.game.game_type import GameType


def test_weak_five_trump_solo_is_rejected_without_control() -> None:
    hand = (
        Card(Suit.SCHELLEN, Rank.OBER),
        Card(Suit.SCHELLEN, Rank.UNTER),
        Card(Suit.HERZ, Rank.NINE),
        Card(Suit.HERZ, Rank.EIGHT),
        Card(Suit.HERZ, Rank.SEVEN),
        Card(Suit.EICHEL, Rank.SEVEN),
        Card(Suit.GRAS, Rank.EIGHT),
        Card(Suit.SCHELLEN, Rank.NINE),
    )
    contract = GameContract(
        GameType.SOLO,
        trump_suit=Suit.HERZ,
        declarer=0,
    )

    evaluation = evaluate_contract(hand, contract)

    assert not evaluation.eligible


def test_strong_controlled_solo_is_eligible() -> None:
    hand = (
        Card(Suit.EICHEL, Rank.OBER),
        Card(Suit.GRAS, Rank.OBER),
        Card(Suit.EICHEL, Rank.UNTER),
        Card(Suit.GRAS, Rank.UNTER),
        Card(Suit.HERZ, Rank.ACE),
        Card(Suit.HERZ, Rank.TEN),
        Card(Suit.SCHELLEN, Rank.ACE),
        Card(Suit.GRAS, Rank.SEVEN),
    )
    contract = GameContract(
        GameType.SOLO,
        trump_suit=Suit.HERZ,
        declarer=0,
    )

    evaluation = evaluate_contract(hand, contract)

    assert evaluation.eligible
    assert 0.0 <= evaluation.confidence <= 1.0


def test_wenz_values_high_unter_quality_more_than_low_unter_quality() -> None:
    strong_hand = (
        Card(Suit.EICHEL, Rank.UNTER),
        Card(Suit.GRAS, Rank.UNTER),
        Card(Suit.EICHEL, Rank.ACE),
        Card(Suit.GRAS, Rank.ACE),
        Card(Suit.HERZ, Rank.TEN),
        Card(Suit.SCHELLEN, Rank.KING),
        Card(Suit.HERZ, Rank.NINE),
        Card(Suit.SCHELLEN, Rank.SEVEN),
    )
    weak_hand = (
        Card(Suit.HERZ, Rank.UNTER),
        Card(Suit.SCHELLEN, Rank.UNTER),
        Card(Suit.EICHEL, Rank.ACE),
        Card(Suit.GRAS, Rank.ACE),
        Card(Suit.HERZ, Rank.TEN),
        Card(Suit.SCHELLEN, Rank.KING),
        Card(Suit.HERZ, Rank.NINE),
        Card(Suit.SCHELLEN, Rank.SEVEN),
    )
    contract = GameContract(GameType.WENZ, declarer=0)

    strong = evaluate_contract(strong_hand, contract)
    weak = evaluate_contract(weak_hand, contract)

    assert strong.confidence > weak.confidence
    assert strong.eligible
    assert not weak.eligible


def test_sauspiel_uses_called_suit_structure_and_side_support() -> None:
    good_hand = (
        Card(Suit.EICHEL, Rank.OBER),
        Card(Suit.GRAS, Rank.UNTER),
        Card(Suit.HERZ, Rank.ACE),
        Card(Suit.HERZ, Rank.TEN),
        Card(Suit.GRAS, Rank.SEVEN),
        Card(Suit.EICHEL, Rank.ACE),
        Card(Suit.SCHELLEN, Rank.ACE),
        Card(Suit.SCHELLEN, Rank.SEVEN),
    )
    weaker_hand = (
        Card(Suit.SCHELLEN, Rank.OBER),
        Card(Suit.SCHELLEN, Rank.UNTER),
        Card(Suit.HERZ, Rank.NINE),
        Card(Suit.GRAS, Rank.TEN),
        Card(Suit.GRAS, Rank.KING),
        Card(Suit.GRAS, Rank.NINE),
        Card(Suit.EICHEL, Rank.SEVEN),
        Card(Suit.SCHELLEN, Rank.EIGHT),
    )
    contract = GameContract(
        GameType.SAUSPIEL,
        called_suit=Suit.GRAS,
        declarer=0,
    )

    good = evaluate_contract(good_hand, contract)
    weaker = evaluate_contract(weaker_hand, contract)

    assert good.confidence > weaker.confidence
    assert good.eligible


def test_solo_requires_margin_over_good_sauspiel() -> None:
    hand = (
        Card(Suit.EICHEL, Rank.OBER),
        Card(Suit.GRAS, Rank.UNTER),
        Card(Suit.HERZ, Rank.ACE),
        Card(Suit.HERZ, Rank.TEN),
        Card(Suit.GRAS, Rank.SEVEN),
        Card(Suit.EICHEL, Rank.ACE),
        Card(Suit.SCHELLEN, Rank.ACE),
        Card(Suit.SCHELLEN, Rank.SEVEN),
    )
    sauspiel = GameContract(
        GameType.SAUSPIEL,
        called_suit=Suit.GRAS,
        declarer=0,
    )
    solo = GameContract(
        GameType.SOLO,
        trump_suit=Suit.HERZ,
        declarer=0,
    )

    config = BiddingHeuristicConfig(solo_margin=0.10)
    choice = preferred_contract(
        hand,
        (sauspiel, solo),
        config=config,
    )

    assert choice is not None
    assert choice.contract == sauspiel


def test_exceptionally_strong_solo_can_beat_sauspiel_margin() -> None:
    hand = (
        Card(Suit.EICHEL, Rank.OBER),
        Card(Suit.GRAS, Rank.OBER),
        Card(Suit.HERZ, Rank.OBER),
        Card(Suit.EICHEL, Rank.UNTER),
        Card(Suit.GRAS, Rank.UNTER),
        Card(Suit.HERZ, Rank.ACE),
        Card(Suit.HERZ, Rank.TEN),
        Card(Suit.GRAS, Rank.SEVEN),
    )
    sauspiel = GameContract(
        GameType.SAUSPIEL,
        called_suit=Suit.GRAS,
        declarer=0,
    )
    solo = GameContract(
        GameType.SOLO,
        trump_suit=Suit.HERZ,
        declarer=0,
    )

    choice = preferred_contract(hand, (sauspiel, solo))

    assert choice is not None
    assert choice.contract == solo
