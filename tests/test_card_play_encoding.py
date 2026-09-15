import pytest

from schafkopf_ai.game.card import Card, Rank, Suit
from schafkopf_ai.game.game_contract import GameContract
from schafkopf_ai.game.game_type import GameType
from schafkopf_ai.game.observation import PlayerObservation
from schafkopf_ai.game.trick import TrickPlay
from schafkopf_ai.training.card_play_encoding import (
    ACTION_COUNT,
    CARD_COUNT,
    OBSERVATION_FEATURE_SIZE,
    action_index_to_card,
    card_to_action_index,
    encode_card_play,
    encode_player_observation,
    legal_action_mask,
    relative_player_index,
)


def test_all_cards_have_unique_stable_action_indices() -> None:
    cards = [Card(suit, rank) for suit in Suit for rank in Rank]
    indices = [card_to_action_index(card) for card in cards]

    assert CARD_COUNT == 32
    assert ACTION_COUNT == 32
    assert len(indices) == 32
    assert set(indices) == set(range(32))

    for card in cards:
        assert action_index_to_card(card_to_action_index(card)) == card


def test_known_card_action_indices_are_stable() -> None:
    assert card_to_action_index(Card(Suit.EICHEL, Rank.SEVEN)) == 0
    assert card_to_action_index(Card(Suit.EICHEL, Rank.ACE)) == 7
    assert card_to_action_index(Card(Suit.GRAS, Rank.SEVEN)) == 8
    assert card_to_action_index(Card(Suit.HERZ, Rank.SEVEN)) == 16
    assert card_to_action_index(Card(Suit.SCHELLEN, Rank.ACE)) == 31


def test_action_index_to_card_rejects_invalid_indices() -> None:
    with pytest.raises(ValueError):
        action_index_to_card(-1)

    with pytest.raises(ValueError):
        action_index_to_card(32)


def test_legal_action_mask_marks_exactly_the_legal_cards() -> None:
    legal_cards = (
        Card(Suit.EICHEL, Rank.ACE),
        Card(Suit.HERZ, Rank.UNTER),
        Card(Suit.SCHELLEN, Rank.SEVEN),
    )

    mask = legal_action_mask(legal_cards)

    assert len(mask) == 32
    assert sum(mask) == 3

    for card in legal_cards:
        assert mask[card_to_action_index(card)] == 1

    assert mask[card_to_action_index(Card(Suit.GRAS, Rank.ACE))] == 0


def test_relative_player_indices_are_observer_relative() -> None:
    observer = 2

    assert relative_player_index(2, observer) == 0
    assert relative_player_index(3, observer) == 1
    assert relative_player_index(0, observer) == 2
    assert relative_player_index(1, observer) == 3


def create_observation() -> PlayerObservation:
    return PlayerObservation(
        player_index=2,
        hand=(
            Card(Suit.EICHEL, Rank.ACE),
            Card(Suit.HERZ, Rank.SEVEN),
        ),
        contract=GameContract(
            game_type=GameType.WENZ,
            declarer=3,
        ),
        current_player=0,
        current_trick=(
            TrickPlay(
                player=3,
                card=Card(Suit.GRAS, Rank.TEN),
            ),
        ),
        completed_tricks=(
            (
                TrickPlay(2, Card(Suit.SCHELLEN, Rank.ACE)),
                TrickPlay(3, Card(Suit.SCHELLEN, Rank.TEN)),
                TrickPlay(0, Card(Suit.SCHELLEN, Rank.KING)),
                TrickPlay(1, Card(Suit.SCHELLEN, Rank.NINE)),
            ),
        ),
        points_by_player=(10, 20, 30, 40),
        called_ace_released=False,
    )


def test_player_observation_has_fixed_numeric_size() -> None:
    observation = create_observation()

    features = encode_player_observation(observation)

    assert OBSERVATION_FEATURE_SIZE == 1254
    assert len(features) == OBSERVATION_FEATURE_SIZE
    assert all(isinstance(value, float) for value in features)


def test_hand_is_encoded_in_first_32_features() -> None:
    observation = create_observation()

    features = encode_player_observation(observation)
    hand_features = features[:CARD_COUNT]

    assert sum(hand_features) == 2.0
    assert hand_features[
        card_to_action_index(Card(Suit.EICHEL, Rank.ACE))
    ] == 1.0
    assert hand_features[
        card_to_action_index(Card(Suit.HERZ, Rank.SEVEN))
    ] == 1.0


def test_combined_card_play_encoding_contains_legal_mask() -> None:
    observation = create_observation()
    legal_cards = (Card(Suit.HERZ, Rank.SEVEN),)

    encoded = encode_card_play(observation, legal_cards)

    assert len(encoded.features) == OBSERVATION_FEATURE_SIZE
    assert len(encoded.legal_action_mask) == ACTION_COUNT
    assert sum(encoded.legal_action_mask) == 1
    assert (
        encoded.legal_action_mask[
            card_to_action_index(Card(Suit.HERZ, Rank.SEVEN))
        ]
        == 1
    )


def test_encoding_is_deterministic() -> None:
    observation = create_observation()

    first = encode_player_observation(observation)
    second = encode_player_observation(observation)

    assert first == second
