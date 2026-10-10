from schafkopf_ai.game.card import Card, Rank, Suit
from schafkopf_ai.game.game_contract import GameContract
from schafkopf_ai.game.game_type import GameType
from schafkopf_ai.game.observation import PlayerObservation
from schafkopf_ai.game.trick import TrickPlay
from schafkopf_ai.training.card_play_encoding import (
    CARD_COUNT,
    PLAYER_COUNT,
    card_to_action_index,
)
from schafkopf_ai.training.card_play_encoding_v2 import OBSERVATION_V2_FEATURE_SIZE
from schafkopf_ai.training.card_play_encoding_v3 import (
    HISTORY_STEP_FEATURE_SIZE,
    MAX_PUBLIC_HISTORY_PLAYS,
    OBSERVATION_V3_FEATURE_SIZE,
    encode_player_observation_v3,
    encode_public_history,
)


def _observation() -> PlayerObservation:
    return PlayerObservation(
        player_index=2,
        hand=(
            Card(Suit.EICHEL, Rank.ACE),
            Card(Suit.HERZ, Rank.SEVEN),
        ),
        contract=GameContract(GameType.WENZ, declarer=3),
        current_player=2,
        current_trick=(
            TrickPlay(3, Card(Suit.GRAS, Rank.TEN)),
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


def test_v3_keeps_v2_prefix_and_fixed_history_size() -> None:
    observation = _observation()

    features = encode_player_observation_v3(observation)

    assert OBSERVATION_V3_FEATURE_SIZE == (
        OBSERVATION_V2_FEATURE_SIZE
        + MAX_PUBLIC_HISTORY_PLAYS * HISTORY_STEP_FEATURE_SIZE
    )
    assert len(features) == OBSERVATION_V3_FEATURE_SIZE


def test_v3_public_history_preserves_play_order() -> None:
    observation = _observation()

    history = encode_public_history(observation)

    first_base = 0
    assert history[
        first_base + card_to_action_index(Card(Suit.SCHELLEN, Rank.ACE))
    ] == 1.0

    # Observer is player 2, so absolute player 2 is relative player 0.
    assert history[first_base + CARD_COUNT] == 1.0

    # First card in its trick.
    position_offset = CARD_COUNT + PLAYER_COUNT
    assert history[first_base + position_offset] == 1.0

    fifth_base = 4 * HISTORY_STEP_FEATURE_SIZE
    assert history[
        fifth_base + card_to_action_index(Card(Suit.GRAS, Rank.TEN))
    ] == 1.0

    # Absolute player 3 is relative player 1 from observer 2.
    assert history[fifth_base + CARD_COUNT + 1] == 1.0

    # The current trick also starts at position zero.
    assert history[fifth_base + position_offset] == 1.0


def test_v3_public_history_zero_pads_future_steps() -> None:
    observation = _observation()

    history = encode_public_history(observation)

    first_unused = 5 * HISTORY_STEP_FEATURE_SIZE
    assert all(value == 0.0 for value in history[first_unused:])
