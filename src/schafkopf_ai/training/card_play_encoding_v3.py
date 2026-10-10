from __future__ import annotations

from schafkopf_ai.game.card import Card
from schafkopf_ai.game.observation import PlayerObservation
from schafkopf_ai.game.trick import TrickPlay

from .card_play_encoding import (
    CARD_COUNT,
    PLAYER_COUNT,
    EncodedCardPlayObservation,
    FeatureVector,
    card_to_action_index,
    legal_action_mask,
    relative_player_index,
)
from .card_play_encoding_v2 import (
    OBSERVATION_V2_FEATURE_SIZE,
    encode_player_observation_v2,
)

MAX_PUBLIC_HISTORY_PLAYS = 31
TRICK_POSITION_COUNT = 4
HISTORY_STEP_FEATURE_SIZE = CARD_COUNT + PLAYER_COUNT + TRICK_POSITION_COUNT
HISTORY_FEATURE_SIZE = MAX_PUBLIC_HISTORY_PLAYS * HISTORY_STEP_FEATURE_SIZE
OBSERVATION_V3_FEATURE_SIZE = OBSERVATION_V2_FEATURE_SIZE + HISTORY_FEATURE_SIZE


def encode_card_play_v3(
    observation: PlayerObservation,
    legal_cards: tuple[Card, ...],
) -> EncodedCardPlayObservation:
    """Encode V2 structured state plus an ordered public-play history."""
    return EncodedCardPlayObservation(
        features=encode_player_observation_v3(observation),
        legal_action_mask=legal_action_mask(legal_cards),
    )


def encode_player_observation_v3(
    observation: PlayerObservation,
) -> FeatureVector:
    """
    Encode the V2 structured observation plus public card-play chronology.

    The history branch stores up to 31 public plays in exact play order. Each
    history step contains only public information:

    - played card (32-way one-hot),
    - player relative to the observer (4-way one-hot),
    - position inside its trick (4-way one-hot).

    Unused future slots are zero padded. The structured V2 prefix remains
    unchanged so the GRU model can combine explicit Schafkopf summaries with
    temporal information.
    """
    structured = encode_player_observation_v2(observation)
    history = encode_public_history(observation)
    features = structured + history

    if len(features) != OBSERVATION_V3_FEATURE_SIZE:
        raise RuntimeError(
            "V3 card-play observation encoding has an unexpected size: "
            f"{len(features)} != {OBSERVATION_V3_FEATURE_SIZE}."
        )

    return features


def encode_public_history(observation: PlayerObservation) -> FeatureVector:
    """Return the padded ordered public-play sequence for the GRU branch."""
    history = [0.0] * HISTORY_FEATURE_SIZE
    plays = _public_plays_with_positions(observation)

    if len(plays) > MAX_PUBLIC_HISTORY_PLAYS:
        raise ValueError(
            "A card-play decision cannot observe more than 31 public plays."
        )

    for step_index, (play, trick_position) in enumerate(plays):
        base = step_index * HISTORY_STEP_FEATURE_SIZE

        history[base + card_to_action_index(play.card)] = 1.0

        relative_player = relative_player_index(
            play.player,
            observation.player_index,
        )
        history[base + CARD_COUNT + relative_player] = 1.0

        position_offset = CARD_COUNT + PLAYER_COUNT
        history[base + position_offset + trick_position] = 1.0

    return tuple(history)


def _public_plays_with_positions(
    observation: PlayerObservation,
) -> tuple[tuple[TrickPlay, int], ...]:
    plays: list[tuple[TrickPlay, int]] = []

    for trick in observation.completed_tricks:
        for trick_position, play in enumerate(trick):
            plays.append((play, trick_position))

    for trick_position, play in enumerate(observation.current_trick):
        plays.append((play, trick_position))

    return tuple(plays)
