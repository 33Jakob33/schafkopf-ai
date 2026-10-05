from __future__ import annotations

from typing import Literal, cast

from schafkopf_ai.game.card import Card
from schafkopf_ai.game.observation import PlayerObservation

from .card_play_encoding import (
    OBSERVATION_FEATURE_SIZE,
    EncodedCardPlayObservation,
    encode_card_play,
    encode_player_observation,
)
from .card_play_encoding_v2 import (
    OBSERVATION_V2_FEATURE_SIZE,
    encode_card_play_v2,
    encode_player_observation_v2,
)

ObservationVersion = Literal["v1", "v2"]
OBSERVATION_VERSIONS: tuple[ObservationVersion, ...] = ("v1", "v2")


def normalize_observation_version(value: str) -> ObservationVersion:
    normalized = value.lower()
    if normalized not in OBSERVATION_VERSIONS:
        raise ValueError(
            f"Unsupported observation version {value!r}; "
            f"expected one of {OBSERVATION_VERSIONS}."
        )
    return cast(ObservationVersion, normalized)


def feature_size_for_version(version: str) -> int:
    resolved = normalize_observation_version(version)
    if resolved == "v1":
        return OBSERVATION_FEATURE_SIZE
    return OBSERVATION_V2_FEATURE_SIZE


def try_infer_observation_version(input_size: int) -> ObservationVersion | None:
    if input_size == OBSERVATION_FEATURE_SIZE:
        return "v1"
    if input_size == OBSERVATION_V2_FEATURE_SIZE:
        return "v2"
    return None


def infer_observation_version(input_size: int) -> ObservationVersion:
    if input_size == OBSERVATION_FEATURE_SIZE:
        return "v1"
    if input_size == OBSERVATION_V2_FEATURE_SIZE:
        return "v2"

    raise ValueError(
        "Cannot infer card-play observation version from input size "
        f"{input_size}. Expected {OBSERVATION_FEATURE_SIZE} (v1) or "
        f"{OBSERVATION_V2_FEATURE_SIZE} (v2)."
    )


def resolve_checkpoint_observation_version(
    *,
    input_size: int,
    explicit_version: object | None,
) -> ObservationVersion:
    inferred = infer_observation_version(input_size)

    if explicit_version is None:
        return inferred
    if not isinstance(explicit_version, str):
        raise TypeError("Checkpoint observation_version must be a string.")

    resolved = normalize_observation_version(explicit_version)
    if resolved != inferred:
        raise ValueError(
            "Checkpoint observation_version does not match its input size: "
            f"{resolved} vs inferred {inferred}."
        )
    return resolved


def encode_card_play_for_version(
    version: str,
    observation: PlayerObservation,
    legal_cards: tuple[Card, ...],
) -> EncodedCardPlayObservation:
    resolved = normalize_observation_version(version)
    if resolved == "v1":
        return encode_card_play(observation, legal_cards)
    return encode_card_play_v2(observation, legal_cards)


def encode_player_observation_for_version(
    version: str,
    observation: PlayerObservation,
) -> tuple[float, ...]:
    resolved = normalize_observation_version(version)
    if resolved == "v1":
        return encode_player_observation(observation)
    return encode_player_observation_v2(observation)
