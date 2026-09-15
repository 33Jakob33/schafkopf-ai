from __future__ import annotations

from dataclasses import dataclass

from schafkopf_ai.game.card import Card, Rank, Suit
from schafkopf_ai.game.game_type import GameType
from schafkopf_ai.game.observation import PlayerObservation
from schafkopf_ai.game.trick import TrickPlay

SUIT_ORDER: tuple[Suit, ...] = (
    Suit.EICHEL,
    Suit.GRAS,
    Suit.HERZ,
    Suit.SCHELLEN,
)

RANK_ORDER: tuple[Rank, ...] = (
    Rank.SEVEN,
    Rank.EIGHT,
    Rank.NINE,
    Rank.UNTER,
    Rank.OBER,
    Rank.KING,
    Rank.TEN,
    Rank.ACE,
)

GAME_TYPE_ORDER: tuple[GameType, ...] = (
    GameType.SAUSPIEL,
    GameType.SOLO,
    GameType.WENZ,
    GameType.GEIER,
    GameType.RAMSCH,
)

PLAYER_COUNT = 4
MAX_TRICKS = 8
PLAYS_PER_TRICK = 4

CARD_COUNT = len(SUIT_ORDER) * len(RANK_ORDER)
ACTION_COUNT = CARD_COUNT

# One played-card slot contains:
# - 32-dimensional card one-hot,
# - 4-dimensional relative-player one-hot,
# - 1 occupied flag.
PLAY_FEATURE_SIZE = CARD_COUNT + PLAYER_COUNT + 1
HISTORY_FEATURE_SIZE = MAX_TRICKS * PLAYS_PER_TRICK * PLAY_FEATURE_SIZE

GAME_TYPE_FEATURE_SIZE = len(GAME_TYPE_ORDER)
OPTIONAL_SUIT_FEATURE_SIZE = len(SUIT_ORDER) + 1
OPTIONAL_PLAYER_FEATURE_SIZE = PLAYER_COUNT + 1
POINTS_FEATURE_SIZE = PLAYER_COUNT
TRICK_NUMBER_FEATURE_SIZE = MAX_TRICKS

OBSERVATION_FEATURE_SIZE = (
    CARD_COUNT
    + HISTORY_FEATURE_SIZE
    + GAME_TYPE_FEATURE_SIZE
    + OPTIONAL_SUIT_FEATURE_SIZE
    + OPTIONAL_SUIT_FEATURE_SIZE
    + OPTIONAL_PLAYER_FEATURE_SIZE
    + OPTIONAL_PLAYER_FEATURE_SIZE
    + POINTS_FEATURE_SIZE
    + 1
    + TRICK_NUMBER_FEATURE_SIZE
)

type FeatureVector = tuple[float, ...]
type ActionMask = tuple[int, ...]


@dataclass(frozen=True, slots=True)
class EncodedCardPlayObservation:
    """Numerical representation consumed by a future policy network."""

    features: FeatureVector
    legal_action_mask: ActionMask


def card_to_action_index(card: Card) -> int:
    """
    Map a card to its stable action index in the range 0..31.

    Layout:
        Eichel   -> 0..7
        Gras     -> 8..15
        Herz     -> 16..23
        Schellen -> 24..31

    Within each suit the rank order is:
        7, 8, 9, Unter, Ober, König, 10, Ass
    """
    suit_index = SUIT_ORDER.index(card.suit)
    rank_index = RANK_ORDER.index(card.rank)
    return suit_index * len(RANK_ORDER) + rank_index


def action_index_to_card(action_index: int) -> Card:
    """Return the card represented by a stable action index."""
    if not 0 <= action_index < ACTION_COUNT:
        raise ValueError(f"Card action index must be between 0 and {ACTION_COUNT - 1}.")

    suit_index, rank_index = divmod(action_index, len(RANK_ORDER))

    return Card(
        suit=SUIT_ORDER[suit_index],
        rank=RANK_ORDER[rank_index],
    )


def legal_action_mask(legal_cards: tuple[Card, ...]) -> ActionMask:
    """Return a 32-dimensional mask with 1 for every legal card action."""
    mask = [0] * ACTION_COUNT

    for card in legal_cards:
        mask[card_to_action_index(card)] = 1

    return tuple(mask)


def encode_card_play(
    observation: PlayerObservation,
    legal_cards: tuple[Card, ...],
) -> EncodedCardPlayObservation:
    """Encode a card-play observation together with its legal action mask."""
    return EncodedCardPlayObservation(
        features=encode_player_observation(observation),
        legal_action_mask=legal_action_mask(legal_cards),
    )


def encode_player_observation(
    observation: PlayerObservation,
) -> FeatureVector:
    """
    Convert public card-play information into a fixed-size feature vector.

    Player identities are encoded relative to the observing player:
        0 = self
        1 = next seat
        2 = opposite seat
        3 = previous seat

    The vector contains, in order:
        - own hand: 32 binary features,
        - complete public trick history: 1,184 features,
        - game type: 5 one-hot features,
        - configured trump suit / none: 5 one-hot features,
        - called suit / none: 5 one-hot features,
        - declarer / none: 5 one-hot features,
        - current player / none: 5 one-hot features,
        - Augen by relative player: 4 normalized features,
        - called-Ace-released flag: 1 feature,
        - current trick number: 8 one-hot features.
    """
    features: list[float] = []

    features.extend(_card_mask(observation.hand))
    features.extend(_encode_trick_history(observation))
    features.extend(_one_hot(GAME_TYPE_ORDER.index(observation.contract.game_type), 5))
    features.extend(_encode_optional_suit(observation.contract.trump_suit))
    features.extend(_encode_optional_suit(observation.contract.called_suit))
    features.extend(
        _encode_optional_player(
            observation.contract.declarer,
            observer=observation.player_index,
        )
    )
    features.extend(
        _encode_optional_player(
            observation.current_player,
            observer=observation.player_index,
        )
    )
    features.extend(_encode_points(observation))
    features.append(1.0 if observation.called_ace_released else 0.0)
    features.extend(_one_hot(observation.trick_number - 1, MAX_TRICKS))

    if len(features) != OBSERVATION_FEATURE_SIZE:
        raise RuntimeError(
            "Card-play observation encoding has an unexpected size: "
            f"{len(features)} != {OBSERVATION_FEATURE_SIZE}."
        )

    return tuple(features)


def relative_player_index(player: int, observer: int) -> int:
    """Map an absolute player index to a seat relative to the observer."""
    if not 0 <= player < PLAYER_COUNT:
        raise ValueError("Player index must be between 0 and 3.")

    if not 0 <= observer < PLAYER_COUNT:
        raise ValueError("Observer index must be between 0 and 3.")

    return (player - observer) % PLAYER_COUNT


def _card_mask(cards: tuple[Card, ...]) -> list[float]:
    mask = [0.0] * CARD_COUNT

    for card in cards:
        mask[card_to_action_index(card)] = 1.0

    return mask


def _encode_trick_history(observation: PlayerObservation) -> list[float]:
    encoded = [0.0] * HISTORY_FEATURE_SIZE

    tricks: list[tuple[TrickPlay, ...]] = list(observation.completed_tricks)

    if observation.current_trick:
        tricks.append(observation.current_trick)

    if len(tricks) > MAX_TRICKS:
        raise ValueError("Observation contains more than eight tricks.")

    for trick_index, trick in enumerate(tricks):
        if len(trick) > PLAYS_PER_TRICK:
            raise ValueError("A trick cannot contain more than four plays.")

        for play_index, play in enumerate(trick):
            base = (
                trick_index * PLAYS_PER_TRICK * PLAY_FEATURE_SIZE
                + play_index * PLAY_FEATURE_SIZE
            )

            card_index = card_to_action_index(play.card)
            encoded[base + card_index] = 1.0

            relative_player = relative_player_index(
                play.player,
                observation.player_index,
            )
            encoded[base + CARD_COUNT + relative_player] = 1.0

            occupied_index = base + CARD_COUNT + PLAYER_COUNT
            encoded[occupied_index] = 1.0

    return encoded


def _encode_optional_suit(suit: Suit | None) -> list[float]:
    # Position 0 represents None; suits occupy positions 1..4.
    index = 0 if suit is None else SUIT_ORDER.index(suit) + 1
    return _one_hot(index, OPTIONAL_SUIT_FEATURE_SIZE)


def _encode_optional_player(
    player: int | None,
    *,
    observer: int,
) -> list[float]:
    # Position 0 represents None; relative players occupy positions 1..4.
    index = 0 if player is None else relative_player_index(player, observer) + 1
    return _one_hot(index, OPTIONAL_PLAYER_FEATURE_SIZE)


def _encode_points(observation: PlayerObservation) -> list[float]:
    points: list[float] = []

    for relative_player in range(PLAYER_COUNT):
        absolute_player = (observation.player_index + relative_player) % PLAYER_COUNT
        points.append(observation.points_by_player[absolute_player] / 120.0)

    return points


def _one_hot(index: int, size: int) -> list[float]:
    if not 0 <= index < size:
        raise ValueError(f"One-hot index {index} is outside vector size {size}.")

    values = [0.0] * size
    values[index] = 1.0
    return values
