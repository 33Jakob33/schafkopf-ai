from __future__ import annotations

from schafkopf_ai.game.card import Card, Suit
from schafkopf_ai.game.game_type import GameType
from schafkopf_ai.game.observation import PlayerObservation
from schafkopf_ai.game.scoring import card_points
from schafkopf_ai.game.trick import TrickPlay, winning_play
from schafkopf_ai.game.trump import is_trump, trump_order

from .card_play_encoding import (
    ACTION_COUNT,
    CARD_COUNT,
    GAME_TYPE_ORDER,
    MAX_TRICKS,
    OPTIONAL_PLAYER_FEATURE_SIZE,
    OPTIONAL_SUIT_FEATURE_SIZE,
    PLAYER_COUNT,
    PLAYS_PER_TRICK,
    RANK_ORDER,
    SUIT_ORDER,
    ActionMask,
    EncodedCardPlayObservation,
    FeatureVector,
    card_to_action_index,
    legal_action_mask,
    relative_player_index,
)

CURRENT_TRICK_PLAY_FEATURE_SIZE = CARD_COUNT + PLAYER_COUNT + 1
CURRENT_TRICK_FEATURE_SIZE = PLAYS_PER_TRICK * CURRENT_TRICK_PLAY_FEATURE_SIZE
KNOWN_VOID_CATEGORY_COUNT = 1 + len(SUIT_ORDER)  # trump + four plain suits

# V2 deliberately replaces the 1,184-feature raw trick history with compact,
# contract-aware summaries derived only from public information.
OBSERVATION_V2_FEATURE_SIZE = (
    CARD_COUNT  # own hand
    + CURRENT_TRICK_FEATURE_SIZE
    + 1  # current trick points
    + OPTIONAL_PLAYER_FEATURE_SIZE  # current winning player
    + len(GAME_TYPE_ORDER)
    + OPTIONAL_SUIT_FEATURE_SIZE  # configured trump suit / none
    + OPTIONAL_SUIT_FEATURE_SIZE  # called suit / none
    + OPTIONAL_PLAYER_FEATURE_SIZE  # declarer / none
    + 1  # called Ace released
    + 1  # normalized trick progress
    + PLAYER_COUNT  # points by player
    + 1  # team information known
    + PLAYER_COUNT  # observer-team membership by relative player
    + 2  # observer team / opposing team points
    + CARD_COUNT  # unseen trump-card mask
    + 1  # unseen trump count
    + len(SUIT_ORDER)  # unseen plain-card count per suit
    + len(SUIT_ORDER)  # unseen plain-card points per suit
    + PLAYER_COUNT * KNOWN_VOID_CATEGORY_COUNT
)

_ALL_CARDS: tuple[Card, ...] = tuple(
    Card(suit=suit, rank=rank) for suit in SUIT_ORDER for rank in RANK_ORDER
)


def encode_card_play_v2(
    observation: PlayerObservation,
    legal_cards: tuple[Card, ...],
) -> EncodedCardPlayObservation:
    """Encode one card-play decision with the compact V2 observation."""
    return EncodedCardPlayObservation(
        features=encode_player_observation_v2(observation),
        legal_action_mask=legal_action_mask(legal_cards),
    )


def encode_player_observation_v2(
    observation: PlayerObservation,
) -> FeatureVector:
    """
    Encode a compact, contract-aware card-play observation.

    The vector contains only the observer's private hand plus information that
    can be derived from public play:

    - own hand,
    - current trick and current winner,
    - contract/declarer information,
    - points by player and team points when team membership is known,
    - unseen trumps (mask + count),
    - unseen non-trump card counts and Augen by suit,
    - known void categories for every player.

    Historical trick slots are not included. Completed tricks are summarized
    into points, remaining-card information, and known voids instead.
    """
    features: list[float] = []

    unseen_cards = _unseen_cards(observation)
    team_known, observer_team = _observer_team(observation)

    features.extend(_card_mask(observation.hand))
    features.extend(_encode_current_trick(observation))
    features.append(_current_trick_points(observation))
    features.extend(_encode_current_winner(observation))

    features.extend(
        _one_hot(GAME_TYPE_ORDER.index(observation.contract.game_type), len(GAME_TYPE_ORDER))
    )
    features.extend(_encode_optional_suit(observation.contract.trump_suit))
    features.extend(_encode_optional_suit(observation.contract.called_suit))
    features.extend(
        _encode_optional_player(
            observation.contract.declarer,
            observer=observation.player_index,
        )
    )

    features.append(1.0 if observation.called_ace_released else 0.0)
    features.append((observation.trick_number - 1) / (MAX_TRICKS - 1))
    features.extend(_encode_points_by_player(observation))

    features.append(1.0 if team_known else 0.0)
    features.extend(_encode_team_membership(observation, observer_team))
    features.extend(_encode_team_points(observation, observer_team))

    features.extend(_encode_unseen_trumps(observation, unseen_cards))
    features.append(_normalized_unseen_trump_count(observation, unseen_cards))
    features.extend(_encode_unseen_suit_counts(observation, unseen_cards))
    features.extend(_encode_unseen_suit_points(observation, unseen_cards))

    features.extend(_encode_known_voids(observation, unseen_cards))

    if len(features) != OBSERVATION_V2_FEATURE_SIZE:
        raise RuntimeError(
            "V2 card-play observation encoding has an unexpected size: "
            f"{len(features)} != {OBSERVATION_V2_FEATURE_SIZE}."
        )

    return tuple(features)


def _card_mask(cards: tuple[Card, ...]) -> list[float]:
    mask = [0.0] * CARD_COUNT
    for card in cards:
        mask[card_to_action_index(card)] = 1.0
    return mask


def _encode_current_trick(observation: PlayerObservation) -> list[float]:
    encoded = [0.0] * CURRENT_TRICK_FEATURE_SIZE

    for play_index, play in enumerate(observation.current_trick):
        base = play_index * CURRENT_TRICK_PLAY_FEATURE_SIZE
        encoded[base + card_to_action_index(play.card)] = 1.0

        relative_player = relative_player_index(
            play.player,
            observation.player_index,
        )
        encoded[base + CARD_COUNT + relative_player] = 1.0
        encoded[base + CARD_COUNT + PLAYER_COUNT] = 1.0

    return encoded


def _current_trick_points(observation: PlayerObservation) -> float:
    return sum(card_points(play.card) for play in observation.current_trick) / 120.0


def _encode_current_winner(observation: PlayerObservation) -> list[float]:
    if not observation.current_trick:
        return _encode_optional_player(None, observer=observation.player_index)

    winner = winning_play(observation.current_trick, observation.contract)
    return _encode_optional_player(
        winner.player,
        observer=observation.player_index,
    )


def _unseen_cards(observation: PlayerObservation) -> tuple[Card, ...]:
    known_cards = set(observation.hand)
    known_cards.update(observation.cards_played)

    return tuple(
        card
        for card in _ALL_CARDS
        if card not in known_cards
    )


def _observer_team(
    observation: PlayerObservation,
) -> tuple[bool, frozenset[int]]:
    contract = observation.contract
    observer = observation.player_index

    if contract.game_type is GameType.RAMSCH:
        return False, frozenset()

    declarer = contract.declarer
    if declarer is None:
        return False, frozenset()

    if contract.game_type in {GameType.SOLO, GameType.WENZ, GameType.GEIER}:
        if observer == declarer:
            return True, frozenset({declarer})
        return True, frozenset(set(range(PLAYER_COUNT)) - {declarer})

    if contract.game_type is not GameType.SAUSPIEL or contract.called_suit is None:
        return False, frozenset()

    called_ace = Card(contract.called_suit, RANK_ORDER[-1])
    partner: int | None = None

    for trick in (*observation.completed_tricks, observation.current_trick):
        for play in trick:
            if play.card == called_ace:
                partner = play.player
                break
        if partner is not None:
            break

    if partner is None and called_ace in observation.hand:
        partner = observer

    if partner is None:
        return False, frozenset()

    declarer_team = frozenset({declarer, partner})
    if observer in declarer_team:
        return True, declarer_team

    return True, frozenset(set(range(PLAYER_COUNT)) - set(declarer_team))


def _encode_team_membership(
    observation: PlayerObservation,
    observer_team: frozenset[int],
) -> list[float]:
    if not observer_team:
        return [0.0] * PLAYER_COUNT

    membership: list[float] = []
    for relative_player in range(PLAYER_COUNT):
        absolute_player = (observation.player_index + relative_player) % PLAYER_COUNT
        membership.append(1.0 if absolute_player in observer_team else 0.0)
    return membership


def _encode_team_points(
    observation: PlayerObservation,
    observer_team: frozenset[int],
) -> list[float]:
    if not observer_team:
        return [0.0, 0.0]

    own_points = sum(
        observation.points_by_player[player] for player in observer_team
    )
    opposing_points = sum(observation.points_by_player) - own_points
    return [own_points / 120.0, opposing_points / 120.0]


def _encode_points_by_player(observation: PlayerObservation) -> list[float]:
    points: list[float] = []
    for relative_player in range(PLAYER_COUNT):
        absolute_player = (observation.player_index + relative_player) % PLAYER_COUNT
        points.append(observation.points_by_player[absolute_player] / 120.0)
    return points


def _encode_unseen_trumps(
    observation: PlayerObservation,
    unseen_cards: tuple[Card, ...],
) -> list[float]:
    unseen = set(unseen_cards)
    mask = [0.0] * CARD_COUNT

    for trump in trump_order(observation.contract):
        if trump in unseen:
            mask[card_to_action_index(trump)] = 1.0

    return mask


def _normalized_unseen_trump_count(
    observation: PlayerObservation,
    unseen_cards: tuple[Card, ...],
) -> float:
    trumps = trump_order(observation.contract)
    if not trumps:
        return 0.0

    count = sum(1 for card in unseen_cards if is_trump(card, observation.contract))
    return count / len(trumps)


def _encode_unseen_suit_counts(
    observation: PlayerObservation,
    unseen_cards: tuple[Card, ...],
) -> list[float]:
    counts: list[float] = []

    for suit in SUIT_ORDER:
        count = sum(
            1
            for card in unseen_cards
            if card.suit is suit and not is_trump(card, observation.contract)
        )
        counts.append(count / len(RANK_ORDER))

    return counts


def _encode_unseen_suit_points(
    observation: PlayerObservation,
    unseen_cards: tuple[Card, ...],
) -> list[float]:
    points: list[float] = []

    for suit in SUIT_ORDER:
        suit_points = sum(
            card_points(card)
            for card in unseen_cards
            if card.suit is suit and not is_trump(card, observation.contract)
        )
        # A complete printed suit is worth at most 30 Augen.
        points.append(suit_points / 30.0)

    return points


def _encode_known_voids(
    observation: PlayerObservation,
    unseen_cards: tuple[Card, ...],
) -> list[float]:
    # Rows are relative players. Columns are:
    # trump, Eichel, Gras, Herz, Schellen.
    voids = [
        [False] * KNOWN_VOID_CATEGORY_COUNT
        for _ in range(PLAYER_COUNT)
    ]

    observer_relative = 0
    voids[observer_relative][0] = not any(
        is_trump(card, observation.contract) for card in observation.hand
    )
    for suit_index, suit in enumerate(SUIT_ORDER, start=1):
        voids[observer_relative][suit_index] = not any(
            card.suit is suit and not is_trump(card, observation.contract)
            for card in observation.hand
        )

    tricks = (*observation.completed_tricks, observation.current_trick)
    for trick in tricks:
        _apply_follow_information(
            observation=observation,
            trick=trick,
            voids=voids,
        )

    # If no unseen opponent card exists in a category, every opponent is
    # necessarily void in that category.
    unseen_trump_exists = any(
        is_trump(card, observation.contract) for card in unseen_cards
    )
    if not unseen_trump_exists:
        for relative_player in range(1, PLAYER_COUNT):
            voids[relative_player][0] = True

    for suit_index, suit in enumerate(SUIT_ORDER, start=1):
        unseen_suit_exists = any(
            card.suit is suit and not is_trump(card, observation.contract)
            for card in unseen_cards
        )
        if not unseen_suit_exists:
            for relative_player in range(1, PLAYER_COUNT):
                voids[relative_player][suit_index] = True

    return [
        1.0 if value else 0.0
        for player_voids in voids
        for value in player_voids
    ]


def _apply_follow_information(
    *,
    observation: PlayerObservation,
    trick: tuple[TrickPlay, ...],
    voids: list[list[bool]],
) -> None:
    if len(trick) < 2:
        return

    lead = trick[0].card
    lead_is_trump = is_trump(lead, observation.contract)

    for play in trick[1:]:
        relative_player = relative_player_index(
            play.player,
            observation.player_index,
        )
        played_is_trump = is_trump(play.card, observation.contract)

        if lead_is_trump:
            if not played_is_trump:
                voids[relative_player][0] = True
            continue

        lead_suit_index = SUIT_ORDER.index(lead.suit) + 1
        followed_plain_suit = (
            not played_is_trump and play.card.suit is lead.suit
        )
        if not followed_plain_suit:
            voids[relative_player][lead_suit_index] = True


def _encode_optional_suit(suit: Suit | None) -> list[float]:
    index = 0 if suit is None else SUIT_ORDER.index(suit) + 1
    return _one_hot(index, OPTIONAL_SUIT_FEATURE_SIZE)


def _encode_optional_player(
    player: int | None,
    *,
    observer: int,
) -> list[float]:
    index = 0 if player is None else relative_player_index(player, observer) + 1
    return _one_hot(index, OPTIONAL_PLAYER_FEATURE_SIZE)


def _one_hot(index: int, size: int) -> list[float]:
    if not 0 <= index < size:
        raise ValueError(f"One-hot index {index} is outside vector size {size}.")

    values = [0.0] * size
    values[index] = 1.0
    return values
