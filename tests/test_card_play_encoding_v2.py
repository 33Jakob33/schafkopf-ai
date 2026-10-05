import pytest

from schafkopf_ai.game.card import Card, Rank, Suit
from schafkopf_ai.game.game_contract import GameContract
from schafkopf_ai.game.game_type import GameType
from schafkopf_ai.game.observation import PlayerObservation
from schafkopf_ai.game.trick import TrickPlay
from schafkopf_ai.training.card_play_encoding import card_to_action_index
from schafkopf_ai.training.card_play_encoding_v2 import (
    OBSERVATION_V2_FEATURE_SIZE,
    _encode_known_voids,
    _encode_team_points,
    _encode_unseen_suit_counts,
    _encode_unseen_suit_points,
    _encode_unseen_trumps,
    _normalized_unseen_trump_count,
    _observer_team,
    _unseen_cards,
    encode_player_observation_v2,
)


def _observation(
    *,
    player_index: int = 0,
    hand: tuple[Card, ...] = (),
    contract: GameContract | None = None,
    current_trick: tuple[TrickPlay, ...] = (),
    completed_tricks: tuple[tuple[TrickPlay, ...], ...] = (),
    points: tuple[int, int, int, int] = (0, 0, 0, 0),
) -> PlayerObservation:
    return PlayerObservation(
        player_index=player_index,
        hand=hand,
        contract=contract or GameContract(GameType.WENZ, declarer=0),
        current_player=player_index,
        current_trick=current_trick,
        completed_tricks=completed_tricks,
        points_by_player=points,
        called_ace_released=False,
    )


def test_v2_has_compact_fixed_size() -> None:
    observation = _observation(
        hand=(
            Card(Suit.EICHEL, Rank.ACE),
            Card(Suit.HERZ, Rank.SEVEN),
        )
    )

    features = encode_player_observation_v2(observation)

    assert OBSERVATION_V2_FEATURE_SIZE == 280
    assert len(features) == 280
    assert all(isinstance(value, float) for value in features)


def test_v2_unseen_trumps_exclude_own_and_played_cards() -> None:
    own_trump = Card(Suit.EICHEL, Rank.UNTER)
    played_trump = Card(Suit.GRAS, Rank.UNTER)
    observation = _observation(
        hand=(own_trump,),
        completed_tricks=(
            (
                TrickPlay(0, Card(Suit.EICHEL, Rank.ACE)),
                TrickPlay(1, played_trump),
                TrickPlay(2, Card(Suit.EICHEL, Rank.KING)),
                TrickPlay(3, Card(Suit.EICHEL, Rank.SEVEN)),
            ),
        ),
    )
    unseen = _unseen_cards(observation)
    mask = _encode_unseen_trumps(observation, unseen)

    assert mask[card_to_action_index(own_trump)] == 0.0
    assert mask[card_to_action_index(played_trump)] == 0.0
    assert mask[card_to_action_index(Card(Suit.HERZ, Rank.UNTER))] == 1.0
    assert mask[card_to_action_index(Card(Suit.SCHELLEN, Rank.UNTER))] == 1.0
    assert _normalized_unseen_trump_count(observation, unseen) == pytest.approx(0.5)


def test_v2_plain_suit_summary_is_contract_aware() -> None:
    observation = _observation(
        hand=(Card(Suit.EICHEL, Rank.UNTER),),
    )
    unseen = _unseen_cards(observation)

    counts = _encode_unseen_suit_counts(observation, unseen)
    points = _encode_unseen_suit_points(observation, unseen)

    # In Wenz only Unter are trump. The own Eichel Unter is therefore excluded
    # from the unseen pool, while the other seven Eichel cards remain unseen.
    assert counts[0] == pytest.approx(7 / 8)
    assert points[0] == pytest.approx(28 / 30)


def test_v2_infers_opponent_void_when_they_fail_to_follow_suit() -> None:
    observation = _observation(
        hand=(Card(Suit.HERZ, Rank.ACE),),
        completed_tricks=(
            (
                TrickPlay(0, Card(Suit.EICHEL, Rank.ACE)),
                TrickPlay(1, Card(Suit.GRAS, Rank.ACE)),
                TrickPlay(2, Card(Suit.EICHEL, Rank.KING)),
                TrickPlay(3, Card(Suit.EICHEL, Rank.SEVEN)),
            ),
        ),
    )
    unseen = _unseen_cards(observation)
    voids = _encode_known_voids(observation, unseen)

    # Five categories per relative player: trump, Eichel, Gras, Herz, Schellen.
    player_one_eichel_void = 1 * 5 + 1
    assert voids[player_one_eichel_void] == 1.0


def test_v2_team_points_are_encoded_when_team_is_public() -> None:
    observation = _observation(
        player_index=2,
        contract=GameContract(GameType.SOLO, trump_suit=Suit.HERZ, declarer=3),
        points=(10, 20, 30, 40),
    )

    known, team = _observer_team(observation)

    assert known is True
    assert team == frozenset({0, 1, 2})
    assert _encode_team_points(observation, team) == pytest.approx([0.5, 1 / 3])


def test_v2_sauspiel_team_stays_unknown_until_partner_is_known() -> None:
    contract = GameContract(
        GameType.SAUSPIEL,
        called_suit=Suit.EICHEL,
        declarer=0,
    )
    hidden = _observation(
        player_index=2,
        hand=(Card(Suit.GRAS, Rank.ACE),),
        contract=contract,
    )
    partner = _observation(
        player_index=2,
        hand=(Card(Suit.EICHEL, Rank.ACE),),
        contract=contract,
    )

    hidden_known, hidden_team = _observer_team(hidden)
    partner_known, partner_team = _observer_team(partner)

    assert hidden_known is False
    assert hidden_team == frozenset()
    assert partner_known is True
    assert partner_team == frozenset({0, 2})
