from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from schafkopf_ai.game.card import Card, Rank
from schafkopf_ai.game.game_type import GameType
from schafkopf_ai.game.observation import PlayerObservation


class PlayerRole(Enum):
    """Strategic role known to the observing player from legal information."""

    DECLARER = auto()
    PARTNER = auto()
    DEFENDER = auto()
    RAMSCH = auto()


@dataclass(frozen=True, slots=True)
class TeamContext:
    """Exact team information when it is publicly/private-to-self knowable."""

    own_team: frozenset[int]
    opponent_team: frozenset[int]
    own_is_declarer_side: bool


@dataclass(frozen=True, slots=True)
class ScoreSituation:
    """Point thresholds relevant to winning and Schneider tactics."""

    own_points: int
    opponent_points: int
    win_target: int
    own_avoid_schneider_target: int
    opponent_avoid_schneider_target: int

    def trick_clinches_game(self, trick_points: int) -> bool:
        return self.own_points < self.win_target <= self.own_points + trick_points

    def trick_avoids_schneider(self, trick_points: int) -> bool:
        return (
            self.own_points
            < self.own_avoid_schneider_target
            <= self.own_points + trick_points
        )

    def losing_trick_breaks_schneider(self, trick_points: int) -> bool:
        return (
            self.opponent_points
            < self.opponent_avoid_schneider_target
            <= self.opponent_points + trick_points
        )


def called_ace(observation: PlayerObservation) -> Card | None:
    if observation.contract.game_type is not GameType.SAUSPIEL:
        return None
    if observation.contract.called_suit is None:
        return None
    return Card(observation.contract.called_suit, Rank.ACE)


def known_sauspiel_partner(observation: PlayerObservation) -> int | None:
    """Return the called-Ace holder when it is knowable without hidden hands."""
    ace = called_ace(observation)
    if ace is None:
        return None

    if ace in observation.hand:
        return observation.player_index

    for trick in observation.completed_tricks:
        for play in trick:
            if play.card == ace:
                return play.player

    for play in observation.current_trick:
        if play.card == ace:
            return play.player

    return None


def called_ace_has_been_played(observation: PlayerObservation) -> bool:
    ace = called_ace(observation)
    if ace is None:
        return False
    return any(
        play.card == ace for trick in observation.completed_tricks for play in trick
    ) or any(play.card == ace for play in observation.current_trick)


def player_role(observation: PlayerObservation) -> PlayerRole:
    """
    Determine the observer's strategic role using only information they know.

    In Sauspiel a non-declarer knows they are the partner if they personally
    hold the called Ace. If they do not hold it, they know they are a defender
    even before the Ace is publicly revealed.
    """
    contract = observation.contract
    player = observation.player_index

    if contract.game_type is GameType.RAMSCH:
        return PlayerRole.RAMSCH

    if contract.declarer == player:
        return PlayerRole.DECLARER

    if contract.game_type is GameType.SAUSPIEL:
        ace = called_ace(observation)
        if ace is not None and ace in observation.hand:
            return PlayerRole.PARTNER

        partner = known_sauspiel_partner(observation)
        if partner == player:
            return PlayerRole.PARTNER
        return PlayerRole.DEFENDER

    return PlayerRole.DEFENDER


def exact_team_context(observation: PlayerObservation) -> TeamContext | None:
    """Return exact teams when they are known to the observer."""
    contract = observation.contract
    declarer = contract.declarer
    player = observation.player_index

    if declarer is None or contract.game_type is GameType.RAMSCH:
        return None

    if contract.game_type in {GameType.SOLO, GameType.WENZ, GameType.GEIER}:
        declarer_team = frozenset({declarer})
    elif contract.game_type is GameType.SAUSPIEL:
        partner = known_sauspiel_partner(observation)
        if partner is None:
            return None
        declarer_team = frozenset({declarer, partner})
    else:
        return None

    defender_team = frozenset(index for index in range(4) if index not in declarer_team)

    if player in declarer_team:
        return TeamContext(
            own_team=declarer_team,
            opponent_team=defender_team,
            own_is_declarer_side=True,
        )

    return TeamContext(
        own_team=defender_team,
        opponent_team=declarer_team,
        own_is_declarer_side=False,
    )


def score_situation(observation: PlayerObservation) -> ScoreSituation | None:
    context = exact_team_context(observation)
    if context is None:
        return None

    own_points = sum(
        observation.points_by_player[player] for player in context.own_team
    )
    opponent_points = sum(
        observation.points_by_player[player] for player in context.opponent_team
    )

    if context.own_is_declarer_side:
        # Declarer side wins from 61. To avoid being Schneider when it loses,
        # it must reach 31. Defenders avoid Schneider against a winning
        # declarer by reaching 30.
        win_target = 61
        own_avoid = 31
        opponent_avoid = 30
    else:
        # Defenders win from 60. They avoid Schneider when they lose by
        # reaching 30; the declarer avoids Schneider in a defender win at 31.
        win_target = 60
        own_avoid = 30
        opponent_avoid = 31

    return ScoreSituation(
        own_points=own_points,
        opponent_points=opponent_points,
        win_target=win_target,
        own_avoid_schneider_target=own_avoid,
        opponent_avoid_schneider_target=opponent_avoid,
    )
