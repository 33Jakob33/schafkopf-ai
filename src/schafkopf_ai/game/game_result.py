from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .game_contract import GameContract
from .game_type import GameType

if TYPE_CHECKING:
    from .game_state import GameState


PLAYER_COUNT = 4
TRICKS_PER_GAME = 8
TOTAL_POINTS = 120


@dataclass(frozen=True, slots=True)
class GameResult:
    """
    Final result of a completed Schafkopf round.

    For Sauspiel / Solo / Wenz / Geier:
        points contains:
            (winning_party_points, losing_party_points)

    For Ramsch:
        points contains the individual points:
            (player_0, player_1, player_2, player_3)
    """

    game_type: GameType

    winner_players: tuple[int, ...]
    loser_players: tuple[int, ...]

    player_points: tuple[int, int, int, int]
    points: tuple[int, ...]

    declarer: int | None
    declarer_team: tuple[int, ...] | None
    defender_team: tuple[int, ...] | None
    declarer_won: bool | None

    schneider: bool
    schwarz: bool


def determine_game_result(
    *,
    contract: GameContract,
    player_points: tuple[int, int, int, int],
    trick_winners: Sequence[int],
    called_ace_player: int | None = None,
) -> GameResult:
    """
    Determine the result of a completed game.

    Args:
        contract:
            Contract being played.

        player_points:
            Augen collected by players 0 through 3.

        trick_winners:
            Player index of the winner of each of the eight tricks.

        called_ace_player:
            Holder of the called Ace in a Sauspiel.
    """
    _validate_complete_game(
        player_points,
        trick_winners,
    )

    if contract.game_type is GameType.RAMSCH:
        return _determine_ramsch_result(player_points)

    return _determine_team_game_result(
        contract=contract,
        player_points=player_points,
        trick_winners=trick_winners,
        called_ace_player=called_ace_player,
    )


def result_from_game_state(
    state: GameState,
) -> GameResult:
    """
    Determine the result directly from a completed GameState.
    """
    if not state.is_complete:
        raise ValueError("Cannot determine the result before the game is complete.")

    trick_winners = tuple(
        trick.winner(state.contract).player for trick in state.completed_tricks
    )

    return determine_game_result(
        contract=state.contract,
        player_points=state.player_points,
        trick_winners=trick_winners,
        called_ace_player=state.called_ace_player,
    )


def _determine_team_game_result(
    *,
    contract: GameContract,
    player_points: tuple[int, int, int, int],
    trick_winners: Sequence[int],
    called_ace_player: int | None,
) -> GameResult:
    declarer = contract.declarer

    if declarer is None:
        raise ValueError(
            f"{contract.game_type.value} requires a declarer "
            "to determine the game result."
        )

    if contract.game_type is GameType.SAUSPIEL:
        if called_ace_player is None:
            raise ValueError(
                "A Sauspiel requires the called-Ace player to determine teams."
            )

        if called_ace_player == declarer:
            raise ValueError("The declarer cannot also be the called-Ace player.")

        declarer_team: tuple[int, ...]

        if contract.game_type is GameType.SAUSPIEL:
            if called_ace_player is None:
                raise ValueError(
                    "A Sauspiel requires the called-Ace player to determine teams."
                )

            if called_ace_player == declarer:
                raise ValueError("The declarer cannot also be the called-Ace player.")

            declarer_team = (
                declarer,
                called_ace_player,
            )

    elif contract.game_type in {
        GameType.SOLO,
        GameType.WENZ,
        GameType.GEIER,
    }:
        declarer_team = (declarer,)

    else:
        raise ValueError(f"Unsupported team game: {contract.game_type.value}")

    defender_team = tuple(
        player for player in range(PLAYER_COUNT) if player not in declarer_team
    )

    declarer_points = sum(player_points[player] for player in declarer_team)

    defender_points = sum(player_points[player] for player in defender_team)

    declarer_won = declarer_points >= 61

    winner_players: tuple[int, ...]
    loser_players: tuple[int, ...]
    if declarer_won:
        winner_players = declarer_team
        loser_players = defender_team

        winning_points = declarer_points
        losing_points = defender_points

        # Defenders need at least 30 Augen to escape Schneider.
        schneider = defender_points <= 29

    else:
        winner_players = defender_team
        loser_players = declarer_team

        winning_points = defender_points
        losing_points = declarer_points

        # Declarer side needs at least 31 Augen to escape Schneider.
        schneider = declarer_points <= 30

    losing_side_tricks = sum(1 for winner in trick_winners if winner in loser_players)

    schwarz = losing_side_tricks == 0

    return GameResult(
        game_type=contract.game_type,
        winner_players=winner_players,
        loser_players=loser_players,
        player_points=player_points,
        points=(
            winning_points,
            losing_points,
        ),
        declarer=declarer,
        declarer_team=declarer_team,
        defender_team=defender_team,
        declarer_won=declarer_won,
        schneider=schneider,
        schwarz=schwarz,
    )


def _determine_ramsch_result(
    player_points: tuple[int, int, int, int],
) -> GameResult:
    """
    Determine a basic Ramsch result.

    The player(s) with the fewest Augen have the best result.
    The player(s) with the most Augen have the worst result.

    Special variants such as Durchmarsch or Jungfrau are deliberately
    not handled here yet.
    """
    lowest_points = min(player_points)
    highest_points = max(player_points)

    winner_players = tuple(
        player for player, points in enumerate(player_points) if points == lowest_points
    )

    loser_players = tuple(
        player
        for player, points in enumerate(player_points)
        if points == highest_points
    )

    return GameResult(
        game_type=GameType.RAMSCH,
        winner_players=winner_players,
        loser_players=loser_players,
        player_points=player_points,
        points=player_points,
        declarer=None,
        declarer_team=None,
        defender_team=None,
        declarer_won=None,
        schneider=False,
        schwarz=False,
    )


def _validate_complete_game(
    player_points: tuple[int, int, int, int],
    trick_winners: Sequence[int],
) -> None:
    if sum(player_points) != TOTAL_POINTS:
        raise ValueError(f"A completed game must contain exactly {TOTAL_POINTS} Augen.")

    if len(trick_winners) != TRICKS_PER_GAME:
        raise ValueError(
            f"A completed game must contain exactly {TRICKS_PER_GAME} trick winners."
        )

    if any(not 0 <= winner < PLAYER_COUNT for winner in trick_winners):
        raise ValueError("Trick winner indices must be between 0 and 3.")
