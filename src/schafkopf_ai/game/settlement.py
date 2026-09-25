from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .card import Card
from .game_contract import GameContract
from .game_result import GameResult
from .game_type import GameType
from .trump import trump_order

if TYPE_CHECKING:
    from .game_state import GameState


PLAYER_COUNT = 4


@dataclass(frozen=True, slots=True)
class GameValueRules:
    """
    Configurable monetary/value rules for one Schafkopf table.

    All values use the same unit, e.g. cents or abstract game points.
    """

    sauspiel_base: int = 10
    solo_base: int = 20
    wenz_base: int = 20
    geier_base: int = 20

    schneider_bonus: int = 10
    schwarz_bonus: int = 10

    laufende_value: int = 10

    sauspiel_laufende_minimum: int = 3
    solo_laufende_minimum: int = 3
    wenz_laufende_minimum: int = 2
    geier_laufende_minimum: int = 2

    ramsch_base: int = 10
    jungfrau_multiplier: int = 2

    def base_value(
        self,
        game_type: GameType,
    ) -> int:
        """Return the base value for a non-Ramsch game."""
        if game_type is GameType.SAUSPIEL:
            return self.sauspiel_base

        if game_type is GameType.SOLO:
            return self.solo_base

        if game_type is GameType.WENZ:
            return self.wenz_base

        if game_type is GameType.GEIER:
            return self.geier_base

        raise ValueError(f"{game_type.value} does not use a normal base value.")

    def laufende_minimum(
        self,
        game_type: GameType,
    ) -> int:
        """Return the minimum number required for Laufende to count."""
        if game_type is GameType.SAUSPIEL:
            return self.sauspiel_laufende_minimum

        if game_type is GameType.SOLO:
            return self.solo_laufende_minimum

        if game_type is GameType.WENZ:
            return self.wenz_laufende_minimum

        if game_type is GameType.GEIER:
            return self.geier_laufende_minimum

        raise ValueError(f"{game_type.value} does not use Laufende.")


DEFAULT_GAME_VALUE_RULES = GameValueRules()


@dataclass(frozen=True, slots=True)
class GameSettlement:
    """
    Monetary/value settlement of a completed round.

    payments:
        Net payment per player.

        Positive value:
            player receives money/points.

        Negative value:
            player pays money/points.

        The four payments always sum to zero.
    """

    game_type: GameType

    base_value: int
    game_value: int

    schneider_bonus: int
    schwarz_bonus: int

    laufende_count: int
    laufende_bonus: int

    jungfrau_players: tuple[int, ...]

    payments: tuple[int, int, int, int]


def calculate_laufende(
    *,
    contract: GameContract,
    initial_hands: tuple[tuple[Card, ...], ...],
    declarer_team: tuple[int, ...],
    rules: GameValueRules = DEFAULT_GAME_VALUE_RULES,
) -> int:
    """
    Determine the number of Laufende.

    Starting with the highest trump, cards are counted while they
    continuously belong to the same party.

    This naturally handles both:
        "mit Laufenden"
    and:
        "ohne Laufende"

    The result is zero if the table-specific minimum is not reached.
    """
    if contract.game_type is GameType.RAMSCH:
        return 0

    _validate_initial_hands(initial_hands)

    owners = {
        card: player_index
        for player_index, hand in enumerate(initial_hands)
        for card in hand
    }

    trumps = trump_order(contract)

    if not trumps:
        return 0

    strongest_trump_owner = owners[trumps[0]]

    strongest_is_declarer_team = strongest_trump_owner in declarer_team

    count = 0

    for trump in trumps:
        owner = owners[trump]

        owner_is_declarer_team = owner in declarer_team

        if owner_is_declarer_team != strongest_is_declarer_team:
            break

        count += 1

    minimum = rules.laufende_minimum(contract.game_type)

    if count < minimum:
        return 0

    return count


def settle_game(
    *,
    result: GameResult,
    contract: GameContract,
    initial_hands: tuple[tuple[Card, ...], ...],
    trick_winners: tuple[int, ...],
    rules: GameValueRules = DEFAULT_GAME_VALUE_RULES,
) -> GameSettlement:
    """
    Calculate the final payment/value settlement.

    Normal team games:
        Every loser pays every winner the final game value.

    Ramsch:
        Every loser pays every other player the Ramsch value.
        Each Jungfrau doubles that value again.
    """
    if result.game_type is GameType.RAMSCH:
        return _settle_ramsch(
            result=result,
            trick_winners=trick_winners,
            rules=rules,
        )

    return _settle_team_game(
        result=result,
        contract=contract,
        initial_hands=initial_hands,
        rules=rules,
    )


def settlement_from_game_state(
    state: GameState,
    result: GameResult,
    rules: GameValueRules = DEFAULT_GAME_VALUE_RULES,
) -> GameSettlement:
    """Calculate settlement directly from a completed GameState."""
    if not state.is_complete:
        raise ValueError("Cannot settle a game before it is complete.")

    trick_winners = tuple(
        trick.winner(state.contract).player for trick in state.completed_tricks
    )

    return settle_game(
        result=result,
        contract=state.contract,
        initial_hands=state.initial_hands,
        trick_winners=trick_winners,
        rules=rules,
    )


def _settle_team_game(
    *,
    result: GameResult,
    contract: GameContract,
    initial_hands: tuple[tuple[Card, ...], ...],
    rules: GameValueRules,
) -> GameSettlement:
    if result.declarer_team is None:
        raise ValueError("A team game requires a declarer team.")

    base_value = rules.base_value(contract.game_type)

    laufende_count = calculate_laufende(
        contract=contract,
        initial_hands=initial_hands,
        declarer_team=result.declarer_team,
        rules=rules,
    )

    laufende_bonus = laufende_count * rules.laufende_value

    schneider_bonus = rules.schneider_bonus if result.schneider else 0

    schwarz_bonus = rules.schwarz_bonus if result.schwarz else 0

    game_value = base_value + laufende_bonus + schneider_bonus + schwarz_bonus

    payments = [0] * PLAYER_COUNT

    for loser in result.loser_players:
        for winner in result.winner_players:
            payments[loser] -= game_value
            payments[winner] += game_value

    return GameSettlement(
        game_type=contract.game_type,
        base_value=base_value,
        game_value=game_value,
        schneider_bonus=schneider_bonus,
        schwarz_bonus=schwarz_bonus,
        laufende_count=laufende_count,
        laufende_bonus=laufende_bonus,
        jungfrau_players=(),
        payments=(
            payments[0],
            payments[1],
            payments[2],
            payments[3],
        ),
    )


def _settle_ramsch(
    *,
    result: GameResult,
    trick_winners: tuple[int, ...],
    rules: GameValueRules,
) -> GameSettlement:
    if len(trick_winners) != 8:
        raise ValueError("A Ramsch settlement requires exactly eight tricks.")

    tricks_won = [0] * PLAYER_COUNT

    for player in trick_winners:
        if not 0 <= player < PLAYER_COUNT:
            raise ValueError("Trick winner must be between 0 and 3.")

        tricks_won[player] += 1

    jungfrau_players = tuple(
        player for player, count in enumerate(tricks_won) if count == 0
    )

    game_value = rules.ramsch_base

    for _ in jungfrau_players:
        game_value *= rules.jungfrau_multiplier

    payments = [0] * PLAYER_COUNT

    for loser in result.loser_players:
        for recipient in range(PLAYER_COUNT):
            if recipient == loser:
                continue

            payments[loser] -= game_value
            payments[recipient] += game_value

    return GameSettlement(
        game_type=GameType.RAMSCH,
        base_value=rules.ramsch_base,
        game_value=game_value,
        schneider_bonus=0,
        schwarz_bonus=0,
        laufende_count=0,
        laufende_bonus=0,
        jungfrau_players=jungfrau_players,
        payments=(
            payments[0],
            payments[1],
            payments[2],
            payments[3],
        ),
    )


def _validate_initial_hands(
    initial_hands: tuple[tuple[Card, ...], ...],
) -> None:
    if len(initial_hands) != PLAYER_COUNT:
        raise ValueError("Exactly four initial hands are required.")

    all_cards = [card for hand in initial_hands for card in hand]

    if len(all_cards) != 32:
        raise ValueError("Initial hands must contain exactly 32 cards.")

    if len(set(all_cards)) != 32:
        raise ValueError("Initial hands must contain 32 unique cards.")
