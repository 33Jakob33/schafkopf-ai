from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from schafkopf_ai.agents.agent import Agent
from schafkopf_ai.game.bidding import (
    DEFAULT_BID_VALUES,
    BiddingResult,
    BiddingState,
    BidValues,
)
from schafkopf_ai.game.card import Card
from schafkopf_ai.game.deck import Deck
from schafkopf_ai.game.game_contract import GameContract
from schafkopf_ai.game.game_result import (
    GameResult,
    result_from_game_state,
)
from schafkopf_ai.game.game_state import GameState
from schafkopf_ai.game.rules import (
    STANDARD_RULES,
    GameRules,
)
from schafkopf_ai.game.settlement import (
    DEFAULT_GAME_VALUE_RULES,
    GameSettlement,
    GameValueRules,
    settlement_from_game_state,
)

from .bidding_runner import BiddingRunner
from .game_runner import GameRunner

PLAYER_COUNT = 4


@dataclass(frozen=True, slots=True)
class RoundResult:
    """
    Complete result of one Schafkopf round.

    Includes bidding, played game, Augen result, and settlement.
    """

    game_state: GameState
    game_result: GameResult
    settlement: GameSettlement

    bidding_results: tuple[BiddingResult, ...]

    redeals: int

    @property
    def contract(self) -> GameContract:
        """Return the contract that was eventually played."""
        return self.game_state.contract

    @property
    def player_points(
        self,
    ) -> tuple[int, int, int, int]:
        """Return the final Augen collected by every player."""
        return self.game_result.player_points

    @property
    def payments(
        self,
    ) -> tuple[int, int, int, int]:
        payments: tuple[int, int, int, int] = self.settlement.payments
        return payments


class RoundRunner:
    """
    Run one complete Schafkopf round.

    The runner handles:

    - shuffling and dealing,
    - bidding,
    - optional redeals,
    - playing all eight tricks,
    - calculating Augen and winners,
    - calculating the final settlement.
    """

    PLAYER_COUNT = 4

    def __init__(
        self,
        *,
        agents: Sequence[Agent],
        rules: GameRules = STANDARD_RULES,
        starting_player: int = 0,
        rng: random.Random | None = None,
        bid_values: BidValues = DEFAULT_BID_VALUES,
        value_rules: GameValueRules = DEFAULT_GAME_VALUE_RULES,
        max_redeals: int = 100,
    ) -> None:
        if len(agents) != self.PLAYER_COUNT:
            raise ValueError("RoundRunner requires exactly four agents.")

        if not 0 <= starting_player < self.PLAYER_COUNT:
            raise ValueError("Starting player must be between 0 and 3.")

        if max_redeals < 0:
            raise ValueError("Maximum redeals cannot be negative.")

        self.agents: tuple[Agent, ...] = tuple(agents)

        self.rules = rules
        self.starting_player = starting_player

        self.rng = rng or random.Random()

        self.bid_values = bid_values
        self.value_rules = value_rules

        self.max_redeals = max_redeals

    def run(self) -> RoundResult:
        """
        Run a complete round from deal through settlement.

        If everyone passes and table rules specify REDEAL,
        another deal is started automatically.
        """
        bidding_results: list[BiddingResult] = []

        redeals = 0

        while True:
            hands = self._deal()

            bidding_result = self._run_bidding(hands)

            bidding_results.append(bidding_result)

            if bidding_result.requires_redeal:
                redeals += 1

                if redeals > self.max_redeals:
                    raise RuntimeError("Maximum number of redeals exceeded.")

                continue

            contract = bidding_result.contract

            if contract is None:
                raise RuntimeError(
                    "Completed bidding produced neither a contract nor a redeal."
                )

            state = GameState.from_hands(
                hands=hands,
                contract=contract,
                starting_player=self.starting_player,
            )

            game_runner = GameRunner(
                state=state,
                agents=self.agents,
            )

            game_runner.run()

            if not state.is_complete:
                raise RuntimeError("GameRunner returned before the game was complete.")

            game_result = result_from_game_state(state)

            settlement = settlement_from_game_state(
                state,
                game_result,
                rules=self.value_rules,
            )

            return RoundResult(
                game_state=state,
                game_result=game_result,
                settlement=settlement,
                bidding_results=tuple(bidding_results),
                redeals=redeals,
            )

    def _deal(
        self,
    ) -> tuple[
        tuple,
        ...,
    ]:
        deck = Deck()

        deck.shuffle(self.rng)

        return deck.deal()

    def _run_bidding(
        self,
        hands: tuple[
            tuple,
            ...,
        ],
    ) -> BiddingResult:
        bidding_state = BiddingState(
            hands=hands,
            rules=self.rules,
            starting_player=self.starting_player,
            bid_values=self.bid_values,
        )

        bidding_runner = BiddingRunner(
            state=bidding_state,
            agents=self.agents,
        )

        return bidding_runner.run()
