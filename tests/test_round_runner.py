import random

from schafkopf_ai.agents.agent import Agent
from schafkopf_ai.game.bidding import (
    BiddingAction,
    BiddingActionType,
    BiddingObservation,
    BiddingPhase,
)
from schafkopf_ai.game.card import Card
from schafkopf_ai.game.game_type import GameType
from schafkopf_ai.game.observation import PlayerObservation
from schafkopf_ai.game.rules import RAMSCH_RULES, STANDARD_RULES
from schafkopf_ai.runner.round_runner import RoundResult, RoundRunner


class PassAgent(Agent):
    """Always passes during bidding and plays the first legal card."""

    def choose_bidding_action(
        self,
        observation: BiddingObservation,
        legal_actions: tuple[BiddingAction, ...],
    ) -> BiddingAction:
        del observation

        return next(
            action
            for action in legal_actions
            if action.action_type is BiddingActionType.PASS
        )

    def choose_card(
        self,
        observation: PlayerObservation,
        legal_cards: tuple[Card, ...],
    ) -> Card:
        del observation
        return legal_cards[0]


class WenzAgent(Agent):
    """Declares interest, announces Wenz, and plays the first legal card."""

    def choose_bidding_action(
        self,
        observation: BiddingObservation,
        legal_actions: tuple[BiddingAction, ...],
    ) -> BiddingAction:
        if observation.phase is BiddingPhase.INTEREST:
            return next(
                action
                for action in legal_actions
                if action.action_type is BiddingActionType.PLAY
            )

        if observation.phase is BiddingPhase.ANNOUNCEMENT:
            return next(
                action
                for action in legal_actions
                if (
                    action.contract is not None
                    and action.contract.game_type is GameType.WENZ
                )
            )

        return legal_actions[0]

    def choose_card(
        self,
        observation: PlayerObservation,
        legal_cards: tuple[Card, ...],
    ) -> Card:
        del observation
        return legal_cards[0]


class PassOnceThenWenzAgent(Agent):
    """Passes the first deal, then plays Wenz after the redeal."""

    def __init__(self) -> None:
        self.interest_calls = 0

    def choose_bidding_action(
        self,
        observation: BiddingObservation,
        legal_actions: tuple[BiddingAction, ...],
    ) -> BiddingAction:
        if observation.phase is BiddingPhase.INTEREST:
            self.interest_calls += 1

            desired_action = (
                BiddingActionType.PASS
                if self.interest_calls == 1
                else BiddingActionType.PLAY
            )

            return next(
                action
                for action in legal_actions
                if action.action_type is desired_action
            )

        if observation.phase is BiddingPhase.ANNOUNCEMENT:
            return next(
                action
                for action in legal_actions
                if (
                    action.contract is not None
                    and action.contract.game_type is GameType.WENZ
                )
            )

        return legal_actions[0]

    def choose_card(
        self,
        observation: PlayerObservation,
        legal_cards: tuple[Card, ...],
    ) -> Card:
        del observation
        return legal_cards[0]


def run_wenz_round(seed: int = 42) -> RoundResult:
    agents: tuple[Agent, ...] = (
        WenzAgent(),
        PassAgent(),
        PassAgent(),
        PassAgent(),
    )

    return RoundRunner(
        agents=agents,
        rules=STANDARD_RULES,
        rng=random.Random(seed),
    ).run()


def test_round_runner_completes_deterministic_wenz_round() -> None:
    result = run_wenz_round()

    assert result.game_state.is_complete
    assert result.contract.game_type is GameType.WENZ
    assert result.contract.declarer == 0
    assert result.redeals == 0
    assert len(result.bidding_results) == 1


def test_round_runner_completes_eight_tricks() -> None:
    result = run_wenz_round()

    assert len(result.game_state.completed_tricks) == 8
    assert all(trick.is_complete for trick in result.game_state.completed_tricks)


def test_round_runner_plays_all_32_unique_cards() -> None:
    result = run_wenz_round()

    played_cards = [
        play.card
        for trick in result.game_state.completed_tricks
        for play in trick.plays
    ]

    assert len(played_cards) == 32
    assert len(set(played_cards)) == 32


def test_round_runner_finishes_with_120_augen() -> None:
    result = run_wenz_round()

    assert sum(result.player_points) == 120


def test_round_runner_payments_sum_to_zero() -> None:
    result = run_wenz_round()

    assert sum(result.payments) == 0


def test_all_pass_starts_ramsch_when_enabled() -> None:
    agents: tuple[Agent, ...] = (
        PassAgent(),
        PassAgent(),
        PassAgent(),
        PassAgent(),
    )

    result = RoundRunner(
        agents=agents,
        rules=RAMSCH_RULES,
        rng=random.Random(42),
    ).run()

    assert result.game_state.is_complete
    assert result.contract.game_type is GameType.RAMSCH
    assert result.contract.declarer is None
    assert result.redeals == 0
    assert len(result.bidding_results) == 1
    assert not result.bidding_results[0].requires_redeal
    assert sum(result.player_points) == 120
    assert sum(result.payments) == 0


def test_all_pass_redeals_then_runs_next_game() -> None:
    agents: tuple[Agent, ...] = (
        PassOnceThenWenzAgent(),
        PassAgent(),
        PassAgent(),
        PassAgent(),
    )

    result = RoundRunner(
        agents=agents,
        rules=STANDARD_RULES,
        rng=random.Random(42),
    ).run()

    assert result.redeals == 1
    assert len(result.bidding_results) == 2
    assert result.bidding_results[0].requires_redeal
    assert not result.bidding_results[1].requires_redeal
    assert result.contract.game_type is GameType.WENZ
    assert result.contract.declarer == 0
    assert result.game_state.is_complete


def test_same_seed_produces_same_round_result() -> None:
    first = run_wenz_round(seed=42)
    second = run_wenz_round(seed=42)

    assert first.contract == second.contract
    assert first.player_points == second.player_points
    assert first.game_result.winner_players == second.game_result.winner_players
    assert first.game_result.loser_players == second.game_result.loser_players
    assert first.settlement.game_value == second.settlement.game_value
    assert first.payments == second.payments
