import random

import pytest

from schafkopf_ai.agents.agent import Agent
from schafkopf_ai.agents.random_agent import RandomAgent
from schafkopf_ai.game.card import Card, Rank, Suit
from schafkopf_ai.game.deck import Deck
from schafkopf_ai.game.game_contract import GameContract
from schafkopf_ai.game.game_state import GameState
from schafkopf_ai.game.game_type import GameType
from schafkopf_ai.game.observation import PlayerObservation
from schafkopf_ai.runner.game_runner import GameRunner


def create_state() -> GameState:
    deck = Deck()
    deck.shuffle(random.Random(42))

    hands = deck.deal()

    return GameState.from_hands(
        hands=hands,
        contract=GameContract(GameType.WENZ),
        starting_player=0,
    )


def create_random_agents() -> tuple[RandomAgent, ...]:
    return tuple(
        RandomAgent(
            rng=random.Random(seed)
        )
        for seed in range(4)
    )


def test_runner_requires_four_agents() -> None:
    state = create_state()

    agents = [
        RandomAgent(),
        RandomAgent(),
        RandomAgent(),
    ]

    with pytest.raises(
        ValueError,
        match="exactly 4 agents",
    ):
        GameRunner(
            state=state,
            agents=agents,
        )


def test_runner_starts_with_zero_turns() -> None:
    runner = GameRunner(
        state=create_state(),
        agents=create_random_agents(),
    )

    assert runner.turns_played == 0


def test_play_turn_plays_exactly_one_card() -> None:
    state = create_state()

    runner = GameRunner(
        state=state,
        agents=create_random_agents(),
    )

    total_cards_before = sum(
        len(player)
        for player in state.players
    )

    runner.play_turn()

    total_cards_after = sum(
        len(player)
        for player in state.players
    )

    assert total_cards_before == 32
    assert total_cards_after == 31
    assert runner.turns_played == 1


def test_play_turn_returns_played_card() -> None:
    state = create_state()

    runner = GameRunner(
        state=state,
        agents=create_random_agents(),
    )

    played_card = runner.play_turn()

    assert isinstance(played_card, Card)

    current_trick = state.current_trick

    assert current_trick is not None

    assert current_trick.plays[0].card == played_card


def test_four_turns_complete_first_trick() -> None:
    state = create_state()

    runner = GameRunner(
        state=state,
        agents=create_random_agents(),
    )

    for _ in range(4):
        runner.play_turn()

    assert runner.turns_played == 4
    assert len(state.completed_tricks) == 1


def test_runner_completes_full_game() -> None:
    state = create_state()

    runner = GameRunner(
        state=state,
        agents=create_random_agents(),
    )

    result = runner.run()

    assert result.is_complete
    assert runner.turns_played == 32
    assert len(result.completed_tricks) == 8


def test_all_players_have_empty_hands_after_game() -> None:
    runner = GameRunner(
        state=create_state(),
        agents=create_random_agents(),
    )

    result = runner.run()

    assert all(
        len(player) == 0
        for player in result.players
    )


def test_all_32_cards_are_present_in_completed_tricks() -> None:
    runner = GameRunner(
        state=create_state(),
        agents=create_random_agents(),
    )

    result = runner.run()

    played_cards = [
        play.card
        for trick in result.completed_tricks
        for play in trick.plays
    ]

    assert len(played_cards) == 32
    assert len(set(played_cards)) == 32


def test_cannot_play_turn_after_game_complete() -> None:
    runner = GameRunner(
        state=create_state(),
        agents=create_random_agents(),
    )

    runner.run()

    with pytest.raises(
        ValueError,
        match="game is complete",
    ):
        runner.play_turn()

class InvalidCardAgent(Agent):
    """
    Deliberately returns a card that is not among its legal moves
    whenever possible.
    """

    def choose_card(
        self,
        observation: PlayerObservation,
        legal_cards: tuple[Card, ...],
    ) -> Card:
        del observation

        for suit in Suit:
            for rank in Rank:
                card = Card(suit, rank)

                if card not in legal_cards:
                    return card

        raise RuntimeError(
            "Could not find an illegal card."
        )


def test_runner_rejects_agent_illegal_choice() -> None:
    state = create_state()

    agents: list[Agent] = [
        InvalidCardAgent(),
        RandomAgent(),
        RandomAgent(),
        RandomAgent(),
    ]

    runner = GameRunner(
        state=state,
        agents=agents,
    )

    with pytest.raises(
        ValueError,
        match="selected illegal card",
    ):
        runner.play_turn()

    assert runner.turns_played == 0