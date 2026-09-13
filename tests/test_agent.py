import pytest

from schafkopf_ai.agents.agent import Agent
from schafkopf_ai.game.card import Card, Rank, Suit
from schafkopf_ai.game.game_contract import GameContract
from schafkopf_ai.game.game_type import GameType
from schafkopf_ai.game.observation import PlayerObservation


def test_dummy_agent_rejects_empty_legal_actions() -> None:
    card = Card(
        Suit.EICHEL,
        Rank.ACE,
    )

    observation = PlayerObservation(
        player_index=0,
        hand=(card,),
        contract=GameContract(
            GameType.WENZ
        ),
        current_player=0,
        current_trick=(),
        completed_tricks=(),
        called_ace_released=False,
    )

    agent = DummyAgent()

    with pytest.raises(
        ValueError,
        match="No legal cards",
    ):
        agent.choose_card(
            observation=observation,
            legal_cards=(),
        )

def test_concrete_agent_can_choose_card() -> None:
    card = Card(
        Suit.EICHEL,
        Rank.ACE,
    )

    observation = PlayerObservation(
        player_index=0,
        hand=(card,),
        contract=GameContract(
            GameType.WENZ
        ),
        current_player=0,
        current_trick=(),
        completed_tricks=(),
        called_ace_released=False,
    )

    agent = DummyAgent()

    chosen = agent.choose_card(
        observation=observation,
        legal_cards=(card,),
    )

    assert chosen == card

def test_agent_cannot_be_instantiated_directly() -> None:
    with pytest.raises(TypeError):
        Agent() # type: ignore


class IncompleteAgent(Agent):
    pass


def test_agent_without_choose_card_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        IncompleteAgent()  # type: ignore


class DummyAgent(Agent):
    def choose_card(
        self,
        observation: PlayerObservation,
        legal_cards: tuple[Card, ...],
    ) -> Card:
        del observation

        if not legal_cards:
            raise ValueError("No legal cards available.")

        return legal_cards[0]

    