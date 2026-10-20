from pathlib import Path

import torch

from schafkopf_ai.agents.q_card_play_agent import QCardPlayAgent
from schafkopf_ai.game.deck import Deck
from schafkopf_ai.game.game_contract import GameContract
from schafkopf_ai.game.game_state import GameState
from schafkopf_ai.game.game_type import GameType
from schafkopf_ai.training.action_values import CardPlayQNetwork


def _observation_and_legal_cards():
    deck = Deck()
    hands = deck.deal()
    state = GameState.from_hands(
        hands=hands,
        contract=GameContract(GameType.WENZ, declarer=0),
        starting_player=0,
    )
    return state.observation_for(0), state.legal_moves(0)


def test_q_agent_selects_legal_card_and_exposes_payment_values() -> None:
    model = CardPlayQNetwork(hidden_sizes=(32, 16))
    for parameter in model.parameters():
        torch.nn.init.constant_(parameter, 0.0)

    observation, legal_cards = _observation_and_legal_cards()
    agent = QCardPlayAgent(model=model, target_scale=100.0)

    chosen = agent.choose_card(observation, legal_cards)

    assert chosen in legal_cards
    assert len(agent.last_q_values) == 32
    assert set(agent.last_q_values) == {0.0}


def test_q_agent_loads_checkpoint(tmp_path: Path) -> None:
    model = CardPlayQNetwork(hidden_sizes=(32, 16))
    checkpoint = tmp_path / "q.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "architecture": model.architecture,
            "input_size": model.input_size,
            "hidden_sizes": model.hidden_sizes,
            "action_count": model.action_count,
            "observation_version": model.observation_version,
            "target_scale": 80.0,
        },
        checkpoint,
    )

    agent = QCardPlayAgent.from_checkpoint(checkpoint)

    assert agent.model.hidden_sizes == (32, 16)
    assert agent.target_scale == 80.0
