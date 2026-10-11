from pathlib import Path

import torch

from schafkopf_ai.agents.multihead_card_play_agent import MultiHeadCardPlayAgent
from schafkopf_ai.game.card import Card, Rank, Suit
from schafkopf_ai.game.game_contract import GameContract
from schafkopf_ai.game.game_type import GameType
from schafkopf_ai.game.observation import PlayerObservation
from schafkopf_ai.training.card_play_encoding import ACTION_COUNT
from schafkopf_ai.training.multihead_policy import MultiHeadCardPlayPolicyNetwork


def _observation() -> PlayerObservation:
    return PlayerObservation(
        player_index=0,
        hand=(
            Card(Suit.EICHEL, Rank.SEVEN),
            Card(Suit.GRAS, Rank.SEVEN),
        ),
        contract=GameContract(GameType.WENZ, declarer=0),
        current_player=0,
        current_trick=(),
        completed_tricks=(),
        points_by_player=(0, 0, 0, 0),
        called_ace_released=False,
    )


def test_multihead_policy_returns_one_logit_vector_per_example() -> None:
    model = MultiHeadCardPlayPolicyNetwork(
        shared_hidden_sizes=(32, 24),
        head_hidden_size=16,
    )
    features = torch.zeros((4, 222), dtype=torch.float32)
    roles = torch.tensor([0, 3, 5, 7], dtype=torch.long)

    logits = model(features, roles)

    assert logits.shape == (4, ACTION_COUNT)


def test_multihead_agent_loads_checkpoint(tmp_path: Path) -> None:
    model = MultiHeadCardPlayPolicyNetwork(
        shared_hidden_sizes=(32, 24),
        head_hidden_size=16,
    )
    checkpoint = tmp_path / "multihead.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "architecture": model.architecture,
            "input_size": model.input_size,
            "shared_hidden_sizes": model.shared_hidden_sizes,
            "head_hidden_size": model.head_hidden_size,
            "action_count": model.action_count,
            "observation_version": model.observation_version,
            "role_count": model.role_count,
        },
        checkpoint,
    )

    agent = MultiHeadCardPlayAgent.from_checkpoint(checkpoint)

    assert agent.model.architecture == "multihead_mlp_v1"
    assert agent.model.shared_hidden_sizes == (32, 24)


def test_multihead_agent_masks_illegal_cards() -> None:
    model = MultiHeadCardPlayPolicyNetwork(
        shared_hidden_sizes=(32, 24),
        head_hidden_size=16,
    )
    for parameter in model.parameters():
        torch.nn.init.constant_(parameter, 0.0)

    observation = _observation()
    agent = MultiHeadCardPlayAgent(model=model)

    chosen = agent.choose_card(observation, observation.hand)

    assert chosen == observation.hand[0]
