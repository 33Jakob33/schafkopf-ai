from pathlib import Path

import torch
from torch.optim import Adam

from schafkopf_ai.agents.ppo_card_play_agent import PPOCardPlayAgent
from schafkopf_ai.game.card import Card, Rank, Suit
from schafkopf_ai.game.game_contract import GameContract
from schafkopf_ai.game.game_type import GameType
from schafkopf_ai.game.observation import PlayerObservation
from schafkopf_ai.training.behavior_cloning import CardPlayPolicyNetwork
from schafkopf_ai.training.card_play_encoding import (
    ACTION_COUNT,
    OBSERVATION_FEATURE_SIZE,
)
from schafkopf_ai.training.ppo import (
    ActorCriticCardPlayNetwork,
    PPORolloutBuffer,
    PPOStep,
    initialize_from_behavior_checkpoint,
    initialize_from_checkpoint,
    ppo_update,
    save_ppo_checkpoint,
)


def _observation(hand: tuple[Card, ...]) -> PlayerObservation:
    return PlayerObservation(
        player_index=0,
        hand=hand,
        contract=GameContract(GameType.WENZ, declarer=0),
        current_player=0,
        current_trick=(),
        completed_tricks=(),
        points_by_player=(0, 0, 0, 0),
        called_ace_released=False,
    )


def test_actor_critic_has_policy_and_value_outputs() -> None:
    model = ActorCriticCardPlayNetwork()
    features = torch.zeros((3, OBSERVATION_FEATURE_SIZE), dtype=torch.float32)

    logits, values = model(features)

    assert logits.shape == (3, ACTION_COUNT)
    assert values.shape == (3,)


def test_behavior_checkpoint_initializes_identical_policy_logits(
    tmp_path: Path,
) -> None:
    torch.manual_seed(7)
    behavior_model = CardPlayPolicyNetwork()
    checkpoint = tmp_path / "behavior.pt"
    torch.save(
        {
            "model_state_dict": behavior_model.state_dict(),
            "input_size": behavior_model.input_size,
            "hidden_sizes": behavior_model.hidden_sizes,
            "action_count": behavior_model.action_count,
        },
        checkpoint,
    )

    ppo_model = ActorCriticCardPlayNetwork()
    initialize_from_behavior_checkpoint(
        model=ppo_model,
        checkpoint_path=checkpoint,
        device=torch.device("cpu"),
    )

    features = torch.randn((4, OBSERVATION_FEATURE_SIZE), dtype=torch.float32)
    behavior_logits = behavior_model(features)
    ppo_logits, ppo_values = ppo_model(features)

    assert torch.allclose(behavior_logits, ppo_logits)
    assert torch.count_nonzero(ppo_values) == 0


def test_existing_ppo_checkpoint_restores_full_actor_critic(
    tmp_path: Path,
) -> None:
    torch.manual_seed(11)
    source = ActorCriticCardPlayNetwork()
    checkpoint = tmp_path / "ppo.pt"
    save_ppo_checkpoint(
        path=checkpoint,
        model=source,
        iteration=12,
        mean_payment=3.5,
        validation_delta=-1.25,
        seed=42,
        initialized_from="dagger.pt",
    )

    torch.manual_seed(99)
    restored = ActorCriticCardPlayNetwork()
    source_kind = initialize_from_checkpoint(
        model=restored,
        checkpoint_path=checkpoint,
        device=torch.device("cpu"),
    )

    assert source_kind == "PPO"
    for source_tensor, restored_tensor in zip(
        source.state_dict().values(),
        restored.state_dict().values(),
        strict=True,
    ):
        assert torch.equal(source_tensor, restored_tensor)


def test_ppo_agent_never_selects_illegal_card() -> None:
    first = Card(Suit.EICHEL, Rank.SEVEN)
    second = Card(Suit.GRAS, Rank.SEVEN)
    observation = _observation((first, second))
    model = ActorCriticCardPlayNetwork()

    for parameter in model.parameters():
        torch.nn.init.constant_(parameter, 0.0)

    agent = PPOCardPlayAgent(model=model, stochastic=False)
    chosen = agent.choose_card(observation, (first, second))

    assert chosen == first


def test_ppo_update_runs_on_small_rollout() -> None:
    model = ActorCriticCardPlayNetwork(
        input_size=4,
        hidden_sizes=(8, 4),
        action_count=2,
    )
    rollout = PPORolloutBuffer()
    legal_mask = torch.tensor([True, True])
    log_probability = -torch.log(torch.tensor(2.0)).item()

    rollout.add_episode(
        [
            PPOStep(
                features=torch.zeros(4),
                legal_mask=legal_mask,
                action=0,
                log_probability=log_probability,
                value=0.0,
            ),
            PPOStep(
                features=torch.ones(4),
                legal_mask=legal_mask,
                action=1,
                log_probability=log_probability,
                value=0.0,
            ),
        ],
        terminal_return=0.2,
    )

    optimizer = Adam(model.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(3)
    metrics = ppo_update(
        model=model,
        optimizer=optimizer,
        rollout=rollout,
        device=torch.device("cpu"),
        epochs=1,
        minibatch_size=2,
        clip_epsilon=0.2,
        value_coefficient=0.5,
        entropy_coefficient=0.01,
        max_grad_norm=0.5,
        generator=generator,
    )

    assert metrics.samples == 2
    assert torch.isfinite(torch.tensor(metrics.policy_loss))
    assert torch.isfinite(torch.tensor(metrics.value_loss))
