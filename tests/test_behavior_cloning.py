from pathlib import Path

import numpy as np
import torch

from schafkopf_ai.agents.neural_card_play_agent import NeuralCardPlayAgent
from schafkopf_ai.game.card import Card, Rank, Suit
from schafkopf_ai.game.game_contract import GameContract
from schafkopf_ai.game.game_type import GameType
from schafkopf_ai.game.observation import PlayerObservation
from schafkopf_ai.training.behavior_cloning import (
    BehaviorCloningArrays,
    CardPlayPolicyNetwork,
    load_behavior_cloning_arrays,
    mask_illegal_logits,
    save_behavior_cloning_arrays,
    split_indices_by_game,
)
from schafkopf_ai.training.card_play_encoding import (
    ACTION_COUNT,
    OBSERVATION_FEATURE_SIZE,
    card_to_action_index,
)
from schafkopf_ai.training.demonstrations import BehaviorCloningCollector


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


def test_card_play_policy_network_has_32_logits() -> None:
    model = CardPlayPolicyNetwork()
    features = torch.zeros((3, OBSERVATION_FEATURE_SIZE), dtype=torch.float32)

    logits = model(features)

    assert logits.shape == (3, ACTION_COUNT)


def test_mask_illegal_logits_prevents_illegal_argmax() -> None:
    logits = torch.arange(ACTION_COUNT, dtype=torch.float32)
    legal_mask = torch.zeros(ACTION_COUNT, dtype=torch.bool)
    legal_mask[3] = True
    legal_mask[7] = True

    masked = mask_illegal_logits(logits, legal_mask)

    assert int(masked.argmax().item()) == 7
    assert masked[31] < masked[3]


def test_collector_records_heuristic_target_as_legal_action() -> None:
    first = Card(Suit.EICHEL, Rank.SEVEN)
    second = Card(Suit.GRAS, Rank.SEVEN)
    observation = _observation((first, second))
    collector = BehaviorCloningCollector(expected_examples=1)

    collector.record(
        game_id=12,
        observation=observation,
        legal_cards=(first, second),
        chosen_card=second,
    )
    arrays = collector.arrays()

    assert arrays.example_count == 1
    assert arrays.game_ids.tolist() == [12]
    assert arrays.legal_counts.tolist() == [2]
    assert int(arrays.targets[0]) == card_to_action_index(second)
    assert arrays.legal_masks[0, card_to_action_index(second)]


def test_behavior_cloning_dataset_round_trip(tmp_path: Path) -> None:
    features = np.zeros((2, OBSERVATION_FEATURE_SIZE), dtype=np.float16)
    legal_masks = np.zeros((2, ACTION_COUNT), dtype=np.bool_)
    legal_masks[0, 0] = True
    legal_masks[1, 8] = True
    arrays = BehaviorCloningArrays(
        features=features,
        legal_masks=legal_masks,
        targets=np.asarray([0, 8], dtype=np.uint8),
        game_ids=np.asarray([0, 1], dtype=np.uint32),
        legal_counts=np.asarray([1, 1], dtype=np.uint8),
    )
    path = tmp_path / "bc.npz"

    save_behavior_cloning_arrays(path, arrays)
    loaded = load_behavior_cloning_arrays(path)

    assert np.array_equal(loaded.features, arrays.features)
    assert np.array_equal(loaded.legal_masks, arrays.legal_masks)
    assert np.array_equal(loaded.targets, arrays.targets)
    assert np.array_equal(loaded.game_ids, arrays.game_ids)


def test_split_indices_keeps_games_together() -> None:
    game_ids = np.asarray([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.uint32)

    train_indices, validation_indices = split_indices_by_game(
        game_ids,
        validation_fraction=0.25,
        seed=7,
    )

    train_games = {int(game_ids[index]) for index in train_indices}
    validation_games = {int(game_ids[index]) for index in validation_indices}

    assert train_games
    assert validation_games
    assert train_games.isdisjoint(validation_games)


def test_neural_agent_masks_illegal_cards() -> None:
    first = Card(Suit.EICHEL, Rank.SEVEN)
    second = Card(Suit.GRAS, Rank.SEVEN)
    observation = _observation((first, second))
    model = CardPlayPolicyNetwork()

    for parameter in model.parameters():
        torch.nn.init.constant_(parameter, 0.0)

    agent = NeuralCardPlayAgent(model=model)

    chosen = agent.choose_card(observation, (first, second))

    assert chosen == first
