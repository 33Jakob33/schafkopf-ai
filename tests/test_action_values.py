from pathlib import Path

import numpy as np
import torch

from schafkopf_ai.training.action_values import (
    ActionValueArrays,
    ActionValueDataset,
    CardPlayQNetwork,
    load_action_value_arrays,
    save_action_value_arrays,
)


def _arrays() -> ActionValueArrays:
    legal_masks = np.zeros((2, 32), dtype=np.bool_)
    legal_masks[0, [0, 1]] = True
    legal_masks[1, [2, 3, 4]] = True

    q_values = np.full((2, 32), np.nan, dtype=np.float32)
    q_values[0, [0, 1]] = [10.0, 20.0]
    q_values[1, [2, 3, 4]] = [-5.0, 15.0, 5.0]

    q_stds = np.full((2, 32), np.nan, dtype=np.float32)
    q_stds[0, [0, 1]] = [2.0, 3.0]
    q_stds[1, [2, 3, 4]] = [4.0, 5.0, 6.0]

    q_counts = np.zeros((2, 32), dtype=np.uint16)
    q_counts[legal_masks] = 8

    return ActionValueArrays(
        features=np.zeros((2, 222), dtype=np.float16),
        legal_masks=legal_masks,
        q_values=q_values,
        q_stds=q_stds,
        q_counts=q_counts,
        heuristic_actions=np.asarray([0, 2], dtype=np.uint8),
        rollout_actions=np.asarray([1, 3], dtype=np.uint8),
        game_ids=np.asarray([10, 11], dtype=np.uint32),
        legal_counts=np.asarray([2, 3], dtype=np.uint8),
        contract_ids=np.asarray([0, 2], dtype=np.uint8),
        role_ids=np.asarray([0, 6], dtype=np.uint8),
    )


def test_action_value_arrays_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "q_values.npz"
    arrays = _arrays()

    save_action_value_arrays(path, arrays)
    loaded = load_action_value_arrays(path)

    assert np.array_equal(loaded.features, arrays.features)
    assert np.array_equal(loaded.legal_masks, arrays.legal_masks)
    assert np.array_equal(loaded.q_values, arrays.q_values, equal_nan=True)
    assert np.array_equal(loaded.q_stds, arrays.q_stds, equal_nan=True)
    assert np.array_equal(loaded.q_counts, arrays.q_counts)
    assert np.array_equal(loaded.rollout_actions, arrays.rollout_actions)


def test_action_value_dataset_exposes_rollout_best_action() -> None:
    dataset = ActionValueDataset(_arrays())

    features, legal, q_values, q_stds, q_counts, rollout_action = dataset[0]

    assert features.shape == (222,)
    assert legal.shape == (32,)
    assert q_values.shape == (32,)
    assert q_stds.shape == (32,)
    assert q_counts.shape == (32,)
    assert int(rollout_action.item()) == 1


def test_q_network_outputs_one_value_per_card() -> None:
    model = CardPlayQNetwork(hidden_sizes=(64, 32))
    features = torch.zeros((4, 222), dtype=torch.float32)

    q_values = model(features)

    assert q_values.shape == (4, 32)
    assert model.architecture == "q_mlp_v1"
