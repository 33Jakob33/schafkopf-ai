from pathlib import Path

import numpy as np

from scripts.generate_bc_dataset import run_generation
from schafkopf_ai.training.behavior_cloning import load_behavior_cloning_arrays


def test_focal_random_generation_records_eight_decisions_per_game(
    tmp_path: Path,
) -> None:
    output = tmp_path / "focal_random.npz"

    run_generation(
        games=4,
        seed=123,
        output=output,
        progress_every=0,
        observation_version="v2",
        generation_mode="focal-random",
    )

    arrays = load_behavior_cloning_arrays(output)

    assert arrays.example_count == 32
    assert arrays.contract_ids is not None
    unique_ids, counts = np.unique(arrays.game_ids, return_counts=True)
    assert unique_ids.tolist() == [0, 1, 2, 3]
    assert counts.tolist() == [8, 8, 8, 8]
