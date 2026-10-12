from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset

from schafkopf_ai.agents.heuristic_agent import HeuristicAgent, HeuristicConfig
from schafkopf_ai.game.card import Card
from schafkopf_ai.game.observation import PlayerObservation
from schafkopf_ai.search.rollout import MonteCarloActionEvaluator

from .card_play_encoding import (
    ACTION_COUNT,
    GAME_TYPE_ORDER,
    card_to_action_index,
)
from .card_play_encoding_v2 import OBSERVATION_V2_FEATURE_SIZE
from .card_play_roles import infer_card_play_role
from .observation_encoding import encode_card_play_for_version


@dataclass(frozen=True, slots=True)
class ActionValueArrays:
    """Dense V2 states with Monte-Carlo Q targets for all legal actions."""

    features: np.ndarray
    legal_masks: np.ndarray
    q_values: np.ndarray
    q_stds: np.ndarray
    q_counts: np.ndarray
    heuristic_actions: np.ndarray
    rollout_actions: np.ndarray
    game_ids: np.ndarray
    legal_counts: np.ndarray
    contract_ids: np.ndarray
    role_ids: np.ndarray

    @property
    def example_count(self) -> int:
        return int(self.features.shape[0])


class ActionValueDataset(
    Dataset[tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]
):
    """PyTorch view over Q-value supervision."""

    def __init__(self, arrays: ActionValueArrays) -> None:
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays.example_count

    def __getitem__(
        self,
        index: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        return (
            torch.from_numpy(self.arrays.features[index]).to(torch.float32),
            torch.from_numpy(self.arrays.legal_masks[index]).to(torch.bool),
            torch.from_numpy(self.arrays.q_values[index]).to(torch.float32),
            torch.from_numpy(self.arrays.q_stds[index]).to(torch.float32),
            torch.from_numpy(self.arrays.q_counts[index].astype(np.int64)).to(
                torch.long
            ),
        )


class CardPlayQNetwork(nn.Module):
    """V2 MLP that predicts expected payment for each physical card action."""

    architecture = "q_mlp_v1"

    def __init__(
        self,
        *,
        input_size: int = OBSERVATION_V2_FEATURE_SIZE,
        hidden_sizes: tuple[int, int] = (512, 256),
        action_count: int = ACTION_COUNT,
    ) -> None:
        super().__init__()

        if input_size != OBSERVATION_V2_FEATURE_SIZE:
            raise ValueError(
                "Q policy currently requires V2 observations: "
                f"{input_size} != {OBSERVATION_V2_FEATURE_SIZE}."
            )

        first_hidden, second_hidden = hidden_sizes
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.action_count = action_count
        self.observation_version = "v2"

        self.network = nn.Sequential(
            nn.Linear(input_size, first_hidden),
            nn.ReLU(),
            nn.Linear(first_hidden, second_hidden),
            nn.ReLU(),
            nn.Linear(second_hidden, action_count),
        )

    def forward(self, features: Tensor) -> Tensor:
        return cast(Tensor, self.network(features))


class ActionValueCollector:
    """Preallocated collector for expensive rollout-supervised V2 states."""

    def __init__(self, expected_examples: int) -> None:
        if expected_examples <= 0:
            raise ValueError("expected_examples must be greater than zero.")

        self.features = np.empty(
            (expected_examples, OBSERVATION_V2_FEATURE_SIZE),
            dtype=np.float16,
        )
        self.legal_masks = np.empty((expected_examples, ACTION_COUNT), dtype=np.bool_)
        self.q_values = np.full(
            (expected_examples, ACTION_COUNT),
            np.nan,
            dtype=np.float32,
        )
        self.q_stds = np.full(
            (expected_examples, ACTION_COUNT),
            np.nan,
            dtype=np.float32,
        )
        self.q_counts = np.zeros(
            (expected_examples, ACTION_COUNT),
            dtype=np.uint16,
        )
        self.heuristic_actions = np.empty(expected_examples, dtype=np.uint8)
        self.rollout_actions = np.empty(expected_examples, dtype=np.uint8)
        self.game_ids = np.empty(expected_examples, dtype=np.uint32)
        self.legal_counts = np.empty(expected_examples, dtype=np.uint8)
        self.contract_ids = np.empty(expected_examples, dtype=np.uint8)
        self.role_ids = np.empty(expected_examples, dtype=np.uint8)

        self._expected_examples = expected_examples
        self._count = 0

    @property
    def count(self) -> int:
        return self._count

    @property
    def is_full(self) -> bool:
        return self._count >= self._expected_examples

    def record(
        self,
        *,
        game_id: int,
        observation: PlayerObservation,
        legal_cards: tuple[Card, ...],
        heuristic_card: Card,
        evaluator: MonteCarloActionEvaluator,
    ) -> None:
        if self.is_full:
            return
        if len(legal_cards) <= 1:
            raise ValueError("Action-value datasets should contain non-trivial states.")
        if heuristic_card not in legal_cards:
            raise ValueError("Heuristic action must be legal.")

        encoded = encode_card_play_for_version("v2", observation, legal_cards)
        estimates = evaluator.evaluate(observation, legal_cards)
        by_card = {estimate.card: estimate for estimate in estimates}

        best_value = max(estimate.mean_payment for estimate in estimates)
        best_cards = tuple(
            card
            for card in legal_cards
            if by_card[card].mean_payment == best_value
        )
        rollout_card = (
            heuristic_card
            if heuristic_card in best_cards
            else next(card for card in legal_cards if card in best_cards)
        )

        index = self._count
        self.features[index] = np.asarray(encoded.features, dtype=np.float16)
        self.legal_masks[index] = np.asarray(
            encoded.legal_action_mask,
            dtype=np.bool_,
        )
        self.heuristic_actions[index] = card_to_action_index(heuristic_card)
        self.rollout_actions[index] = card_to_action_index(rollout_card)
        self.game_ids[index] = game_id
        self.legal_counts[index] = len(legal_cards)
        self.contract_ids[index] = GAME_TYPE_ORDER.index(
            observation.contract.game_type
        )
        self.role_ids[index] = int(infer_card_play_role(observation))

        for card in legal_cards:
            action = card_to_action_index(card)
            estimate = by_card[card]
            self.q_values[index, action] = estimate.mean_payment
            self.q_stds[index, action] = estimate.std_payment
            self.q_counts[index, action] = estimate.sample_count

        self._count += 1

    def arrays(self) -> ActionValueArrays:
        if self._count == 0:
            raise ValueError("No action-value examples have been recorded.")

        return ActionValueArrays(
            features=self.features[: self._count],
            legal_masks=self.legal_masks[: self._count],
            q_values=self.q_values[: self._count],
            q_stds=self.q_stds[: self._count],
            q_counts=self.q_counts[: self._count],
            heuristic_actions=self.heuristic_actions[: self._count],
            rollout_actions=self.rollout_actions[: self._count],
            game_ids=self.game_ids[: self._count],
            legal_counts=self.legal_counts[: self._count],
            contract_ids=self.contract_ids[: self._count],
            role_ids=self.role_ids[: self._count],
        )


class RecordingActionValueAgent(HeuristicAgent):
    """
    Heuristic actor that labels its non-trivial states with rollout action values.

    The actual played card remains the heuristic choice. Search is used only to
    create supervision, keeping state generation separate from target generation.
    """

    def __init__(
        self,
        *,
        collector: ActionValueCollector,
        evaluator: MonteCarloActionEvaluator,
        game_id: int,
        config: HeuristicConfig | None = None,
    ) -> None:
        super().__init__(config=config)
        self.collector = collector
        self.evaluator = evaluator
        self.game_id = game_id

    def choose_card(
        self,
        observation: PlayerObservation,
        legal_cards: tuple[Card, ...],
    ) -> Card:
        heuristic_card = super().choose_card(observation, legal_cards)

        if len(legal_cards) > 1 and not self.collector.is_full:
            self.collector.record(
                game_id=self.game_id,
                observation=observation,
                legal_cards=legal_cards,
                heuristic_card=heuristic_card,
                evaluator=self.evaluator,
            )

        return heuristic_card


def save_action_value_arrays(
    path: str | Path,
    arrays: ActionValueArrays,
) -> None:
    _validate_action_value_arrays(arrays)

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        features=arrays.features,
        legal_masks=arrays.legal_masks,
        q_values=arrays.q_values,
        q_stds=arrays.q_stds,
        q_counts=arrays.q_counts,
        heuristic_actions=arrays.heuristic_actions,
        rollout_actions=arrays.rollout_actions,
        game_ids=arrays.game_ids,
        legal_counts=arrays.legal_counts,
        contract_ids=arrays.contract_ids,
        role_ids=arrays.role_ids,
    )


def load_action_value_arrays(path: str | Path) -> ActionValueArrays:
    with np.load(Path(path), allow_pickle=False) as data:
        arrays = ActionValueArrays(
            features=data["features"],
            legal_masks=data["legal_masks"],
            q_values=data["q_values"],
            q_stds=data["q_stds"],
            q_counts=data["q_counts"],
            heuristic_actions=data["heuristic_actions"],
            rollout_actions=data["rollout_actions"],
            game_ids=data["game_ids"],
            legal_counts=data["legal_counts"],
            contract_ids=data["contract_ids"],
            role_ids=data["role_ids"],
        )

    _validate_action_value_arrays(arrays)
    return arrays


def _validate_action_value_arrays(arrays: ActionValueArrays) -> None:
    count = arrays.example_count
    if count <= 0:
        raise ValueError("Action-value dataset cannot be empty.")

    if arrays.features.shape != (count, OBSERVATION_V2_FEATURE_SIZE):
        raise ValueError(f"Unexpected action-value feature shape: {arrays.features.shape}.")
    if arrays.legal_masks.shape != (count, ACTION_COUNT):
        raise ValueError(f"Unexpected legal-mask shape: {arrays.legal_masks.shape}.")
    if arrays.q_values.shape != (count, ACTION_COUNT):
        raise ValueError(f"Unexpected q_values shape: {arrays.q_values.shape}.")
    if arrays.q_stds.shape != (count, ACTION_COUNT):
        raise ValueError(f"Unexpected q_stds shape: {arrays.q_stds.shape}.")
    if arrays.q_counts.shape != (count, ACTION_COUNT):
        raise ValueError(f"Unexpected q_counts shape: {arrays.q_counts.shape}.")

    for name, values in (
        ("heuristic_actions", arrays.heuristic_actions),
        ("rollout_actions", arrays.rollout_actions),
        ("game_ids", arrays.game_ids),
        ("legal_counts", arrays.legal_counts),
        ("contract_ids", arrays.contract_ids),
        ("role_ids", arrays.role_ids),
    ):
        if values.shape != (count,):
            raise ValueError(f"Unexpected {name} shape: {values.shape}.")

    if np.any(arrays.legal_counts <= 1):
        raise ValueError("Action-value dataset must contain only non-trivial states.")
    if np.any(arrays.contract_ids >= 5):
        raise ValueError("Action-value dataset contains an invalid contract id.")
    if np.any(arrays.role_ids >= 8):
        raise ValueError("Action-value dataset contains an invalid role id.")

    rows = np.arange(count)
    for actions, name in (
        (arrays.heuristic_actions, "heuristic"),
        (arrays.rollout_actions, "rollout"),
    ):
        if np.any(actions >= ACTION_COUNT):
            raise ValueError(f"Invalid {name} action index.")
        if not np.all(arrays.legal_masks[rows, actions.astype(np.int64)]):
            raise ValueError(f"Every {name} action must be legal.")

    legal = arrays.legal_masks
    illegal = ~legal
    if not np.all(np.isfinite(arrays.q_values[legal])):
        raise ValueError("Every legal action requires a finite Q target.")
    if not np.all(np.isfinite(arrays.q_stds[legal])):
        raise ValueError("Every legal action requires a finite Q standard deviation.")
    if np.any(arrays.q_counts[legal] <= 0):
        raise ValueError("Every legal action requires at least one rollout sample.")
    if not np.all(np.isnan(arrays.q_values[illegal])):
        raise ValueError("Illegal actions must have NaN Q targets.")
    if not np.all(np.isnan(arrays.q_stds[illegal])):
        raise ValueError("Illegal actions must have NaN Q standard deviations.")
    if np.any(arrays.q_counts[illegal] != 0):
        raise ValueError("Illegal actions must have zero rollout samples.")
