from __future__ import annotations

import numpy as np

from schafkopf_ai.agents.heuristic_agent import HeuristicAgent
from schafkopf_ai.game.card import Card
from schafkopf_ai.game.observation import PlayerObservation

from .behavior_cloning import BehaviorCloningArrays
from .card_play_encoding import (
    ACTION_COUNT,
    OBSERVATION_FEATURE_SIZE,
    card_to_action_index,
    encode_card_play,
)


class BehaviorCloningCollector:
    """Preallocated collector for card-play demonstrations."""

    def __init__(self, expected_examples: int) -> None:
        if expected_examples <= 0:
            raise ValueError("expected_examples must be greater than zero.")

        self.features = np.empty(
            (expected_examples, OBSERVATION_FEATURE_SIZE),
            dtype=np.float16,
        )
        self.legal_masks = np.empty(
            (expected_examples, ACTION_COUNT),
            dtype=np.bool_,
        )
        self.targets = np.empty(expected_examples, dtype=np.uint8)
        self.game_ids = np.empty(expected_examples, dtype=np.uint32)
        self.legal_counts = np.empty(expected_examples, dtype=np.uint8)

        self._expected_examples = expected_examples
        self._count = 0

    @property
    def count(self) -> int:
        return self._count

    @property
    def expected_examples(self) -> int:
        return self._expected_examples

    def record(
        self,
        *,
        game_id: int,
        observation: PlayerObservation,
        legal_cards: tuple[Card, ...],
        chosen_card: Card,
    ) -> None:
        if self._count >= self._expected_examples:
            raise RuntimeError("Behavior-cloning collector capacity exceeded.")

        if chosen_card not in legal_cards:
            raise ValueError("Demonstration target must be one of the legal cards.")

        encoded = encode_card_play(observation, legal_cards)
        target = card_to_action_index(chosen_card)

        index = self._count
        self.features[index] = np.asarray(encoded.features, dtype=np.float16)
        self.legal_masks[index] = np.asarray(
            encoded.legal_action_mask,
            dtype=np.bool_,
        )
        self.targets[index] = target
        self.game_ids[index] = game_id
        self.legal_counts[index] = len(legal_cards)
        self._count += 1

    def arrays(self) -> BehaviorCloningArrays:
        """Return exactly the examples recorded so far."""
        if self._count == 0:
            raise ValueError("No demonstrations have been recorded.")

        return BehaviorCloningArrays(
            features=self.features[: self._count],
            legal_masks=self.legal_masks[: self._count],
            targets=self.targets[: self._count],
            game_ids=self.game_ids[: self._count],
            legal_counts=self.legal_counts[: self._count],
        )


class RecordingHeuristicAgent(HeuristicAgent):
    """Heuristic teacher that records every card-play decision it makes."""

    def __init__(
        self,
        *,
        collector: BehaviorCloningCollector,
        game_id: int,
    ) -> None:
        super().__init__()
        self.collector = collector
        self.game_id = game_id

    def choose_card(
        self,
        observation: PlayerObservation,
        legal_cards: tuple[Card, ...],
    ) -> Card:
        chosen_card = super().choose_card(observation, legal_cards)
        self.collector.record(
            game_id=self.game_id,
            observation=observation,
            legal_cards=legal_cards,
            chosen_card=chosen_card,
        )
        return chosen_card
