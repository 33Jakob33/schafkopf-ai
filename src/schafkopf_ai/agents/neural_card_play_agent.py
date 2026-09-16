from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from schafkopf_ai.game.bidding import BiddingAction, BiddingObservation
from schafkopf_ai.game.card import Card
from schafkopf_ai.game.observation import PlayerObservation
from schafkopf_ai.training.behavior_cloning import (
    CardPlayPolicyNetwork,
    mask_illegal_logits,
)
from schafkopf_ai.training.card_play_encoding import (
    ACTION_COUNT,
    OBSERVATION_FEATURE_SIZE,
    action_index_to_card,
    encode_card_play,
)

from .agent import Agent
from .heuristic_agent import HeuristicAgent


class NeuralCardPlayAgent(Agent):
    """
    Learned card-play policy with heuristic bidding.

    The engine still computes legal moves. The policy network only scores the
    32 physical cards, after which illegal logits are masked before selection.
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        device: str | torch.device = "cpu",
        bidding_agent: Agent | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.bidding_agent = (
            HeuristicAgent() if bidding_agent is None else bidding_agent
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> NeuralCardPlayAgent:
        checkpoint = torch.load(
            Path(checkpoint_path),
            map_location=device,
            weights_only=True,
        )

        if not isinstance(checkpoint, dict):
            raise TypeError("Behavior-cloning checkpoint must be a dictionary.")

        input_size = int(checkpoint.get("input_size", OBSERVATION_FEATURE_SIZE))
        action_count = int(checkpoint.get("action_count", ACTION_COUNT))
        hidden_sizes_raw = checkpoint.get("hidden_sizes", (512, 256))

        if (
            not isinstance(hidden_sizes_raw, (tuple, list))
            or len(hidden_sizes_raw) != 2
        ):
            raise ValueError("Checkpoint hidden_sizes must contain exactly two values.")

        hidden_sizes = (int(hidden_sizes_raw[0]), int(hidden_sizes_raw[1]))

        if input_size != OBSERVATION_FEATURE_SIZE:
            raise ValueError(
                "Checkpoint observation size does not match the current encoder: "
                f"{input_size} != {OBSERVATION_FEATURE_SIZE}."
            )
        if action_count != ACTION_COUNT:
            raise ValueError(
                "Checkpoint action count does not match the current card mapping: "
                f"{action_count} != {ACTION_COUNT}."
            )

        model = CardPlayPolicyNetwork(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            action_count=action_count,
        )

        state_dict: Any = checkpoint.get("model_state_dict")
        if not isinstance(state_dict, dict):
            raise TypeError("Checkpoint does not contain model_state_dict.")
        model.load_state_dict(state_dict)

        return cls(model=model, device=device)

    def choose_bidding_action(
        self,
        observation: BiddingObservation,
        legal_actions: tuple[BiddingAction, ...],
    ) -> BiddingAction:
        return self.bidding_agent.choose_bidding_action(observation, legal_actions)

    def choose_card(
        self,
        observation: PlayerObservation,
        legal_cards: tuple[Card, ...],
    ) -> Card:
        if not legal_cards:
            raise ValueError("No legal cards available.")

        if len(legal_cards) == 1:
            return legal_cards[0]

        encoded = encode_card_play(observation, legal_cards)
        features = torch.tensor(
            encoded.features,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        legal_mask = torch.tensor(
            encoded.legal_action_mask,
            dtype=torch.bool,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(features)
            masked_logits = mask_illegal_logits(logits, legal_mask)
            action_index = int(masked_logits.argmax(dim=1).item())

        chosen_card = action_index_to_card(action_index)
        if chosen_card not in legal_cards:
            raise RuntimeError("Neural policy selected an illegal card after masking.")

        return chosen_card
