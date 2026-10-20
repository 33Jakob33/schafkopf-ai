from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from schafkopf_ai.game.bidding import BiddingAction, BiddingObservation
from schafkopf_ai.game.card import Card
from schafkopf_ai.game.observation import PlayerObservation
from schafkopf_ai.training.action_values import CardPlayQNetwork
from schafkopf_ai.training.card_play_encoding import (
    ACTION_COUNT,
    action_index_to_card,
)
from schafkopf_ai.training.card_play_encoding_v2 import OBSERVATION_V2_FEATURE_SIZE
from schafkopf_ai.training.observation_encoding import encode_card_play_for_version

from .agent import Agent
from .heuristic_agent import HeuristicAgent


class QCardPlayAgent(Agent):
    """Greedy card-play agent over learned expected-payment Q values."""

    def __init__(
        self,
        *,
        model: CardPlayQNetwork,
        target_scale: float = 100.0,
        device: str | torch.device = "cpu",
        bidding_agent: Agent | None = None,
    ) -> None:
        if target_scale <= 0.0:
            raise ValueError("target_scale must be positive.")
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.target_scale = target_scale
        self.bidding_agent = (
            HeuristicAgent() if bidding_agent is None else bidding_agent
        )
        self.last_q_values: tuple[float, ...] = ()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> QCardPlayAgent:
        checkpoint = torch.load(
            Path(checkpoint_path),
            map_location=device,
            weights_only=True,
        )
        if not isinstance(checkpoint, dict):
            raise TypeError("Q checkpoint must be a dictionary.")
        if checkpoint.get("architecture") != CardPlayQNetwork.architecture:
            raise ValueError("Checkpoint is not a card-play Q network.")

        input_size = int(checkpoint.get("input_size", OBSERVATION_V2_FEATURE_SIZE))
        hidden_raw = checkpoint.get("hidden_sizes", (512, 256))
        if not isinstance(hidden_raw, (tuple, list)) or len(hidden_raw) != 2:
            raise ValueError("Q checkpoint hidden_sizes must contain two values.")

        model = CardPlayQNetwork(
            input_size=input_size,
            hidden_sizes=(int(hidden_raw[0]), int(hidden_raw[1])),
            action_count=int(checkpoint.get("action_count", ACTION_COUNT)),
        )

        state_dict: Any = checkpoint.get("model_state_dict")
        if not isinstance(state_dict, dict):
            raise TypeError("Q checkpoint does not contain model_state_dict.")
        model.load_state_dict(state_dict)

        return cls(
            model=model,
            target_scale=float(checkpoint.get("target_scale", 100.0)),
            device=device,
        )

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
            self.last_q_values = ()
            return legal_cards[0]

        encoded = encode_card_play_for_version("v2", observation, legal_cards)
        features = torch.tensor(
            encoded.features,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        legal_mask = torch.tensor(
            encoded.legal_action_mask,
            dtype=torch.bool,
            device=self.device,
        )

        with torch.no_grad():
            scaled_q = self.model(features).squeeze(0)
            payment_q = scaled_q * self.target_scale
            self.last_q_values = tuple(
                float(value) for value in payment_q.cpu().tolist()
            )

            minimum = torch.finfo(scaled_q.dtype).min
            masked_q = scaled_q.masked_fill(~legal_mask, minimum)
            action_index = int(masked_q.argmax().item())

        chosen = action_index_to_card(action_index)
        if chosen not in legal_cards:
            raise RuntimeError("Q policy selected an illegal card.")
        return chosen
