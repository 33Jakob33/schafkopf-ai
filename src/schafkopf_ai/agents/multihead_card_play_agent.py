from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from schafkopf_ai.game.bidding import BiddingAction, BiddingObservation
from schafkopf_ai.game.card import Card
from schafkopf_ai.game.observation import PlayerObservation
from schafkopf_ai.training.behavior_cloning import mask_illegal_logits
from schafkopf_ai.training.card_play_encoding import ACTION_COUNT, action_index_to_card
from schafkopf_ai.training.card_play_encoding_v2 import OBSERVATION_V2_FEATURE_SIZE
from schafkopf_ai.training.card_play_roles import infer_card_play_role
from schafkopf_ai.training.multihead_policy import MultiHeadCardPlayPolicyNetwork
from schafkopf_ai.training.observation_encoding import encode_card_play_for_version

from .agent import Agent
from .heuristic_agent import HeuristicAgent


class MultiHeadCardPlayAgent(Agent):
    """Card-play agent using a shared representation and role-specific policy heads."""

    def __init__(
        self,
        *,
        model: MultiHeadCardPlayPolicyNetwork,
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
    ) -> MultiHeadCardPlayAgent:
        checkpoint = torch.load(
            Path(checkpoint_path),
            map_location=device,
            weights_only=True,
        )
        if not isinstance(checkpoint, dict):
            raise TypeError("Multi-head checkpoint must be a dictionary.")
        if checkpoint.get("architecture") != MultiHeadCardPlayPolicyNetwork.architecture:
            raise ValueError("Checkpoint is not a multi-head card-play policy.")

        input_size = int(checkpoint.get("input_size", OBSERVATION_V2_FEATURE_SIZE))
        shared_raw = checkpoint.get("shared_hidden_sizes", (512, 512))
        if not isinstance(shared_raw, (tuple, list)) or len(shared_raw) != 2:
            raise ValueError("shared_hidden_sizes must contain exactly two values.")

        model = MultiHeadCardPlayPolicyNetwork(
            input_size=input_size,
            shared_hidden_sizes=(int(shared_raw[0]), int(shared_raw[1])),
            head_hidden_size=int(checkpoint.get("head_hidden_size", 256)),
            action_count=int(checkpoint.get("action_count", ACTION_COUNT)),
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
        ).unsqueeze(0)
        role_id = torch.tensor(
            [int(infer_card_play_role(observation))],
            dtype=torch.long,
            device=self.device,
        )

        with torch.no_grad():
            logits = self.model(features, role_id)
            masked_logits = mask_illegal_logits(logits, legal_mask)
            action_index = int(masked_logits.argmax(dim=1).item())

        chosen = action_index_to_card(action_index)
        if chosen not in legal_cards:
            raise RuntimeError("Multi-head policy selected an illegal card.")
        return chosen
