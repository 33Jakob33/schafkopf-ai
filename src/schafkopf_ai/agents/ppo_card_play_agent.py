from __future__ import annotations

from pathlib import Path

import torch
from torch.distributions import Categorical

from schafkopf_ai.game.bidding import BiddingAction, BiddingObservation
from schafkopf_ai.game.card import Card
from schafkopf_ai.game.observation import PlayerObservation
from schafkopf_ai.training.card_play_encoding import (
    action_index_to_card,
    encode_card_play,
)
from schafkopf_ai.training.ppo import (
    ActorCriticCardPlayNetwork,
    PPOStep,
    load_ppo_model,
)
from schafkopf_ai.training.behavior_cloning import mask_illegal_logits

from .agent import Agent
from .heuristic_agent import HeuristicAgent


class PPOCardPlayAgent(Agent):
    """Actor-critic card-play agent with heuristic bidding."""

    def __init__(
        self,
        *,
        model: ActorCriticCardPlayNetwork,
        device: str | torch.device = "cpu",
        stochastic: bool = False,
        bidding_agent: Agent | None = None,
        record_trajectory: bool = False,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.stochastic = stochastic
        self.bidding_agent = (
            HeuristicAgent() if bidding_agent is None else bidding_agent
        )
        self.record_trajectory = record_trajectory
        self._trajectory: list[PPOStep] = []

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device = "cpu",
        stochastic: bool = False,
    ) -> PPOCardPlayAgent:
        model = load_ppo_model(checkpoint_path, device=device)
        return cls(model=model, device=device, stochastic=stochastic)

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
            logits, values = self.model(features)
            masked_logits = mask_illegal_logits(logits, legal_mask)
            distribution = Categorical(logits=masked_logits)

            if self.stochastic:
                action_tensor = distribution.sample()
            else:
                action_tensor = masked_logits.argmax(dim=1)

            action_index = int(action_tensor.item())
            log_probability = float(
                distribution.log_prob(action_tensor).item()
            )
            value = float(values.item())

        chosen_card = action_index_to_card(action_index)
        if chosen_card not in legal_cards:
            raise RuntimeError("PPO policy selected an illegal card after masking.")

        if self.record_trajectory:
            self._trajectory.append(
                PPOStep(
                    features=features.squeeze(0).detach().cpu(),
                    legal_mask=legal_mask.squeeze(0).detach().cpu(),
                    action=action_index,
                    log_probability=log_probability,
                    value=value,
                )
            )

        return chosen_card

    def drain_trajectory(self) -> list[PPOStep]:
        trajectory = self._trajectory
        self._trajectory = []
        return trajectory
