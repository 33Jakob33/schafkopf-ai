from __future__ import annotations

import json
from pathlib import Path

import torch

from schafkopf_ai.game.bidding import BiddingAction, BiddingObservation
from schafkopf_ai.game.card import Card
from schafkopf_ai.game.observation import PlayerObservation
from schafkopf_ai.training.card_play_roles import (
    ROLE_KEYS,
    CardPlayRole,
    infer_card_play_role,
    normalize_card_play_role,
)

from .agent import Agent
from .heuristic_agent import HeuristicAgent
from .neural_card_play_agent import NeuralCardPlayAgent

SPECIALIST_MANIFEST_FORMAT = "specialist_card_play_v1"


class SpecialistCardPlayAgent(Agent):
    """Route each known contract/role situation to its dedicated neural policy."""

    def __init__(
        self,
        *,
        specialists: dict[CardPlayRole, NeuralCardPlayAgent],
        bidding_agent: Agent | None = None,
    ) -> None:
        missing = [role.key for role in CardPlayRole if role not in specialists]
        if missing:
            raise ValueError(
                "Missing specialist policies for roles: " + ", ".join(missing)
            )

        self.specialists = dict(specialists)
        self.bidding_agent = (
            HeuristicAgent() if bidding_agent is None else bidding_agent
        )

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> SpecialistCardPlayAgent:
        path = Path(manifest_path)
        with path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)

        if not isinstance(manifest, dict):
            raise TypeError("Specialist manifest must contain a JSON object.")
        if manifest.get("format") != SPECIALIST_MANIFEST_FORMAT:
            raise ValueError(
                f"Unsupported specialist manifest format: {manifest.get('format')!r}."
            )

        raw_roles = manifest.get("roles")
        if not isinstance(raw_roles, dict):
            raise TypeError("Specialist manifest must contain a roles object.")

        specialists: dict[CardPlayRole, NeuralCardPlayAgent] = {}
        for role_key in ROLE_KEYS:
            raw_checkpoint = raw_roles.get(role_key)
            if not isinstance(raw_checkpoint, str):
                raise TypeError(
                    f"Specialist manifest has no checkpoint for {role_key!r}."
                )
            checkpoint = Path(raw_checkpoint)
            if not checkpoint.is_absolute():
                checkpoint = path.parent / checkpoint

            role = normalize_card_play_role(role_key)
            metadata = torch.load(
                checkpoint,
                map_location="cpu",
                weights_only=True,
            )
            if not isinstance(metadata, dict):
                raise TypeError(
                    f"Specialist checkpoint {checkpoint} must contain a dictionary."
                )
            checkpoint_role = metadata.get("specialist_role")
            if checkpoint_role != role.key:
                raise ValueError(
                    f"Checkpoint {checkpoint} is tagged for role "
                    f"{checkpoint_role!r}, expected {role.key!r}."
                )

            agent = NeuralCardPlayAgent.from_checkpoint(
                checkpoint,
                device=device,
            )
            specialists[role] = agent

        return cls(specialists=specialists)

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
        role = infer_card_play_role(observation)
        return self.specialists[role].choose_card(observation, legal_cards)
