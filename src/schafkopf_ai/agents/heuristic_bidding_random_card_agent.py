from __future__ import annotations

import random

from schafkopf_ai.game.bidding import BiddingAction, BiddingObservation
from schafkopf_ai.game.card import Card
from schafkopf_ai.game.observation import PlayerObservation

from .agent import Agent
from .heuristic_agent import HeuristicAgent, HeuristicConfig
from .heuristic_bidding import (
    BiddingHeuristicConfig,
    DEFAULT_BIDDING_HEURISTIC_CONFIG,
)
from .random_agent import RandomAgent


class HeuristicBiddingRandomCardAgent(Agent):
    """
    Use realistic heuristic bidding while keeping card play random.

    This is useful when evaluating or generating card-play data: the contract
    distribution comes from the bidding heuristic, while opponent card play
    remains stochastic and intentionally weak.
    """

    def __init__(
        self,
        rng: random.Random | None = None,
        *,
        bidding_config: BiddingHeuristicConfig = DEFAULT_BIDDING_HEURISTIC_CONFIG,
    ) -> None:
        self.bidding_agent = HeuristicAgent(
            config=HeuristicConfig(bidding=bidding_config)
        )
        self.card_agent = RandomAgent(rng=rng)

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
        return self.card_agent.choose_card(observation, legal_cards)
