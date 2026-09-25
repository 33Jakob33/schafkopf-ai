from __future__ import annotations

import numpy as np

from schafkopf_ai.agents.agent import Agent
from schafkopf_ai.game.bidding import BiddingAction, BiddingObservation
from schafkopf_ai.game.card import Card
from schafkopf_ai.game.observation import PlayerObservation

from .behavior_cloning import BehaviorCloningArrays
from .demonstrations import BehaviorCloningCollector


class DAggerRecordingAgent(Agent):
    """
    Execute a learner policy while asking an expert to label visited states.

    Card play returned to the game comes from ``policy_agent``. The expert's
    card is only used as the supervised target stored in ``collector``. Bidding
    is delegated to the learner; for ``NeuralCardPlayAgent`` this is still the
    heuristic bidder, so DAgger changes only the card-play distribution.
    """

    def __init__(
        self,
        *,
        policy_agent: Agent,
        expert_agent: Agent,
        collector: BehaviorCloningCollector,
        game_id: int,
    ) -> None:
        self.policy_agent = policy_agent
        self.expert_agent = expert_agent
        self.collector = collector
        self.game_id = game_id
        self.decisions = 0
        self.disagreements = 0
        self.nontrivial_decisions = 0
        self.nontrivial_disagreements = 0

    def choose_bidding_action(
        self,
        observation: BiddingObservation,
        legal_actions: tuple[BiddingAction, ...],
    ) -> BiddingAction:
        return self.policy_agent.choose_bidding_action(observation, legal_actions)

    def choose_card(
        self,
        observation: PlayerObservation,
        legal_cards: tuple[Card, ...],
    ) -> Card:
        if not legal_cards:
            raise ValueError("No legal cards available.")

        policy_card = self.policy_agent.choose_card(observation, legal_cards)
        expert_card = self.expert_agent.choose_card(observation, legal_cards)

        self.collector.record(
            game_id=self.game_id,
            observation=observation,
            legal_cards=legal_cards,
            chosen_card=expert_card,
        )

        self.decisions += 1
        disagreement = policy_card != expert_card
        self.disagreements += int(disagreement)

        if len(legal_cards) > 1:
            self.nontrivial_decisions += 1
            self.nontrivial_disagreements += int(disagreement)

        return policy_card


def aggregate_dagger_arrays(
    base: BehaviorCloningArrays,
    additions: BehaviorCloningArrays,
) -> BehaviorCloningArrays:
    """
    Append DAgger examples while remapping their game ids after the base data.

    Keeping every newly collected game id distinct preserves the game-level
    train/validation split used by behavioral-cloning training.
    """
    if base.example_count <= 0 or additions.example_count <= 0:
        raise ValueError("Both base and DAgger datasets must contain examples.")

    next_game_id = int(base.game_ids.max()) + 1
    addition_ids = additions.game_ids.astype(np.uint64)
    remapped_ids = addition_ids + next_game_id

    if int(remapped_ids.max()) > np.iinfo(np.uint32).max:
        raise OverflowError("Aggregated DAgger game ids exceed uint32 capacity.")

    return BehaviorCloningArrays(
        features=np.concatenate((base.features, additions.features), axis=0),
        legal_masks=np.concatenate(
            (base.legal_masks, additions.legal_masks),
            axis=0,
        ),
        targets=np.concatenate((base.targets, additions.targets), axis=0),
        game_ids=np.concatenate(
            (
                base.game_ids.astype(np.uint32, copy=False),
                remapped_ids.astype(np.uint32),
            ),
            axis=0,
        ),
        legal_counts=np.concatenate(
            (base.legal_counts, additions.legal_counts),
            axis=0,
        ),
    )
