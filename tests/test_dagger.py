import numpy as np

from schafkopf_ai.agents.agent import Agent
from schafkopf_ai.game.bidding import BiddingAction, BiddingObservation
from schafkopf_ai.game.card import Card, Rank, Suit
from schafkopf_ai.game.game_contract import GameContract
from schafkopf_ai.game.game_type import GameType
from schafkopf_ai.game.observation import PlayerObservation
from schafkopf_ai.training.behavior_cloning import BehaviorCloningArrays
from schafkopf_ai.training.card_play_encoding import (
    ACTION_COUNT,
    OBSERVATION_FEATURE_SIZE,
    card_to_action_index,
)
from schafkopf_ai.training.dagger import DAggerRecordingAgent, aggregate_dagger_arrays
from schafkopf_ai.training.demonstrations import BehaviorCloningCollector


class FixedCardAgent(Agent):
    def __init__(self, card: Card) -> None:
        self.card = card

    def choose_card(
        self,
        observation: PlayerObservation,
        legal_cards: tuple[Card, ...],
    ) -> Card:
        del observation
        if self.card not in legal_cards:
            raise ValueError("Configured fixed card is not legal.")
        return self.card

    def choose_bidding_action(
        self,
        observation: BiddingObservation,
        legal_actions: tuple[BiddingAction, ...],
    ) -> BiddingAction:
        del observation
        if not legal_actions:
            raise ValueError("No legal bidding actions available.")
        return legal_actions[0]


def _observation(hand: tuple[Card, ...]) -> PlayerObservation:
    return PlayerObservation(
        player_index=0,
        hand=hand,
        contract=GameContract(GameType.WENZ, declarer=0),
        current_player=0,
        current_trick=(),
        completed_tricks=(),
        points_by_player=(0, 0, 0, 0),
        called_ace_released=False,
    )


def _arrays(game_ids: list[int], targets: list[int]) -> BehaviorCloningArrays:
    example_count = len(game_ids)
    features = np.zeros(
        (example_count, OBSERVATION_FEATURE_SIZE),
        dtype=np.float16,
    )
    legal_masks = np.zeros((example_count, ACTION_COUNT), dtype=np.bool_)
    for index, target in enumerate(targets):
        legal_masks[index, target] = True

    return BehaviorCloningArrays(
        features=features,
        legal_masks=legal_masks,
        targets=np.asarray(targets, dtype=np.uint8),
        game_ids=np.asarray(game_ids, dtype=np.uint32),
        legal_counts=np.ones(example_count, dtype=np.uint8),
    )


def test_dagger_executes_policy_action_but_records_expert_target() -> None:
    policy_card = Card(Suit.EICHEL, Rank.SEVEN)
    expert_card = Card(Suit.GRAS, Rank.SEVEN)
    observation = _observation((policy_card, expert_card))
    collector = BehaviorCloningCollector(expected_examples=1)
    agent = DAggerRecordingAgent(
        policy_agent=FixedCardAgent(policy_card),
        expert_agent=FixedCardAgent(expert_card),
        collector=collector,
        game_id=7,
    )

    chosen = agent.choose_card(observation, observation.hand)
    arrays = collector.arrays()

    assert chosen == policy_card
    assert int(arrays.targets[0]) == card_to_action_index(expert_card)
    assert agent.decisions == 1
    assert agent.disagreements == 1
    assert agent.nontrivial_decisions == 1
    assert agent.nontrivial_disagreements == 1


def test_aggregate_dagger_arrays_remaps_game_ids_after_base() -> None:
    base = _arrays([0, 1], [0, 1])
    additions = _arrays([0, 1], [2, 3])

    aggregated = aggregate_dagger_arrays(base, additions)

    assert aggregated.example_count == 4
    assert aggregated.game_ids.tolist() == [0, 1, 2, 3]
    assert aggregated.targets.tolist() == [0, 1, 2, 3]
