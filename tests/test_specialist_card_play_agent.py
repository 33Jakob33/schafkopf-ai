from typing import cast

from schafkopf_ai.agents.agent import Agent
from schafkopf_ai.agents.neural_card_play_agent import NeuralCardPlayAgent
from schafkopf_ai.agents.specialist_card_play_agent import SpecialistCardPlayAgent
from schafkopf_ai.game.bidding import BiddingAction, BiddingObservation
from schafkopf_ai.game.card import Card, Rank, Suit
from schafkopf_ai.game.game_contract import GameContract
from schafkopf_ai.game.game_type import GameType
from schafkopf_ai.game.observation import PlayerObservation
from schafkopf_ai.training.card_play_roles import CardPlayRole


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
            raise ValueError("Fixed card is not legal.")
        return self.card

    def choose_bidding_action(
        self,
        observation: BiddingObservation,
        legal_actions: tuple[BiddingAction, ...],
    ) -> BiddingAction:
        del observation
        return legal_actions[0]


def test_specialist_agent_routes_by_contract_and_role() -> None:
    first = Card(Suit.EICHEL, Rank.SEVEN)
    second = Card(Suit.GRAS, Rank.SEVEN)

    default = cast(NeuralCardPlayAgent, FixedCardAgent(first))
    solo_declarer = cast(NeuralCardPlayAgent, FixedCardAgent(second))
    specialists = {role: default for role in CardPlayRole}
    specialists[CardPlayRole.SOLO_DECLARER] = solo_declarer

    agent = SpecialistCardPlayAgent(specialists=specialists)
    observation = PlayerObservation(
        player_index=2,
        hand=(first, second),
        contract=GameContract(
            GameType.SOLO,
            trump_suit=Suit.HERZ,
            declarer=2,
        ),
        current_player=2,
        current_trick=(),
        completed_tricks=(),
        points_by_player=(0, 0, 0, 0),
        called_ace_released=False,
    )

    assert agent.choose_card(observation, (first, second)) == second
