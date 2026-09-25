import random

from schafkopf_ai.agents.random_agent import RandomAgent
from schafkopf_ai.game.bidding import BiddingState
from schafkopf_ai.game.deck import Deck
from schafkopf_ai.game.rules import RAMSCH_RULES
from schafkopf_ai.runner.bidding_runner import BiddingRunner


def test_random_agents_complete_bidding() -> None:
    deck = Deck()
    deck.shuffle(random.Random(42))

    state = BiddingState(
        hands=deck.deal(),
        rules=RAMSCH_RULES,
        starting_player=0,
    )

    agents = [RandomAgent(rng=random.Random(100 + player)) for player in range(4)]

    runner = BiddingRunner(
        state=state,
        agents=agents,
    )

    result = runner.run()

    assert state.is_complete

    assert result.contract is not None or result.requires_redeal
