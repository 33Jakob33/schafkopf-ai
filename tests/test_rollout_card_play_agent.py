from schafkopf_ai.agents.rollout_card_play_agent import RolloutCardPlayAgent
from schafkopf_ai.game.deck import Deck
from schafkopf_ai.game.game_contract import GameContract
from schafkopf_ai.game.game_state import GameState
from schafkopf_ai.game.game_type import GameType
from schafkopf_ai.search.rollout import MonteCarloActionEvaluator


def _state() -> GameState:
    deck = Deck()
    hands = deck.deal()
    return GameState.from_hands(
        hands=hands,
        contract=GameContract(GameType.WENZ, declarer=0),
        starting_player=0,
    )


def test_action_evaluator_returns_payment_samples_for_each_card() -> None:
    state = _state()
    observation = state.observation_for(0)
    legal_cards = state.legal_moves(0)[:2]

    evaluator = MonteCarloActionEvaluator(
        samples_per_action=2,
        seed=321,
    )
    estimates = evaluator.evaluate(observation, legal_cards)

    assert tuple(estimate.card for estimate in estimates) == legal_cards
    assert all(len(estimate.payments) == 2 for estimate in estimates)
    assert all(
        estimate.mean_payment == sum(estimate.payments) / 2 for estimate in estimates
    )


def test_rollout_agent_selects_a_legal_card() -> None:
    state = _state()
    observation = state.observation_for(0)
    legal_cards = state.legal_moves(0)[:2]

    agent = RolloutCardPlayAgent(samples_per_action=1, seed=999)
    chosen = agent.choose_card(observation, legal_cards)

    assert chosen in legal_cards
    assert len(agent.last_action_values) == len(legal_cards)
