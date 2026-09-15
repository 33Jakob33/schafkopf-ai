from schafkopf_ai.agents.heuristic_agent import HeuristicAgent, infer_voids
from schafkopf_ai.game.bidding import BiddingState
from schafkopf_ai.game.card import Card, Rank, Suit
from schafkopf_ai.game.deck import Deck
from schafkopf_ai.game.game_contract import GameContract
from schafkopf_ai.game.game_state import GameState
from schafkopf_ai.game.game_type import GameType
from schafkopf_ai.game.observation import PlayerObservation
from schafkopf_ai.game.rules import RAMSCH_RULES
from schafkopf_ai.game.trick import TrickPlay


def test_heuristic_agent_returns_legal_bidding_action() -> None:
    hands = Deck().deal()
    state = BiddingState(
        hands=hands,
        rules=RAMSCH_RULES,
        starting_player=0,
    )
    agent = HeuristicAgent()

    player = state.current_player
    assert player is not None

    observation = state.observation_for(player)
    legal_actions = state.legal_actions(player)
    action = agent.choose_bidding_action(observation, legal_actions)

    assert action in legal_actions


def test_heuristic_agent_returns_legal_card() -> None:
    hands = Deck().deal()
    state = GameState.from_hands(
        hands=hands,
        contract=GameContract(
            game_type=GameType.WENZ,
            declarer=0,
        ),
        starting_player=0,
    )
    agent = HeuristicAgent()

    player = state.current_player
    assert player is not None

    observation = state.observation_for(player)
    legal_cards = state.legal_moves(player)
    card = agent.choose_card(observation, legal_cards)

    assert card in legal_cards


def test_infer_voids_detects_plain_suit_void() -> None:
    observation = PlayerObservation(
        player_index=0,
        hand=(),
        contract=GameContract(
            game_type=GameType.WENZ,
            declarer=0,
        ),
        current_player=0,
        current_trick=(),
        completed_tricks=(
            (
                TrickPlay(0, Card(Suit.EICHEL, Rank.ACE)),
                TrickPlay(1, Card(Suit.GRAS, Rank.SEVEN)),
                TrickPlay(2, Card(Suit.EICHEL, Rank.TEN)),
                TrickPlay(3, Card(Suit.EICHEL, Rank.KING)),
            ),
        ),
        points_by_player=(0, 0, 0, 0),
        called_ace_released=False,
    )

    voids = infer_voids(observation)

    assert Suit.EICHEL in voids.plain_suits[1]
    assert Suit.EICHEL not in voids.plain_suits[2]
    assert Suit.EICHEL not in voids.plain_suits[3]


def test_infer_voids_detects_trump_void() -> None:
    observation = PlayerObservation(
        player_index=0,
        hand=(),
        contract=GameContract(
            game_type=GameType.WENZ,
            declarer=0,
        ),
        current_player=0,
        current_trick=(),
        completed_tricks=(
            (
                TrickPlay(0, Card(Suit.EICHEL, Rank.UNTER)),
                TrickPlay(1, Card(Suit.GRAS, Rank.SEVEN)),
                TrickPlay(2, Card(Suit.GRAS, Rank.UNTER)),
                TrickPlay(3, Card(Suit.HERZ, Rank.SEVEN)),
            ),
        ),
        points_by_player=(0, 0, 0, 0),
        called_ace_released=False,
    )

    voids = infer_voids(observation)

    assert 1 in voids.trump_void_players
    assert 3 in voids.trump_void_players
    assert 2 not in voids.trump_void_players
