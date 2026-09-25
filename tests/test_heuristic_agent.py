from schafkopf_ai.agents.heuristic_agent import HeuristicAgent
from schafkopf_ai.agents.heuristic_knowledge import (
    PublicCardKnowledge,
    infer_voids,
)
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


def test_public_knowledge_counts_remaining_wenz_trumps() -> None:
    contract = GameContract(GameType.WENZ, declarer=0)
    observation = PlayerObservation(
        player_index=0,
        hand=(
            Card(Suit.EICHEL, Rank.UNTER),
            Card(Suit.GRAS, Rank.UNTER),
        ),
        contract=contract,
        current_player=0,
        current_trick=(),
        completed_tricks=(
            (
                TrickPlay(0, Card(Suit.EICHEL, Rank.ACE)),
                TrickPlay(1, Card(Suit.HERZ, Rank.UNTER)),
                TrickPlay(2, Card(Suit.EICHEL, Rank.TEN)),
                TrickPlay(3, Card(Suit.EICHEL, Rank.KING)),
            ),
        ),
        points_by_player=(0, 0, 0, 0),
        called_ace_released=False,
    )

    knowledge = PublicCardKnowledge.from_observation(observation)

    assert knowledge.remaining_trump_count == 1
    assert knowledge.unseen_trumps == (Card(Suit.SCHELLEN, Rank.UNTER),)


def test_public_knowledge_probability_excludes_known_void_player() -> None:
    called_ace = Card(Suit.GRAS, Rank.ACE)
    observation = PlayerObservation(
        player_index=0,
        hand=(Card(Suit.SCHELLEN, Rank.SEVEN),),
        contract=GameContract(
            game_type=GameType.SAUSPIEL,
            called_suit=Suit.GRAS,
            declarer=0,
        ),
        current_player=0,
        current_trick=(),
        completed_tricks=(
            (
                TrickPlay(0, Card(Suit.GRAS, Rank.SEVEN)),
                TrickPlay(1, Card(Suit.EICHEL, Rank.SEVEN)),
                TrickPlay(2, Card(Suit.GRAS, Rank.TEN)),
                TrickPlay(3, Card(Suit.GRAS, Rank.KING)),
            ),
        ),
        points_by_player=(0, 0, 0, 0),
        called_ace_released=False,
    )

    probabilities = PublicCardKnowledge.from_observation(
        observation
    ).holder_probabilities(called_ace)

    assert probabilities[0] == 0.0
    assert probabilities[1] == 0.0
    assert probabilities[2] > 0.0
    assert probabilities[3] > 0.0
    assert abs(sum(probabilities) - 1.0) < 1e-9


def test_sauspiel_declarer_searches_for_called_ace() -> None:
    agent = HeuristicAgent()
    contract = GameContract(
        game_type=GameType.SAUSPIEL,
        called_suit=Suit.GRAS,
        declarer=0,
    )
    search_card = Card(Suit.GRAS, Rank.SEVEN)
    observation = PlayerObservation(
        player_index=0,
        hand=(
            search_card,
            Card(Suit.EICHEL, Rank.OBER),
            Card(Suit.HERZ, Rank.ACE),
            Card(Suit.SCHELLEN, Rank.ACE),
        ),
        contract=contract,
        current_player=0,
        current_trick=(),
        completed_tricks=(),
        points_by_player=(0, 0, 0, 0),
        called_ace_released=False,
    )

    legal_cards = observation.hand

    assert agent.choose_card(observation, legal_cards) == search_card


def test_wenz_declarer_draws_trumps_with_top_control() -> None:
    agent = HeuristicAgent()
    contract = GameContract(GameType.WENZ, declarer=0)
    top_trump = Card(Suit.EICHEL, Rank.UNTER)
    observation = PlayerObservation(
        player_index=0,
        hand=(
            top_trump,
            Card(Suit.GRAS, Rank.UNTER),
            Card(Suit.EICHEL, Rank.ACE),
            Card(Suit.GRAS, Rank.ACE),
        ),
        contract=contract,
        current_player=0,
        current_trick=(),
        completed_tricks=(),
        points_by_player=(0, 0, 0, 0),
        called_ace_released=False,
    )

    chosen = agent.choose_card(observation, observation.hand)

    assert chosen in {
        Card(Suit.EICHEL, Rank.UNTER),
        Card(Suit.GRAS, Rank.UNTER),
    }


def test_follow_uses_cheapest_sufficient_trump_when_last_to_play() -> None:
    agent = HeuristicAgent()
    contract = GameContract(GameType.WENZ, declarer=0)
    low_trump = Card(Suit.SCHELLEN, Rank.UNTER)
    high_trump = Card(Suit.EICHEL, Rank.UNTER)
    observation = PlayerObservation(
        player_index=0,
        hand=(
            low_trump,
            high_trump,
            Card(Suit.SCHELLEN, Rank.SEVEN),
        ),
        contract=contract,
        current_player=0,
        current_trick=(
            TrickPlay(1, Card(Suit.GRAS, Rank.ACE)),
            TrickPlay(2, Card(Suit.GRAS, Rank.TEN)),
            TrickPlay(3, Card(Suit.GRAS, Rank.KING)),
        ),
        completed_tricks=(),
        points_by_player=(0, 0, 0, 0),
        called_ace_released=False,
    )

    chosen = agent.choose_card(
        observation,
        (low_trump, high_trump, Card(Suit.SCHELLEN, Rank.SEVEN)),
    )

    assert chosen == low_trump


def test_follow_secures_valuable_trick_when_overtrump_risk_remains() -> None:
    agent = HeuristicAgent()
    contract = GameContract(GameType.WENZ, declarer=0)
    low_trump = Card(Suit.SCHELLEN, Rank.UNTER)
    high_trump = Card(Suit.EICHEL, Rank.UNTER)
    observation = PlayerObservation(
        player_index=0,
        hand=(
            low_trump,
            high_trump,
            Card(Suit.SCHELLEN, Rank.SEVEN),
        ),
        contract=contract,
        current_player=0,
        current_trick=(
            TrickPlay(2, Card(Suit.GRAS, Rank.ACE)),
            TrickPlay(3, Card(Suit.GRAS, Rank.TEN)),
        ),
        completed_tricks=(),
        points_by_player=(0, 0, 0, 0),
        called_ace_released=False,
    )

    chosen = agent.choose_card(
        observation,
        (low_trump, high_trump, Card(Suit.SCHELLEN, Rank.SEVEN)),
    )

    assert chosen == high_trump


def test_ramsch_dumps_points_when_other_player_is_winning() -> None:
    agent = HeuristicAgent()
    contract = GameContract(GameType.RAMSCH)
    ten = Card(Suit.SCHELLEN, Rank.TEN)
    seven = Card(Suit.SCHELLEN, Rank.SEVEN)
    observation = PlayerObservation(
        player_index=0,
        hand=(ten, seven),
        contract=contract,
        current_player=0,
        current_trick=(
            TrickPlay(2, Card(Suit.EICHEL, Rank.ACE)),
            TrickPlay(3, Card(Suit.EICHEL, Rank.KING)),
        ),
        completed_tricks=(),
        points_by_player=(0, 0, 25, 0),
        called_ace_released=False,
    )

    chosen = agent.choose_card(observation, (ten, seven))

    assert chosen == ten
