from schafkopf_ai.agents.heuristic_knowledge import PublicCardKnowledge
from schafkopf_ai.agents.heuristic_strategy import (
    PlayerRole,
    player_role,
    score_situation,
)
from schafkopf_ai.agents.strategic_heuristic_agent import StrategicHeuristicAgent
from schafkopf_ai.game.card import Card, Rank, Suit
from schafkopf_ai.game.game_contract import GameContract
from schafkopf_ai.game.game_type import GameType
from schafkopf_ai.game.observation import PlayerObservation
from schafkopf_ai.game.trick import TrickPlay


def _observation(
    *,
    player_index: int,
    hand: tuple[Card, ...],
    contract: GameContract,
    current_trick: tuple[TrickPlay, ...] = (),
    completed_tricks: tuple[tuple[TrickPlay, ...], ...] = (),
    points_by_player: tuple[int, int, int, int] = (0, 0, 0, 0),
    called_ace_released: bool = False,
) -> PlayerObservation:
    return PlayerObservation(
        player_index=player_index,
        hand=hand,
        contract=contract,
        current_player=player_index,
        current_trick=current_trick,
        completed_tricks=completed_tricks,
        points_by_player=points_by_player,
        called_ace_released=called_ace_released,
    )


def test_sauspiel_defender_searches_called_suit() -> None:
    agent = StrategicHeuristicAgent()
    contract = GameContract(
        GameType.SAUSPIEL,
        called_suit=Suit.GRAS,
        declarer=0,
    )
    search_card = Card(Suit.GRAS, Rank.SEVEN)
    observation = _observation(
        player_index=1,
        hand=(
            search_card,
            Card(Suit.EICHEL, Rank.ACE),
            Card(Suit.SCHELLEN, Rank.SEVEN),
        ),
        contract=contract,
    )

    chosen = agent.choose_card(observation, observation.hand)

    assert player_role(observation) is PlayerRole.DEFENDER
    assert chosen == search_card


def test_sauspiel_declarer_does_not_search_called_suit_when_alternative_exists() -> None:
    agent = StrategicHeuristicAgent()
    contract = GameContract(
        GameType.SAUSPIEL,
        called_suit=Suit.GRAS,
        declarer=0,
    )
    called_suit_card = Card(Suit.GRAS, Rank.SEVEN)
    alternative = Card(Suit.SCHELLEN, Rank.SEVEN)
    observation = _observation(
        player_index=0,
        hand=(called_suit_card, alternative),
        contract=contract,
    )

    chosen = agent.choose_card(observation, observation.hand)

    assert player_role(observation) is PlayerRole.DECLARER
    assert chosen == alternative


def test_sauspiel_called_ace_holder_knows_they_are_partner() -> None:
    contract = GameContract(
        GameType.SAUSPIEL,
        called_suit=Suit.GRAS,
        declarer=0,
    )
    observation = _observation(
        player_index=2,
        hand=(Card(Suit.GRAS, Rank.ACE),),
        contract=contract,
    )

    assert player_role(observation) is PlayerRole.PARTNER


def test_public_knowledge_identifies_definite_master_trump() -> None:
    contract = GameContract(GameType.WENZ, declarer=0)
    top_trump = Card(Suit.EICHEL, Rank.UNTER)
    lower_trump = Card(Suit.SCHELLEN, Rank.UNTER)
    observation = _observation(
        player_index=0,
        hand=(top_trump, lower_trump),
        contract=contract,
    )

    knowledge = PublicCardKnowledge.from_observation(observation)

    assert knowledge.is_definite_master_trump(top_trump)
    assert not knowledge.is_definite_master_trump(lower_trump)


def test_known_void_opponent_makes_plain_ace_less_safe() -> None:
    contract = GameContract(GameType.WENZ, declarer=0)
    ace = Card(Suit.GRAS, Rank.ACE)
    observation = _observation(
        player_index=0,
        hand=(ace, Card(Suit.EICHEL, Rank.UNTER)),
        contract=contract,
        completed_tricks=(
            (
                TrickPlay(0, Card(Suit.GRAS, Rank.SEVEN)),
                TrickPlay(1, Card(Suit.EICHEL, Rank.SEVEN)),
                TrickPlay(2, Card(Suit.GRAS, Rank.TEN)),
                TrickPlay(3, Card(Suit.GRAS, Rank.KING)),
            ),
        ),
    )

    knowledge = PublicCardKnowledge.from_observation(observation)
    safety = knowledge.ace_safety_probability(ace, {1, 2, 3})

    assert safety < 1.0


def test_score_situation_detects_game_clinching_trick() -> None:
    contract = GameContract(GameType.WENZ, declarer=0)
    observation = _observation(
        player_index=0,
        hand=(Card(Suit.EICHEL, Rank.UNTER),),
        contract=contract,
        points_by_player=(55, 20, 20, 15),
    )

    situation = score_situation(observation)

    assert situation is not None
    assert situation.trick_clinches_game(6)


def test_score_situation_detects_schneider_escape() -> None:
    contract = GameContract(GameType.WENZ, declarer=0)
    observation = _observation(
        player_index=1,
        hand=(Card(Suit.EICHEL, Rank.ACE),),
        contract=contract,
        points_by_player=(80, 10, 10, 8),
    )

    situation = score_situation(observation)

    assert situation is not None
    assert situation.trick_avoids_schneider(2)
