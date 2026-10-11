import random

from schafkopf_ai.game.card import Card, Rank, Suit
from schafkopf_ai.game.deck import Deck
from schafkopf_ai.game.game_contract import GameContract
from schafkopf_ai.game.game_state import GameState
from schafkopf_ai.game.game_type import GameType
from schafkopf_ai.search.hidden_state import HiddenStateSampler


def _wenz_state() -> GameState:
    deck = Deck()
    hands = deck.deal()
    return GameState.from_hands(
        hands=hands,
        contract=GameContract(GameType.WENZ, declarer=0),
        starting_player=0,
    )


def test_hidden_state_sample_reconstructs_same_observation() -> None:
    state = _wenz_state()

    for _ in range(7):
        player = state.current_player
        assert player is not None
        state.play_card(player, state.legal_moves(player)[0])

    player = state.current_player
    assert player is not None
    observation = state.observation_for(player)

    sampler = HiddenStateSampler(rng=random.Random(123))
    determinization = sampler.sample(observation)
    rebuilt = determinization.build_state(observation)

    assert rebuilt.observation_for(player) == observation
    assert rebuilt.player(player).cards == observation.hand
    all_cards = [card for hand in determinization.initial_hands for card in hand]
    assert len(all_cards) == 32
    assert len(set(all_cards)) == 32


def test_hidden_state_sampler_does_not_assign_sauspiel_ace_to_declarer() -> None:
    called_suit = Suit.EICHEL
    called_ace = Card(called_suit, Rank.ACE)
    deck = Deck()
    remaining = [card for card in deck.cards if card != called_ace]

    hands = (
        tuple(remaining[:8]),
        (called_ace,) + tuple(remaining[8:15]),
        tuple(remaining[15:23]),
        tuple(remaining[23:31]),
    )
    state = GameState.from_hands(
        hands=hands,
        contract=GameContract(
            GameType.SAUSPIEL,
            called_suit=called_suit,
            declarer=0,
        ),
        starting_player=0,
    )
    observation = state.observation_for(2)

    sampler = HiddenStateSampler(rng=random.Random(456))
    determinizations = sampler.sample_many(observation, 8)

    for determinization in determinizations:
        rebuilt = determinization.build_state(observation)
        assert rebuilt.called_ace_player != 0


def test_hidden_state_sampler_handles_late_sauspiel_information() -> None:
    called_suit = Suit.EICHEL
    called_ace = Card(called_suit, Rank.ACE)
    deck = Deck()
    remaining = [card for card in deck.cards if card != called_ace]

    hands = (
        tuple(remaining[:8]),
        (called_ace,) + tuple(remaining[8:15]),
        tuple(remaining[15:23]),
        tuple(remaining[23:31]),
    )
    state = GameState.from_hands(
        hands=hands,
        contract=GameContract(
            GameType.SAUSPIEL,
            called_suit=called_suit,
            declarer=0,
        ),
        starting_player=0,
    )

    for _ in range(20):
        player = state.current_player
        assert player is not None
        state.play_card(player, state.legal_moves(player)[0])

    player = state.current_player
    assert player is not None
    observation = state.observation_for(player)

    sampler = HiddenStateSampler(
        rng=random.Random(20261011),
        max_attempts_per_sample=50,
    )
    determinizations = sampler.sample_many(observation, 8)

    assert len(determinizations) == 8
    for determinization in determinizations:
        rebuilt = determinization.build_state(observation)
        assert rebuilt.observation_for(player) == observation
