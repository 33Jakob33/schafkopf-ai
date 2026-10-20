from schafkopf_ai.game.card import Card, Rank, Suit
from schafkopf_ai.game.deck import Deck
from schafkopf_ai.game.game_contract import GameContract
from schafkopf_ai.game.game_state import GameState
from schafkopf_ai.game.game_type import GameType
from schafkopf_ai.game.observation import PlayerObservation


def create_state(
    starting_player: int = 0,
) -> GameState:
    deck = Deck()
    hands = deck.deal()

    return GameState.from_hands(
        hands=hands,
        contract=GameContract(GameType.WENZ),
        starting_player=starting_player,
    )


def test_observation_contains_player_index() -> None:
    state = create_state()

    observation = state.observation_for(2)

    assert observation.player_index == 2


def test_observation_contains_only_own_hand() -> None:
    state = create_state()

    observation = state.observation_for(0)

    assert observation.hand == state.player(0).cards


def test_observation_does_not_expose_players() -> None:
    state = create_state()

    observation = state.observation_for(0)

    assert not hasattr(observation, "players")


def test_initial_observation_has_zero_points() -> None:
    state = create_state()

    observation = state.observation_for(0)

    assert observation.points_by_player == (
        0,
        0,
        0,
        0,
    )


def test_observation_does_not_expose_other_hands() -> None:
    state = create_state()

    observation = state.observation_for(0)

    other_cards = {
        card for player_index in (1, 2, 3) for card in state.player(player_index).cards
    }

    assert not any(card in other_cards for card in observation.hand)


def test_initial_observation_has_empty_current_trick() -> None:
    state = create_state()

    observation = state.observation_for(0)

    assert observation.current_trick == ()


def test_initial_observation_has_no_completed_tricks() -> None:
    state = create_state()

    observation = state.observation_for(0)

    assert observation.completed_tricks == ()


def test_observation_contains_current_player() -> None:
    state = create_state(starting_player=3)

    observation = state.observation_for(1)

    assert observation.current_player == 3


def test_played_card_appears_in_current_trick() -> None:
    state = create_state()

    player_index = state.current_player

    assert player_index is not None

    card = state.legal_moves(player_index)[0]

    state.play_card(
        player_index,
        card,
    )

    observation = state.observation_for(1)

    assert len(observation.current_trick) == 1

    assert observation.current_trick[0].card == card
    assert observation.current_trick[0].player == player_index


def test_observation_contains_current_points() -> None:
    state = create_state()

    for _ in range(4):
        player = state.current_player

        assert player is not None

        state.play_card(
            player,
            state.legal_moves(player)[0],
        )

    observation = state.observation_for(0)

    assert observation.points_by_player == state.player_points
    assert sum(observation.points_by_player) > 0


def test_played_card_is_removed_from_own_hand() -> None:
    state = create_state()

    player_index = state.current_player

    assert player_index is not None

    card = state.legal_moves(player_index)[0]

    state.play_card(
        player_index,
        card,
    )

    observation = state.observation_for(player_index)

    assert card not in observation.hand
    assert len(observation.hand) == 7


def test_completed_observation_contains_120_points() -> None:
    state = create_state()

    while not state.is_complete:
        player = state.current_player

        assert player is not None

        state.play_card(
            player,
            state.legal_moves(player)[0],
        )

    observation = state.observation_for(0)

    assert sum(observation.points_by_player) == 120


def test_completed_trick_appears_in_observation() -> None:
    state = create_state()

    for _ in range(4):
        player_index = state.current_player

        assert player_index is not None

        card = state.legal_moves(player_index)[0]

        state.play_card(
            player_index,
            card,
        )

    observation = state.observation_for(0)

    assert len(observation.completed_tricks) == 1
    assert len(observation.completed_tricks[0]) == 4


def test_current_trick_is_empty_after_trick_finishes() -> None:
    state = create_state()

    for _ in range(4):
        player_index = state.current_player

        assert player_index is not None

        state.play_card(
            player_index,
            state.legal_moves(player_index)[0],
        )

    observation = state.observation_for(0)

    assert observation.current_trick == ()


def test_cards_played_contains_publicly_played_cards() -> None:
    state = create_state()

    played_cards: list[Card] = []

    for _ in range(6):
        player_index = state.current_player

        assert player_index is not None

        card = state.legal_moves(player_index)[0]

        played_cards.append(card)

        state.play_card(
            player_index,
            card,
        )

    observation = state.observation_for(0)

    assert observation.cards_played == tuple(played_cards)


def test_initial_trick_number_is_one() -> None:
    state = create_state()

    observation = state.observation_for(0)

    assert observation.trick_number == 1


def test_trick_number_advances() -> None:
    state = create_state()

    for _ in range(4):
        player_index = state.current_player

        assert player_index is not None

        state.play_card(
            player_index,
            state.legal_moves(player_index)[0],
        )

    observation = state.observation_for(0)

    assert observation.trick_number == 2


def test_observation_contains_contract() -> None:
    state = create_state()

    observation = state.observation_for(0)

    assert observation.contract == state.contract


def test_observation_type() -> None:
    state = create_state()

    observation = state.observation_for(0)

    assert isinstance(
        observation,
        PlayerObservation,
    )


def _sauspiel_state(*, running_away: bool = False) -> GameState:
    contract = GameContract(
        GameType.SAUSPIEL,
        called_suit=Suit.EICHEL,
        declarer=0,
    )
    called_ace = Card(Suit.EICHEL, Rank.ACE)

    if running_away:
        called_hand = (
            called_ace,
            Card(Suit.EICHEL, Rank.TEN),
            Card(Suit.EICHEL, Rank.KING),
            Card(Suit.EICHEL, Rank.NINE),
        )
        reserved = set(called_hand)
        remaining = [card for card in Deck().cards if card not in reserved]
        hands = (
            tuple(remaining[:8]),
            called_hand + tuple(remaining[8:12]),
            tuple(remaining[12:20]),
            tuple(remaining[20:28]),
        )
    else:
        remaining = [card for card in Deck().cards if card != called_ace]
        hands = (
            tuple(remaining[:8]),
            (called_ace,) + tuple(remaining[8:15]),
            tuple(remaining[15:23]),
            tuple(remaining[23:31]),
        )

    return GameState.from_hands(
        hands=hands,
        contract=contract,
        starting_player=1,
    )


def test_sauspiel_partner_is_private_until_publicly_known() -> None:
    state = _sauspiel_state()

    holder_observation = state.observation_for(1)
    other_observation = state.observation_for(2)

    assert holder_observation.known_called_ace_player == 1
    assert other_observation.known_called_ace_player is None


def test_playing_called_ace_reveals_sauspiel_partner() -> None:
    state = _sauspiel_state()
    called_ace = Card(Suit.EICHEL, Rank.ACE)

    assert called_ace in state.legal_moves(1)
    state.play_card(1, called_ace)

    for player in range(4):
        assert state.observation_for(player).known_called_ace_player == 1


def test_davonlaufen_reveals_sauspiel_partner() -> None:
    state = _sauspiel_state(running_away=True)
    lower_called_suit = Card(Suit.EICHEL, Rank.TEN)

    assert lower_called_suit in state.legal_moves(1)
    state.play_card(1, lower_called_suit)

    assert state.called_ace_released is True
    for player in range(4):
        assert state.observation_for(player).known_called_ace_player == 1
