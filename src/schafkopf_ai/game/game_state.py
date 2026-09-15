from collections.abc import Sequence
from .scoring import points_by_player
from .card import Card, Rank
from .game_contract import GameContract
from .game_type import GameType
from .legal_moves import (
    LegalMoveContext,
)
from .legal_moves import (
    legal_moves as determine_legal_moves,
)
from .observation import PlayerObservation
from .player import Player
from .trick import Trick, TrickPlay
from .trump import is_trump


class GameState:
    """
    Mutable state of the card-playing phase of one Schafkopf round.

    Bidding has already finished when a GameState is created.
    """

    PLAYER_COUNT = 4
    CARDS_PER_PLAYER = 8
    TRICKS_PER_GAME = 8

    def __init__(
        self,
        players: Sequence[Player],
        contract: GameContract,
        starting_player: int = 0,
    ) -> None:
        self._validate_players(players)

        if not 0 <= starting_player < self.PLAYER_COUNT:
            raise ValueError(
                f"Starting player must be between 0 and {self.PLAYER_COUNT - 1}."
            )

        players_by_index = {player.index: player for player in players}

        self.players: tuple[Player, ...] = tuple(
            players_by_index[index] for index in range(self.PLAYER_COUNT)
        )

        self._initial_hands: tuple[
            tuple[Card, ...],
            ...,
        ] = tuple(player.cards for player in self.players)

        self.contract = contract

        self.called_ace_player = self._determine_called_ace_player()

        self.current_trick: Trick | None = Trick(starting_player=starting_player)

        self._completed_tricks: list[Trick] = []
        self.called_ace_released = False

    @classmethod
    def from_hands(
        cls,
        hands: Sequence[Sequence[Card]],
        contract: GameContract,
        starting_player: int = 0,
    ) -> "GameState":
        """
        Create a GameState directly from four hands.

        Each player must have exactly eight cards and all 32 cards
        must be unique.
        """
        if len(hands) != cls.PLAYER_COUNT:
            raise ValueError(
                f"A Schafkopf game requires exactly {cls.PLAYER_COUNT} hands."
            )

        players = [
            Player(
                index=index,
                hand=list(hand),
            )
            for index, hand in enumerate(hands)
        ]

        return cls(
            players=players,
            contract=contract,
            starting_player=starting_player,
        )

    @property
    def initial_hands(
        self,
    ) -> tuple[tuple[Card, ...], ...]:
        """Return an immutable snapshot of the original deal."""
        return self._initial_hands

    @staticmethod
    def _validate_players(
        players: Sequence[Player],
    ) -> None:
        if len(players) != GameState.PLAYER_COUNT:
            raise ValueError(
                f"A Schafkopf game requires exactly {GameState.PLAYER_COUNT} players."
            )

        indices = [player.index for player in players]

        if set(indices) != set(range(GameState.PLAYER_COUNT)):
            raise ValueError("Players must have the unique indices 0, 1, 2, and 3.")

        for player in players:
            if len(player) != GameState.CARDS_PER_PLAYER:
                raise ValueError(
                    f"Player {player.index} must start with exactly "
                    f"{GameState.CARDS_PER_PLAYER} cards."
                )

        all_cards = [card for player in players for card in player.hand]

        if len(set(all_cards)) != len(all_cards):
            raise ValueError("The players' hands contain duplicate cards.")

        if len(all_cards) != (GameState.PLAYER_COUNT * GameState.CARDS_PER_PLAYER):
            raise ValueError("A Schafkopf game must contain exactly 32 cards.")

    @property
    def completed_tricks(self) -> tuple[Trick, ...]:
        """Return all completed tricks."""
        return tuple(self._completed_tricks)

    @property
    def current_player(self) -> int | None:
        """Return the player whose turn it currently is."""
        if self.current_trick is None:
            return None

        return self.current_trick.next_player

    @property
    def player_points(
        self,
    ) -> tuple[int, int, int, int]:
        """
        Return the Augen currently collected by each player.

        Only completed tricks count.
        """
        return points_by_player(
            self._completed_tricks,
            self.contract,
        )

    @property
    def is_complete(self) -> bool:
        """Return whether all eight tricks have been played."""
        return self.current_trick is None

    def player(self, index: int) -> Player:
        """Return a player by index."""
        if not 0 <= index < self.PLAYER_COUNT:
            raise ValueError(
                f"Player index must be between 0 and {self.PLAYER_COUNT - 1}."
            )

        return self.players[index]

    def _determine_called_ace_player(
        self,
    ) -> int | None:
        """
        Determine who holds the called Ace at the start of a Sauspiel.

        The information is stored because hands become empty during play.
        """
        if self.contract.game_type is not GameType.SAUSPIEL:
            return None

        called_suit = self.contract.called_suit

        if called_suit is None:
            raise ValueError("A Sauspiel requires a called suit.")

        called_ace = Card(
            suit=called_suit,
            rank=Rank.ACE,
        )

        holders = [
            player.index for player in self.players if player.has_card(called_ace)
        ]

        if len(holders) != 1:
            raise ValueError("The called Ace must belong to exactly one player.")

        holder = holders[0]

        if self.contract.declarer is not None and holder == self.contract.declarer:
            raise ValueError("The Sauspiel declarer cannot hold the called Ace.")

        return holder

    def observation_for(
        self,
        player_index: int,
    ) -> PlayerObservation:
        """
        Create an observation from one player's perspective.

        Only information legitimately visible to that player is included.
        Other players' remaining hands are never exposed.
        """
        player = self.player(player_index)

        current_trick: tuple[TrickPlay, ...]

        if self.current_trick is None:
            current_trick = ()
        else:
            current_trick = tuple(self.current_trick.plays)

        completed_tricks: tuple[
            tuple[TrickPlay, ...],
            ...,
        ] = tuple(tuple(trick.plays) for trick in self._completed_tricks)

        return PlayerObservation(
            player_index=player_index,
            hand=player.cards,
            contract=self.contract,
            current_player=self.current_player,
            current_trick=current_trick,
            completed_tricks=completed_tricks,
            points_by_player=self.player_points,
            called_ace_released=self.called_ace_released,
        )

    def legal_moves(
        self,
        player_index: int,
    ) -> tuple[Card, ...]:
        """Return all legal moves for the current player."""
        if self.is_complete:
            return ()

        if player_index != self.current_player:
            raise ValueError(
                f"It is player {self.current_player}'s turn, "
                f"not player {player_index}'s."
            )

        if self.current_trick is None:
            return ()

        player = self.player(player_index)

        context = LegalMoveContext(
            called_ace_released=self.called_ace_released,
        )

        return determine_legal_moves(
            hand=player.hand,
            trick=self.current_trick,
            contract=self.contract,
            context=context,
        )

    def play_card(
        self,
        player_index: int,
        card: Card,
    ) -> None:
        """
        Play one card and update the game state.

        The method:
        - checks turn order,
        - verifies that the player owns the card,
        - verifies move legality,
        - detects Davonlaufen,
        - removes the card,
        - updates the trick,
        - starts the next trick after a trick is completed.
        """
        if self.is_complete or self.current_trick is None:
            raise ValueError("Cannot play a card: the game is already complete.")

        if player_index != self.current_player:
            raise ValueError(
                f"It is player {self.current_player}'s turn, "
                f"not player {player_index}'s."
            )

        player = self.player(player_index)

        if not player.has_card(card):
            raise ValueError(f"Player {player_index} does not hold {card}.")

        allowed_cards = self.legal_moves(player_index)

        if card not in allowed_cards:
            raise ValueError(f"{card} is not a legal move for player {player_index}.")

        if self._is_davonlaufen(player, card):
            self.called_ace_released = True

        player.remove_card(card)

        self.current_trick.play_card(
            player=player_index,
            card=card,
        )

        if self.current_trick.is_complete:
            self._finish_current_trick()

    def _finish_current_trick(self) -> None:
        """Finish a trick and start the next one."""
        if self.current_trick is None:
            raise RuntimeError("There is no active trick.")

        winner = self.current_trick.winner(self.contract)

        self._completed_tricks.append(self.current_trick)

        if len(self._completed_tricks) == self.TRICKS_PER_GAME:
            self.current_trick = None
            return

        self.current_trick = Trick(starting_player=winner.player)

    def _is_davonlaufen(
        self,
        player: Player,
        card: Card,
    ) -> bool:
        """
        Return whether this move constitutes Davonlaufen.

        Davonlaufen happens when the holder of the called Ace leads
        a lower card of the called suit while holding at least four
        non-trump cards of that suit including the Ace.
        """
        if self.contract.game_type is not GameType.SAUSPIEL:
            return False

        if self.called_ace_released:
            return False

        if self.current_trick is None:
            return False

        if self.current_trick.plays:
            return False

        called_suit = self.contract.called_suit

        if called_suit is None:
            return False

        called_ace = Card(
            suit=called_suit,
            rank=Rank.ACE,
        )

        if not player.has_card(called_ace):
            return False

        if card == called_ace:
            return False

        if card.suit is not called_suit:
            return False

        if is_trump(card, self.contract):
            return False

        called_suit_cards = [
            hand_card
            for hand_card in player.hand
            if (
                hand_card.suit is called_suit
                and not is_trump(
                    hand_card,
                    self.contract,
                )
            )
        ]

        return len(called_suit_cards) >= 4
