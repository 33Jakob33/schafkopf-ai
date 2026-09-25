from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, auto

from .card import Card, Rank, Suit
from .game_contract import GameContract
from .game_type import GameType
from .rules import AllPassAction, GameRules

PLAYER_COUNT = 4
CARDS_PER_PLAYER = 8


class BiddingPhase(Enum):
    """Current phase of the bidding process."""

    INTEREST = auto()
    NEGOTIATION = auto()
    ANNOUNCEMENT = auto()
    COMPLETE = auto()


class BiddingActionType(Enum):
    """Possible public actions during bidding."""

    PASS = auto()
    PLAY = auto()
    HOLD = auto()
    RAISE = auto()
    ANNOUNCE = auto()


@dataclass(frozen=True, slots=True)
class BidValues:
    """
    Configurable relative values used during bidding.

    Wenz and Geier deliberately remain separate game types even
    though they currently have the same bidding value.
    """

    sauspiel: int = 1
    wenz: int = 2
    geier: int = 2
    solo: int = 3

    def value(
        self,
        game_type: GameType,
    ) -> int:
        if game_type is GameType.SAUSPIEL:
            return self.sauspiel

        if game_type is GameType.WENZ:
            return self.wenz

        if game_type is GameType.GEIER:
            return self.geier

        if game_type is GameType.SOLO:
            return self.solo

        raise ValueError(f"{game_type.value} cannot be bid.")

    def enabled_values(
        self,
        rules: GameRules,
    ) -> tuple[int, ...]:
        """
        Return the distinct bidding values currently available.

        Example:
            Sauspiel = 1
            Wenz = 2
            Geier = 2
            Solo = 3

        becomes:
            (1, 2, 3)
        """
        values = {
            self.value(game_type)
            for game_type in rules.enabled_game_types
            if game_type.can_be_bid
        }

        return tuple(sorted(values))


DEFAULT_BID_VALUES = BidValues()


@dataclass(frozen=True, slots=True)
class BiddingAction:
    """One action taken during bidding."""

    action_type: BiddingActionType
    bid_value: int | None = None
    contract: GameContract | None = None

    def __post_init__(self) -> None:
        if self.action_type is BiddingActionType.RAISE:
            if self.bid_value is None:
                raise ValueError("A RAISE action requires a bid value.")

            if self.bid_value <= 0:
                raise ValueError("Bid value must be positive.")

            if self.contract is not None:
                raise ValueError("A RAISE action cannot contain a contract.")

            return

        if self.action_type is BiddingActionType.ANNOUNCE:
            if self.contract is None:
                raise ValueError("An ANNOUNCE action requires a contract.")

            if self.bid_value is not None:
                raise ValueError("An ANNOUNCE action cannot specify a bid value.")

            return

        if self.bid_value is not None:
            raise ValueError(f"{self.action_type.name} cannot specify a bid value.")

        if self.contract is not None:
            raise ValueError(f"{self.action_type.name} cannot contain a contract.")

    @classmethod
    def pass_action(cls) -> "BiddingAction":
        return cls(BiddingActionType.PASS)

    @classmethod
    def play(cls) -> "BiddingAction":
        return cls(BiddingActionType.PLAY)

    @classmethod
    def hold(cls) -> "BiddingAction":
        return cls(BiddingActionType.HOLD)

    @classmethod
    def raise_to(
        cls,
        value: int,
    ) -> "BiddingAction":
        return cls(
            BiddingActionType.RAISE,
            bid_value=value,
        )

    @classmethod
    def announce(
        cls,
        contract: GameContract,
    ) -> "BiddingAction":
        return cls(
            BiddingActionType.ANNOUNCE,
            contract=contract,
        )


@dataclass(frozen=True, slots=True)
class BiddingEvent:
    """Public record of one bidding action."""

    player_index: int
    action: BiddingAction


@dataclass(frozen=True, slots=True)
class BiddingObservation:
    """
    Public bidding information from one player's perspective.

    Only the observing player's hand is included.
    """

    player_index: int
    hand: tuple[Card, ...]

    phase: BiddingPhase
    starting_player: int
    current_player: int | None

    interested_players: tuple[int, ...]
    incumbent: int | None
    challenger: int | None

    current_bid_value: int | None

    history: tuple[BiddingEvent, ...]

    rules: GameRules
    bid_values: BidValues


@dataclass(frozen=True, slots=True)
class BiddingResult:
    """
    Final result of bidding.

    contract:
        Selected game contract.

        None means the cards must be redealt.

    requires_redeal:
        True when everybody passed and the table rules specify
        redealing instead of Ramsch.
    """

    contract: GameContract | None
    requires_redeal: bool
    interested_players: tuple[int, ...]
    history: tuple[BiddingEvent, ...]


def callable_sauspiel_suits(
    hand: Sequence[Card],
) -> tuple[Suit, ...]:
    """
    Return all suits that this player may legally call.

    The player:
    - cannot call Herz,
    - cannot hold the called Ace,
    - must hold at least one ordinary card of the called suit.

    Ober and Unter do not count because they are trump.
    """
    callable_suits: list[Suit] = []

    for suit in (
        Suit.EICHEL,
        Suit.GRAS,
        Suit.SCHELLEN,
    ):
        ace = Card(
            suit=suit,
            rank=Rank.ACE,
        )

        if ace in hand:
            continue

        has_plain_card = any(
            card.suit is suit
            and card.rank
            not in {
                Rank.OBER,
                Rank.UNTER,
            }
            for card in hand
        )

        if has_plain_card:
            callable_suits.append(suit)

    return tuple(callable_suits)


def legal_contracts(
    *,
    hand: Sequence[Card],
    rules: GameRules,
    declarer: int,
) -> tuple[GameContract, ...]:
    """Return every exact contract a player may announce."""
    contracts: list[GameContract] = []

    if rules.is_enabled(GameType.SAUSPIEL):
        for suit in callable_sauspiel_suits(hand):
            contracts.append(
                GameContract(
                    game_type=GameType.SAUSPIEL,
                    called_suit=suit,
                    declarer=declarer,
                )
            )

    if rules.is_enabled(GameType.WENZ):
        contracts.append(
            GameContract(
                game_type=GameType.WENZ,
                declarer=declarer,
            )
        )

    if rules.is_enabled(GameType.GEIER):
        contracts.append(
            GameContract(
                game_type=GameType.GEIER,
                declarer=declarer,
            )
        )

    if rules.is_enabled(GameType.SOLO):
        for suit in Suit:
            contracts.append(
                GameContract(
                    game_type=GameType.SOLO,
                    trump_suit=suit,
                    declarer=declarer,
                )
            )

    return tuple(contracts)


class BiddingState:
    """
    Stateful, turn-by-turn Schafkopf bidding process.

    Stage 1:
        All four players say PLAY or PASS in seating order.

    Stage 2:
        Interested players negotiate. The earlier-seated incumbent
        has priority at equal bidding value. A challenger must raise
        to a strictly higher value.

    Stage 3:
        The remaining player announces the exact contract.
    """

    def __init__(
        self,
        *,
        hands: Sequence[Sequence[Card]],
        rules: GameRules,
        starting_player: int = 0,
        bid_values: BidValues = DEFAULT_BID_VALUES,
    ) -> None:
        self._validate_hands(hands)

        if not 0 <= starting_player < PLAYER_COUNT:
            raise ValueError("Starting player must be between 0 and 3.")

        self.hands: tuple[
            tuple[Card, ...],
            ...,
        ] = tuple(tuple(hand) for hand in hands)

        self.rules = rules
        self.starting_player = starting_player
        self.bid_values = bid_values

        self.phase = BiddingPhase.INTEREST

        self._interest_offset = 0
        self._interested_players: list[int] = []

        self._history: list[BiddingEvent] = []

        self.incumbent: int | None = None
        self.challenger: int | None = None

        self.current_bid_value: int | None = None

        self._challenger_position = 0
        self._awaiting_incumbent_response = False

        self._contract: GameContract | None = None
        self._requires_redeal = False

    @property
    def interested_players(self) -> tuple[int, ...]:
        return tuple(self._interested_players)

    @property
    def history(self) -> tuple[BiddingEvent, ...]:
        return tuple(self._history)

    @property
    def is_complete(self) -> bool:
        return self.phase is BiddingPhase.COMPLETE

    @property
    def current_player(self) -> int | None:
        if self.phase is BiddingPhase.COMPLETE:
            return None

        if self.phase is BiddingPhase.INTEREST:
            return (self.starting_player + self._interest_offset) % PLAYER_COUNT

        if self.phase is BiddingPhase.ANNOUNCEMENT:
            return self.incumbent

        if self.phase is BiddingPhase.NEGOTIATION:
            if self._awaiting_incumbent_response:
                return self.incumbent

            return self.challenger

        raise RuntimeError("Unsupported bidding phase.")

    def observation_for(
        self,
        player_index: int,
    ) -> BiddingObservation:
        if not 0 <= player_index < PLAYER_COUNT:
            raise ValueError("Player index must be between 0 and 3.")

        return BiddingObservation(
            player_index=player_index,
            hand=self.hands[player_index],
            phase=self.phase,
            starting_player=self.starting_player,
            current_player=self.current_player,
            interested_players=self.interested_players,
            incumbent=self.incumbent,
            challenger=self.challenger,
            current_bid_value=self.current_bid_value,
            history=self.history,
            rules=self.rules,
            bid_values=self.bid_values,
        )

    def legal_actions(
        self,
        player_index: int,
    ) -> tuple[BiddingAction, ...]:
        """
        Return exactly the actions the current player may take.
        """
        if self.is_complete:
            return ()

        if player_index != self.current_player:
            raise ValueError(
                f"It is player {self.current_player}'s bidding turn, "
                f"not player {player_index}'s."
            )

        if self.phase is BiddingPhase.INTEREST:
            actions = [BiddingAction.pass_action()]

            contracts = legal_contracts(
                hand=self.hands[player_index],
                rules=self.rules,
                declarer=player_index,
            )

            if contracts:
                actions.append(BiddingAction.play())

            return tuple(actions)

        if self.phase is BiddingPhase.NEGOTIATION:
            return self._negotiation_actions(player_index)

        if self.phase is BiddingPhase.ANNOUNCEMENT:
            return self._announcement_actions(player_index)

        raise RuntimeError("Unsupported bidding phase.")

    def apply_action(
        self,
        player_index: int,
        action: BiddingAction,
    ) -> None:
        """Apply one legal bidding action."""
        legal = self.legal_actions(player_index)

        if action not in legal:
            raise ValueError(
                f"{action} is not a legal bidding action for player {player_index}."
            )

        self._history.append(
            BiddingEvent(
                player_index=player_index,
                action=action,
            )
        )

        if self.phase is BiddingPhase.INTEREST:
            self._apply_interest_action(
                player_index,
                action,
            )
            return

        if self.phase is BiddingPhase.NEGOTIATION:
            self._apply_negotiation_action(
                player_index,
                action,
            )
            return

        if self.phase is BiddingPhase.ANNOUNCEMENT:
            self._apply_announcement(action)
            return

        raise RuntimeError("Cannot apply action in completed bidding.")

    def result(self) -> BiddingResult:
        """Return the final bidding result."""
        if not self.is_complete:
            raise ValueError("Bidding is not complete yet.")

        return BiddingResult(
            contract=self._contract,
            requires_redeal=self._requires_redeal,
            interested_players=self.interested_players,
            history=self.history,
        )

    def _negotiation_actions(
        self,
        player_index: int,
    ) -> tuple[BiddingAction, ...]:
        current_value = self.current_bid_value

        if current_value is None:
            raise RuntimeError("Negotiation has no current bid value.")

        if self._awaiting_incumbent_response:
            actions = [BiddingAction.pass_action()]

            max_value = self._maximum_legal_value(player_index)

            if max_value is not None and max_value >= current_value:
                actions.append(BiddingAction.hold())

            return tuple(actions)

        actions = [BiddingAction.pass_action()]

        max_value = self._maximum_legal_value(player_index)

        if max_value is None:
            return tuple(actions)

        for value in self.bid_values.enabled_values(self.rules):
            if value > current_value and value <= max_value:
                actions.append(BiddingAction.raise_to(value))

        return tuple(actions)

    def _announcement_actions(
        self,
        player_index: int,
    ) -> tuple[BiddingAction, ...]:
        current_value = self.current_bid_value

        if current_value is None:
            raise RuntimeError("Final announcement has no minimum bid value.")

        contracts = legal_contracts(
            hand=self.hands[player_index],
            rules=self.rules,
            declarer=player_index,
        )

        eligible_contracts = tuple(
            contract
            for contract in contracts
            if (self.bid_values.value(contract.game_type) >= current_value)
        )

        if not eligible_contracts:
            raise RuntimeError(
                "Winning bidder has no contract matching the committed bidding value."
            )

        return tuple(
            BiddingAction.announce(contract) for contract in eligible_contracts
        )

    def _apply_interest_action(
        self,
        player_index: int,
        action: BiddingAction,
    ) -> None:
        if action.action_type is BiddingActionType.PLAY:
            self._interested_players.append(player_index)

        self._interest_offset += 1

        if self._interest_offset == PLAYER_COUNT:
            self._finish_interest_round()

    def _finish_interest_round(
        self,
    ) -> None:
        if not self._interested_players:
            self._finish_all_passed()
            return

        enabled_values = self.bid_values.enabled_values(self.rules)

        if not enabled_values:
            raise RuntimeError(
                "Players declared interest but no bid games are enabled."
            )

        self.current_bid_value = enabled_values[0]

        self.incumbent = self._interested_players[0]

        if len(self._interested_players) == 1:
            self.phase = BiddingPhase.ANNOUNCEMENT
            return

        self._challenger_position = 1

        self.challenger = self._interested_players[self._challenger_position]

        self._awaiting_incumbent_response = False

        self.phase = BiddingPhase.NEGOTIATION

    def _apply_negotiation_action(
        self,
        player_index: int,
        action: BiddingAction,
    ) -> None:
        del player_index

        if self._awaiting_incumbent_response:
            if action.action_type is BiddingActionType.HOLD:
                self._awaiting_incumbent_response = False
                return

            if action.action_type is BiddingActionType.PASS:
                if self.challenger is None:
                    raise RuntimeError("No challenger exists.")

                self.incumbent = self.challenger

                self._advance_to_next_challenger()
                return

            raise RuntimeError("Unexpected incumbent action.")

        if action.action_type is BiddingActionType.RAISE:
            if action.bid_value is None:
                raise RuntimeError("Raise has no value.")

            self.current_bid_value = action.bid_value

            self._awaiting_incumbent_response = True
            return

        if action.action_type is BiddingActionType.PASS:
            self._advance_to_next_challenger()
            return

        raise RuntimeError("Unexpected challenger action.")

    def _advance_to_next_challenger(
        self,
    ) -> None:
        self._challenger_position += 1

        if self._challenger_position >= len(self._interested_players):
            self.challenger = None
            self._awaiting_incumbent_response = False
            self.phase = BiddingPhase.ANNOUNCEMENT
            return

        self.challenger = self._interested_players[self._challenger_position]

        self._awaiting_incumbent_response = False

    def _apply_announcement(
        self,
        action: BiddingAction,
    ) -> None:
        if action.contract is None:
            raise RuntimeError("Announcement has no contract.")

        self._contract = action.contract
        self.phase = BiddingPhase.COMPLETE

    def _finish_all_passed(
        self,
    ) -> None:
        if self.rules.all_pass_action is AllPassAction.RAMSCH:
            self._contract = GameContract(GameType.RAMSCH)

            self._requires_redeal = False

        elif self.rules.all_pass_action is AllPassAction.REDEAL:
            self._contract = None
            self._requires_redeal = True

        else:
            raise RuntimeError("Unsupported all-pass action.")

        self.phase = BiddingPhase.COMPLETE

    def _maximum_legal_value(
        self,
        player_index: int,
    ) -> int | None:
        contracts = legal_contracts(
            hand=self.hands[player_index],
            rules=self.rules,
            declarer=player_index,
        )

        if not contracts:
            return None

        return max(self.bid_values.value(contract.game_type) for contract in contracts)

    @staticmethod
    def _validate_hands(
        hands: Sequence[Sequence[Card]],
    ) -> None:
        if len(hands) != PLAYER_COUNT:
            raise ValueError("Bidding requires exactly four hands.")

        for hand in hands:
            if len(hand) != CARDS_PER_PLAYER:
                raise ValueError("Each player must have exactly eight cards.")

        all_cards = [card for hand in hands for card in hand]

        if len(all_cards) != 32:
            raise ValueError("Bidding requires exactly 32 cards.")

        if len(set(all_cards)) != 32:
            raise ValueError("Bidding hands must contain 32 unique cards.")
