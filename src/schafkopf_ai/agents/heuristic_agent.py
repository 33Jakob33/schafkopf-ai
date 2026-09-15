from __future__ import annotations

from dataclasses import dataclass

from schafkopf_ai.game.bidding import (
    BiddingAction,
    BiddingActionType,
    BiddingObservation,
    BiddingPhase,
    legal_contracts,
)
from schafkopf_ai.game.card import Card, Rank, Suit
from schafkopf_ai.game.game_contract import GameContract
from schafkopf_ai.game.game_type import GameType
from schafkopf_ai.game.observation import PlayerObservation
from schafkopf_ai.game.scoring import card_points
from schafkopf_ai.game.trick import card_beats, plain_card_strength, winning_play
from schafkopf_ai.game.trump import is_trump, trump_order, trump_strength

from .agent import Agent


BID_SCORE_THRESHOLDS: dict[GameType, float] = {
    GameType.SAUSPIEL: 17.0,
    GameType.WENZ: 12.0,
    GameType.GEIER: 12.0,
    GameType.SOLO: 22.0,
}


@dataclass(frozen=True, slots=True)
class InferredVoids:
    """Publicly inferable information about players who failed to follow."""

    plain_suits: tuple[frozenset[Suit], ...]
    trump_void_players: frozenset[int]


class HeuristicAgent(Agent):
    """
    Deterministic, interpretable baseline agent.

    The agent deliberately uses only information contained in the public
    observation. Its card-play rules are intentionally simple:

    - take valuable tricks with the cheapest winning card,
    - conserve trumps, especially high trumps,
    - avoid overtaking a known teammate,
    - smear points to a known teammate when last to play,
    - infer void suits from completed public tricks,
    - prefer safe plain Aces when leading,
    - apply basic Sauspiel partner logic.
    """

    def choose_bidding_action(
        self,
        observation: BiddingObservation,
        legal_actions: tuple[BiddingAction, ...],
    ) -> BiddingAction:
        if not legal_actions:
            raise ValueError("No legal bidding actions available.")

        if observation.phase is BiddingPhase.INTEREST:
            return self._choose_interest_action(observation, legal_actions)

        if observation.phase is BiddingPhase.NEGOTIATION:
            return self._choose_negotiation_action(observation, legal_actions)

        if observation.phase is BiddingPhase.ANNOUNCEMENT:
            return self._choose_announcement_action(observation, legal_actions)

        raise ValueError("Cannot choose a bidding action after bidding is complete.")

    def choose_card(
        self,
        observation: PlayerObservation,
        legal_cards: tuple[Card, ...],
    ) -> Card:
        if not legal_cards:
            raise ValueError("No legal cards available.")

        if len(legal_cards) == 1:
            return legal_cards[0]

        if not observation.current_trick:
            return self._choose_lead(observation, legal_cards)

        return self._choose_follow(observation, legal_cards)

    def _choose_interest_action(
        self,
        observation: BiddingObservation,
        legal_actions: tuple[BiddingAction, ...],
    ) -> BiddingAction:
        contracts = legal_contracts(
            hand=observation.hand,
            rules=observation.rules,
            declarer=observation.player_index,
        )

        wants_to_play = any(
            self._contract_score(observation.hand, contract)
            >= BID_SCORE_THRESHOLDS[contract.game_type]
            for contract in contracts
        )

        desired_type = (
            BiddingActionType.PLAY if wants_to_play else BiddingActionType.PASS
        )

        return self._action_of_type(legal_actions, desired_type)

    def _choose_negotiation_action(
        self,
        observation: BiddingObservation,
        legal_actions: tuple[BiddingAction, ...],
    ) -> BiddingAction:
        desired_value = self._desired_bid_value(observation)

        if desired_value is None:
            return self._action_of_type(legal_actions, BiddingActionType.PASS)

        hold_action = next(
            (
                action
                for action in legal_actions
                if action.action_type is BiddingActionType.HOLD
            ),
            None,
        )

        if hold_action is not None:
            current_value = observation.current_bid_value
            if current_value is not None and desired_value >= current_value:
                return hold_action

            return self._action_of_type(legal_actions, BiddingActionType.PASS)

        raises = [
            action
            for action in legal_actions
            if (
                action.action_type is BiddingActionType.RAISE
                and action.bid_value is not None
                and action.bid_value <= desired_value
            )
        ]

        if raises:
            return max(
                raises,
                key=lambda action: action.bid_value or 0,
            )

        return self._action_of_type(legal_actions, BiddingActionType.PASS)

    def _choose_announcement_action(
        self,
        observation: BiddingObservation,
        legal_actions: tuple[BiddingAction, ...],
    ) -> BiddingAction:
        announcements = [
            action
            for action in legal_actions
            if (
                action.action_type is BiddingActionType.ANNOUNCE
                and action.contract is not None
            )
        ]

        if not announcements:
            raise ValueError("No legal contract announcement available.")

        return max(
            announcements,
            key=lambda action: self._contract_score(
                observation.hand,
                self._require_contract(action),
            ),
        )

    def _desired_bid_value(
        self,
        observation: BiddingObservation,
    ) -> int | None:
        contracts = legal_contracts(
            hand=observation.hand,
            rules=observation.rules,
            declarer=observation.player_index,
        )

        qualified = [
            contract
            for contract in contracts
            if (
                self._contract_score(observation.hand, contract)
                >= BID_SCORE_THRESHOLDS[contract.game_type]
            )
        ]

        if not qualified:
            return None

        return max(
            observation.bid_values.value(contract.game_type)
            for contract in qualified
        )

    def _contract_score(
        self,
        hand: tuple[Card, ...],
        contract: GameContract,
    ) -> float:
        """Estimate how suitable a hand is for one exact contract."""
        trumps = [card for card in hand if is_trump(card, contract)]
        trump_count = len(trumps)
        trump_total = len(trump_order(contract))

        score = 0.0

        for card in trumps:
            normalized_strength = trump_strength(card, contract) / trump_total
            score += 3.0 + 2.0 * normalized_strength

        for card in hand:
            if is_trump(card, contract):
                continue

            if card.rank is Rank.ACE:
                score += 2.5
            elif card.rank is Rank.TEN:
                score += 1.25
            elif card.rank is Rank.KING:
                score += 0.25

        if contract.game_type in {GameType.WENZ, GameType.GEIER}:
            score += trump_count * 1.5

        if contract.game_type is GameType.SOLO and trump_count >= 5:
            score += 3.0

        if contract.game_type is GameType.SAUSPIEL and trump_count >= 4:
            score += 1.5

        return score

    def _choose_lead(
        self,
        observation: PlayerObservation,
        legal_cards: tuple[Card, ...],
    ) -> Card:
        called_ace = self._called_ace(observation)

        if (
            called_ace is not None
            and called_ace in observation.hand
            and called_ace in legal_cards
            and observation.trick_number <= 3
        ):
            return called_ace

        voids = infer_voids(observation)
        opponents = self._possible_opponents(observation)

        safe_plain_aces = [
            card
            for card in legal_cards
            if (
                card.rank is Rank.ACE
                and not is_trump(card, observation.contract)
                and all(
                    card.suit not in voids.plain_suits[player]
                    for player in opponents
                )
            )
        ]

        if safe_plain_aces:
            return min(
                safe_plain_aces,
                key=lambda card: self._card_cost(card, observation.contract),
            )

        non_trumps = [
            card for card in legal_cards if not is_trump(card, observation.contract)
        ]

        if non_trumps:
            return min(
                non_trumps,
                key=lambda card: self._discard_cost(card, observation.contract),
            )

        return min(
            legal_cards,
            key=lambda card: self._card_cost(card, observation.contract),
        )

    def _choose_follow(
        self,
        observation: PlayerObservation,
        legal_cards: tuple[Card, ...],
    ) -> Card:
        current_winner = winning_play(
            observation.current_trick,
            observation.contract,
        )
        lead_card = observation.current_trick[0].card

        winning_cards = tuple(
            card
            for card in legal_cards
            if card_beats(
                challenger=card,
                current_winner=current_winner.card,
                lead_card=lead_card,
                contract=observation.contract,
            )
        )

        known_teammates = self._known_teammates(observation)
        teammate_winning = current_winner.player in known_teammates
        last_to_play = len(observation.current_trick) == 3

        if teammate_winning:
            losing_cards = tuple(card for card in legal_cards if card not in winning_cards)

            if losing_cards:
                if last_to_play:
                    return max(
                        losing_cards,
                        key=lambda card: (
                            card_points(card),
                            -self._card_cost(card, observation.contract),
                        ),
                    )

                return min(
                    losing_cards,
                    key=lambda card: self._discard_cost(
                        card,
                        observation.contract,
                    ),
                )

        if winning_cards:
            cheapest_winner = min(
                winning_cards,
                key=lambda card: self._card_cost(card, observation.contract),
            )

            trick_value = sum(
                card_points(play.card) for play in observation.current_trick
            )
            projected_value = trick_value + card_points(cheapest_winner)

            should_take = (
                last_to_play
                or projected_value >= 10
                or not is_trump(cheapest_winner, observation.contract)
                or observation.trick_number >= 7
            )

            if should_take:
                return cheapest_winner

        return min(
            legal_cards,
            key=lambda card: self._discard_cost(card, observation.contract),
        )

    def _known_teammates(
        self,
        observation: PlayerObservation,
    ) -> frozenset[int]:
        contract = observation.contract
        player = observation.player_index
        declarer = contract.declarer

        if contract.game_type is GameType.RAMSCH or declarer is None:
            return frozenset({player})

        if contract.game_type in {
            GameType.SOLO,
            GameType.WENZ,
            GameType.GEIER,
        }:
            if player == declarer:
                return frozenset({player})

            return frozenset(index for index in range(4) if index != declarer)

        if contract.game_type is not GameType.SAUSPIEL:
            return frozenset({player})

        partner = self._known_sauspiel_partner(observation)

        if partner is None:
            return frozenset({player})

        declarer_team = frozenset({declarer, partner})

        if player in declarer_team:
            return declarer_team

        return frozenset(index for index in range(4) if index not in declarer_team)

    def _possible_opponents(
        self,
        observation: PlayerObservation,
    ) -> frozenset[int]:
        teammates = self._known_teammates(observation)
        return frozenset(
            player
            for player in range(4)
            if player != observation.player_index and player not in teammates
        )

    def _known_sauspiel_partner(
        self,
        observation: PlayerObservation,
    ) -> int | None:
        called_ace = self._called_ace(observation)

        if called_ace is None:
            return None

        if called_ace in observation.hand:
            return observation.player_index

        for trick in observation.completed_tricks:
            for play in trick:
                if play.card == called_ace:
                    return play.player

        for play in observation.current_trick:
            if play.card == called_ace:
                return play.player

        return None

    @staticmethod
    def _called_ace(observation: PlayerObservation) -> Card | None:
        if observation.contract.game_type is not GameType.SAUSPIEL:
            return None

        called_suit = observation.contract.called_suit

        if called_suit is None:
            return None

        return Card(called_suit, Rank.ACE)

    @staticmethod
    def _card_cost(card: Card, contract: GameContract) -> float:
        if is_trump(card, contract):
            return 100.0 + trump_strength(card, contract) * 5.0 + card_points(card)

        return card_points(card) * 3.0 + plain_card_strength(card, contract)

    @staticmethod
    def _discard_cost(card: Card, contract: GameContract) -> float:
        if is_trump(card, contract):
            return 100.0 + trump_strength(card, contract) * 5.0 + card_points(card)

        return card_points(card) * 10.0 + plain_card_strength(card, contract)

    @staticmethod
    def _action_of_type(
        legal_actions: tuple[BiddingAction, ...],
        action_type: BiddingActionType,
    ) -> BiddingAction:
        for action in legal_actions:
            if action.action_type is action_type:
                return action

        raise ValueError(f"No legal {action_type.name} action available.")

    @staticmethod
    def _require_contract(action: BiddingAction) -> GameContract:
        if action.contract is None:
            raise ValueError("Bidding action does not contain a contract.")

        return action.contract


def infer_voids(observation: PlayerObservation) -> InferredVoids:
    """
    Infer suit/trump voids solely from public play history.

    If a player legally fails to follow a led plain suit, that player is
    known to be void in that suit. If a player fails to play trump after
    trump is led, that player is known to be void in trump.
    """
    plain_voids: list[set[Suit]] = [set() for _ in range(4)]
    trump_voids: set[int] = set()

    tricks = list(observation.completed_tricks)
    if observation.current_trick:
        tricks.append(observation.current_trick)

    for trick in tricks:
        if len(trick) < 2:
            continue

        lead = trick[0].card
        lead_is_trump = is_trump(lead, observation.contract)

        for play in trick[1:]:
            played_is_trump = is_trump(play.card, observation.contract)

            if lead_is_trump:
                if not played_is_trump:
                    trump_voids.add(play.player)
                continue

            if played_is_trump or play.card.suit is not lead.suit:
                plain_voids[play.player].add(lead.suit)

    return InferredVoids(
        plain_suits=tuple(frozenset(suits) for suits in plain_voids),
        trump_void_players=frozenset(trump_voids),
    )
