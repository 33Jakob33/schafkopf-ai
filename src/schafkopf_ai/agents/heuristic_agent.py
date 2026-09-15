from __future__ import annotations

from collections import Counter
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
from .heuristic_knowledge import (
    InferredVoids,
    PublicCardKnowledge,
    infer_voids,
)

__all__ = [
    "HeuristicAgent",
    "HeuristicConfig",
    "InferredVoids",
    "PublicCardKnowledge",
    "infer_voids",
]


BID_SCORE_THRESHOLDS: dict[GameType, float] = {
    GameType.SAUSPIEL: 18.0,
    GameType.WENZ: 14.0,
    GameType.GEIER: 14.0,
    GameType.SOLO: 25.0,
}


@dataclass(frozen=True, slots=True)
class HeuristicConfig:
    """Tunable constants for the deterministic heuristic baseline."""

    valuable_trick_points: int = 10
    search_called_ace_until_trick: int = 4
    draw_trumps_until_trick: int = 5
    safe_ruff_risk: float = 0.30
    teammate_confidence: float = 0.70
    low_overtake_risk: float = 0.20
    long_suit_minimum: int = 2


class HeuristicAgent(Agent):
    """
    Contract-aware deterministic Schafkopf baseline.

    The agent uses only its private hand plus public information. It tracks
    remaining trumps and high cards, estimates locations of unseen cards,
    reasons about players still behind in the trick, draws trumps when it has
    control, establishes long suits, infers Sauspiel partnership, changes
    tactics for declarer/defender roles, reacts to score pressure, and has a
    dedicated Ramsch policy.

    It is intentionally interpretable rather than optimal. Multi-trick
    planning is heuristic: trump control, suit establishment, void creation,
    master-card preservation, and score-aware sacrifice are planned across
    future tricks without enumerating complete hidden hands.
    """

    def __init__(self, config: HeuristicConfig = HeuristicConfig()) -> None:
        self.config = config

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

        knowledge = PublicCardKnowledge.from_observation(observation)

        if observation.contract.game_type is GameType.RAMSCH:
            if not observation.current_trick:
                return self._choose_ramsch_lead(observation, legal_cards, knowledge)
            return self._choose_ramsch_follow(observation, legal_cards, knowledge)

        if not observation.current_trick:
            return self._choose_lead(observation, legal_cards, knowledge)

        return self._choose_follow(observation, legal_cards, knowledge)

    # ------------------------------------------------------------------
    # Bidding
    # ------------------------------------------------------------------

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
            return max(raises, key=lambda action: action.bid_value or 0)

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

    def _desired_bid_value(self, observation: BiddingObservation) -> int | None:
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
        """Estimate hand strength for one exact contract."""
        trumps = tuple(card for card in hand if is_trump(card, contract))
        trump_ordering = trump_order(contract)
        score = 0.0

        for card in trumps:
            normalized = trump_strength(card, contract) / len(trump_ordering)
            score += 3.0 + 3.0 * normalized

        held = set(hand)
        consecutive_top_trumps = 0
        for trump in trump_ordering:
            if trump not in held:
                break
            consecutive_top_trumps += 1
        score += consecutive_top_trumps * 2.5

        suit_lengths: Counter[Suit] = Counter()
        for card in hand:
            if is_trump(card, contract):
                continue
            suit_lengths[card.suit] += 1
            if card.rank is Rank.ACE:
                score += 3.0
            elif card.rank is Rank.TEN:
                score += 1.5
            elif card.rank is Rank.KING:
                score += 0.4

        short_plain_suits = sum(1 for length in suit_lengths.values() if length == 1)

        if contract.game_type in {GameType.WENZ, GameType.GEIER}:
            score += len(trumps) * 1.75
            score += short_plain_suits * 0.5

        if contract.game_type is GameType.SOLO:
            if len(trumps) >= 5:
                score += 4.0
            if len(trumps) >= 6:
                score += 3.0

        if contract.game_type is GameType.SAUSPIEL:
            if len(trumps) >= 4:
                score += 2.0
            called_suit = contract.called_suit
            if called_suit is not None:
                called_plain = sum(
                    1
                    for card in hand
                    if (
                        card.suit is called_suit
                        and not is_trump(card, contract)
                    )
                )
                # One or two called-suit cards make it easier to search for the
                # partner without being overloaded in that suit.
                if 1 <= called_plain <= 2:
                    score += 1.5

        return score

    # ------------------------------------------------------------------
    # Normal-game leading strategy
    # ------------------------------------------------------------------

    def _choose_lead(
        self,
        observation: PlayerObservation,
        legal_cards: tuple[Card, ...],
        knowledge: PublicCardKnowledge,
    ) -> Card:
        contract = observation.contract

        sauspiel_move = self._sauspiel_lead_plan(observation, legal_cards, knowledge)
        if sauspiel_move is not None:
            return sauspiel_move

        trump_draw = self._trump_draw_plan(observation, legal_cards, knowledge)
        if trump_draw is not None:
            return trump_draw

        opponents = self._possible_opponents(observation, knowledge)

        master_cards = [
            card
            for card in legal_cards
            if (
                not is_trump(card, contract)
                and knowledge.is_master_card(card)
                and knowledge.probability_suit_gets_trumped(
                    card.suit,
                    opponents,
                )
                <= self.config.safe_ruff_risk
            )
        ]
        if master_cards:
            return max(
                master_cards,
                key=lambda card: (
                    card_points(card),
                    self._own_plain_suit_length(observation, card.suit),
                ),
            )

        establishment = self._long_suit_establishment_plan(
            observation,
            legal_cards,
            knowledge,
        )
        if establishment is not None:
            return establishment

        safe_plain_aces = [
            card
            for card in legal_cards
            if (
                card.rank is Rank.ACE
                and not is_trump(card, contract)
                and knowledge.probability_suit_gets_trumped(
                    card.suit,
                    opponents,
                )
                <= self.config.safe_ruff_risk
            )
        ]
        if safe_plain_aces:
            return max(
                safe_plain_aces,
                key=lambda card: self._future_control_value(
                    observation,
                    card,
                    knowledge,
                ),
            )

        non_trumps = [card for card in legal_cards if not is_trump(card, contract)]
        if non_trumps:
            return max(
                non_trumps,
                key=lambda card: self._lead_plan_score(
                    observation,
                    card,
                    knowledge,
                ),
            )

        return min(
            legal_cards,
            key=lambda card: self._card_cost(card, contract),
        )

    def _sauspiel_lead_plan(
        self,
        observation: PlayerObservation,
        legal_cards: tuple[Card, ...],
        knowledge: PublicCardKnowledge,
    ) -> Card | None:
        if observation.contract.game_type is not GameType.SAUSPIEL:
            return None

        contract = observation.contract
        called_suit = contract.called_suit
        declarer = contract.declarer
        if called_suit is None or declarer is None:
            return None

        called_ace = Card(called_suit, Rank.ACE)
        partner = self._known_sauspiel_partner(observation)

        if called_ace in observation.hand and called_ace in legal_cards:
            opponents = self._possible_opponents(observation, knowledge)
            ruff_risk = knowledge.probability_suit_gets_trumped(
                called_suit,
                opponents,
            )
            if (
                observation.trick_number <= self.config.search_called_ace_until_trick
                and ruff_risk <= self.config.safe_ruff_risk
            ):
                return called_ace

        if (
            observation.player_index == declarer
            and partner is None
            and not observation.called_ace_released
            and observation.trick_number <= self.config.search_called_ace_until_trick
        ):
            search_cards = [
                card
                for card in legal_cards
                if (
                    card.suit is called_suit
                    and not is_trump(card, contract)
                    and card != called_ace
                )
            ]
            if search_cards:
                return min(
                    search_cards,
                    key=lambda card: self._card_cost(card, contract),
                )

        return None

    def _trump_draw_plan(
        self,
        observation: PlayerObservation,
        legal_cards: tuple[Card, ...],
        knowledge: PublicCardKnowledge,
    ) -> Card | None:
        contract = observation.contract
        declarer = contract.declarer
        if declarer is None or observation.trick_number > self.config.draw_trumps_until_trick:
            return None

        own_trumps = tuple(card for card in legal_cards if is_trump(card, contract))
        if len(own_trumps) < 2 or knowledge.remaining_trump_count == 0:
            return None

        controlling = tuple(
            card for card in own_trumps if not knowledge.unseen_higher_trumps(card)
        )
        if not controlling:
            return None

        cheapest_controller = min(
            controlling,
            key=lambda card: trump_strength(card, contract),
        )

        if observation.player_index == declarer:
            minimum_trumps = {
                GameType.SAUSPIEL: 5,
                GameType.SOLO: 4,
                GameType.WENZ: 2,
                GameType.GEIER: 2,
            }.get(contract.game_type, 99)

            total_own_trumps = len(knowledge.own_trumps)
            if total_own_trumps >= minimum_trumps:
                return cheapest_controller
            return None

        if contract.game_type in {GameType.WENZ, GameType.GEIER}:
            probability_declarer_has_trump = knowledge.probability_players_have_trump(
                {declarer}
            )
            if probability_declarer_has_trump >= 0.55:
                return cheapest_controller

        return None

    def _long_suit_establishment_plan(
        self,
        observation: PlayerObservation,
        legal_cards: tuple[Card, ...],
        knowledge: PublicCardKnowledge,
    ) -> Card | None:
        if observation.trick_number >= 7 or self._score_urgency(observation) >= 2.0:
            return None

        contract = observation.contract
        by_suit: dict[Suit, list[Card]] = {suit: [] for suit in Suit}
        for card in observation.hand:
            if not is_trump(card, contract):
                by_suit[card.suit].append(card)

        plans: list[tuple[float, Card]] = []
        opponents = self._possible_opponents(observation, knowledge)

        for suit, cards in by_suit.items():
            if len(cards) < self.config.long_suit_minimum:
                continue

            legal_in_suit = [card for card in legal_cards if card in cards]
            if not legal_in_suit:
                continue

            strongest = max(
                cards,
                key=lambda card: plain_card_strength(card, contract),
            )
            higher_unseen = knowledge.unseen_higher_plain_cards(strongest)
            if not higher_unseen:
                continue

            ruff_risk = knowledge.probability_suit_gets_trumped(suit, opponents)
            if ruff_risk > self.config.safe_ruff_risk:
                continue

            sacrifice = min(
                legal_in_suit,
                key=lambda card: self._card_cost(card, contract),
            )

            future_value = (
                card_points(strongest)
                + len(cards) * 2.0
                - len(higher_unseen)
                - ruff_risk * 10.0
            )
            plans.append((future_value, sacrifice))

        if not plans:
            return None

        return max(plans, key=lambda plan: plan[0])[1]

    # ------------------------------------------------------------------
    # Normal-game follow strategy
    # ------------------------------------------------------------------

    def _choose_follow(
        self,
        observation: PlayerObservation,
        legal_cards: tuple[Card, ...],
        knowledge: PublicCardKnowledge,
    ) -> Card:
        contract = observation.contract
        current_winner = winning_play(observation.current_trick, contract)
        lead_card = observation.current_trick[0].card
        players_behind = self._players_behind(observation)

        winning_cards = tuple(
            card
            for card in legal_cards
            if card_beats(
                challenger=card,
                current_winner=current_winner.card,
                lead_card=lead_card,
                contract=contract,
            )
        )
        losing_cards = tuple(card for card in legal_cards if card not in winning_cards)

        teammate_probability = self._teammate_probability(
            observation,
            knowledge,
            current_winner.player,
        )
        last_to_play = not players_behind
        trick_value = sum(card_points(play.card) for play in observation.current_trick)
        urgency = self._score_urgency(observation)

        if teammate_probability >= self.config.teammate_confidence and losing_cards:
            if last_to_play:
                return max(
                    losing_cards,
                    key=lambda card: (
                        card_points(card),
                        -self._card_cost(card, contract),
                    ),
                )
            return self._best_discard(observation, losing_cards, knowledge)

        if winning_cards:
            best_winner = self._best_winning_card(
                observation,
                winning_cards,
                knowledge,
                lead_card,
                players_behind,
                trick_value,
                urgency,
            )
            overtake_risk = knowledge.probability_overtaken(
                candidate=best_winner,
                lead_card=lead_card,
                players_behind=players_behind,
            )

            projected_value = trick_value + card_points(best_winner)
            must_secure_score = urgency >= 2.0
            should_take = (
                last_to_play
                or projected_value >= self.config.valuable_trick_points
                or must_secure_score
                or observation.trick_number >= 7
                or (
                    not is_trump(best_winner, contract)
                    and overtake_risk <= self.config.low_overtake_risk
                )
            )

            if (
                not should_take
                and losing_cards
                and is_trump(best_winner, contract)
            ):
                return self._best_discard(observation, losing_cards, knowledge)

            if should_take:
                return best_winner

        if losing_cards:
            return self._best_discard(observation, losing_cards, knowledge)

        return self._best_winning_card(
            observation,
            legal_cards,
            knowledge,
            lead_card,
            players_behind,
            trick_value,
            urgency,
        )

    def _best_winning_card(
        self,
        observation: PlayerObservation,
        cards: tuple[Card, ...],
        knowledge: PublicCardKnowledge,
        lead_card: Card,
        players_behind: tuple[int, ...],
        trick_value: int,
        urgency: float,
    ) -> Card:
        contract = observation.contract

        def score(card: Card) -> float:
            risk = knowledge.probability_overtaken(
                candidate=card,
                lead_card=lead_card,
                players_behind=players_behind,
            )
            risk_weight = 12.0 + trick_value * 1.2 + urgency * 12.0
            return self._card_cost(card, contract) + risk * risk_weight

        return min(cards, key=score)

    def _best_discard(
        self,
        observation: PlayerObservation,
        cards: tuple[Card, ...],
        knowledge: PublicCardKnowledge,
    ) -> Card:
        contract = observation.contract

        def score(card: Card) -> float:
            base = self._discard_cost(card, contract)
            if is_trump(card, contract):
                return base

            suit_length = self._own_plain_suit_length(observation, card.suit)
            create_void_bonus = 4.0 if suit_length == 1 and knowledge.own_trumps else 0.0

            called_suit_penalty = 0.0
            if (
                contract.game_type is GameType.SAUSPIEL
                and contract.called_suit is card.suit
                and self._known_sauspiel_partner(observation) is None
            ):
                called_suit_penalty = 2.0

            return base + suit_length * 0.4 + called_suit_penalty - create_void_bonus

        return min(cards, key=score)

    # ------------------------------------------------------------------
    # Ramsch strategy
    # ------------------------------------------------------------------

    def _choose_ramsch_lead(
        self,
        observation: PlayerObservation,
        legal_cards: tuple[Card, ...],
        knowledge: PublicCardKnowledge,
    ) -> Card:
        opponents = tuple(
            player for player in range(4) if player != observation.player_index
        )

        def shed_score(card: Card) -> float:
            probability_beaten = knowledge.probability_overtaken(
                candidate=card,
                lead_card=card,
                players_behind=opponents,
            )
            points = card_points(card)
            trump_penalty = 5.0 if is_trump(card, observation.contract) else 0.0
            return (
                probability_beaten * (16.0 + points * 2.5)
                + points * probability_beaten
                - (1.0 - probability_beaten) * points * 2.0
                - trump_penalty
            )

        return max(legal_cards, key=shed_score)

    def _choose_ramsch_follow(
        self,
        observation: PlayerObservation,
        legal_cards: tuple[Card, ...],
        knowledge: PublicCardKnowledge,
    ) -> Card:
        current_winner = winning_play(
            observation.current_trick,
            observation.contract,
        )
        lead_card = observation.current_trick[0].card
        players_behind = self._players_behind(observation)

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
        losing_cards = tuple(card for card in legal_cards if card not in winning_cards)

        if losing_cards:
            current_points = observation.points_by_player[current_winner.player]
            highest_points = max(observation.points_by_player)
            target_bonus = 5.0 if current_points == highest_points else 0.0
            return max(
                losing_cards,
                key=lambda card: (
                    card_points(card) * 3.0
                    + target_bonus
                    + self._future_ramsch_danger(card, observation)
                ),
            )

        if players_behind:
            return max(
                legal_cards,
                key=lambda card: (
                    knowledge.probability_overtaken(
                        candidate=card,
                        lead_card=lead_card,
                        players_behind=players_behind,
                    )
                    * 20.0
                    - card_points(card)
                    - self._card_cost(card, observation.contract) * 0.02
                ),
            )

        return min(
            legal_cards,
            key=lambda card: (
                card_points(card),
                self._card_cost(card, observation.contract),
            ),
        )

    # ------------------------------------------------------------------
    # Public-information team / score reasoning
    # ------------------------------------------------------------------

    def _known_teammates(self, observation: PlayerObservation) -> frozenset[int]:
        contract = observation.contract
        player = observation.player_index
        declarer = contract.declarer

        if contract.game_type is GameType.RAMSCH or declarer is None:
            return frozenset({player})

        if contract.game_type in {GameType.SOLO, GameType.WENZ, GameType.GEIER}:
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

    def _teammate_probability(
        self,
        observation: PlayerObservation,
        knowledge: PublicCardKnowledge,
        other_player: int,
    ) -> float:
        player = observation.player_index
        contract = observation.contract
        declarer = contract.declarer

        if other_player == player:
            return 1.0
        if contract.game_type is GameType.RAMSCH or declarer is None:
            return 0.0

        if contract.game_type in {GameType.SOLO, GameType.WENZ, GameType.GEIER}:
            if player == declarer:
                return 0.0
            return 0.0 if other_player == declarer else 1.0

        if contract.game_type is not GameType.SAUSPIEL:
            return 0.0

        known_partner = self._known_sauspiel_partner(observation)
        if known_partner is not None:
            return 1.0 if other_player in self._known_teammates(observation) else 0.0

        called_ace = self._called_ace(observation)
        if called_ace is None:
            return 0.0

        partner_probabilities = knowledge.holder_probabilities(called_ace)

        if player == declarer:
            return partner_probabilities[other_player]

        if called_ace in observation.hand:
            return 1.0 if other_player == declarer else 0.0

        if other_player == declarer:
            return 0.0

        return 1.0 - partner_probabilities[other_player]

    def _possible_opponents(
        self,
        observation: PlayerObservation,
        knowledge: PublicCardKnowledge,
    ) -> frozenset[int]:
        return frozenset(
            player
            for player in range(4)
            if (
                player != observation.player_index
                and self._teammate_probability(
                    observation,
                    knowledge,
                    player,
                )
                < 0.5
            )
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

    def _score_urgency(self, observation: PlayerObservation) -> float:
        team = self._exact_team_context(observation)
        if team is None:
            return 0.0

        own_team, opponent_team, own_is_declarer_side = team
        own_points = sum(observation.points_by_player[player] for player in own_team)
        opponent_points = sum(
            observation.points_by_player[player] for player in opponent_team
        )
        win_target = 61 if own_is_declarer_side else 60
        points_needed = max(0, win_target - own_points)

        urgency = 0.0
        if observation.trick_number >= 6:
            urgency += 0.5
        if points_needed <= 15:
            urgency += 1.0
        if points_needed <= 8:
            urgency += 0.5

        own_schneider_border = 30 if own_is_declarer_side else 29
        opponent_schneider_border = 29 if own_is_declarer_side else 30
        if observation.trick_number >= 6 and own_points <= own_schneider_border:
            urgency += 0.75
        if observation.trick_number >= 6 and opponent_points <= opponent_schneider_border:
            urgency += 0.5

        return min(3.0, urgency)

    def _exact_team_context(
        self,
        observation: PlayerObservation,
    ) -> tuple[frozenset[int], frozenset[int], bool] | None:
        contract = observation.contract
        declarer = contract.declarer
        if declarer is None or contract.game_type is GameType.RAMSCH:
            return None

        if contract.game_type in {GameType.SOLO, GameType.WENZ, GameType.GEIER}:
            declarer_team = frozenset({declarer})
        elif contract.game_type is GameType.SAUSPIEL:
            partner = self._known_sauspiel_partner(observation)
            if partner is None:
                return None
            declarer_team = frozenset({declarer, partner})
        else:
            return None

        defender_team = frozenset(
            player for player in range(4) if player not in declarer_team
        )
        if observation.player_index in declarer_team:
            return declarer_team, defender_team, True
        return defender_team, declarer_team, False

    # ------------------------------------------------------------------
    # Planning / utility helpers
    # ------------------------------------------------------------------

    def _players_behind(self, observation: PlayerObservation) -> tuple[int, ...]:
        if not observation.current_trick:
            return ()

        count = 3 - len(observation.current_trick)
        return tuple(
            (observation.player_index + offset) % 4
            for offset in range(1, count + 1)
        )

    def _lead_plan_score(
        self,
        observation: PlayerObservation,
        card: Card,
        knowledge: PublicCardKnowledge,
    ) -> float:
        contract = observation.contract
        score = self._future_control_value(observation, card, knowledge)
        score -= self._card_cost(card, contract) * 0.08

        if not is_trump(card, contract):
            opponents = self._possible_opponents(observation, knowledge)
            ruff_risk = knowledge.probability_suit_gets_trumped(card.suit, opponents)
            score -= ruff_risk * 8.0

            if (
                contract.game_type is GameType.SAUSPIEL
                and contract.called_suit is card.suit
                and contract.declarer != observation.player_index
                and self._known_sauspiel_partner(observation) is None
            ):
                score -= 5.0

        return score

    def _future_control_value(
        self,
        observation: PlayerObservation,
        card: Card,
        knowledge: PublicCardKnowledge,
    ) -> float:
        contract = observation.contract
        value = 0.0

        if knowledge.is_master_card(card):
            value += 5.0 + card_points(card) * 0.25

        if is_trump(card, contract):
            if not knowledge.unseen_higher_trumps(card):
                value += 4.0
            return value

        suit_length = self._own_plain_suit_length(observation, card.suit)
        if suit_length >= self.config.long_suit_minimum:
            value += suit_length * 1.25

        if suit_length == 1 and knowledge.own_trumps:
            value += 2.5

        return value

    def _future_ramsch_danger(
        self,
        card: Card,
        observation: PlayerObservation,
    ) -> float:
        contract = observation.contract
        if is_trump(card, contract):
            return trump_strength(card, contract) * 0.7
        return plain_card_strength(card, contract) * 0.5

    @staticmethod
    def _own_plain_suit_length(
        observation: PlayerObservation,
        suit: Suit,
    ) -> int:
        return sum(
            1
            for card in observation.hand
            if (
                card.suit is suit
                and not is_trump(card, observation.contract)
            )
        )

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
