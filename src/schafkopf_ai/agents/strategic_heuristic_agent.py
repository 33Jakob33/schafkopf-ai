from __future__ import annotations

from schafkopf_ai.game.card import Card, Rank
from schafkopf_ai.game.game_type import GameType
from schafkopf_ai.game.observation import PlayerObservation
from schafkopf_ai.game.scoring import card_points
from schafkopf_ai.game.trick import card_beats, winning_play
from schafkopf_ai.game.trump import is_trump, trump_strength

from .heuristic_agent import HeuristicAgent, HeuristicConfig
from .heuristic_knowledge import PublicCardKnowledge
from .heuristic_strategy import (
    PlayerRole,
    called_ace,
    called_ace_has_been_played,
    player_role,
    score_situation,
)


class StrategicHeuristicAgent(HeuristicAgent):
    """
    Stronger, role-aware extension of :class:`HeuristicAgent`.

    The base agent remains available as a stable benchmark. This version adds
    explicit declarer/partner/defender policy, corrected Sauspiel search play,
    exact master-trump reasoning, Ace-safety estimates, teammate-behind
    reasoning, game-clinching/Schneider priorities, and deliberate sacrifice
    decisions. It still uses only the acting player's hand and public history.
    """

    def __init__(self, config: HeuristicConfig | None = None) -> None:
        super().__init__(config=config)

    def _choose_lead(
        self,
        observation: PlayerObservation,
        legal_cards: tuple[Card, ...],
        knowledge: PublicCardKnowledge,
    ) -> Card:
        role_move = self._role_specific_lead(observation, legal_cards, knowledge)
        if role_move is not None:
            return role_move

        return super()._choose_lead(observation, legal_cards, knowledge)

    def _sauspiel_lead_plan(
        self,
        observation: PlayerObservation,
        legal_cards: tuple[Card, ...],
        knowledge: PublicCardKnowledge,
    ) -> Card | None:
        """
        Apply the core Sauspiel role distinction explicitly.

        Defenders normally search the called Ace by leading the called suit.
        Declarer and called-Ace partner do not deliberately search their own
        side; they prefer trump/control play unless another tactical rule wins.
        """
        if observation.contract.game_type is not GameType.SAUSPIEL:
            return None

        contract = observation.contract
        called_suit = contract.called_suit
        if called_suit is None:
            return None

        role = player_role(observation)
        ace = called_ace(observation)

        if (
            role is PlayerRole.DEFENDER
            and ace is not None
            and not called_ace_has_been_played(observation)
            and not observation.called_ace_released
            and observation.trick_number <= self.config.search_called_ace_until_trick
        ):
            search_cards = tuple(
                card
                for card in legal_cards
                if (
                    card.suit is called_suit
                    and not is_trump(card, contract)
                    and card != ace
                )
            )
            if search_cards:
                return min(
                    search_cards,
                    key=lambda card: self._card_cost(card, contract),
                )

        return None

    def _role_specific_lead(
        self,
        observation: PlayerObservation,
        legal_cards: tuple[Card, ...],
        knowledge: PublicCardKnowledge,
    ) -> Card | None:
        contract = observation.contract
        role = player_role(observation)

        if contract.game_type is GameType.SAUSPIEL:
            search = self._sauspiel_lead_plan(observation, legal_cards, knowledge)
            if search is not None:
                return search

        if role is PlayerRole.DECLARER:
            master_trumps = tuple(
                card for card in legal_cards if knowledge.is_definite_master_trump(card)
            )

            # A declarer with exact trump control can safely pull enemy trumps.
            if (
                master_trumps
                and knowledge.remaining_trump_count > 0
                and len(knowledge.own_trumps) >= 2
                and observation.trick_number <= self.config.draw_trumps_until_trick
            ):
                return min(
                    master_trumps,
                    key=lambda card: trump_strength(card, contract),
                )

            safe_aces = self._safe_aces(observation, legal_cards, knowledge)
            if safe_aces:
                return max(safe_aces, key=card_points)

            # In Sauspiel, the declarer should not voluntarily search their own
            # partner. Prefer another suit when a reasonable alternative exists.
            if (
                contract.game_type is GameType.SAUSPIEL
                and not called_ace_has_been_played(observation)
                and not observation.called_ace_released
            ):
                alternatives = tuple(
                    card
                    for card in legal_cards
                    if (
                        not is_trump(card, contract)
                        and card.suit is not contract.called_suit
                    )
                )
                if alternatives:
                    return min(
                        alternatives,
                        key=lambda card: self._card_cost(card, contract),
                    )

        if role is PlayerRole.PARTNER:
            safe_aces = tuple(
                card
                for card in self._safe_aces(observation, legal_cards, knowledge)
                if card.suit is not contract.called_suit
            )
            if safe_aces:
                return max(safe_aces, key=card_points)

            if (
                contract.game_type is GameType.SAUSPIEL
                and not called_ace_has_been_played(observation)
                and not observation.called_ace_released
            ):
                alternatives = tuple(
                    card
                    for card in legal_cards
                    if (
                        not is_trump(card, contract)
                        and card.suit is not contract.called_suit
                    )
                )
                if alternatives:
                    return min(
                        alternatives,
                        key=lambda card: self._card_cost(card, contract),
                    )

        if role is PlayerRole.DEFENDER:
            safe_aces = self._safe_aces(observation, legal_cards, knowledge)
            if safe_aces:
                return max(safe_aces, key=card_points)

            # Against Solo/Wenz/Geier, force the declarer to spend trump when
            # they are publicly known void in the led suit.
            declarer = contract.declarer
            if declarer is not None:
                forcing = tuple(
                    card
                    for card in legal_cards
                    if (
                        not is_trump(card, contract)
                        and card.suit in knowledge.voids.plain_suits[declarer]
                    )
                )
                if forcing:
                    return min(
                        forcing,
                        key=lambda card: self._card_cost(card, contract),
                    )

        return None

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

        trick_points_now = sum(
            card_points(play.card) for play in observation.current_trick
        )
        score = score_situation(observation)

        if winning_cards and score is not None:
            max_added = max(card_points(card) for card in winning_cards)
            projected = trick_points_now + max_added
            high_priority = score.trick_clinches_game(
                projected
            ) or score.trick_avoids_schneider(projected)
            if high_priority:
                return self._safest_winner(
                    observation,
                    winning_cards,
                    knowledge,
                    lead_card,
                    players_behind,
                )

        teammate_probability = self._teammate_probability(
            observation,
            knowledge,
            current_winner.player,
        )

        # If a teammate behind us has a good chance to beat the current winner,
        # avoid wasting a stronger winning card ourselves unless score pressure
        # makes the trick urgent.
        teammate_behind_can_win = self._teammate_behind_can_still_win(
            observation,
            knowledge,
            current_winner.card,
            lead_card,
            players_behind,
        )
        urgent = False
        if score is not None:
            maximum_trick = trick_points_now + max(
                (card_points(card) for card in legal_cards),
                default=0,
            )
            urgent = score.trick_clinches_game(
                maximum_trick
            ) or score.trick_avoids_schneider(maximum_trick)

        if (
            teammate_probability < self.config.teammate_confidence
            and teammate_behind_can_win
            and losing_cards
            and not urgent
        ):
            return self._best_sacrifice(observation, losing_cards, knowledge)

        if teammate_probability >= self.config.teammate_confidence and losing_cards:
            if not players_behind:
                return max(
                    losing_cards,
                    key=lambda card: (
                        card_points(card),
                        -self._card_cost(card, contract),
                    ),
                )

            if teammate_behind_can_win and not urgent:
                return self._best_sacrifice(observation, losing_cards, knowledge)

        if winning_cards:
            safest = self._safest_winner(
                observation,
                winning_cards,
                knowledge,
                lead_card,
                players_behind,
            )
            risk = knowledge.probability_overtaken(
                candidate=safest,
                lead_card=lead_card,
                players_behind=players_behind,
            )

            projected = trick_points_now + card_points(safest)
            preserve_schneider = False
            if score is not None:
                preserve_schneider = score.losing_trick_breaks_schneider(projected)

            should_take = (
                not players_behind
                or projected >= self.config.valuable_trick_points
                or preserve_schneider
                or observation.trick_number >= 7
                or risk <= self.config.low_overtake_risk
            )

            if should_take:
                return safest

        if losing_cards:
            return self._best_sacrifice(observation, losing_cards, knowledge)

        return self._safest_winner(
            observation,
            legal_cards,
            knowledge,
            lead_card,
            players_behind,
        )

    def _safe_aces(
        self,
        observation: PlayerObservation,
        legal_cards: tuple[Card, ...],
        knowledge: PublicCardKnowledge,
    ) -> tuple[Card, ...]:
        opponents = self._possible_opponents(observation, knowledge)
        return tuple(
            card
            for card in legal_cards
            if (
                card.rank is Rank.ACE
                and not is_trump(card, observation.contract)
                and knowledge.ace_safety_probability(card, opponents)
                >= 1.0 - self.config.safe_ruff_risk
            )
        )

    def _safest_winner(
        self,
        observation: PlayerObservation,
        winning_cards: tuple[Card, ...],
        knowledge: PublicCardKnowledge,
        lead_card: Card,
        players_behind: tuple[int, ...],
    ) -> Card:
        contract = observation.contract

        def key(card: Card) -> tuple[float, float]:
            risk = knowledge.probability_overtaken(
                candidate=card,
                lead_card=lead_card,
                players_behind=players_behind,
            )
            return risk, self._card_cost(card, contract)

        return min(winning_cards, key=key)

    def _teammate_behind_can_still_win(
        self,
        observation: PlayerObservation,
        knowledge: PublicCardKnowledge,
        current_winner: Card,
        lead_card: Card,
        players_behind: tuple[int, ...],
    ) -> bool:
        teammate_players = tuple(
            player
            for player in players_behind
            if self._teammate_probability(observation, knowledge, player)
            >= self.config.teammate_confidence
        )
        if not teammate_players:
            return False

        beating_cards = tuple(
            card
            for card in knowledge.unseen_cards
            if card_beats(
                challenger=card,
                current_winner=current_winner,
                lead_card=lead_card,
                contract=observation.contract,
            )
        )
        probability = knowledge.probability_any_card_with_players(
            beating_cards,
            teammate_players,
        )
        return probability >= 0.55

    def _best_sacrifice(
        self,
        observation: PlayerObservation,
        cards: tuple[Card, ...],
        knowledge: PublicCardKnowledge,
    ) -> Card:
        """
        Deliberately lose cheaply while improving future hand shape.

        Prefer creating a void when trumps remain, preserve master trumps and
        safe Aces, and avoid discarding high point cards unless a teammate has
        the trick secured.
        """
        contract = observation.contract
        opponents = self._possible_opponents(observation, knowledge)

        def cost(card: Card) -> float:
            base = self._discard_cost(card, contract)

            if knowledge.is_definite_master_trump(card):
                base += 35.0

            if (
                card.rank is Rank.ACE
                and not is_trump(card, contract)
                and knowledge.ace_safety_probability(card, opponents) >= 0.70
            ):
                base += 25.0

            if not is_trump(card, contract):
                suit_length = self._own_plain_suit_length(observation, card.suit)
                if suit_length == 1 and knowledge.own_trumps:
                    base -= 8.0

            return base

        return min(cards, key=cost)
