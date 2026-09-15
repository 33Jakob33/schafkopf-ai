from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import prod

from schafkopf_ai.game.card import Card, Rank, Suit
from schafkopf_ai.game.game_type import GameType
from schafkopf_ai.game.observation import PlayerObservation
from schafkopf_ai.game.trick import card_beats, plain_card_strength
from schafkopf_ai.game.trump import is_trump, trump_strength

PLAYER_COUNT = 4
CARDS_PER_PLAYER = 8
ALL_CARDS: tuple[Card, ...] = tuple(
    Card(suit=suit, rank=rank) for suit in Suit for rank in Rank
)


@dataclass(frozen=True, slots=True)
class InferredVoids:
    """Suit and trump voids that are provable from public trick history."""

    plain_suits: tuple[frozenset[Suit], ...]
    trump_void_players: frozenset[int]


@dataclass(frozen=True, slots=True)
class PublicCardKnowledge:
    """
    Public-information card model from one player's perspective.

    `unseen_cards` never contains cards in the observing player's hand or
    cards that have already been played. Holder probabilities are approximate:
    they distribute each unseen card across players who could still hold it,
    weighted by those players' remaining hand sizes and constrained by proven
    suit/trump voids.

    Exact statements are kept separate from probabilistic ones. For example,
    `is_definite_master_trump` is exact given the visible/own cards, while
    `ace_safety_probability` is an approximate hidden-hand estimate.
    """

    observation: PlayerObservation
    voids: InferredVoids
    unseen_cards: frozenset[Card]
    remaining_cards_by_player: tuple[int, int, int, int]

    @classmethod
    def from_observation(
        cls,
        observation: PlayerObservation,
    ) -> PublicCardKnowledge:
        played = frozenset(observation.cards_played)
        hand = frozenset(observation.hand)

        unseen_cards = frozenset(
            card for card in ALL_CARDS if card not in played and card not in hand
        )

        plays_by_player = [0] * PLAYER_COUNT
        for trick in observation.completed_tricks:
            for play in trick:
                plays_by_player[play.player] += 1
        for play in observation.current_trick:
            plays_by_player[play.player] += 1

        remaining = tuple(
            CARDS_PER_PLAYER - plays_by_player[player] for player in range(PLAYER_COUNT)
        )

        return cls(
            observation=observation,
            voids=infer_voids(observation),
            unseen_cards=unseen_cards,
            remaining_cards_by_player=(
                remaining[0],
                remaining[1],
                remaining[2],
                remaining[3],
            ),
        )

    @property
    def unseen_trumps(self) -> tuple[Card, ...]:
        contract = self.observation.contract
        return tuple(card for card in self.unseen_cards if is_trump(card, contract))

    @property
    def own_trumps(self) -> tuple[Card, ...]:
        contract = self.observation.contract
        return tuple(card for card in self.observation.hand if is_trump(card, contract))

    @property
    def remaining_trump_count(self) -> int:
        """Return trumps that remain outside the observer's own hand."""
        return len(self.unseen_trumps)

    def unseen_higher_trumps(self, card: Card) -> tuple[Card, ...]:
        contract = self.observation.contract
        if not is_trump(card, contract):
            raise ValueError("Higher-trump lookup requires a trump card.")

        strength = trump_strength(card, contract)
        return tuple(
            other
            for other in self.unseen_trumps
            if trump_strength(other, contract) > strength
        )

    def unseen_higher_plain_cards(self, card: Card) -> tuple[Card, ...]:
        contract = self.observation.contract
        if is_trump(card, contract):
            raise ValueError("Plain-card lookup requires a non-trump card.")

        strength = plain_card_strength(card, contract)
        return tuple(
            other
            for other in self.unseen_cards
            if (
                other.suit is card.suit
                and not is_trump(other, contract)
                and plain_card_strength(other, contract) > strength
            )
        )

    def is_master_card(self, card: Card) -> bool:
        """Return whether no unseen card can beat this card in its category."""
        contract = self.observation.contract

        if is_trump(card, contract):
            return not self.unseen_higher_trumps(card)

        return not self.unseen_higher_plain_cards(card)

    def is_definite_master_trump(self, card: Card) -> bool:
        """Return whether no card outside our hand can overtrump this trump."""
        if not is_trump(card, self.observation.contract):
            return False
        return not self.unseen_higher_trumps(card)

    def possible_holders(self, card: Card) -> tuple[int, ...]:
        """Return players who could hold a particular unseen card."""
        if card not in self.unseen_cards:
            return ()

        observation = self.observation
        contract = observation.contract

        candidates: list[int] = []
        for player in range(PLAYER_COUNT):
            if player == observation.player_index:
                continue

            if self.remaining_cards_by_player[player] <= 0:
                continue

            if is_trump(card, contract):
                if player in self.voids.trump_void_players:
                    continue
            elif card.suit in self.voids.plain_suits[player]:
                continue

            if (
                contract.game_type is GameType.SAUSPIEL
                and card.rank is Rank.ACE
                and card.suit is contract.called_suit
                and contract.declarer == player
            ):
                continue

            candidates.append(player)

        if candidates:
            return tuple(candidates)

        # Partially constructed test observations can be inconsistent. Keep
        # probability helpers robust rather than inventing certainty from that.
        return tuple(
            player
            for player in range(PLAYER_COUNT)
            if (
                player != observation.player_index
                and self.remaining_cards_by_player[player] > 0
            )
        )

    def holder_probabilities(self, card: Card) -> tuple[float, float, float, float]:
        """Approximate P(player holds card) for one unseen card."""
        candidates = self.possible_holders(card)
        if not candidates:
            return (0.0, 0.0, 0.0, 0.0)

        total_slots = sum(
            self.remaining_cards_by_player[player] for player in candidates
        )
        if total_slots <= 0:
            return (0.0, 0.0, 0.0, 0.0)

        probabilities = [0.0] * PLAYER_COUNT
        for player in candidates:
            probabilities[player] = self.remaining_cards_by_player[player] / total_slots

        return (
            probabilities[0],
            probabilities[1],
            probabilities[2],
            probabilities[3],
        )

    def probability_card_with_players(
        self,
        card: Card,
        players: Iterable[int],
    ) -> float:
        selected = frozenset(players)
        probabilities = self.holder_probabilities(card)
        return sum(
            probabilities[player] for player in selected if 0 <= player < PLAYER_COUNT
        )

    def probability_any_card_with_players(
        self,
        cards: Iterable[Card],
        players: Iterable[int],
    ) -> float:
        """
        Approximate probability that selected players hold at least one card.

        Individual unseen-card locations are treated as independent. This is a
        deliberately lightweight approximation, not a full hand enumerator.
        """
        selected = frozenset(players)
        card_probabilities = [
            self.probability_card_with_players(card, selected)
            for card in cards
            if card in self.unseen_cards
        ]

        if not card_probabilities:
            return 0.0

        probability_none = prod(1.0 - min(1.0, p) for p in card_probabilities)
        return 1.0 - probability_none

    def probability_players_have_trump(self, players: Iterable[int]) -> float:
        return self.probability_any_card_with_players(
            self.unseen_trumps,
            players,
        )

    def probability_player_has_plain_suit(self, player: int, suit: Suit) -> float:
        """Approximate probability that one player can still follow a plain suit."""
        if player == self.observation.player_index:
            return float(
                any(
                    card.suit is suit
                    and not is_trump(card, self.observation.contract)
                    for card in self.observation.hand
                )
            )

        if suit in self.voids.plain_suits[player]:
            return 0.0

        remaining_suit_cards = tuple(
            card
            for card in self.unseen_cards
            if (
                card.suit is suit
                and not is_trump(card, self.observation.contract)
            )
        )
        return self.probability_any_card_with_players(
            remaining_suit_cards,
            {player},
        )

    def probability_player_can_trump_suit(self, player: int, suit: Suit) -> float:
        """Approximate probability that `player` can ruff a lead of `suit`."""
        if player == self.observation.player_index:
            has_plain = any(
                card.suit is suit
                and not is_trump(card, self.observation.contract)
                for card in self.observation.hand
            )
            has_trump = bool(self.own_trumps)
            return float(not has_plain and has_trump)

        if player in self.voids.trump_void_players:
            return 0.0

        if suit in self.voids.plain_suits[player]:
            probability_void = 1.0
        else:
            probability_void = 1.0 - self.probability_player_has_plain_suit(
                player,
                suit,
            )

        probability_trump = self.probability_players_have_trump({player})
        return min(1.0, probability_void * probability_trump)

    def probability_overtaken(
        self,
        *,
        candidate: Card,
        lead_card: Card,
        players_behind: Iterable[int],
    ) -> float:
        """Approximate probability that a player behind can beat candidate."""
        contract = self.observation.contract
        beating_cards = tuple(
            card
            for card in self.unseen_cards
            if card_beats(
                challenger=card,
                current_winner=candidate,
                lead_card=lead_card,
                contract=contract,
            )
        )

        return self.probability_any_card_with_players(
            beating_cards,
            players_behind,
        )

    def probability_suit_gets_trumped(
        self,
        suit: Suit,
        opponents: Iterable[int],
    ) -> float:
        """Estimate probability that at least one opponent can ruff this suit."""
        probabilities = [
            self.probability_player_can_trump_suit(player, suit)
            for player in frozenset(opponents)
            if player != self.observation.player_index
        ]
        if not probabilities:
            return 0.0

        probability_none = prod(1.0 - min(1.0, p) for p in probabilities)
        return 1.0 - probability_none

    def ace_safety_probability(
        self,
        card: Card,
        opponents: Iterable[int],
    ) -> float:
        """Estimate the chance a plain Ace survives without being ruffed."""
        if card.rank is not Rank.ACE or is_trump(card, self.observation.contract):
            raise ValueError("Ace safety requires a plain Ace.")

        return 1.0 - self.probability_suit_gets_trumped(card.suit, opponents)


def infer_voids(observation: PlayerObservation) -> InferredVoids:
    """
    Infer suit/trump voids solely from public play history.

    If a player legally fails to follow a led plain suit, that player is known
    to be void in that suit. If a player fails to play trump after trump is led,
    that player is known to be void in trump.
    """
    plain_voids: list[set[Suit]] = [set() for _ in range(PLAYER_COUNT)]
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
