from __future__ import annotations

import random
from dataclasses import dataclass

from schafkopf_ai.agents.heuristic_knowledge import infer_voids
from schafkopf_ai.game.card import Card, Rank, Suit
from schafkopf_ai.game.deck import Deck
from schafkopf_ai.game.game_state import GameState
from schafkopf_ai.game.game_type import GameType
from schafkopf_ai.game.observation import PlayerObservation
from schafkopf_ai.game.trump import is_trump


class HiddenStateSamplingError(RuntimeError):
    """Raised when no hidden deal consistent with public information is found."""


@dataclass(frozen=True, slots=True)
class Determinization:
    """
    One complete hidden deal consistent with a player's public observation.

    The initial hands stay inside the search layer. Agents only receive them
    indirectly through Monte-Carlo outcomes and never through PlayerObservation.
    """

    initial_hands: tuple[tuple[Card, ...], ...]

    def build_state(self, observation: PlayerObservation) -> GameState:
        """Reconstruct the exact public position for this hidden deal."""
        return reconstruct_state(
            observation=observation,
            initial_hands=self.initial_hands,
        )


def _public_plays(observation: PlayerObservation) -> tuple[tuple[int, Card], ...]:
    plays: list[tuple[int, Card]] = []
    for trick in observation.completed_tricks:
        plays.extend((play.player, play.card) for play in trick)
    plays.extend((play.player, play.card) for play in observation.current_trick)
    return tuple(plays)


def _starting_player(observation: PlayerObservation) -> int:
    if observation.completed_tricks:
        return observation.completed_tricks[0][0].player
    if observation.current_trick:
        return observation.current_trick[0].player
    if observation.current_player is not None:
        return observation.current_player
    raise ValueError("Cannot reconstruct a completed observation without play history.")


def reconstruct_state(
    *,
    observation: PlayerObservation,
    initial_hands: tuple[tuple[Card, ...], ...],
) -> GameState:
    """
    Rebuild a mutable GameState and replay all publicly observed cards.

    Replay uses the normal legality engine. Therefore a determinization that
    would make any historical public action illegal is rejected by construction.
    """
    state = GameState.from_hands(
        hands=initial_hands,
        contract=observation.contract,
        starting_player=_starting_player(observation),
    )

    for player, card in _public_plays(observation):
        state.play_card(player, card)

    rebuilt = state.observation_for(observation.player_index)
    if rebuilt != observation:
        raise ValueError("Reconstructed state does not match the source observation.")

    return state


class HiddenStateSampler:
    """
    Sample complete deals consistent with one player's information set.

    Sampling starts from uniformly shuffled allocations of unseen cards into
    opponents' remaining hand slots and rejects allocations that violate
    proven voids, Sauspiel partner knowledge, or historical legality. The real
    hidden hands from the environment are never inspected.
    """

    def __init__(
        self,
        *,
        rng: random.Random | None = None,
        max_attempts_per_sample: int = 5_000,
    ) -> None:
        if max_attempts_per_sample <= 0:
            raise ValueError("max_attempts_per_sample must be positive.")
        self.rng = random.Random() if rng is None else rng
        self.max_attempts_per_sample = max_attempts_per_sample
        self._all_cards = Deck().cards
        self._card_order = {card: index for index, card in enumerate(self._all_cards)}

    def sample(self, observation: PlayerObservation) -> Determinization:
        """Return one determinization consistent with the observation."""
        played_by_player: list[list[Card]] = [[] for _ in range(4)]
        played_cards: set[Card] = set()

        for player, card in _public_plays(observation):
            if card in played_cards:
                raise ValueError("Observation contains a duplicated played card.")
            played_cards.add(card)
            played_by_player[player].append(card)

        own_cards = set(observation.hand)
        if own_cards & played_cards:
            raise ValueError("Observation hand contains an already played card.")

        remaining_counts = [
            GameState.CARDS_PER_PLAYER - len(played_by_player[player])
            for player in range(4)
        ]
        if remaining_counts[observation.player_index] != len(observation.hand):
            raise ValueError("Observation hand size is inconsistent with play history.")

        unseen = [
            card
            for card in self._all_cards
            if card not in played_cards and card not in own_cards
        ]
        opponent_slots = sum(
            remaining_counts[player]
            for player in range(4)
            if player != observation.player_index
        )
        if len(unseen) != opponent_slots:
            raise ValueError("Observation does not account for exactly 32 cards.")

        voids = infer_voids(observation)

        for _ in range(self.max_attempts_per_sample):
            current_hands = self._constraint_current_hands(
                observation=observation,
                unseen=unseen,
                remaining_counts=remaining_counts,
                plain_suits=voids.plain_suits,
                trump_void_players=voids.trump_void_players,
            )
            if current_hands is None:
                break

            initial_hands = tuple(
                tuple(current_hands[player]) + tuple(played_by_player[player])
                for player in range(4)
            )
            if not self._respects_contract_legality(
                observation=observation,
                initial_hands=initial_hands,
            ):
                continue

            determinization = Determinization(initial_hands=initial_hands)

            try:
                determinization.build_state(observation)
            except (ValueError, RuntimeError):
                continue

            return determinization

        raise HiddenStateSamplingError(
            "Could not sample a hidden deal consistent with the observation. "
            f"game={observation.contract.game_type.value}, "
            f"trick={observation.trick_number}, "
            f"played={len(observation.cards_played)}, "
            f"known_called_ace_player={observation.known_called_ace_player}."
        )

    def sample_many(
        self,
        observation: PlayerObservation,
        count: int,
    ) -> tuple[Determinization, ...]:
        if count <= 0:
            raise ValueError("count must be positive.")
        return tuple(self.sample(observation) for _ in range(count))

    def _constraint_current_hands(
        self,
        *,
        observation: PlayerObservation,
        unseen: list[Card],
        remaining_counts: list[int],
        plain_suits: tuple[frozenset[Suit], ...],
        trump_void_players: frozenset[int],
    ) -> list[list[Card]] | None:
        """
        Fill opponent hand slots with randomized backtracking.

        Candidate holders are constrained before assignment, so late-game
        determinizations do not rely on repeatedly shuffling into a tiny valid
        subset of all deals.
        """
        focal_player = observation.player_index
        opponents = tuple(player for player in range(4) if player != focal_player)

        candidate_holders = {
            card: self._candidate_holders(
                card=card,
                observation=observation,
                opponents=opponents,
                plain_suits=plain_suits,
                trump_void_players=trump_void_players,
            )
            for card in unseen
        }
        if any(not holders for holders in candidate_holders.values()):
            return None

        random_priority = {card: self.rng.random() for card in unseen}
        ordered_cards = sorted(
            unseen,
            key=lambda card: (
                len(candidate_holders[card]),
                random_priority[card],
            ),
        )

        current_hands: list[list[Card]] = [[] for _ in range(4)]
        current_hands[focal_player] = list(observation.hand)
        slots = {player: remaining_counts[player] for player in opponents}

        def can_still_fill(start_index: int) -> bool:
            remaining_cards = ordered_cards[start_index:]
            for player in opponents:
                if slots[player] <= 0:
                    continue
                compatible = sum(
                    player in candidate_holders[card] for card in remaining_cards
                )
                if compatible < slots[player]:
                    return False
            return True

        def assign(index: int) -> bool:
            if index == len(ordered_cards):
                return all(slots[player] == 0 for player in opponents)

            card = ordered_cards[index]
            holders = [
                player for player in candidate_holders[card] if slots[player] > 0
            ]
            self.rng.shuffle(holders)

            for player in holders:
                current_hands[player].append(card)
                slots[player] -= 1

                if can_still_fill(index + 1) and assign(index + 1):
                    return True

                slots[player] += 1
                current_hands[player].pop()

            return False

        if not can_still_fill(0) or not assign(0):
            return None

        for player in opponents:
            current_hands[player].sort(key=self._card_order.__getitem__)

        return current_hands

    @staticmethod
    def _candidate_holders(
        *,
        card: Card,
        observation: PlayerObservation,
        opponents: tuple[int, ...],
        plain_suits: tuple[frozenset[Suit], ...],
        trump_void_players: frozenset[int],
    ) -> tuple[int, ...]:
        contract = observation.contract
        candidates: list[int] = []

        for player in opponents:
            if is_trump(card, contract):
                if player in trump_void_players:
                    continue
            elif card.suit in plain_suits[player]:
                continue

            if (
                contract.game_type is GameType.SAUSPIEL
                and contract.called_suit is not None
                and card == Card(contract.called_suit, Rank.ACE)
            ):
                if player == contract.declarer:
                    continue
                known_holder = observation.known_called_ace_player
                if known_holder is not None and player != known_holder:
                    continue

            candidates.append(player)

        return tuple(candidates)

    @staticmethod
    def _respects_contract_legality(
        *,
        observation: PlayerObservation,
        initial_hands: tuple[tuple[Card, ...], ...],
    ) -> bool:
        contract = observation.contract
        if contract.game_type is not GameType.SAUSPIEL:
            return True
        if contract.declarer is None or contract.called_suit is None:
            return False

        declarer_hand = initial_hands[contract.declarer]
        called_ace = Card(contract.called_suit, Rank.ACE)
        if called_ace in declarer_hand:
            return False

        return any(
            card.suit is contract.called_suit
            and card.rank not in {Rank.OBER, Rank.UNTER}
            for card in declarer_hand
        )
