from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from schafkopf_ai.game.card import Card, Rank, Suit
from schafkopf_ai.game.game_contract import GameContract
from schafkopf_ai.game.game_type import GameType
from schafkopf_ai.game.trump import is_trump, trump_order


@dataclass(frozen=True, slots=True)
class BiddingHeuristicConfig:
    """Thresholds for normalized contract confidence in the heuristic bidder."""

    sauspiel_min_confidence: float = 0.52
    wenz_min_confidence: float = 0.58
    geier_min_confidence: float = 0.58
    solo_min_confidence: float = 0.66
    solo_margin: float = 0.10


DEFAULT_BIDDING_HEURISTIC_CONFIG = BiddingHeuristicConfig()


@dataclass(frozen=True, slots=True)
class ContractEvaluation:
    """Normalized assessment of one exact legal contract."""

    contract: GameContract
    confidence: float
    eligible: bool
    components: tuple[tuple[str, float], ...]


def evaluate_contract(
    hand: tuple[Card, ...],
    contract: GameContract,
    *,
    config: BiddingHeuristicConfig = DEFAULT_BIDDING_HEURISTIC_CONFIG,
) -> ContractEvaluation:
    """Evaluate an exact contract on a comparable 0.0..1.0 confidence scale."""
    if len(hand) != 8:
        raise ValueError("A Schafkopf bidding hand must contain exactly eight cards.")

    if contract.game_type is GameType.SAUSPIEL:
        return _evaluate_sauspiel(hand, contract, config)
    if contract.game_type is GameType.WENZ:
        return _evaluate_wenz_like(hand, contract, config, trump_rank=Rank.UNTER)
    if contract.game_type is GameType.GEIER:
        return _evaluate_wenz_like(hand, contract, config, trump_rank=Rank.OBER)
    if contract.game_type is GameType.SOLO:
        return _evaluate_solo(hand, contract, config)

    raise ValueError(f"{contract.game_type.value} cannot be actively bid.")


def evaluate_contracts(
    hand: tuple[Card, ...],
    contracts: Iterable[GameContract],
    *,
    config: BiddingHeuristicConfig = DEFAULT_BIDDING_HEURISTIC_CONFIG,
) -> tuple[ContractEvaluation, ...]:
    return tuple(
        evaluate_contract(hand, contract, config=config) for contract in contracts
    )


def preferred_contract(
    hand: tuple[Card, ...],
    contracts: Iterable[GameContract],
    *,
    config: BiddingHeuristicConfig = DEFAULT_BIDDING_HEURISTIC_CONFIG,
) -> ContractEvaluation | None:
    """
    Return the contract the heuristic is actually willing to bid.

    Solo receives an explicit risk premium: if a safer eligible contract exists,
    the Solo must exceed its normalized confidence by `solo_margin`. This avoids
    upgrading a good Sauspiel/Wenz into a merely marginal Solo.
    """
    evaluations = evaluate_contracts(hand, contracts, config=config)
    eligible = [evaluation for evaluation in evaluations if evaluation.eligible]
    if not eligible:
        return None

    non_solos = [
        evaluation
        for evaluation in eligible
        if evaluation.contract.game_type is not GameType.SOLO
    ]
    solos = [
        evaluation
        for evaluation in eligible
        if evaluation.contract.game_type is GameType.SOLO
    ]

    best_non_solo = max(
        non_solos,
        key=lambda evaluation: evaluation.confidence,
        default=None,
    )
    best_solo = max(
        solos,
        key=lambda evaluation: evaluation.confidence,
        default=None,
    )

    if best_solo is not None:
        if best_non_solo is None:
            return best_solo

        if best_solo.confidence >= best_non_solo.confidence + config.solo_margin:
            return best_solo

    return best_non_solo


def _evaluate_solo(
    hand: tuple[Card, ...],
    contract: GameContract,
    config: BiddingHeuristicConfig,
) -> ContractEvaluation:
    trumps = tuple(card for card in hand if is_trump(card, contract))
    non_trumps = tuple(card for card in hand if not is_trump(card, contract))
    ordering = trump_order(contract)
    held = set(hand)

    trump_count = len(trumps)
    top_four = ordering[:4]
    top_four_count = sum(card in held for card in top_four)
    has_top_trump = ordering[0] in held
    has_second_trump = ordering[1] in held

    side_aces = sum(card.rank is Rank.ACE for card in non_trumps)
    side_tens = sum(card.rank is Rank.TEN for card in non_trumps)
    losers = _estimated_plain_losers(non_trumps)

    trump_length = _clamp((trump_count - 3) / 4)
    top_control = _weighted_top_control(held, ordering, limit=6)
    side_control = _clamp((side_aces + 0.35 * side_tens) / 2.0)
    loser_control = _clamp(1.0 - losers / 4.0)

    confidence = _clamp(
        0.40 * trump_length
        + 0.30 * top_control
        + 0.18 * side_control
        + 0.12 * loser_control
    )

    enough_trumps = trump_count >= 5
    if has_top_trump:
        enough_top_control = top_four_count >= 1
    else:
        # Missing Eichel Ober makes a Solo materially riskier. Compensate with
        # extra length plus at least two of the remaining high trumps.
        enough_top_control = trump_count >= 6 and top_four_count >= 2

    if not has_top_trump and not has_second_trump:
        enough_top_control = trump_count >= 7 and top_four_count >= 2

    enough_side_control = (
        side_aces >= 1 or trump_count >= 7 or (trump_count >= 6 and losers <= 2.0)
    )
    manageable_losers = losers <= 3.0

    eligible = (
        enough_trumps
        and enough_top_control
        and enough_side_control
        and manageable_losers
        and confidence >= config.solo_min_confidence
    )

    return ContractEvaluation(
        contract=contract,
        confidence=confidence,
        eligible=eligible,
        components=(
            ("trump_length", trump_length),
            ("top_control", top_control),
            ("side_control", side_control),
            ("loser_control", loser_control),
        ),
    )


def _evaluate_wenz_like(
    hand: tuple[Card, ...],
    contract: GameContract,
    config: BiddingHeuristicConfig,
    *,
    trump_rank: Rank,
) -> ContractEvaluation:
    trumps = tuple(card for card in hand if card.rank is trump_rank)
    non_trumps = tuple(card for card in hand if card.rank is not trump_rank)

    quality_weights = {
        Suit.EICHEL: 1.00,
        Suit.GRAS: 0.78,
        Suit.HERZ: 0.55,
        Suit.SCHELLEN: 0.35,
    }
    trump_quality = _clamp(sum(quality_weights[card.suit] for card in trumps) / 1.60)
    trump_length = len(trumps) / 4.0

    side_aces = sum(card.rank is Rank.ACE for card in non_trumps)
    side_tens = sum(card.rank is Rank.TEN for card in non_trumps)
    side_control = _clamp((side_aces + 0.30 * side_tens) / 2.5)

    plain_suit_lengths = _plain_suit_lengths(non_trumps)
    short_suits = sum(length <= 1 for length in plain_suit_lengths.values())
    shape = _clamp(short_suits / 3.0)

    confidence = _clamp(
        0.42 * trump_quality + 0.20 * trump_length + 0.25 * side_control + 0.13 * shape
    )

    trump_count = len(trumps)
    has_eichel = Card(Suit.EICHEL, trump_rank) in hand
    has_gras = Card(Suit.GRAS, trump_rank) in hand

    enough_trumps = trump_count >= 2
    if trump_count == 2:
        enough_quality = has_eichel or has_gras
    else:
        enough_quality = has_eichel or has_gras or trump_count == 4

    min_confidence = (
        config.wenz_min_confidence
        if contract.game_type is GameType.WENZ
        else config.geier_min_confidence
    )

    eligible = enough_trumps and enough_quality and confidence >= min_confidence

    return ContractEvaluation(
        contract=contract,
        confidence=confidence,
        eligible=eligible,
        components=(
            ("trump_quality", trump_quality),
            ("trump_length", trump_length),
            ("side_control", side_control),
            ("shape", shape),
        ),
    )


def _evaluate_sauspiel(
    hand: tuple[Card, ...],
    contract: GameContract,
    config: BiddingHeuristicConfig,
) -> ContractEvaluation:
    called_suit = contract.called_suit
    if called_suit is None:
        raise ValueError("Sauspiel evaluation requires a called suit.")

    trumps = tuple(card for card in hand if is_trump(card, contract))
    non_trumps = tuple(card for card in hand if not is_trump(card, contract))
    ordering = trump_order(contract)
    held = set(hand)

    trump_count = len(trumps)
    trump_length = _clamp((trump_count - 2) / 4.0)
    top_control = _weighted_top_control(held, ordering, limit=8)

    side_aces = sum(
        card.rank is Rank.ACE and card.suit is not called_suit for card in non_trumps
    )
    side_control = _clamp(side_aces / 2.0)

    called_cards = tuple(card for card in non_trumps if card.suit is called_suit)
    called_count = len(called_cards)
    count_quality = {
        1: 1.00,
        2: 0.75,
        3: 0.40,
        4: 0.15,
    }.get(called_count, 0.0)

    called_rank_quality = {
        Rank.SEVEN: 1.00,
        Rank.EIGHT: 0.95,
        Rank.NINE: 0.90,
        Rank.KING: 0.65,
        Rank.TEN: 0.35,
        Rank.ACE: 0.0,
        Rank.UNTER: 0.0,
        Rank.OBER: 0.0,
    }
    best_call_structure = max(
        (called_rank_quality[card.rank] for card in called_cards),
        default=0.0,
    )
    called_structure = 0.65 * count_quality + 0.35 * best_call_structure

    lengths = _plain_suit_lengths(non_trumps)
    useful_short_suits = sum(
        length <= 1 for suit, length in lengths.items() if suit is not called_suit
    )
    shape = _clamp(useful_short_suits / 2.0)

    confidence = _clamp(
        0.30 * trump_length
        + 0.25 * top_control
        + 0.18 * side_control
        + 0.19 * called_structure
        + 0.08 * shape
    )

    enough_trump_base = trump_count >= 3 or (
        trump_count >= 2 and top_control >= 0.60 and side_aces >= 1
    )
    callable_shape = 1 <= called_count <= 3
    has_support = side_aces >= 1 or top_control >= 0.45 or trump_count >= 4

    eligible = (
        enough_trump_base
        and callable_shape
        and has_support
        and confidence >= config.sauspiel_min_confidence
    )

    return ContractEvaluation(
        contract=contract,
        confidence=confidence,
        eligible=eligible,
        components=(
            ("trump_length", trump_length),
            ("top_control", top_control),
            ("side_control", side_control),
            ("called_structure", called_structure),
            ("shape", shape),
        ),
    )


def _weighted_top_control(
    held: set[Card],
    ordering: tuple[Card, ...],
    *,
    limit: int,
) -> float:
    weights = (1.00, 0.85, 0.72, 0.60, 0.50, 0.42, 0.35, 0.30)
    selected = ordering[:limit]
    selected_weights = weights[: len(selected)]
    maximum = sum(selected_weights[: min(3, len(selected_weights))])
    if maximum == 0:
        return 0.0

    held_weight = sum(
        weight
        for card, weight in zip(selected, selected_weights, strict=True)
        if card in held
    )
    return _clamp(held_weight / maximum)


def _estimated_plain_losers(cards: tuple[Card, ...]) -> float:
    by_suit: dict[Suit, tuple[Card, ...]] = {
        suit: tuple(card for card in cards if card.suit is suit) for suit in Suit
    }

    losers = 0.0
    for suit_cards in by_suit.values():
        if not suit_cards:
            continue

        has_ace = any(card.rank is Rank.ACE for card in suit_cards)
        for card in suit_cards:
            if card.rank is Rank.ACE:
                continue
            if has_ace and card.rank is Rank.TEN:
                losers += 0.20
            elif has_ace and card.rank is Rank.KING:
                losers += 0.45
            elif has_ace:
                losers += 0.65
            elif card.rank is Rank.TEN:
                losers += 0.70
            else:
                losers += 1.0

    return losers


def _plain_suit_lengths(cards: tuple[Card, ...]) -> dict[Suit, int]:
    return {suit: sum(card.suit is suit for card in cards) for suit in Suit}


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
