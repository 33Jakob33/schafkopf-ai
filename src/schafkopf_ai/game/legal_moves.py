from collections.abc import Sequence
from dataclasses import dataclass

from .card import Card, Rank
from .game_contract import GameContract
from .game_type import GameType
from .trick import Trick
from .trump import is_trump


@dataclass(frozen=True, slots=True)
class LegalMoveContext:
    """
    Additional game state relevant for move legality.

    called_ace_released becomes True after the called player
    successfully runs away (Davonlaufen).
    """

    called_ace_released: bool = False


def _matching_lead_cards(
    hand: Sequence[Card],
    lead_card: Card,
    contract: GameContract,
) -> list[Card]:
    """
    Return cards that satisfy the normal Bedienpflicht.

    If trump was led:
        Return all trumps.

    If a non-trump suit was led:
        Return all non-trump cards of that suit.
    """
    if is_trump(lead_card, contract):
        return [card for card in hand if is_trump(card, contract)]

    return [
        card
        for card in hand
        if (not is_trump(card, contract) and card.suit is lead_card.suit)
    ]


def _called_ace(contract: GameContract) -> Card | None:
    """Return the called Ace for a Sauspiel."""
    if contract.game_type is not GameType.SAUSPIEL:
        return None

    if contract.called_suit is None:
        raise ValueError("A Sauspiel requires a called suit.")

    return Card(
        suit=contract.called_suit,
        rank=Rank.ACE,
    )


def _called_suit_cards(
    hand: Sequence[Card],
    contract: GameContract,
) -> list[Card]:
    """
    Return non-trump cards belonging to the called suit.
    """
    if contract.called_suit is None:
        return []

    return [
        card
        for card in hand
        if (card.suit is contract.called_suit and not is_trump(card, contract))
    ]


def _apply_sauspiel_rules(
    hand: Sequence[Card],
    legal: list[Card],
    trick: Trick,
    contract: GameContract,
    context: LegalMoveContext,
) -> list[Card]:
    """
    Apply the special called-Ace rules of a Sauspiel.

    Before Davonlaufen:
    - The called Ace must be played when its suit is led.
    - The Ace cannot normally be discarded on another suit.
    - The called player may lead the Ace at any time.
    - The called player may lead below the Ace when holding at least
      four cards of the called suit, including the Ace.

    After Davonlaufen:
    - The called Ace behaves like an ordinary suit card.
    """
    called_ace = _called_ace(contract)

    if called_ace is None:
        return legal

    if called_ace not in hand:
        # This player is not the called player, or the Ace has
        # already been played.
        return legal

    if context.called_ace_released:
        return legal

    # ------------------------------------------------------------
    # Called player is leading the trick.
    # ------------------------------------------------------------
    if not trick.plays:
        called_suit_cards = _called_suit_cards(
            hand,
            contract,
        )

        can_run_away = len(called_suit_cards) >= 4

        if can_run_away:
            return legal

        # The player may lead the called Ace itself, but may not
        # lead another card of the called suit.
        return [
            card
            for card in legal
            if (
                card == called_ace
                or card.suit is not contract.called_suit
                or is_trump(card, contract)
            )
        ]

    lead_card = trick.plays[0].card

    # ------------------------------------------------------------
    # Someone has led the called suit.
    # The called player must play the called Ace.
    # ------------------------------------------------------------
    called_suit_was_led = (
        not is_trump(lead_card, contract) and lead_card.suit is contract.called_suit
    )

    if called_suit_was_led:
        return [called_ace]

    # ------------------------------------------------------------
    # On another suit/trump, the called Ace may not be discarded
    # before the final trick.
    #
    # If it is the only card left in hand, we are in trick eight.
    # ------------------------------------------------------------
    if len(hand) == 1:
        return legal

    without_called_ace = [card for card in legal if card != called_ace]

    # In a valid Schafkopf state there should always be another
    # legal card before the final trick. Keep this explicit so a
    # corrupt game state fails rather than silently allowing an
    # illegal move.
    if not without_called_ace:
        raise ValueError(
            "Invalid game state: the called Ace would be the only "
            "legal move before the final trick."
        )

    return without_called_ace


def legal_moves(
    hand: Sequence[Card],
    trick: Trick,
    contract: GameContract,
    context: LegalMoveContext | None = None,
) -> tuple[Card, ...]:
    """
    Return all cards the current player may legally play.

    The original hand order is preserved.
    """
    if not hand:
        return ()

    if trick.is_complete:
        raise ValueError("Cannot determine legal moves for a completed trick.")

    if context is None:
        context = LegalMoveContext()

    # ------------------------------------------------------------
    # No lead card yet: normally every card can be played.
    # ------------------------------------------------------------
    if not trick.plays:
        legal = list(hand)

    else:
        lead_card = trick.plays[0].card

        matching_cards = _matching_lead_cards(
            hand,
            lead_card,
            contract,
        )

        if matching_cards:
            legal = matching_cards
        else:
            # Cannot follow suit/trump -> any card may be played.
            legal = list(hand)

    # ------------------------------------------------------------
    # Sauspiel has additional rules around the called Ace.
    # ------------------------------------------------------------
    if contract.game_type is GameType.SAUSPIEL:
        legal = _apply_sauspiel_rules(
            hand=hand,
            legal=legal,
            trick=trick,
            contract=contract,
            context=context,
        )

    return tuple(legal)
