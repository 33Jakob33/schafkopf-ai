from __future__ import annotations

from enum import IntEnum

from schafkopf_ai.game.game_type import GameType
from schafkopf_ai.game.observation import PlayerObservation


class CardPlayRole(IntEnum):
    """Strategic role used to route card-play decisions to specialist models."""

    SAUSPIEL_DECLARER = 0
    SAUSPIEL_PARTNER = 1
    SAUSPIEL_DEFENDER = 2
    SOLO_DECLARER = 3
    SOLO_DEFENDER = 4
    WENZ_DECLARER = 5
    WENZ_DEFENDER = 6
    RAMSCH = 7

    @property
    def key(self) -> str:
        return self.name.lower()


ROLE_BY_KEY: dict[str, CardPlayRole] = {role.key: role for role in CardPlayRole}
ROLE_KEYS: tuple[str, ...] = tuple(role.key for role in CardPlayRole)


def normalize_card_play_role(value: str | CardPlayRole) -> CardPlayRole:
    if isinstance(value, CardPlayRole):
        return value

    try:
        return ROLE_BY_KEY[value.lower()]
    except KeyError as exc:
        supported = ", ".join(ROLE_KEYS)
        raise ValueError(
            f"Unsupported card-play role {value!r}. Supported roles: {supported}."
        ) from exc


def infer_card_play_role(observation: PlayerObservation) -> CardPlayRole:
    """
    Return the observer's known strategic role for the current contract.

    Routing uses only information the observer is allowed to know. In a
    Sauspiel, the called-Ace holder knows their own role from the start, while
    other players do not learn the holder's identity before it becomes public.
    """
    contract = observation.contract
    player = observation.player_index
    declarer = contract.declarer

    if contract.game_type is GameType.RAMSCH:
        return CardPlayRole.RAMSCH

    if contract.game_type is GameType.SOLO:
        if declarer is None:
            raise ValueError("Solo requires a declarer.")
        return (
            CardPlayRole.SOLO_DECLARER
            if player == declarer
            else CardPlayRole.SOLO_DEFENDER
        )

    if contract.game_type is GameType.WENZ:
        if declarer is None:
            raise ValueError("Wenz requires a declarer.")
        return (
            CardPlayRole.WENZ_DECLARER
            if player == declarer
            else CardPlayRole.WENZ_DEFENDER
        )

    if contract.game_type is GameType.SAUSPIEL:
        if declarer is None:
            raise ValueError("Sauspiel requires a declarer.")
        if player == declarer:
            return CardPlayRole.SAUSPIEL_DECLARER
        if observation.known_called_ace_player == player:
            return CardPlayRole.SAUSPIEL_PARTNER
        return CardPlayRole.SAUSPIEL_DEFENDER

    raise ValueError(
        f"No specialist role is configured for {contract.game_type.value}."
    )
