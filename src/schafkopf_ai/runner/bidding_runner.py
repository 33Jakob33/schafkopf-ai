from collections.abc import Sequence

from schafkopf_ai.agents.agent import Agent
from schafkopf_ai.game.bidding import (
    BiddingResult,
    BiddingState,
)


class BiddingRunner:
    """Connect four agents to a BiddingState."""

    PLAYER_COUNT = 4
    MAX_ACTIONS = 32

    def __init__(
        self,
        *,
        state: BiddingState,
        agents: Sequence[Agent],
    ) -> None:
        if len(agents) != self.PLAYER_COUNT:
            raise ValueError("BiddingRunner requires exactly four agents.")

        self.state = state
        self.agents: tuple[Agent, ...] = tuple(agents)

        self.actions_played = 0

    def play_action(self) -> None:
        """Execute one bidding action."""
        if self.state.is_complete:
            raise ValueError("Bidding is already complete.")

        player_index = self.state.current_player

        if player_index is None:
            raise RuntimeError("Incomplete bidding has no current player.")

        observation = self.state.observation_for(player_index)

        legal_actions = self.state.legal_actions(player_index)

        if not legal_actions:
            raise RuntimeError(f"Player {player_index} has no legal bidding actions.")

        action = self.agents[player_index].choose_bidding_action(
            observation=observation,
            legal_actions=legal_actions,
        )

        if action not in legal_actions:
            raise ValueError(
                f"Agent for player {player_index} selected "
                f"illegal bidding action {action}."
            )

        self.state.apply_action(
            player_index,
            action,
        )

        self.actions_played += 1

    def run(self) -> BiddingResult:
        """Run bidding until a contract or redeal is decided."""
        while not self.state.is_complete:
            if self.actions_played >= self.MAX_ACTIONS:
                raise RuntimeError("Bidding exceeded maximum action count.")

            self.play_action()

        return self.state.result()
