from collections.abc import Sequence

from schafkopf_ai.agents.agent import Agent
from schafkopf_ai.game.card import Card
from schafkopf_ai.game.game_state import GameState


class GameRunner:
    """
    Runs the card-playing phase of a Schafkopf round.

    The runner connects agents to the game engine. It gives each
    agent only that player's observation and legal moves.
    """

    PLAYER_COUNT = 4
    CARDS_PER_GAME = 32

    def __init__(
        self,
        state: GameState,
        agents: Sequence[Agent],
    ) -> None:
        if len(agents) != self.PLAYER_COUNT:
            raise ValueError(
                f"A GameRunner requires exactly {self.PLAYER_COUNT} agents."
            )

        self.state = state
        self.agents: tuple[Agent, ...] = tuple(agents)
        self.turns_played = 0

    def play_turn(self) -> Card:
        """
        Let the current player's agent choose and play one card.

        Returns:
            The card that was played.

        Raises:
            ValueError:
                If the game is already complete or the agent returns
                a card that is not one of the legal actions.
        """
        if self.state.is_complete:
            raise ValueError("Cannot play another turn: the game is complete.")

        player_index = self.state.current_player

        if player_index is None:
            raise RuntimeError("Game is not complete but has no current player.")

        observation = self.state.observation_for(player_index)

        legal_cards = self.state.legal_moves(player_index)

        if not legal_cards:
            raise RuntimeError(f"Player {player_index} has no legal moves.")

        agent = self.agents[player_index]

        chosen_card = agent.choose_card(
            observation=observation,
            legal_cards=legal_cards,
        )

        if chosen_card not in legal_cards:
            raise ValueError(
                f"Agent for player {player_index} selected illegal card {chosen_card}."
            )

        self.state.play_card(
            player_index=player_index,
            card=chosen_card,
        )

        self.turns_played += 1

        return chosen_card

    def run(self) -> GameState:
        """
        Play the round until all eight tricks are complete.

        Returns:
            The completed GameState.
        """
        while not self.state.is_complete:
            if self.turns_played >= self.CARDS_PER_GAME:
                raise RuntimeError("Game exceeded 32 turns without completing.")

            self.play_turn()

        return self.state
