"""Soft evidence model for opponent plays in sampled hidden deals.

All hard legality constraints are handled by HiddenStateSampler. This module
only *ranks* valid determinizations using an imperfect behavioral model.
"""
from __future__ import annotations

import math

from schafkopf_ai.agents.heuristic_agent import HeuristicAgent
from schafkopf_ai.game.game_state import GameState
from schafkopf_ai.game.observation import PlayerObservation
from schafkopf_ai.search.hidden_state import Determinization, _public_plays, _starting_player


def observed_action_log_likelihood(
    observation: PlayerObservation,
    deal: Determinization,
    *,
    agreement_probability: float = 0.7,
) -> float:
    """Score public opponent plays against a heuristic-with-noise model.

    Only opponents' actions are used. All alternative legal moves retain
    positive probability, preventing false certainty about player intentions.
    """
    if not 0.0 < agreement_probability < 1.0:
        raise ValueError("agreement_probability must be strictly between 0 and 1")
    state = GameState.from_hands(
        hands=deal.initial_hands,
        contract=observation.contract,
        starting_player=_starting_player(observation),
    )
    heuristic = HeuristicAgent()
    score = 0.0
    for player, card in _public_plays(observation):
        legal = state.legal_moves(player)
        if player != observation.player_index and len(legal) > 1:
            recommended = heuristic.choose_card(state.observation_for(player), legal)
            # Mixture of behavioral heuristic and uniform random play.
            p = (1.0 - agreement_probability) / len(legal)
            if recommended == card:
                p += agreement_probability
            score += math.log(p)
        state.play_card(player, card)
    return score
