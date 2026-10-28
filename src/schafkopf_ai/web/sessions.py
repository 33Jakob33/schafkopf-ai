"""In-memory single-user game sessions, backed by the real Schafkopf engine.

Development-only storage: sessions disappear when the API restarts.
Only the human player's hand is serialized. Other hands stay server-side.
"""
from __future__ import annotations

import random
import secrets
from dataclasses import dataclass

from schafkopf_ai.agents.agent import Agent
from schafkopf_ai.agents.heuristic_agent import HeuristicAgent
from schafkopf_ai.agents.random_agent import RandomAgent
from schafkopf_ai.agents.rollout_card_play_agent import RolloutCardPlayAgent
from schafkopf_ai.game.bidding import BiddingAction, BiddingState
from schafkopf_ai.game.card import Card
from schafkopf_ai.game.deck import Deck
from schafkopf_ai.game.game_state import GameState
from schafkopf_ai.game.game_result import result_from_game_state
from schafkopf_ai.game.rules import RAMSCH_RULES
from schafkopf_ai.game.settlement import settlement_from_game_state

AGENT_KINDS = ("heuristic", "random", "rollout")
HUMAN = 0
MAX_SESSIONS = 100


def make_agent(kind: str, seed: int) -> Agent:
    if kind == "heuristic":
        return HeuristicAgent()
    if kind == "random":
        return RandomAgent(rng=random.Random(seed))
    if kind == "rollout":
        return RolloutCardPlayAgent(samples_per_action=8, seed=seed)
    raise ValueError(f"Unknown agent kind: {kind}")


def card_id(card: Card) -> str:
    return f"{card.suit.name}:{card.rank.name}"


def card_json(card: Card) -> dict[str, str]:
    return {"id": card_id(card), "suit": card.suit.value, "rank": card.rank.value}


def action_label(action: BiddingAction) -> str:
    if action.contract is not None:
        contract = action.contract
        extra = contract.called_suit or contract.trump_suit
        return f"{contract.game_type.value}" + (f" {extra.value}" if extra else "")
    if action.bid_value is not None:
        return f"Steigern auf {action.bid_value}"
    return {
        "PASS": "Weiter", "PLAY": "Ich spiele",
        "HOLD": "Halten",
    }.get(action.action_type.name, action.action_type.name)


@dataclass
class Session:
    agents: tuple[Agent, Agent, Agent, Agent]
    bidding: BiddingState
    state: GameState | None
    seed: int
    names: tuple[str, str, str, str]
    redeals: int = 0
    last_trick: list[dict] | None = None
    announcement: str | None = None

    def advance(self, max_actions: int = 500) -> None:
        """Run automatic agents only until the next human action/end."""
        for _ in range(max_actions):
            if self.state is None:
                if self.bidding.is_complete:
                    result = self.bidding.result()
                    if result.requires_redeal:
                        self.redeals += 1
                        if self.redeals > 100:
                            raise RuntimeError("Too many redeals")
                        rng = random.Random(self.seed + self.redeals * 1009)
                        deck = Deck()
                        deck.shuffle(rng)
                        self.bidding = BiddingState(
                            hands=deck.deal(), rules=RAMSCH_RULES, starting_player=0,
                        )
                        continue
                    if result.contract is None:
                        raise RuntimeError("No contract after bidding")
                    self.announcement = f'Gespielt wird: {action_label(BiddingAction.announce(result.contract))}'
                    self.state = GameState.from_hands(
                        self.bidding.hands, contract=result.contract, starting_player=0,
                    )
                    continue
                player = self.bidding.current_player
                if player == HUMAN:
                    return
                if player is None:
                    continue
                legal = self.bidding.legal_actions(player)
                action = self.agents[player].choose_bidding_action(
                    self.bidding.observation_for(player), legal
                )
                self.bidding.apply_action(player, action)
                self.announcement = f'{self.names[player]}: {action_label(action)}'
                continue
            if self.state.is_complete or self.state.current_player == HUMAN:
                return
            player = self.state.current_player
            if player is None:
                return
            cards = self.state.legal_moves(player)
            chosen = self.agents[player].choose_card(
                self.state.observation_for(player), cards
            )
            if chosen not in cards:
                raise RuntimeError(f"Agent {player} returned illegal card")
            self.state.play_card(player, chosen)
            self.announcement = f'{self.names[player]} spielt {chosen}'
            self._capture_trick()
        raise RuntimeError("Auto-play exceeded safe action limit")

    def _capture_trick(self) -> None:
        if self.state is None:
            return
        if self.state.completed_tricks:
            trick = self.state.completed_tricks[-1]
            plays = [{"player": play.player, "card": card_json(play.card)} for play in trick.plays]
            self.last_trick = plays

    def view(self, session_id: str) -> dict:
        state = self.state
        if state is None:
            legal = self.bidding.legal_actions(HUMAN) if self.bidding.current_player == HUMAN else ()
            return {
                "id": session_id, "phase": "bidding", "current_player": self.bidding.current_player,
                "hand": [card_json(c) for c in self.bidding.hands[HUMAN]],
                "legal_cards": [], "bidding_actions": [
                    {"id": str(i), "label": action_label(a)}
                    for i, a in enumerate(legal)
                ],
                "players": [
                    {"name": self.names[i], "cards": 8, "points": 0}
                    for i in range(4)
                ],
                "trick": [], "last_trick": self.last_trick or [],
                "contract": None, "payments": None,
                "trick_number": 0, "redeals": self.redeals,
                "announcement": self.announcement,
            "bidding_history": [
                    {"player": event.player_index, "action": action_label(event.action)}
                    for event in self.bidding.history
                ],
            }
        completed = state.is_complete
        observation = state.observation_for(HUMAN)
        legal_cards = (
            state.legal_moves(HUMAN) if state.current_player == HUMAN else ()
        )
        contract = state.contract
        result = result_from_game_state(state) if completed else None
        settlement = (
            settlement_from_game_state(state, result) if result is not None else None
        )
        return {
            "id": session_id, "phase": "finished" if completed else "playing",
            "current_player": state.current_player,
            "hand": [card_json(c) for c in observation.hand],
            "legal_cards": [card_id(c) for c in legal_cards],
            "bidding_actions": [],
            "players": [
                {"name": self.names[i], "cards": len(state.player(i).cards) if i == HUMAN else
                 8 - sum(1 for trick in state.completed_tricks for p in trick.plays if p.player == i) -
                 (sum(1 for p in state.current_trick.plays if p.player == i) if state.current_trick else 0),
                 "points": state.player_points[i]}
                for i in range(4)
            ],
            "trick": (
                [{"player": p.player, "card": card_json(p.card)}
                 for p in state.current_trick.plays] if state.current_trick else []
            ),
            "last_trick": self.last_trick or [],
            "contract": {
                "type": contract.game_type.value,
                "declarer": contract.declarer,
                "called_suit": contract.called_suit.value if contract.called_suit else None,
                "trump_suit": contract.trump_suit.value if contract.trump_suit else None,
            },
            "payments": list(settlement.payments) if settlement else None,
            "trick_number": len(state.completed_tricks) + (0 if completed else 1),
            "redeals": self.redeals,
            "bidding_history": [
                {"player": e.player_index, "action": action_label(e.action)}
                for e in self.bidding.history
            ],
        }


sessions: dict[str, Session] = {}


def new_session(opponents: list[str], seed: int | None = None) -> tuple[str, dict]:
    if len(opponents) != 3 or any(kind not in AGENT_KINDS for kind in opponents):
        raise ValueError("Exactly three supported opponent models required")
    if len(sessions) >= MAX_SESSIONS:
        raise RuntimeError("Too many active sessions; restart development server")
    chosen_seed = seed if seed is not None else secrets.randbelow(2**31)
    rng = random.Random(chosen_seed)
    deck = Deck()
    deck.shuffle(rng)
    hands = deck.deal()
    agents = (HeuristicAgent(), *(make_agent(kind, chosen_seed + i * 997) for i, kind in enumerate(opponents, 1)))
    session = Session(
        agents=agents, bidding=BiddingState(hands=hands, rules=RAMSCH_RULES),
        state=None, seed=chosen_seed,
        names=("Du", *[kind.capitalize() for kind in opponents]),
    )
    session_id = secrets.token_urlsafe(18)
    sessions[session_id] = session
    session.advance(max_actions=1)
    return session_id, session.view(session_id)


def play(session_id: str, action_id: str | None, card: str | None) -> dict:
    session = sessions[session_id]
    if session.state is None:
        if action_id is None or card is not None or session.bidding.current_player != HUMAN:
            raise ValueError("Expected a bidding action")
        legal = session.bidding.legal_actions(HUMAN)
        try:
            action = legal[int(action_id)]
        except (ValueError, IndexError) as exc:
            raise ValueError("Invalid bidding action") from exc
        session.bidding.apply_action(HUMAN, action)
    else:
        if session.state.is_complete:
            raise ValueError("Game already complete")
        if card is None or action_id is not None or session.state.current_player != HUMAN:
            raise ValueError("Expected a legal card")
        legal_cards = session.state.legal_moves(HUMAN)
        selected = next((c for c in legal_cards if card_id(c) == card), None)
        if selected is None:
            raise ValueError("Card is not legal")
        session.state.play_card(HUMAN, selected)
        session._capture_trick()
    session.advance(max_actions=1)
    return session.view(session_id)


def advance_one(session_id: str) -> dict:
    session = sessions[session_id]
    if session.state is None and session.bidding.current_player == HUMAN:
        return session.view(session_id)
    if session.state is not None and (session.state.is_complete or session.state.current_player == HUMAN):
        return session.view(session_id)
    session.advance(max_actions=1)
    return session.view(session_id)
