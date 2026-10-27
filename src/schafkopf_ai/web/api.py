"""Minimal FastAPI adapter: engine metadata without leaking hidden hands."""
from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from schafkopf_ai.game.deck import Deck
from schafkopf_ai.game.game_type import GameType
from schafkopf_ai.web.sessions import AGENT_KINDS, new_session, play, sessions

app = FastAPI(title="Schafkopf AI API", version="0.1.0")

# Local development: Vite serves on 5173 while the API serves on 8000.
origins = os.environ.get("SCHAFKOPF_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in origins.split(",") if origin.strip()],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class HealthResponse(BaseModel):
    status: str
    service: str


class EngineInfoResponse(BaseModel):
    name: str
    player_count: int
    cards_per_player: int
    deck_size: int
    game_types: list[str]
    available_agents: list[str]


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="schafkopf-ai")


@app.get("/api/engine", response_model=EngineInfoResponse)
def engine_info() -> EngineInfoResponse:
    """Expose supported engine capabilities, not any player's private state."""
    return EngineInfoResponse(
        name="Bavarian Schafkopf",
        player_count=4,
        cards_per_player=8,
        deck_size=len(Deck().cards),
        game_types=[game_type.value for game_type in GameType],
        available_agents=list(AGENT_KINDS),
    )


class NewGameRequest(BaseModel):
    opponents: list[str]
    seed: int | None = None


class PlayRequest(BaseModel):
    action_id: str | None = None
    card: str | None = None


@app.post("/api/games")
def create_game(body: NewGameRequest) -> dict:
    try:
        _, view = new_session(body.opponents, body.seed)
        return view
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/games/{session_id}")
def get_game(session_id: str) -> dict:
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Game not found")
    return session.view(session_id)


@app.post("/api/games/{session_id}/actions")
def submit_action(session_id: str, body: PlayRequest) -> dict:
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Game not found")
    try:
        return play(session_id, body.action_id, body.card)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
