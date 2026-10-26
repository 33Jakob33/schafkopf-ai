"""Minimal FastAPI adapter: engine metadata without leaking hidden hands."""
from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from schafkopf_ai.game.deck import Deck
from schafkopf_ai.game.game_type import GameType

app = FastAPI(title="Schafkopf AI API", version="0.1.0")

# Local development: Vite serves on 5173 while the API serves on 8000.
origins = os.environ.get("SCHAFKOPF_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in origins.split(",") if origin.strip()],
    allow_credentials=False,
    allow_methods=["GET"],
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
        available_agents=["random", "heuristic"],
    )
