"""Smoke tests for the optional web adapter (run with FastAPI + httpx)."""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from schafkopf_ai.web.api import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "schafkopf-ai"}


def test_engine_reports_real_game_types() -> None:
    from schafkopf_ai.game.game_type import GameType

    response = client.get("/api/engine")
    assert response.status_code == 200
    payload = response.json()
    assert payload["deck_size"] == 32
    assert payload["player_count"] == 4
    assert payload["game_types"] == [game_type.value for game_type in GameType]
    assert "heuristic" in payload["available_agents"]


def test_catalog_reports_missing_checkpoints_without_offering_them() -> None:
    payload = client.get("/api/engine").json()
    catalog = payload["agent_catalog"]
    ids = {entry["id"] for entry in catalog}
    assert {"heuristic", "random", "rollout", "bc", "dagger", "ppo",
            "q", "multihead", "specialist", "strategic"}.issubset(ids)
    assert all(entry["available"] == (entry["id"] in payload["available_agents"])
               for entry in catalog)
    assert all(entry["reason"] for entry in catalog if not entry["available"])
