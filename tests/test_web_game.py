"""End-to-end web session tests with the real game engine."""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient
from schafkopf_ai.web.api import app

client = TestClient(app)


def test_game_can_be_played_to_settlement() -> None:
    response = client.post("/api/games", json={
        "opponents": ["heuristic", "random", "heuristic"], "seed": 12345,
    })
    assert response.status_code == 200, response.text
    game = response.json()
    assert len(game["hand"]) == 8
    assert all("hand" not in opponent for opponent in game["players"])
    for _ in range(160):
        if game["phase"] == "finished":
            break
        if game["current_player"] != 0:
            response = client.post(f"/api/games/{game['id']}/advance")
            assert response.status_code == 200, response.text
            game = response.json()
            continue
        if game["phase"] == "bidding":
            assert game["bidding_actions"]
            action = {"action_id": game["bidding_actions"][0]["id"]}
        else:
            assert game["legal_cards"]
            action = {"card": game["legal_cards"][0]}
        response = client.post(f"/api/games/{game['id']}/actions", json=action)
        assert response.status_code == 200, response.text
        game = response.json()
    assert game["phase"] == "finished"
    assert sum(game["payments"]) == 0
    assert sum(player["points"] for player in game["players"]) == 120


def test_bad_model_is_rejected() -> None:
    response = client.post("/api/games", json={
        "opponents": ["heuristic", "nonexistent", "random"]
    })
    assert response.status_code == 422


def test_invalid_card_rejected() -> None:
    response = client.post("/api/games", json={
        "opponents": ["random", "random", "random"], "seed": 17,
    })
    game = response.json()
    response = client.post(f"/api/games/{game['id']}/actions", json={"card": "HERZ:ACE"})
    assert response.status_code == 422
