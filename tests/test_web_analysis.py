"""Analysis mode shows all cards only on explicit request and steps exactly once."""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient
from schafkopf_ai.web.api import app

client = TestClient(app)


def test_analysis_preview_is_stable_and_each_click_advances_one_decision() -> None:
    created = client.post("/api/games", json={
        "opponents": ["random", "heuristic", "random"],
        "seed": 123, "analysis": True,
    })
    assert created.status_code == 200, created.text
    game = created.json()
    assert game["analysis"]
    assert len(game["all_hands"]) == 4
    assert [len(hand) for hand in game["all_hands"]] == [8, 8, 8, 8]
    assert game["step_index"] == 0
    preview = game["analysis_preview"]
    assert preview and preview["kind"] == "bidding"
    assert exactly_one_selected(preview["options"])

    again = client.get(f"/api/games/{game['id']}")
    assert again.status_code == 200
    assert again.json()["analysis_preview"] == preview

    moved = client.post(f"/api/games/{game['id']}/advance")
    assert moved.status_code == 200, moved.text
    next_view = moved.json()
    assert next_view["step_index"] == 1
    assert len(next_view["bidding_history"]) == 1
    assert next_view["bidding_history"][0]["action"] == preview["chosen"]


def exactly_one_selected(options: list[dict]) -> bool:
    return sum(option["selected"] for option in options) == 1


def test_analysis_does_not_leak_cards_after_switching_to_normal() -> None:
    response = client.post("/api/games", json={
        "opponents": ["heuristic", "random", "heuristic"],
        "analysis": True, "seed": 884,
    })
    game = response.json()
    assert game["all_hands"] is not None

    switch = client.post(f"/api/games/{game['id']}/mode", json={"analysis": False})
    assert switch.status_code == 200
    normal = switch.json()
    assert normal["analysis"] is False
    assert normal["all_hands"] is None
    assert normal["analysis_preview"] is None
    assert all("hand" not in player for player in normal["players"])
    saved = client.get(f"/api/games/{game['id']}").json()
    assert saved["all_hands"] is None


def test_analysis_rejects_direct_human_actions() -> None:
    game = client.post("/api/games", json={
        "opponents": ["random", "random", "random"],
        "analysis": True, "seed": 993,
    }).json()
    response = client.post(f"/api/games/{game['id']}/actions",
                           json={"action_id": "0"})
    assert response.status_code == 422
