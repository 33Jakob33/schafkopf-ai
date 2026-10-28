# Schafkopf AI web table

Playable local prototype on `feature/web-frontend`. Three independently
configurable opponents and one human player. The actual Python Schafkopf
bidding, legal moves and settlement drive the interface; it does not simulate
the rules in React.

## Start

Terminal 1, repository root:

```powershell
uv run --with fastapi --with uvicorn uvicorn schafkopf_ai.web.api:app --reload
```

Terminal 2:

```powershell
cd frontend
npm.cmd install
npm.cmd run dev
```

Visit http://localhost:5173. Select the three agents and click **Spiel
starten**. Choose legal bidding actions, then highlighted playable cards.
The API automatically makes each opponent's moves until the human must act
again. At game end, the table displays each player's settlement.

## Available opponents

- `heuristic`: rule-based HeuristicAgent
- `random`: RandomAgent
- `rollout`: RolloutCardPlayAgent with 8 samples per action (may block an
  API request for a noticeable time)

Trained BC, DAgger, PPO, Q and specialist agents are **not** wired into this
frontend yet; they require loading model checkpoints and providing safe
configuration. The selector intentionally lists only working choices.

## API

- `GET /api/health`
- `GET /api/engine`
- `POST /api/games` with `{"opponents":["heuristic","random","rollout"]}`
- `GET /api/games/{id}`
- `POST /api/games/{id}/actions` with `{"action_id":"0"}` during
  bidding or `{"card":"HERZ:ACE"}` during card play.

The action IDs and card IDs are supplied by the server's current observation;
the server validates legal actions. The UI never receives opponent hands.

## Testing

```powershell
uv run --with fastapi --with httpx pytest tests/test_web_api.py tests/test_web_game.py
cd frontend
npm.cmd run typecheck
npm.cmd run build
```

Development limitations: sessions are stored **in memory**, reset on API
restart, and are not durable or suitable for public hosting. The code is a
single-user local prototype; multiplayer, authentication, rate limiting,
persistent match history, and production-grade session isolation are not
implemented. Moves run synchronously; rollout models can be slow. The cards
use a **stylized Bavarian-suit representation** (Eichel, Gras, Herz, Schellen),
not licensed scans of a traditional Bavarian deck.
