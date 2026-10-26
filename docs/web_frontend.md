# Web frontend: development setup

The web interface lives on `feature/web-frontend`, branched from `main`.

## Requirements

- Python 3.12+ and [uv](https://docs.astral.sh/uv/)
- Node.js 20+ and npm (Node 22 LTS recommended)

## Terminal 1: API (repository root)

FastAPI and Uvicorn are **optional** dependencies. For now, `uv --with`
installs them into uv's execution environment without modifying the existing
Python ML dependency lockfile.

```powershell
uv sync
uv run --with fastapi --with uvicorn uvicorn schafkopf_ai.web.api:app --reload
```

API: http://127.0.0.1:8000/api/health

Interactive OpenAPI documentation: http://127.0.0.1:8000/docs

## Terminal 2: React (repository root)

```powershell
cd frontend
npm install
npm run dev
```

Frontend: http://localhost:5173

Vite proxies browser `/api` requests to FastAPI on port 8000. A CORS
allowlist is also configured for local development. This app currently
reads only engine metadata; game sessions and card-play API endpoints
will be added in the next phase.

## Verification

```powershell
uv run --with fastapi --with httpx pytest tests/test_web_api.py
cd frontend
npm run typecheck
npm run build
```

`uv run pytest` without FastAPI installed will not be able to import
`tests/test_web_api.py`. Use `uv run --with fastapi --with httpx pytest`
for the **entire** Python suite, or install the optional web dependencies.

## Architecture

- `src/schafkopf_ai/web/api.py`: optional Python/FastAPI HTTP layer
- `frontend/src/api.ts`: typed fetch client
- `frontend/src/App.tsx`: API status and engine metadata
- `frontend/vite.config.ts`: Vite development proxy

Only metadata is transferred in this first milestone. When adding game
sessions, return **player-scoped observations**, never full `GameState`
objects or opponent hands.
