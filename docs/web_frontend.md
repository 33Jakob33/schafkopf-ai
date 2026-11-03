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


## Bavarian card artwork and visible AI turns

The frontend expects the user-supplied 4-row by 8-column Bavarian card-sheet
at `frontend/public/bavarian-deck.jpg`. The artwork is **not committed**;
copy the original image into that location before starting Vite.

Rows must be Eichel, Gras, Herz, Schellen, with columns Ass, König, Ober,
Unter, 10, 9, 8, 7 (as in the supplied image). The CSS sprite uses positions
from that exact 1110 × 694 image; unrelated image dimensions/layouts will
not align.

The client calls `POST /api/games/{id}/advance` to advance **one AI action**
at a time. This makes betting announcements and opponent card plays visible.
The API remains authoritative; the browser cannot inspect hidden hands.

## Reorder the human player's hand

- Drag a card onto another card to move it to that position.
- Use **Nach Farbe** for Bavarian suits (Eichel, Gras, Herz, Schellen)
  and **Nach Wert** for descending rank groups.
- Sorting is purely a browser preference. The API still validates moves
  against actual card IDs, and hidden hands are never sent to the client.
- Played cards disappear without resetting your chosen order; a new round
  starts with the newly dealt order.

## AI model explanations

Every opponent has a custom dropdown. Hover over or keyboard-focus a model
choice for a short description; the menu also explains missing checkpoints.
Unlike native HTML select options, this works in a browser-controlled menu.
A model choice becomes active when you start a **new** round.

## Manual UI check

```powershell
cd frontend
npm.cmd run typecheck
npm.cmd run build
```

In the browser: drag an unplayable card during bidding, sort by suit, start a
round, play one legal card, and verify the order of remaining cards is
preserved. Open each opponent model dropdown, hover a disabled neural agent,
check the description, then select a baseline agent and start a new round.

## Analysemodus (Schritt für Schritt)

Die Taste **Analysemodus: AN/AUS** oberhalb des Spieltisches schaltet in der
aktuellen Partie zwischen normalem Spielen und Analyse um. Bei einer neuen
Runde bleibt der ausgewählte Modus erhalten.

- **Alle vier Hände:** Im Analysemodus werden die echten verbleibenden Karten
  aller Spieler angezeigt. Außerhalb des Analysemodus werden die verdeckten
  Hände **nicht** an den Browser gesendet.
- **Nächster Schritt:** Führt genau eine nächste Ansage oder Kartenaktion aus.
  Kein automatisches Durchspielen. Auch Sitz 0 wird für die Analyse durch
  den HeuristicAgent gespielt; im normalen Modus ist Sitz 0 wieder menschlich.
- **Entscheidungsvorschau:** Vor dem Schritt wird die nächste Modellentscheidung
  berechnet und zwischengespeichert. Der folgende Schritt spielt exakt diese
  Karte bzw. Ansage aus (auch bei RandomAgent).
- **Kartenwerte:** Nur legale Karten werden aufgelistet. RolloutAgent zeigt
  seine eigenen erwarteten Auszahlungen, Q-Agent seine vorhergesagten
  Auszahlungen. Bei anderen Modellen wird eine *separate, nur auf der
  Spieler-Observation beruhende 4-Sample-Rollout-Schätzung* ausgewiesen.
  Diese Zahlen sind **nicht die internen Werte** der Heuristik, PPO- oder
  BC-Modelle und auch kein perfektes Spiel mit Kenntnis aller Hände.
- **Gebote:** Derzeit wird die geplante Ansage gegenüber allen legalen
  Alternativen markiert; für Bieten gibt es noch keine numerischen Rankings.

Für einen lokalen Funktionstest:

```powershell
uv run --with fastapi --with httpx pytest tests/test_web_analysis.py tests/test_web_game.py
cd frontend
npm.cmd run typecheck
npm.cmd run build
```

Anmerkung: Die Analyseentscheidung wird serverseitig aus dem Modellzustand
berechnet und kann bei Rollout-Modellen etwas dauern. Sessionzustand und
Vorschaucache sind in-memory, nicht für öffentliche Bereitstellung gedacht.
