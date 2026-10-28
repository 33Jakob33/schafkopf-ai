import { useEffect, useState } from "react";
import { api, type Card, type EngineInfo, type GameView, type TrickPlay } from "./api";

const SUITS = ["Eichel", "Gras", "Herz", "Schellen"];
const RANKS = ["Ass", "König", "Ober", "Unter", "10", "9", "8", "7"];
const FALLBACK_MODELS: EngineInfo["agent_catalog"] = [
  { id: "heuristic", label: "Heuristic", available: true, reason: "" },
  { id: "random", label: "Random", available: true, reason: "" },
  { id: "rollout", label: "Rollout (8)", available: true, reason: "" },
];

function PlayingCard({ card, clickable, onClick, small = false }: {
  card: Card; clickable?: boolean; onClick?: () => void; small?: boolean;
}) {
  const column = RANKS.indexOf(card.rank);
  const row = SUITS.indexOf(card.suit);
  const spriteStyle = {
    backgroundPosition: `-${228 + column * 83}px -${67 + row * 142}px`,
  };
  return (
    <button
      type="button"
      className={`playing-card playing-card--illustrated ${small ? "playing-card--small" : ""} ${clickable ? "playing-card--legal" : ""}`}
      onClick={onClick} disabled={!clickable}
      title={`${card.suit} ${card.rank}`} aria-label={`${card.suit} ${card.rank}`}
    >
      <span className="card-art-frame">
        <span className="card-art" style={spriteStyle} />
      </span>
      <span className="card-fallback">{card.suit} {card.rank}</span>
    </button>
  );
}

function BackCards({ count }: { count: number }) {
  return <div className="back-row" aria-label={`${count} verdeckte Karten`}>
    {Array.from({ length: count }, (_, i) => <span className="card-back" key={i}>◇</span>)}
  </div>;
}

function TrickCards({ plays }: { plays: TrickPlay[] }) {
  return <div className="trick-layout">
    {plays.map(play => (
      <div key={play.player} className={`trick-slot trick-slot--${play.player}`}>
        <PlayingCard card={play.card} small />
      </div>
    ))}
  </div>;
}

export default function App() {
  const [opponents, setOpponents] = useState(["heuristic", "heuristic", "heuristic"]);
  const [game, setGame] = useState<GameView | null>(null);
  const [catalog, setCatalog] = useState<EngineInfo["agent_catalog"]>(FALLBACK_MODELS);
  const [loading, setLoading] = useState(false);
  const [recentTrick, setRecentTrick] = useState<TrickPlay[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void api.engine().then(info => setCatalog(info.agent_catalog))
      .catch(() => setError("Backend nicht erreichbar. Bitte FastAPI starten."));
  }, []);

  async function start() {
    setLoading(true);
    setError(null);

    try {
      setGame(await api.newGame(opponents));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Spiel konnte nicht gestartet werden");
    } finally { setLoading(false); }
  }

  async function act(action: { card?: string; action_id?: string }) {
    if (!game) return;
    setLoading(true);
    setError(null);
    try {
      setGame(await api.act(game.id, action));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Aktion fehlgeschlagen");
    } finally { setLoading(false); }
  }

  // Advance exactly one AI action at a time: React can animate every card
  // and display every bidding announcement before the next action.
  useEffect(() => {
    if (!game || loading || game.phase === "finished" || game.current_player === 0) return;
    const timer = window.setTimeout(async () => {
      setLoading(true);
      try { setGame(await api.advance(game.id)); }
      catch (err) { setError(err instanceof Error ? err.message : "KI-Zug fehlgeschlagen"); }
      finally { setLoading(false); }
    }, recentTrick ? 1100 : 800);
    return () => window.clearTimeout(timer);
  }, [game, loading, recentTrick]);

  useEffect(() => {
    if (!game?.last_trick.length || game.phase === "bidding") return;
    // Show the completed trick briefly even after the backend has started
    // the next trick. The completed trick remains available separately.
    if (game.trick.length === 0 && game.phase === "playing" && game.trick_number > 1) {
      setRecentTrick(game.last_trick);
      const timer = window.setTimeout(() => setRecentTrick(null), 900);
      return () => window.clearTimeout(timer);
    }
  }, [game?.trick_number, game?.phase]);

  const contract = game?.contract;
  const contractLabel = contract
    ? [contract.type, contract.called_suit ?? contract.trump_suit].filter(Boolean).join(" · ")
    : "Bieten";

  function opponent(index: number, cls: string) {
    const player = game?.players[index];
    return <div className={`opponent opponent--${cls}`}>
      <strong>Gegner {index}</strong>
      <label className="opponent-model">
        <span className="sr-only">KI für Gegner {index}</span>
        <select
          aria-label={`Modell für Gegner ${index}`}
          value={opponents[index - 1]}
          disabled={loading}
          onChange={event => setOpponents(previous => previous.map(
            (value, position) => position === index - 1 ? event.target.value : value
          ))}
        >
          {catalog.map(model => <option key={model.id} value={model.id}
            disabled={!model.available} title={model.reason}>
            {model.label}{model.available ? "" : " (Checkpoint fehlt)"}
          </option>)}
        </select>
      </label>
      <span>{player ? `${player.points} Augen` : "8 Karten"}</span>
      <BackCards count={player?.cards ?? 8} />
    </div>;
  }

  return <main className="app">
    <header className="topbar">
      <h1>Schafkopf AI</h1>
      <span className="subtitle">Bairisches Kartenspiel</span>
    </header>

    <section className="settings settings--compact">
      <button className="new-game-button" type="button" disabled={loading}
        onClick={() => void start()}>{game ? "Neue Runde" : "Spiel starten"}</button>
      <span className="hint">Wähle die KI direkt bei jedem Gegner. Änderungen gelten ab der nächsten Runde.</span>
    </section>

    <section className="table-area">
      <div className="game-info">
        <strong>{game ? contractLabel : "Bereit zum Spielen"}</strong>
        {game && <span>{game.phase === "finished" ? "Runde beendet" : game.phase === "bidding" ? "Bietrunde" : `Stich ${game.trick_number} von 8`}</span>}
      </div>
      {opponent(2, "top")}
      {opponent(1, "left")}
      {opponent(3, "right")}
      {game?.announcement && <div className="table-announcement" role="status" key={game.announcement + game.trick_number}>{game.announcement}</div>}
      <div className="table-center">
        {(recentTrick || game?.trick)?.length ? <TrickCards plays={recentTrick || game!.trick} /> : (
          <div className="table-message">
            {game?.phase === "bidding" ? "Es wird geboten" : game?.phase === "finished" ? "Runde beendet" : "Schafkopf"}
          </div>
        )}
      </div>
      <div className="human">
        <div className="human-heading"><strong>Du</strong><span>{game?.players[0].points ?? 0} Augen</span></div>
        <div className="hand">
          {game?.hand.map(card => <PlayingCard key={card.id} card={card}
            clickable={!loading && game.phase === "playing" && game.legal_cards.includes(card.id)}
            onClick={() => void act({ card: card.id })} />)}
          {!game && <span className="empty-hand">Starte eine Runde, um deine Karten zu sehen.</span>}
        </div>
      </div>
    </section>

    {game?.phase === "bidding" && <section className="action-panel">
      <h2>Deine Ansage</h2>
      <div className="bid-actions">
        {game.bidding_actions.map(action => <button key={action.id} disabled={loading}
          onClick={() => void act({ action_id: action.id })}>{action.label}</button>)}
      </div>
      <p className="hint">{game.bidding_actions.length ? "Du bist an der Reihe." : "Die Gegner bieten …"}</p>
    </section>}
    {game?.phase === "playing" && <p className="turn-hint">{game.current_player === 0 ? "Du bist dran: Wähle eine hervorgehobene Karte." : "Die Gegner spielen …"}</p>}
    {game?.phase === "finished" && <section className="action-panel">
      <h2>Abrechnung</h2>
      <div className="payments">{game.payments?.map((value, i) =>
        <div key={i}><span>{i === 0 ? "Du" : `Gegner ${i}`}</span>
          <strong className={value >= 0 ? "positive" : "negative"}>{value > 0 ? "+" : ""}{value}</strong></div>)}</div>
      <button onClick={() => void start()} disabled={loading}>Noch eine Runde</button>
    </section>}
    {game && <details className="history"><summary>Letzter Stich und Bietverlauf</summary>
      <p>Letzter Stich: {game.last_trick.map(p => `Spieler ${p.player}: ${p.card.suit} ${p.card.rank}`).join(" · ") || "Noch kein Stich"}</p>
      {game.bidding_history.map((h, i) => <p key={i}>Spieler {h.player}: {h.action}</p>)}
    </details>}
    {error && <div className="error" role="alert">{error}</div>}
    <footer>Schafkopf AI · Python Engine + React</footer>
  </main>;
}