import { useEffect, useState } from "react";
import { api, type EngineInfo } from "./api";

type Connection = "connecting" | "online" | "offline";

export default function App() {
  const [connection, setConnection] = useState<Connection>("connecting");
  const [engine, setEngine] = useState<EngineInfo | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function refresh() {
    setConnection("connecting");
    setError(null);
    try {
      const [health, info] = await Promise.all([api.health(), api.engine()]);
      if (health.status !== "ok") throw new Error("Unexpected API health status");
      setEngine(info);
      setConnection("online");
    } catch (caught) {
      setConnection("offline");
      setEngine(null);
      setError(caught instanceof Error ? caught.message : "Unknown connection error");
    }
  }

  useEffect(() => { void refresh(); }, []);

  return (
    <main className="app-shell">
      <header className="header">
        <div className="brand-mark">♠</div>
        <div><p className="eyebrow">SCHAFKOPF AI</p><h1>Your next opponent is learning.</h1></div>
        <span className={`status status--${connection}`}>
          <span className="status-dot" />
          {connection === "online" ? "API connected" : connection === "offline" ? "API offline" : "Connecting"}
        </span>
      </header>
      <section className="hero">
        <div>
          <p className="eyebrow">GAME ENGINE · PYTHON + REACT</p>
          <h2>Schafkopf, reinvented with AI.</h2>
          <p className="intro">A home for playing against agents, inspecting their decisions, and comparing experiments.</p>
          <button onClick={() => void refresh()} type="button">Check API connection</button>
        </div>
        <div className="card-fan" aria-hidden="true">
          <span className="playing-card red">♥<small>A</small></span>
          <span className="playing-card">♠<small>O</small></span>
          <span className="playing-card red">♦<small>U</small></span>
        </div>
      </section>
      <section className="panel">
        <div className="panel-heading"><div><p className="eyebrow">LIVE BACKEND DATA</p><h3>Engine overview</h3></div></div>
        {engine ? (
          <>
            <div className="metrics">
              <article><strong>{engine.player_count}</strong><span>Players</span></article>
              <article><strong>{engine.deck_size}</strong><span>Cards</span></article>
              <article><strong>{engine.cards_per_player}</strong><span>Cards per hand</span></article>
            </div>
            <div className="details"><div><h4>Available game modes</h4><div className="chips">{engine.game_types.map(item => <span key={item}>{item}</span>)}</div></div><div><h4>Available baseline agents</h4><div className="chips">{engine.available_agents.map(item => <span key={item}>{item}</span>)}</div></div></div>
          </>
        ) : <p className="placeholder">{error ?? "Fetching engine information…"}</p>}
      </section>
      <p className="footer-note">Foundation milestone · Interactive gameplay is coming in the next phase.</p>
    </main>
  );
}
