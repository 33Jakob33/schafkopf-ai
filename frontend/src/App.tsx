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
      if (health.status !== "ok") {
        throw new Error("Unerwarteter API-Status");
      }
      setEngine(info);
      setConnection("online");
    } catch (caught) {
      setConnection("offline");
      setEngine(null);
      setError(
        caught instanceof Error ? caught.message : "Unbekannter Verbindungsfehler",
      );
    }
  }

  useEffect(() => {
    void refresh();
  }, []);

  return (
    <main className="app-shell">
      <header className="topbar">
        <h1>Schafkopf AI</h1>
        <div className={`status status--${connection}`}>
          <span className="status-dot" />
          {connection === "online"
            ? "API verbunden"
            : connection === "offline"
              ? "API nicht erreichbar"
              : "Verbindung wird aufgebaut"}
        </div>
      </header>

      <section className="intro-card">
        <div className="intro-text">
          <p className="subtitle">Bayerisches Kartenspiel trifft KI</p>
          <p>
            Dieses Frontend ist das Grundgerüst für eine spielbare Schafkopf-Oberfläche
            mit Python-Backend und verschiedenen Agenten.
          </p>
          <button onClick={() => void refresh()} type="button">
            Verbindung prüfen
          </button>
        </div>

      
      </section>

      <section className="panel">
        <h2>Engine</h2>

        {engine ? (
          <>
            <div className="metrics">
              <article>
                <strong>{engine.player_count}</strong>
                <span>Spieler</span>
              </article>
              <article>
                <strong>{engine.deck_size}</strong>
                <span>Karten</span>
              </article>
              <article>
                <strong>{engine.cards_per_player}</strong>
                <span>Karten pro Hand</span>
              </article>
            </div>

            <div className="lists">
              <div>
                <h3>Spieltypen</h3>
                <div className="chips">
                  {engine.game_types.map((item) => (
                    <span key={item}>{item}</span>
                  ))}
                </div>
              </div>

              <div>
                <h3>Agenten</h3>
                <div className="chips">
                  {engine.available_agents.map((item) => (
                    <span key={item}>{item}</span>
                  ))}
                </div>
              </div>
            </div>
          </>
        ) : (
          <p className="error-text">{error ?? "Lade Engine-Daten ..."}</p>
        )}
      </section>
    </main>
  );
}