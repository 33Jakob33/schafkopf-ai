import { useEffect, useState } from "react";
import { api, type Card, type GameView, type TrickPlay } from "./api";

const SUITS: Record<string, string> = {
  Eichel: "♣", Gras: "♠", Herz: "♥", Schellen: "◆",
};
const RANKS: Record<string, string> = {
  Ass: "A", König: "K", Ober: "O", Unter: "U",
};
const OPTIONS = [
  { id: "heuristic", label: "Heuristic" },
  { id: "random", label: "Random" },
  { id: "rollout", label: "Rollout (8 Samples)" },
];

function PlayingCard({ card, clickable, onClick, small = false }: {
  card: Card; clickable?: boolean; onClick?: () => void; small?: boolean;
}) {
  return (
    <button
      type="button"
      className={`playing-card ${small ? "playing-card--small" : ""} ${clickable ? "playing-card--legal" : ""} ${card.suit === "Herz" || card.suit === "Schellen" ? "playing-card--red" : ""}`}
      onClick={onClick}
      disabled={!clickable}
      title={`${card.suit} ${card.rank}`}
      aria-label={`${card.suit} ${card.rank}`}
    >
      <span className="card-rank">{RANKS[card.rank] ?? card.rank}</span>
      <span className="card-symbol">{SUITS[card.suit] ?? "?"}</span>
      <span className="card-suit-name">{card.suit}</span>
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
  const [available, setAvailable] = useState<string[]>(["heuristic", "random", "rollout"]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void api.engine().then(info => setAvailable(info.available_agents))
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

  const contract = game?.contract;
  const contractLabel = contract
    ? [contract.type, contract.called_suit ?? contract.trump_suit].filter(Boolean).join(" · ")
    : "Bieten";

  function opponent(index: number, cls: string) {
    const player = game?.players[index];
    return <div className={`opponent opponent--${cls}`}>
      <strong>{player?.name ?? `Gegner ${index}`}</strong>
      <span>{player ? `${player.points} Augen` : opponents[index - 1]}</span>
      <BackCards count={player?.cards ?? 8} />
    </div>;
  }

  return <main className="app">
    <header className="topbar">
      <h1>Schafkopf AI</h1>
      <span className="subtitle">Bairisches Kartenspiel</span>
    </header>

    <section className="settings">
      <div className="settings-heading">Gegner auswählen</div>
      <div className="model-controls">
        {opponents.map((kind, i) => (
          <label key={i}>
            Gegner {i + 1}
            <select value={kind} disabled={loading}
              onChange={e => setOpponents(prev => prev.map((v, j) => j === i ? e.target.value : v))}>
              {OPTIONS.filter(o => available.includes(o.id)).map(o =>
                <option key={o.id} value={o.id}>{o.label}</option>)}
            </select>
          </label>
        ))}
        <button type="button" disabled={loading} onClick={() => void start()}>
          {game ? "Neue Runde" : "Spiel starten"}
        </button>
      </div>
      {game && <p className="hint">Die Modellauswahl gilt beim Start einer neuen Runde.</p>}
    </section>

    <section className="table-area">
      <div className="game-info">
        <strong>{game ? contractLabel : "Bereit zum Spielen"}</strong>
        {game && <span>{game.phase === "finished" ? "Runde beendet" : game.phase === "bidding" ? "Bietrunde" : `Stich ${game.trick_number} von 8`}</span>}
      </div>
      {opponent(2, "top")}
      {opponent(1, "left")}
      {opponent(3, "right")}
      <div className="table-center">
        {game?.trick.length ? <TrickCards plays={game.trick} /> : (
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
    {game?.phase === "playing" && <p className="turn-hint">{loading ? "Gegner denken …" : game.current_player === 0 ? "Du bist dran: Wähle eine hervorgehobene Karte." : "Gegner sind am Zug."}</p>}
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
