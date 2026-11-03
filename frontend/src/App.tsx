import { useEffect, useMemo, useState } from "react";
import ModelPicker from "./ModelPicker";
import { moveCard, orderHand, sortHand } from "./handOrder";
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
  const [analysisMode, setAnalysisMode] = useState(false);
  const [game, setGame] = useState<GameView | null>(null);
  const [catalog, setCatalog] = useState<EngineInfo["agent_catalog"]>(FALLBACK_MODELS);
  const [loading, setLoading] = useState(false);
  const [recentTrick, setRecentTrick] = useState<TrickPlay[] | null>(null);
  const [handOrder, setHandOrder] = useState<string[]>([]);
  const [draggedCard, setDraggedCard] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void api.engine().then(info => setCatalog(info.agent_catalog))
      .catch(() => setError("Backend nicht erreichbar. Bitte FastAPI starten."));
  }, []);

  async function start() {
    setLoading(true);
    setError(null);

    try {
      const next = await api.newGame(opponents, analysisMode);
      setHandOrder([]);
      setDraggedCard(null);
      setRecentTrick(null);
      setGame(next);
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
    if (!game || game.analysis || loading || game.phase === "finished" || game.current_player === 0) return;
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

  async function toggleMode() {
    if (loading) return;
    const nextMode = !analysisMode;
    setLoading(true);
    setError(null);
    try {
      if (game) {
        const updated = await api.mode(game.id, nextMode);
        setGame(updated);
      }
      setAnalysisMode(nextMode);
      setRecentTrick(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Moduswechsel fehlgeschlagen");
    } finally { setLoading(false); }
  }

  async function nextAnalysisStep() {
    if (!game || !game.analysis || loading || game.phase === "finished") return;
    setLoading(true);
    setError(null);
    try {
      setRecentTrick(null);
      setGame(await api.advance(game.id));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Analyseschritt fehlgeschlagen");
    } finally { setLoading(false); }
  }

  const orderedHand = useMemo(() => orderHand(game?.hand ?? [], handOrder), [game?.hand, handOrder]);

  function reorder(sourceId: string, targetId: string) {
    if (!game || game.phase === "finished") return;
    const current = orderedHand.map(card => card.id);
    setHandOrder(moveCard(current, sourceId, targetId));
  }

  const contract = game?.contract;
  const contractLabel = contract
    ? [contract.type, contract.called_suit ?? contract.trump_suit].filter(Boolean).join(" · ")
    : "Bieten";

  function opponent(index: number, cls: string) {
    const player = game?.players[index];
    return <div className={`opponent opponent--${cls}`}>
      <strong>Gegner {index}</strong>
      <ModelPicker
        player={index}
        models={catalog}
        value={opponents[index - 1]}
        disabled={loading}
        onChange={modelId => setOpponents(previous => previous.map(
          (value, position) => position === index - 1 ? modelId : value
        ))}
      />
      <span>{player ? `${player.points} Augen` : "8 Karten"}</span>
      {game?.analysis && game.all_hands ? <div className="analysis-opponent-hand">
        {game.all_hands[index].map(card => <PlayingCard key={card.id} card={card} small />)}
      </div> : <BackCards count={player?.cards ?? 8} />}
    </div>;
  }

  return <main className="app">
    <header className="topbar">
      <h1>Schafkopf AI</h1>
      <span className="subtitle">Bairisches Kartenspiel</span>
    </header>

    <section className="settings settings--compact">
      <button type="button" className={`mode-button ${analysisMode ? "mode-button--active" : ""}`}
        onClick={() => void toggleMode()} disabled={loading} aria-pressed={analysisMode}>
        {analysisMode ? "Analysemodus: AN" : "Analysemodus: AUS"}
      </button>
      <button className="new-game-button" type="button" disabled={loading}
        onClick={() => void start()}>{game ? "Neue Runde" : "Spiel starten"}</button>
      <span className="hint">{analysisMode ? "Alle Karten sichtbar · jeden KI-Schritt einzeln prüfen" : "Wähle die KI direkt bei jedem Gegner. Änderungen gelten ab der nächsten Runde."}</span>
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
        <div className="human-heading">
          <strong>Du</strong><span>{game?.players[0].points ?? 0} Augen</span>
          {game && orderedHand.length > 1 && (
            <div className="hand-sort-actions" aria-label="Handkarten sortieren">
              <button type="button" onClick={() => setHandOrder(sortHand(game.hand, "suit"))}>Nach Farbe</button>
              <button type="button" onClick={() => setHandOrder(sortHand(game.hand, "rank"))}>Nach Wert</button>
            </div>
          )}
        </div>
        <div className="hand" aria-label="Deine Karten – per Ziehen sortierbar">
          {orderedHand.map(card => (
            <div
              key={card.id}
              className={`hand-card-slot ${draggedCard === card.id ? "hand-card-slot--dragging" : ""}`}
              onDragOver={event => { event.preventDefault(); event.dataTransfer.dropEffect = "move"; }}
              onDrop={event => {
                event.preventDefault();
                const source = event.dataTransfer.getData("text/plain") || draggedCard;
                if (source) reorder(source, card.id);
                setDraggedCard(null);
              }}
              onDragEnd={() => setDraggedCard(null)}
            >
              <PlayingCard card={card}
                clickable={!analysisMode && !loading && game?.phase === "playing" &&
                  game.current_player === 0 && game.legal_cards.includes(card.id)}
                onClick={() => void act({ card: card.id })} />
              <span
                className="card-drag-handle"
                draggable
                role="img"
                aria-label={`${card.suit} ${card.rank} zum Sortieren ziehen`}
                title="Hier ziehen, um die Karte zu sortieren"
                onDragStart={event => {
                  event.dataTransfer.effectAllowed = "move";
                  event.dataTransfer.setData("text/plain", card.id);
                  setDraggedCard(card.id);
                }}
                onDragEnd={() => setDraggedCard(null)}
              >⠿</span>
            </div>
          ))}
          {!game && <span className="empty-hand">Starte eine Runde, um deine Karten zu sehen.</span>}
        </div>
      </div>
    </section>

    {game?.analysis && game.phase !== "finished" && <section className="analysis-panel">
      <div className="analysis-panel-heading"><div><h2>Entscheidungsanalyse</h2>
        <span>Schritt {game.step_index + 1} · {game.analysis_preview?.player === null ? "Spielübergang" : `Spieler ${(game.analysis_preview?.player ?? 0) + 1} (Sitz ${game.analysis_preview?.player ?? 0})`}</span></div>
        <button disabled={loading} onClick={() => void nextAnalysisStep()}>Nächster Schritt <span aria-hidden="true">›</span></button>
      </div>
      {game.analysis_preview ? <>
        <p className="analysis-decision">Geplante Aktion: <strong>{game.analysis_preview.kind === "card" ?
          game.analysis_preview.options.find(option => option.selected)?.label :
          game.analysis_preview.chosen ?? game.analysis_preview.message}</strong></p>
        <p className="analysis-note">{game.analysis_preview.message}</p>
        {game.analysis_preview.options.length > 0 && <div className="analysis-options">
          {[...game.analysis_preview.options].sort((a, b) =>
            (a.rank ?? 999) - (b.rank ?? 999)).map(option =>
            <div className={`analysis-option ${option.selected ? "analysis-option--chosen" : ""}`} key={option.id}>
              <span>{option.label}</span>
              {option.rank !== null && <span>Rang {option.rank}</span>}
              {option.value !== null && <strong>{option.value > 0 ? "+" : ""}{option.value.toFixed(2)}</strong>}
              {option.selected && <em>Wahl der KI</em>}
            </div>)}
        </div>}
      </> : <p>Keine weitere Entscheidung verfügbar.</p>}
      <p className="analysis-note">Bewertungen sind geschätzte Auszahlungen, keine garantierten Ergebnisse. Die KI erhält weiterhin nur ihre eigene Hand und öffentliche Informationen.</p>
    </section>}

    {game?.phase === "bidding" && !game.analysis && <section className="action-panel">
      <h2>Deine Ansage</h2>
      <div className="bid-actions">
        {game.bidding_actions.map(action => <button key={action.id} disabled={loading}
          onClick={() => void act({ action_id: action.id })}>{action.label}</button>)}
      </div>
      <p className="hint">{game.bidding_actions.length ? "Du bist an der Reihe." : "Die Gegner bieten …"}</p>
    </section>}
    {game?.phase === "playing" && !game.analysis && <p className="turn-hint">{game.current_player === 0 ? `Du bist dran: ${game.legal_cards.length} spielbare Karte(n) sind markiert.` : "Die Gegner spielen …"}</p>}
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