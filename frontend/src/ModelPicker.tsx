import { useEffect, useRef, useState } from "react";
import type { EngineInfo } from "./api";

type Model = EngineInfo["agent_catalog"][number];

const DESCRIPTIONS: Record<string, string> = {
  heuristic: "Regelbasierter Agent mit taktischen Entscheidungen für Bieten und Kartenspiel.",
  strategic: "Erweiterte Heuristik mit rollenbezogener Strategie, Trumpfplanung und Partnerspiel.",
  "heuristic-bid-random": "Bietet mit Heuristik, spielt seine Karten danach zufällig. Nützlicher Vergleichsgegner.",
  random: "Wählt zufällig aus allen erlaubten Aktionen. Einfacher Basisgegner.",
  rollout: "Simuliert mögliche verdeckte Hände und Spielverläufe. Acht Samples pro Kartenentscheidung; kann etwas länger rechnen.",
  bc: "Behavioral Cloning: Ein neuronales Modell, das aus Heuristik-Beispielen gelernt hat.",
  dagger: "Imitation Learning mit DAgger: Trainingsdaten aus Zuständen, die der Agent selbst erreicht.",
  ppo: "Reinforcement Learning mit Proximal Policy Optimization und trainiertem Policy-Netz.",
  q: "Neurale Schätzung des erwarteten Spielwerts legaler Karten anhand von Action-Value-Daten.",
  multihead: "Gemeinsames neuronales Modell mit verschiedenen Entscheidungsköpfen für Spielrollen.",
  specialist: "Separate neuronale Modelle für unterschiedliche Verträge bzw. Spielrollen.",
};

interface Props {
  player: number;
  models: Model[];
  value: string;
  disabled: boolean;
  onChange: (id: string) => void;
}

export default function ModelPicker({ player, models, value, disabled, onChange }: Props) {
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLDivElement>(null);
  const selected = models.find(model => model.id === value);

  useEffect(() => {
    if (!open) return;
    function onPointerDown(event: PointerEvent) {
      if (!root.current?.contains(event.target as Node)) setOpen(false);
    }
    function onEscape(event: KeyboardEvent) {
      if (event.key === "Escape") setOpen(false);
    }
    document.addEventListener("pointerdown", onPointerDown);
    document.addEventListener("keydown", onEscape);
    return () => {
      document.removeEventListener("pointerdown", onPointerDown);
      document.removeEventListener("keydown", onEscape);
    };
  }, [open]);

  return (
    <div className="opponent-model" ref={root}>
      <button
        type="button"
        className="model-trigger"
        aria-label={`Modell für Gegner ${player}: ${selected?.label ?? value}`}
        aria-expanded={open}
        aria-haspopup="listbox"
        disabled={disabled}
        onClick={() => setOpen(previous => !previous)}
      >
        <span>{selected?.label ?? value}</span><span aria-hidden="true">⌄</span>
      </button>
      {open && (
        <div className="model-menu" role="listbox" aria-label={`Modell für Gegner ${player}`}>
          {models.map(model => (
            <div className="model-option-wrap" key={model.id}>
              <button
                type="button"
                role="option"
                aria-selected={model.id === value}
                aria-disabled={!model.available}
                className={`model-option ${model.id === value ? "model-option--selected" : ""} ${!model.available ? "model-option--missing" : ""}`}
                onClick={() => {
                  if (!model.available) return;
                  onChange(model.id);
                  setOpen(false);
                }}
              >
                <span>{model.label}</span>
                {!model.available && <span className="model-lock" aria-label="Checkpoint fehlt">🔒</span>}
              </button>
              <div className="model-tooltip" role="tooltip">
                <strong>{model.label}</strong>
                <p>{DESCRIPTIONS[model.id] ?? "KI-Agent für das Schafkopfspiel."}</p>
                {!model.available && <p className="model-tooltip-warning">{model.reason || "Checkpoint nicht gefunden."}</p>}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
