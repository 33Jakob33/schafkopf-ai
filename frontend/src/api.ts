export interface Card { id: string; suit: string; rank: string }
export interface BidAction { id: string; label: string }
export interface Player { name: string; cards: number; points: number }
export interface TrickPlay { player: number; card: Card }
export interface Contract {
  type: string;
  declarer: number | null;
  called_suit: string | null;
  trump_suit: string | null;
}
export interface GameView {
  id: string;
  phase: "bidding" | "playing" | "finished";
  current_player: number | null;
  hand: Card[];
  legal_cards: string[];
  bidding_actions: BidAction[];
  players: Player[];
  trick: TrickPlay[];
  last_trick: TrickPlay[];
  contract: Contract | null;
  payments: number[] | null;
  trick_number: number;
  redeals: number;
  bidding_history: { player: number; action: string }[];
}
export interface EngineInfo {
  name: string;
  player_count: number;
  deck_size: number;
  cards_per_player: number;
  game_types: string[];
  available_agents: string[];
}
async function request<T>(url: string, method = "GET", body?: unknown): Promise<T> {
  const response = await fetch(url, {
    method,
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!response.ok) {
    let message = `HTTP ${response.status}`;
    try { const data = await response.json() as { detail?: string }; message = data.detail ?? message; }
    catch { /* keep status */ }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}
export const api = {
  engine: () => request<EngineInfo>("/api/engine"),
  newGame: (opponents: string[]) => request<GameView>("/api/games", "POST", { opponents }),
  act: (id: string, action: { card?: string; action_id?: string }) =>
    request<GameView>(`/api/games/${encodeURIComponent(id)}/actions`, "POST", action),
};
