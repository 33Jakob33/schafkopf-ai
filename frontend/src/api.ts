export interface HealthResponse {
  status: string;
  service: string;
}

export interface EngineInfo {
  name: string;
  player_count: number;
  cards_per_player: number;
  deck_size: number;
  game_types: string[];
  available_agents: string[];
}

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(path, { headers: { Accept: "application/json" } });
  if (!response.ok) {
    throw new Error(`API request failed (${response.status})`);
  }
  return (await response.json()) as T;
}

export const api = {
  health: () => getJson<HealthResponse>("/api/health"),
  engine: () => getJson<EngineInfo>("/api/engine"),
};
