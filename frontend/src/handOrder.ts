import type { Card } from "./api";

const SUITS = ["Eichel", "Gras", "Herz", "Schellen"];
const RANKS = ["Ass", "10", "König", "Ober", "Unter", "9", "8", "7"];

export type HandSort = "suit" | "rank";

/** Preserve the player's chosen card order as the backend removes played cards. */
export function orderHand(hand: Card[], cardIds: string[]): Card[] {
  const positions = new Map(cardIds.map((id, index) => [id, index]));
  return [...hand].sort((a, b) =>
    (positions.get(a.id) ?? Number.MAX_SAFE_INTEGER) -
    (positions.get(b.id) ?? Number.MAX_SAFE_INTEGER)
  );
}

export function moveCard(cardIds: string[], sourceId: string, targetId: string): string[] {
  const from = cardIds.indexOf(sourceId);
  const to = cardIds.indexOf(targetId);
  if (from < 0 || to < 0 || from === to) return cardIds;
  const reordered = [...cardIds];
  reordered.splice(from, 1);
  reordered.splice(to, 0, sourceId);
  return reordered;
}

export function sortHand(hand: Card[], by: HandSort): string[] {
  return [...hand].sort((a, b) => {
    const suitA = SUITS.indexOf(a.suit);
    const suitB = SUITS.indexOf(b.suit);
    const rankA = RANKS.indexOf(a.rank);
    const rankB = RANKS.indexOf(b.rank);
    return by === "suit"
      ? (suitA - suitB || rankA - rankB)
      : (rankA - rankB || suitA - suitB);
  }).map(card => card.id);
}
