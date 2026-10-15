import type { Game } from "@/lib/predictions";

const TIME_ZONE = "America/Chicago";

export function formatLine(n: number): string {
  const r = Math.round(n * 10) / 10;
  if (r === 0) return "PK";
  return r > 0 ? `+${r.toFixed(1)}` : `−${Math.abs(r).toFixed(1)}`;
}

export function kickoffTime(iso: string | null): string {
  if (!iso) return "TBD";
  return new Date(iso).toLocaleTimeString("en-US", {
    hour: "numeric",
    minute: "2-digit",
    timeZone: TIME_ZONE,
  });
}

export function kickoffDay(iso: string | null): string {
  if (!iso) return "Date to be announced";
  return new Date(iso).toLocaleDateString("en-US", {
    weekday: "long",
    month: "long",
    day: "numeric",
    timeZone: TIME_ZONE,
  });
}

export function shortDate(iso: string): string {
  return new Date(iso).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    timeZone: TIME_ZONE,
  });
}

/** Games grouped by local kickoff day, in kickoff order. */
export function groupByDay(games: Game[]): { day: string; games: Game[] }[] {
  const sorted = [...games].sort((a, b) => {
    if (!a.kickoff) return 1;
    if (!b.kickoff) return -1;
    return a.kickoff.localeCompare(b.kickoff) || a.home_team.localeCompare(b.home_team);
  });
  const groups: { day: string; games: Game[] }[] = [];
  for (const game of sorted) {
    const day = kickoffDay(game.kickoff);
    const last = groups.at(-1);
    if (last && last.day === day) last.games.push(game);
    else groups.push({ day, games: [game] });
  }
  return groups;
}

/**
 * The side the model takes against the Vegas line. vegas_spread is the home
 * team's number and predicted_margin is home minus away, matching
 * src/cfb/evaluate.py.
 */
export function atsPick(game: Game): { team: string; line: number } | null {
  if (game.predicted_margin == null || game.vegas_spread == null) return null;
  const edge = game.predicted_margin + game.vegas_spread;
  if (edge === 0) return null;
  return edge > 0
    ? { team: game.home_team, line: game.vegas_spread }
    : { team: game.away_team, line: -game.vegas_spread };
}

export function modelFavorite(g: Game): { team: string; by: number } | null {
  if (g.predicted_margin == null) return null;
  return g.predicted_margin >= 0
    ? { team: g.home_team, by: g.predicted_margin }
    : { team: g.away_team, by: -g.predicted_margin };
}

export function vegasFavorite(g: Game): { team: string; by: number } | null {
  if (g.vegas_spread == null) return null;
  return g.vegas_spread <= 0
    ? { team: g.home_team, by: -g.vegas_spread }
    : { team: g.away_team, by: g.vegas_spread };
}

