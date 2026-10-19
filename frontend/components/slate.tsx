"use client";

import { useSearchParams } from "next/navigation";
import GameRow from "@/components/game-row";
import { groupByDay, vegasFavorite } from "@/lib/format";
import type { Game } from "@/lib/predictions";

/** Quick filters. Each one is a URL param set to "1" while it's on. */
const TOGGLES: { key: string; label: string; title: string; test: (g: Game) => boolean }[] = [
  {
    key: "top25",
    label: "Top 25",
    title: "Games with a ranked team",
    test: (g) => g.home_rank != null || g.away_rank != null,
  },
  {
    key: "tossup",
    label: "Toss-ups",
    title: "The model gives neither team more than a 60% chance",
    test: (g) => g.confidence < 0.6,
  },
  {
    key: "upset",
    label: "Upsets",
    title: "The model picks the team Vegas has as the underdog",
    test: (g) => {
      const v = vegasFavorite(g);
      return v != null && v.by > 0 && v.team !== g.predicted_winner;
    },
  },
  {
    key: "edge",
    label: "Vegas gap",
    title: "The model line is 5 or more points off the Vegas line",
    test: (g) =>
      g.predicted_margin != null && g.vegas_spread != null && Math.abs(g.predicted_margin + g.vegas_spread) >= 5,
  },
];

function inConference(g: Game, conference: string): boolean {
  return g.home_conference === conference || g.away_conference === conference;
}

export function GameList({ games }: { games: Game[] }) {
  return groupByDay(games).map((group) => (
    <section key={group.day} className="mt-8">
      <h2 className="mb-2 text-sm font-bold">{group.day}</h2>
      <ul className="overflow-hidden rounded-xl border border-line bg-panel">
        {group.games.map((game, i) => (
          <GameRow key={`${game.id ?? `${game.away_team}-${game.home_team}`}-${i}`} game={game} />
        ))}
      </ul>
    </section>
  ));
}

export default function Slate({ games }: { games: Game[] }) {
  const params = useSearchParams();
  const conference = params.get("conference") ?? "";
  const active = TOGGLES.filter((t) => params.get(t.key) === "1");

  const conferences = [
    ...new Set(games.flatMap((g) => [g.home_conference, g.away_conference]).filter((c): c is string => !!c)),
  ].sort();
  const available = TOGGLES.filter((t) => games.some(t.test));

  const shown = games.filter(
    (g) => (!conference || inConference(g, conference)) && active.every((t) => t.test(g)),
  );

  // Filtering is all client-side, so update the URL in place. Next.js syncs
  // useSearchParams with history.replaceState without fetching the route again.
  // location.pathname (not usePathname) keeps the basePath.
  function navigate(query: URLSearchParams) {
    const qs = query.toString();
    const path = window.location.pathname;
    window.history.replaceState(null, "", qs ? `${path}?${qs}` : path);
  }

  function setConference(value: string) {
    const query = new URLSearchParams(params.toString());
    if (value) query.set("conference", value);
    else query.delete("conference");
    navigate(query);
  }

  function toggle(key: string) {
    const query = new URLSearchParams(params.toString());
    if (query.get(key) === "1") query.delete(key);
    else query.set(key, "1");
    navigate(query);
  }

  function clear() {
    const query = new URLSearchParams(params.toString());
    query.delete("conference");
    for (const t of TOGGLES) query.delete(t.key);
    navigate(query);
  }

  const filtered = conference !== "" || active.length > 0;

  return (
    <>
      <div className="no-scrollbar -mx-5 mt-4 flex items-center gap-1.5 overflow-x-auto px-5 py-1 sm:-mx-1 sm:px-1">
        <label className="sr-only" htmlFor="conference">
          Conference
        </label>
        <div className="relative shrink-0">
          <select
            id="conference"
            value={conference}
            onChange={(e) => setConference(e.target.value)}
            className={`h-8 appearance-none rounded-full border pr-8 pl-3 text-[13px] font-semibold ${
              conference ? "border-ink bg-ink text-white" : "border-line bg-panel text-ink hover:border-muted"
            }`}
          >
            <option value="">Conference</option>
            {conferences.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
          <svg
            viewBox="0 0 12 12"
            className={`pointer-events-none absolute top-1/2 right-3 h-2.5 w-2.5 -translate-y-1/2 ${conference ? "text-white" : "text-muted"}`}
            fill="none"
            stroke="currentColor"
            strokeWidth="1.75"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden
          >
            <path d="M2.5 4.5L6 8l3.5-3.5" />
          </svg>
        </div>
        {available.map((t) => {
          const on = active.includes(t);
          return (
            <button
              key={t.key}
              type="button"
              aria-pressed={on}
              onClick={() => toggle(t.key)}
              title={t.title}
              className={`h-8 shrink-0 rounded-full border px-3 text-[13px] font-semibold whitespace-nowrap ${
                on ? "border-ink bg-ink text-white" : "border-line bg-panel text-ink hover:border-muted"
              }`}
            >
              {t.label}
            </button>
          );
        })}
        {filtered ? (
          <button
            type="button"
            onClick={clear}
            aria-label="Clear filters"
            title="Clear filters"
            className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-muted hover:bg-panel hover:text-ink"
          >
            <svg viewBox="0 0 12 12" className="h-3 w-3" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden>
              <path d="M3 3l6 6M9 3l-6 6" />
            </svg>
          </button>
        ) : null}
      </div>

      {shown.length > 0 ? (
        <GameList games={shown} />
      ) : (
        <p className="mt-8 rounded-xl border border-line bg-panel px-5 py-8 text-center text-muted">
          No games match.{" "}
          <button type="button" onClick={clear} className="font-semibold text-ink underline underline-offset-2">
            Show all
          </button>
        </p>
      )}
    </>
  );
}
