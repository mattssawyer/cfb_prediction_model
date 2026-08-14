import type { Game } from "@/lib/predictions";

function ProbabilityBar({
  homeP,
  awayTeam,
  homeTeam,
}: {
  homeP: number;
  awayTeam: string;
  homeTeam: string;
}) {
  const homePct = Math.round(homeP * 100);
  const awayPct = 100 - homePct;
  const homeFavored = homePct >= awayPct;

  return (
    <div className="w-full">
      <div
        role="img"
        aria-label={`${awayTeam} ${awayPct}%, ${homeTeam} ${homePct}%`}
        className="relative h-1.5 w-full bg-hairline"
      >
        <div
          className="absolute inset-y-0 left-0 bg-accent"
          style={{ width: `${awayPct}%` }}
        />
      </div>
      <div className="mt-1 flex items-baseline justify-between font-mono text-xs tabular-nums">
        <span className={homeFavored ? "text-ink-muted" : "font-semibold text-ink"}>
          {awayPct}%
        </span>
        <span className={homeFavored ? "font-semibold text-ink" : "text-ink-muted"}>
          {homePct}%
        </span>
      </div>
    </div>
  );
}

function kickoffTime(iso: string | null) {
  if (!iso) return "TBD";
  return new Date(iso).toLocaleTimeString("en-US", {
    hour: "numeric",
    minute: "2-digit",
    timeZone: "America/Chicago",
  });
}

export default function Table({ data }: { data: Game[] }) {
  return (
    <div>
      <div className="hidden border-b border-dotted border-hairline py-1.5 font-mono text-[10px] uppercase tracking-[0.15em] text-ink-muted sm:grid sm:grid-cols-[4.5rem_1fr_minmax(11rem,26rem)_1fr] sm:gap-4">
        <span>Kickoff</span>
        <span className="text-right">Away</span>
        <span className="text-center">Win probability</span>
        <span>Home</span>
      </div>
      <div className="divide-y divide-dotted divide-hairline">
        {data.map((game, i) => (
          <div
            key={game.id ?? `${game.away_team}-${game.home_team}-${i}`}
            className="grid grid-cols-1 gap-2 py-2.5 sm:grid-cols-[4.5rem_1fr_minmax(11rem,26rem)_1fr] sm:items-center sm:gap-4"
          >
            <div className="font-mono text-sm font-medium text-ink">
              {kickoffTime(game.kickoff)}
            </div>

            <div className="flex items-baseline justify-between sm:block sm:text-right">
              <div
                title={game.away_team}
                className="truncate text-base font-semibold uppercase leading-none tracking-tight text-ink"
              >
                {game.away_team}
              </div>
              <div
                title={game.away_conference ?? undefined}
                className={`truncate font-mono text-[10px] uppercase tracking-wide text-ink-muted ${game.away_conference ? "" : "invisible"}`}
              >
                {game.away_conference ?? "—"}
              </div>
            </div>

            <ProbabilityBar
              homeP={game.home_win_probability}
              awayTeam={game.away_team}
              homeTeam={game.home_team}
            />

            <div className="flex items-baseline justify-between sm:block">
              <div
                title={game.home_team}
                className="truncate text-base font-semibold uppercase leading-none tracking-tight text-ink"
              >
                {game.home_team}
              </div>
              <div
                title={game.home_conference ?? undefined}
                className={`truncate font-mono text-[10px] uppercase tracking-wide text-ink-muted ${game.home_conference ? "" : "invisible"}`}
              >
                {game.home_conference ?? "—"}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
