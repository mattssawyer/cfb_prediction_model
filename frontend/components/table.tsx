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
      <div className="mt-1 flex items-baseline justify-between font-mono text-[10px] tabular-nums sm:text-xs">
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

function formatAwaySpread(margin: number): string {
  // predicted_margin is home − away. Away-centric line is the same number:
  // home favored by 7 → away +7; away favored by 7 → away −7.
  const n = Math.round(margin * 10) / 10;
  return n > 0 ? `+${n.toFixed(1)}` : n.toFixed(1);
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
      <div className="hidden border-b border-dotted border-hairline py-1.5 font-mono text-[10px] uppercase tracking-[0.15em] text-ink-muted sm:grid sm:grid-cols-[4.5rem_1fr_minmax(10rem,1.2fr)_1fr] sm:gap-4">
        <span>Kickoff</span>
        <span className="text-right">Away</span>
        <span className="text-center">Win probability</span>
        <span>Home</span>
      </div>
      <div className="divide-y divide-dotted divide-hairline">
        {data.map((game, i) => (
          <div
            key={game.id ?? `${game.away_team}-${game.home_team}-${i}`}
            className="grid grid-cols-[minmax(0,1.35fr)_minmax(3.25rem,0.55fr)_minmax(0,1fr)] items-center gap-x-1.5 gap-y-1.5 py-3 sm:grid-cols-[4.5rem_1fr_minmax(10rem,1.2fr)_1fr] sm:gap-4"
          >
            <div className="col-span-3 font-mono text-xs text-ink-muted sm:col-span-1 sm:text-sm sm:font-medium sm:text-ink">
              {kickoffTime(game.kickoff)}
            </div>

            <div className="min-w-0 text-right">
              <div className="mb-0.5 font-mono text-[9px] uppercase tracking-[0.12em] text-ink-muted sm:hidden">
                Away
              </div>
              <div className="flex flex-wrap items-baseline justify-end gap-x-1 gap-y-0.5">
                <div className="min-w-0 text-right text-[13px] font-semibold uppercase leading-tight tracking-tight text-ink sm:text-base sm:leading-none">
                  {game.away_team}
                </div>
                {game.predicted_margin != null ? (
                  <span
                    className={`shrink-0 font-mono text-sm font-semibold tabular-nums leading-none sm:text-lg ${
                      game.predicted_margin > 0
                        ? "text-spread-plus"
                        : game.predicted_margin < 0
                          ? "text-spread-minus"
                          : "text-ink-muted"
                    }`}
                  >
                    {formatAwaySpread(game.predicted_margin)}
                  </span>
                ) : null}
              </div>
              <div
                className={`font-mono text-[10px] uppercase tracking-wide text-ink-muted ${game.away_conference ? "" : "invisible"}`}
              >
                {game.away_conference ?? "—"}
              </div>
            </div>

            <ProbabilityBar
              homeP={game.home_win_probability}
              awayTeam={game.away_team}
              homeTeam={game.home_team}
            />

            <div className="min-w-0">
              <div className="mb-0.5 font-mono text-[9px] uppercase tracking-[0.12em] text-ink-muted sm:hidden">
                Home
              </div>
              <div className="text-[13px] font-semibold uppercase leading-tight tracking-tight text-ink sm:text-base sm:leading-none">
                {game.home_team}
              </div>
              <div
                className={`font-mono text-[10px] uppercase tracking-wide text-ink-muted ${game.home_conference ? "" : "invisible"}`}
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
