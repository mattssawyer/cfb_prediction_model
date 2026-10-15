/* eslint-disable @next/next/no-img-element -- static export can't optimize remote logos */
import { atsPick, formatLine, kickoffTime, modelFavorite, vegasFavorite } from "@/lib/format";
import type { Game } from "@/lib/predictions";
import { team } from "@/lib/teams";

function luminance(hex: string): number {
  const n = parseInt(hex.replace("#", ""), 16);
  const [r, g, b] = [(n >> 16) & 255, (n >> 8) & 255, n & 255].map((v) => {
    const s = v / 255;
    return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

/** The team's primary color, or its alternate if the primary would vanish on white. */
function teamColor(name: string): string {
  const t = team(name);
  for (const color of [t.color, t.alt]) {
    if (color && /^#[0-9a-f]{6}$/i.test(color) && luminance(color) < 0.75) return color;
  }
  return "#8a8f98";
}

function abbr(name: string): string {
  return team(name).abbr ?? name;
}

function Logo({ name }: { name: string }) {
  const logo = team(name).logo;
  return logo ? (
    <img src={logo} alt="" width={32} height={32} className="h-8 w-8 shrink-0 object-contain" loading="lazy" />
  ) : (
    <span
      className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-[10px] font-bold text-white"
      style={{ background: teamColor(name) }}
    >
      {abbr(name).slice(0, 3)}
    </span>
  );
}

/** Green check or red X: whether a pick came true. */
function Result({ hit, label }: { hit: boolean; label: string }) {
  return (
    <span
      role="img"
      aria-label={`${label} ${hit ? "hit" : "missed"}`}
      title={`${label} ${hit ? "hit" : "missed"}`}
      className={`inline-flex h-4 w-4 shrink-0 items-center justify-center rounded-full text-white ${hit ? "bg-hit" : "bg-miss"}`}
    >
      <svg viewBox="0 0 12 12" className="h-2.5 w-2.5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
        {hit ? <path d="M2.5 6.5l2.5 2.5 4.5-5.5" /> : <path d="M3 3l6 6M9 3l-6 6" />}
      </svg>
    </span>
  );
}

function TeamLine({ game, home }: { game: Game; home: boolean }) {
  const name = home ? game.home_team : game.away_team;
  const rank = home ? game.home_rank : game.away_rank;
  const p = home ? game.home_win_probability : 1 - game.home_win_probability;
  const score = home ? game.actual_home_score : game.actual_away_score;
  const favored = game.predicted_winner === name;
  const won = game.actual_winner === name;
  return (
    <div className="flex items-center gap-3">
      <Logo name={name} />
      <div className="min-w-0 flex-1 truncate">
        {rank ? <span className="mr-1 text-xs font-semibold text-muted">{rank}</span> : null}
        <span className={`text-[15px] ${favored ? "font-bold text-ink" : "font-medium text-muted"}`}>
          {name}
        </span>
        <span className="ml-2 hidden text-xs text-muted sm:inline">
          {home ? game.home_conference : game.away_conference}
        </span>
      </div>
      {score != null ? (
        <span className={`w-8 text-right text-lg ${won ? "font-extrabold text-ink" : "font-medium text-muted"}`}>
          {score}
        </span>
      ) : null}
      <span className={`flex w-16 items-center justify-end gap-1.5 text-[15px] ${favored ? "font-bold text-ink" : "font-medium text-muted"}`}>
        {favored && game.correct != null ? <Result hit={game.correct} label="Winner pick" /> : null}
        {Math.round(p * 100)}%
      </span>
    </div>
  );
}

function Detail({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="min-w-0">
      <dt className="text-xs text-muted">{label}</dt>
      <dd className="flex items-center gap-1.5 text-sm font-semibold whitespace-nowrap">{children}</dd>
    </div>
  );
}

function line(fav: { team: string; by: number } | null): string {
  if (!fav) return "\u2014";
  return fav.by === 0 ? "PK" : `${abbr(fav.team)} ${formatLine(-fav.by)}`;
}

export default function GameRow({ game }: { game: Game }) {
  const away = Math.round((1 - game.home_win_probability) * 100);
  const pick = atsPick(game);

  return (
    <li className="grid gap-x-8 gap-y-3 border-t border-line px-4 py-4 first:border-t-0 sm:grid-cols-[minmax(0,1fr)_12rem] sm:px-5">
      <div>
        <div className="mb-2 text-xs text-muted">
          {kickoffTime(game.kickoff)}
          {game.neutral_site ? " (neutral)" : ""}
        </div>
        <div className="space-y-2">
          <TeamLine game={game} home={false} />
          <TeamLine game={game} home />
        </div>
        <div
          className="mt-3 flex h-1.5 overflow-hidden rounded-full"
          role="img"
          aria-label={`Win chance: ${game.away_team} ${away}%, ${game.home_team} ${100 - away}%`}
        >
          <div style={{ width: `${away}%`, background: teamColor(game.away_team) }} />
          <div className="ml-0.5 flex-1" style={{ background: teamColor(game.home_team) }} />
        </div>
      </div>
      <dl className="grid grid-cols-3 content-center gap-x-3 gap-y-2 border-t border-line pt-3 sm:grid-cols-1 sm:border-t-0 sm:border-l sm:pt-0 sm:pl-5">
        <Detail label="Model">{line(modelFavorite(game))}</Detail>
        <Detail label="Vegas">{game.vegas_spread == null ? "\u2014" : line(vegasFavorite(game))}</Detail>
        <Detail label="Spread pick">
          {pick ? (
            <>
              {abbr(pick.team)} {formatLine(pick.line)}
              {game.ats_correct != null ? <Result hit={game.ats_correct} label="Spread pick" /> : null}
            </>
          ) : (
            "\u2014"
          )}
        </Detail>
      </dl>
    </li>
  );
}
