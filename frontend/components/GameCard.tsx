import type { Game } from "@/lib/predictions";

type Props = {
  game: Game;
};

function formatKickoff(iso: string | null): string {
  if (!iso) return "TBD";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString("en-US", {
    weekday: "short",
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
    hour12: true,
  });
}

export function GameCard({ game }: Props) {
  const homeProb = game.home_win_probability;
  const awayProb = 1 - homeProb;
  const homeWins = homeProb >= 0.5;
  const confidencePct = Math.round(game.confidence * 100);

  return (
    <article className="group relative border border-neutral-200 dark:border-neutral-800 rounded-lg p-5 hover:border-neutral-400 dark:hover:border-neutral-600 transition-colors bg-white dark:bg-neutral-950">
      <div className="flex items-center justify-between gap-3 mb-4 text-xs font-mono text-neutral-500 dark:text-neutral-400">
        <span>{formatKickoff(game.kickoff)}</span>
        <div className="flex items-center gap-1.5">
          {game.neutral_site && (
            <span className="px-1.5 py-0.5 rounded bg-amber-100 dark:bg-amber-950/50 text-amber-800 dark:text-amber-300 uppercase tracking-wider text-[10px]">
              neutral
            </span>
          )}
          {game.conference_game && (
            <span className="px-1.5 py-0.5 rounded bg-neutral-100 dark:bg-neutral-800 uppercase tracking-wider text-[10px]">
              conf
            </span>
          )}
        </div>
      </div>

      <TeamRow
        team={game.away_team}
        conference={game.away_conference}
        prob={awayProb}
        favored={!homeWins}
        side={game.neutral_site ? null : "@"}
      />
      <TeamRow
        team={game.home_team}
        conference={game.home_conference}
        prob={homeProb}
        favored={homeWins}
        side={null}
      />

      <div className="mt-4 pt-3 border-t border-neutral-100 dark:border-neutral-900 flex items-baseline justify-between text-xs font-mono text-neutral-500 dark:text-neutral-400">
        <span>
          Pick{" "}
          <span className="text-neutral-900 dark:text-neutral-100 font-semibold">
            {game.predicted_winner}
          </span>
        </span>
        <span>{confidencePct}%</span>
      </div>
    </article>
  );
}

function TeamRow({
  team,
  conference,
  prob,
  favored,
  side,
}: {
  team: string;
  conference: string | null;
  prob: number;
  favored: boolean;
  side: string | null;
}) {
  const pct = Math.round(prob * 100);
  return (
    <div className="mb-3 last:mb-0">
      <div className="flex items-baseline justify-between gap-3 mb-1.5">
        <div className="flex items-baseline gap-2 min-w-0">
          {side && (
            <span className="text-neutral-400 dark:text-neutral-600 font-mono text-sm">
              {side}
            </span>
          )}
          <span
            className={`truncate text-base ${
              favored
                ? "font-semibold text-neutral-900 dark:text-neutral-50"
                : "text-neutral-600 dark:text-neutral-400"
            }`}
          >
            {team}
          </span>
          {conference && (
            <span className="text-[11px] font-mono text-neutral-400 dark:text-neutral-600 shrink-0">
              {conference}
            </span>
          )}
        </div>
        <span
          className={`font-mono tabular-nums text-sm ${
            favored
              ? "text-neutral-900 dark:text-neutral-50 font-semibold"
              : "text-neutral-500 dark:text-neutral-400"
          }`}
        >
          {pct}%
        </span>
      </div>
      <div className="h-1 bg-neutral-100 dark:bg-neutral-900 rounded-full overflow-hidden">
        <div
          className={`h-full ${
            favored ? "bg-neutral-900 dark:bg-neutral-100" : "bg-neutral-300 dark:bg-neutral-700"
          }`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
