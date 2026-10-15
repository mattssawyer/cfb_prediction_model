import Link from "next/link";
import { Suspense } from "react";
import Slate, { GameList } from "@/components/slate";
import type { WeekIndex, WeekPrediction } from "@/lib/predictions";

function WeekTabs({ payload, weeks }: { payload: WeekPrediction; weeks: WeekIndex[] }) {
  const season = weeks.filter((w) => w.season === payload.season);
  return (
    <nav aria-label="Weeks" className="-mx-5 flex gap-1 overflow-x-auto px-5">
      {season.map((w) => {
        const current = w.week === payload.week;
        return (
          <Link
            key={w.week}
            href={`/${w.season}/week/${w.week}`}
            aria-current={current ? "page" : undefined}
            className={`rounded-full px-3.5 py-1.5 text-sm font-semibold whitespace-nowrap ${
              current ? "bg-ink text-white" : "text-muted hover:bg-panel hover:text-ink"
            }`}
          >
            Week {w.week}
          </Link>
        );
      })}
    </nav>
  );
}

/** Share of picks that hit, as a ring. */
function Ring({ hits, total, label }: { hits: number; total: number; label: string }) {
  const share = hits / total;
  const r = 15;
  const circumference = 2 * Math.PI * r;
  return (
    <div className="flex flex-col items-center" title={`${label}: ${hits}\u2013${total - hits}`}>
      <div className="relative h-8 w-8">
        <svg viewBox="0 0 36 36" className="h-8 w-8 -rotate-90" aria-hidden>
          <circle cx="18" cy="18" r={r} fill="none" strokeWidth="3.5" className="stroke-line" />
          <circle
            cx="18"
            cy="18"
            r={r}
            fill="none"
            strokeWidth="3.5"
            strokeLinecap="round"
            strokeDasharray={`${share * circumference} ${circumference}`}
            className="stroke-hit"
          />
        </svg>
        <span className="absolute inset-0 flex items-center justify-center text-[10px] font-bold">
          {Math.round(share * 100)}
        </span>
      </div>
      <span className="mt-0.5 text-[10px] leading-3 font-medium text-muted">
        <span className="sr-only">
          {hits} of {total}{" "}
        </span>
        {label}
      </span>
    </div>
  );
}

export default function WeekView({
  payload,
  weeks,
}: {
  payload: WeekPrediction;
  weeks: WeekIndex[];
}) {
  const graded = payload.games.filter((g) => g.correct != null);
  const right = graded.filter((g) => g.correct).length;
  const ats = graded.filter((g) => g.ats_correct != null);
  const atsRight = ats.filter((g) => g.ats_correct).length;

  return (
    <div>
      <WeekTabs payload={payload} weeks={weeks} />
      <div className="mt-5 flex h-12 items-center justify-between gap-4">
        <h1 className="text-3xl font-extrabold tracking-tight">Week {payload.week} picks</h1>
        {graded.length > 0 ? (
          <div className="flex shrink-0 gap-4">
            <Ring hits={right} total={graded.length} label="Winners" />
            {ats.length > 0 ? <Ring hits={atsRight} total={ats.length} label="Spread" /> : null}
          </div>
        ) : null}
      </div>
      <Suspense fallback={<GameList games={payload.games} />}>
        <Slate games={payload.games} />
      </Suspense>
    </div>
  );
}
