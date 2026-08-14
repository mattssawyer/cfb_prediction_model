import Table from "@/components/table";
import WeekNav from "@/components/week-nav";
import type { WeekIndex, WeekPrediction } from "@/lib/predictions";

export default function WeekView({
  payload,
  weeks,
}: {
  payload: WeekPrediction;
  weeks: WeekIndex[];
}) {
  return (
    <div>
      <div className="mb-8 flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <div className="font-mono text-xs uppercase tracking-[0.2em] text-ink-muted">
            Season {payload.season}
          </div>
          <h1 className="mt-2 text-4xl leading-none font-bold tracking-tight text-ink uppercase sm:text-5xl">
            Week {payload.week}
          </h1>
          <p className="mt-2 text-sm text-ink-muted">
            {payload.games.length} FBS games
          </p>
        </div>
        <WeekNav season={payload.season} week={payload.week} weeks={weeks} />
      </div>
      <Table data={payload.games} />
    </div>
  );
}
