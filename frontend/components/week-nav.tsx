"use client";

import { useRouter } from "next/navigation";
import Link from "next/link";
import type { WeekIndex } from "@/lib/predictions";

type Props = {
  season: number;
  week: number;
  weeks: WeekIndex[];
};

export default function WeekNav({ season, week, weeks }: Props) {
  const router = useRouter();
  const index = weeks.findIndex((w) => w.season === season && w.week === week);
  const prev = index > 0 ? weeks[index - 1] : null;
  const next = index >= 0 && index < weeks.length - 1 ? weeks[index + 1] : null;

  return (
    <div className="flex flex-wrap items-center gap-4 font-mono text-xs uppercase tracking-wide">
      {prev ? (
        <Link href={`/${prev.season}/week/${prev.week}`} className="text-ink hover:text-accent">
          ← Wk {prev.week}
        </Link>
      ) : (
        <span className="text-hairline">← Wk</span>
      )}

      <select
        className="border-2 border-ink bg-canvas px-2 py-1 text-ink"
        value={`${season}-${week}`}
        onChange={(event) => {
          const [s, w] = event.target.value.split("-");
          router.push(`/${s}/week/${w}`);
        }}
      >
        {weeks.map((w) => (
          <option key={`${w.season}-${w.week}`} value={`${w.season}-${w.week}`}>
            {w.season} · Wk {w.week}
          </option>
        ))}
      </select>

      {next ? (
        <Link href={`/${next.season}/week/${next.week}`} className="text-ink hover:text-accent">
          Wk {next.week} →
        </Link>
      ) : (
        <span className="text-hairline">Wk →</span>
      )}
    </div>
  );
}
