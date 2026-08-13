import Link from "next/link";
import { listAllWeeks } from "@/lib/predictions";

export default function ArchivePage() {
  const weeks = listAllWeeks();
  const bySeason = new Map<number, number[]>();
  for (const w of weeks) {
    const arr = bySeason.get(w.season) ?? [];
    arr.push(w.week);
    bySeason.set(w.season, arr);
  }
  const seasons = Array.from(bySeason.entries()).sort((a, b) => b[0] - a[0]);

  if (seasons.length === 0) {
    return (
      <div className="border border-dashed border-neutral-200 dark:border-neutral-800 rounded-lg p-12 text-center">
        <p className="text-neutral-500 dark:text-neutral-400 font-mono text-sm">
          No historical predictions.
        </p>
      </div>
    );
  }

  return (
    <>
      <header className="border-b border-neutral-200 dark:border-neutral-800 pb-6 mb-8">
        <div className="text-xs uppercase tracking-widest text-neutral-500 dark:text-neutral-400 font-mono">
          Archive
        </div>
        <h1 className="text-4xl sm:text-5xl font-bold tracking-tight mt-1">
          All weeks
        </h1>
      </header>
      <div className="space-y-10">
        {seasons.map(([season, weeks]) => (
          <section key={season}>
            <h2 className="text-xl font-semibold mb-3 font-mono">{season}</h2>
            <div className="flex flex-wrap gap-2">
              {weeks
                .slice()
                .sort((a, b) => a - b)
                .map((week) => (
                  <Link
                    key={week}
                    href={`/${season}/week/${week}`}
                    className="px-3 py-1.5 rounded border border-neutral-200 dark:border-neutral-800 hover:border-neutral-400 dark:hover:border-neutral-600 text-sm font-mono transition-colors"
                  >
                    week {week}
                  </Link>
                ))}
            </div>
          </section>
        ))}
      </div>
    </>
  );
}
