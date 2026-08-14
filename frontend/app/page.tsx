import Link from "next/link";
import { listWeeks, loadLatest } from "@/lib/predictions";

export default function HomePage() {
  const latest = loadLatest();
  const weeks = listWeeks();

  const bySeason = new Map<number, typeof weeks>();
  for (const w of [...weeks].reverse()) {
    const list = bySeason.get(w.season) ?? [];
    list.push(w);
    bySeason.set(w.season, list);
  }

  return (
    <div className="space-y-14">
      <section className="border-b-2 border-ink pb-8">
        <h1 className="mt-3 text-6xl leading-[0.85] font-bold tracking-tight text-ink uppercase sm:text-7xl">
          CFB Predictor
        </h1>
        {latest ? (
          <>
            <p className="mt-3 max-w-xl text-ink-muted">
              College football win probabilities
            </p>
            <Link
              href={`/${latest.season}/week/${latest.week}`}
              className="mt-6 inline-flex items-center gap-2 border-2 border-ink px-5 py-2.5 font-mono text-sm uppercase tracking-wide text-ink transition-colors hover:bg-canvas-dim"
            >
              View Week {latest.week} · {latest.season} →
            </Link>
          </>
        ) : (
          <p className="mt-4 max-w-xl text-ink-muted">
            No predictions yet. Run{" "}
            <code className="bg-canvas-dim px-1 font-mono text-sm text-ink">
              python scripts/pipeline.py
            </code>{" "}
            to generate this week&apos;s slate.
          </p>
        )}
      </section>

      <section>
        <h2 className="font-mono text-xs font-medium tracking-[0.15em] text-ink-muted uppercase">
          Browse weeks
        </h2>
        {weeks.length === 0 ? (
          <p className="mt-3 text-sm text-ink-muted">No weekly files on disk.</p>
        ) : (
          <div className="mt-4 space-y-3">
            {[...bySeason.entries()].map(([season, seasonWeeks]) => (
              <details
                key={season}
                open={latest != null && season === latest.season}
                className="group"
              >
                <summary className="flex cursor-pointer list-none items-center gap-2 font-mono text-sm text-ink-muted marker:hidden [&::-webkit-details-marker]:hidden">
                  <span className="inline-block text-xs transition-transform group-open:rotate-90">
                    ›
                  </span>
                  {season}
                </summary>
                <div className="mt-2 flex flex-wrap gap-2 pl-5">
                  {seasonWeeks
                    .slice()
                    .sort((a, b) => a.week - b.week)
                    .map((w) => {
                      const isCurrent =
                        latest != null &&
                        w.season === latest.season &&
                        w.week === latest.week;
                      return (
                        <Link
                          key={`${w.season}-${w.week}`}
                          href={`/${w.season}/week/${w.week}`}
                          className={
                            isCurrent
                              ? "border-2 border-ink bg-ink px-3 py-1.5 font-mono text-sm text-canvas"
                              : "border-2 border-hairline px-3 py-1.5 font-mono text-sm text-ink hover:border-ink"
                          }
                        >
                          Wk {w.week}
                        </Link>
                      );
                    })}
                </div>
              </details>
            ))}
          </div>
        )}
      </section>

      <section className="grid gap-4 sm:grid-cols-2">
        <Link
          href="/performance"
          className="border-2 border-ink p-5 transition-colors hover:bg-canvas-dim"
        >
          <div className="text-lg font-semibold tracking-tight text-ink uppercase">
            Performance
          </div>
          <p className="mt-1 text-sm text-ink-muted">
            How the model did this season, once games are played.
          </p>
        </Link>
        <Link
          href="/models"
          className="border-2 border-ink p-5 transition-colors hover:bg-canvas-dim"
        >
          <div className="text-lg font-semibold tracking-tight text-ink uppercase">
            Models
          </div>
          <p className="mt-1 text-sm text-ink-muted">
            The models used to predict the win probabilities.
          </p>
        </Link>
      </section>
    </div>
  );
}
