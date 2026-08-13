import type { ModelMeta } from "@/lib/predictions";

type Props = {
  season: number;
  week: number;
  generatedAt: string;
  model: ModelMeta;
  gameCount: number;
};

function formatRelative(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const diffMs = Date.now() - d.getTime();
  const hours = Math.floor(diffMs / (1000 * 60 * 60));
  if (hours < 1) return "just now";
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  if (days < 7) return `${days}d ago`;
  return d.toISOString().slice(0, 10);
}

export function WeekHeader({
  season,
  week,
  generatedAt,
  model,
  gameCount,
}: Props) {
  const accuracy =
    model.holdout_accuracy !== null
      ? `${(model.holdout_accuracy * 100).toFixed(1)}%`
      : "—";
  const brier =
    model.holdout_brier !== null
      ? model.holdout_brier.toFixed(3)
      : "—";
  const accuracyLabel =
    model.holdout_variant === "calibrated"
      ? "Calibrated holdout accuracy"
      : "Holdout accuracy";

  return (
    <header className="border-b border-neutral-200 dark:border-neutral-800 pb-6 mb-8">
      <div className="flex items-baseline justify-between gap-6 flex-wrap">
        <div>
          <div className="text-xs uppercase tracking-widest text-neutral-500 dark:text-neutral-400 font-mono">
            Season {season}
          </div>
          <h1 className="text-4xl sm:text-5xl font-bold tracking-tight mt-1">
            Week {week}
          </h1>
        </div>
        <div className="text-right text-sm text-neutral-500 dark:text-neutral-400 font-mono">
          <div>{gameCount} FBS games</div>
          <div>Generated {formatRelative(generatedAt)}</div>
        </div>
      </div>
      <div className="mt-6 flex gap-6 flex-wrap text-sm font-mono">
        <MetricPill label={accuracyLabel} value={accuracy} />
        <MetricPill label="Brier score" value={brier} />
        {model.market_independent ? (
          <span className="inline-flex items-center gap-1.5 text-neutral-500 dark:text-neutral-400">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
            market-independent
          </span>
        ) : (
          <span className="inline-flex items-center gap-1.5 text-neutral-500 dark:text-neutral-400">
            <span className="w-1.5 h-1.5 rounded-full bg-amber-500" />
            uses betting lines
          </span>
        )}
      </div>
    </header>
  );
}

function MetricPill({ label, value }: { label: string; value: string }) {
  return (
    <span className="text-neutral-500 dark:text-neutral-400">
      {label}{" "}
      <span className="text-neutral-900 dark:text-neutral-100 font-semibold">
        {value}
      </span>
    </span>
  );
}
