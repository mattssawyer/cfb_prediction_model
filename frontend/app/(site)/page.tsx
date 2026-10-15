import WeekView from "@/components/week-view";
import { listWeeks, loadLatest } from "@/lib/predictions";

export default function HomePage() {
  const latest = loadLatest();

  if (!latest) {
    return (
      <div className="max-w-xl">
        <h1 className="text-3xl font-extrabold tracking-tight">No picks yet</h1>
        <p className="mt-2 text-muted">
          Run{" "}
          <code className="rounded bg-panel px-1 text-sm text-ink">
            python scripts/pipeline.py
          </code>{" "}
          to generate this week&apos;s predictions.
        </p>
      </div>
    );
  }

  return <WeekView payload={latest} weeks={listWeeks()} />;
}
