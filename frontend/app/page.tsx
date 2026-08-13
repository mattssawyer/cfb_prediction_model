import { GameCard } from "@/components/GameCard";
import { WeekHeader } from "@/components/WeekHeader";
import { loadLatest } from "@/lib/predictions";

export default function Home() {
  const latest = loadLatest();

  if (!latest) {
    return (
      <div className="border border-dashed border-neutral-200 dark:border-neutral-800 rounded-lg p-12 text-center">
        <h1 className="text-2xl font-semibold mb-2">No predictions yet</h1>
        <p className="text-neutral-500 dark:text-neutral-400 font-mono text-sm">
          Run <code className="text-neutral-900 dark:text-neutral-100">python scripts/pipeline.py</code>{" "}
          to generate this week&apos;s predictions.
        </p>
      </div>
    );
  }

  return (
    <>
      <WeekHeader
        season={latest.season}
        week={latest.week}
        generatedAt={latest.generated_at}
        model={latest.model}
        gameCount={latest.games.length}
      />
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {latest.games.map((game, i) => (
          <GameCard key={game.id ?? i} game={game} />
        ))}
      </div>
    </>
  );
}
