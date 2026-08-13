import { notFound } from "next/navigation";
import { GameCard } from "@/components/GameCard";
import { WeekHeader } from "@/components/WeekHeader";
import { listAllWeeks, loadWeek } from "@/lib/predictions";

export function generateStaticParams() {
  return listAllWeeks().map((w) => ({
    season: String(w.season),
    week: String(w.week),
  }));
}

type Params = { season: string; week: string };

export default async function WeekPage({
  params,
}: {
  params: Promise<Params>;
}) {
  const { season, week } = await params;
  const seasonNum = Number(season);
  const weekNum = Number(week);
  if (!Number.isFinite(seasonNum) || !Number.isFinite(weekNum)) notFound();

  const data = loadWeek(seasonNum, weekNum);
  if (!data) notFound();

  return (
    <>
      <WeekHeader
        season={data.season}
        week={data.week}
        generatedAt={data.generated_at}
        model={data.model}
        gameCount={data.games.length}
      />
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {data.games.map((game, i) => (
          <GameCard key={game.id ?? i} game={game} />
        ))}
      </div>
    </>
  );
}
