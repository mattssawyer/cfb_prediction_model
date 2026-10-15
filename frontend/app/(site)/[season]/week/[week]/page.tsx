import WeekView from "@/components/week-view";
import { listWeeks, loadWeek } from "@/lib/predictions";
import { notFound } from "next/navigation";

type Params = { season: string; week: string };

export function generateStaticParams() {
  return listWeeks().map(({ season, week }) => ({
    season: String(season),
    week: String(week),
  }));
}

export default async function WeekPage({
  params,
}: {
  params: Promise<Params>;
}) {
  const { season, week } = await params;
  const payload = loadWeek(Number(season), Number(week));
  if (!payload) notFound();
  return <WeekView payload={payload} weeks={listWeeks()} />;
}
