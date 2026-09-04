import fs from "node:fs";
import path from "node:path";

const ACCURACY_PATH = path.resolve(process.cwd(), "..", "predictions", "accuracy.json");

export type AccuracySummary = {
  games_graded: number;
  binary_accuracy: number | null;
  brier: number | null;
  spread_games_graded: number;
  spread_mae: number | null;
  spread_rmse: number | null;
  spread_sign_accuracy: number | null;
  ats_games_graded?: number;
  ats_accuracy?: number | null;
};

export type WeekAccuracy = AccuracySummary & { week: number };

export type SeasonAccuracy = {
  season_to_date: AccuracySummary;
  weeks: WeekAccuracy[];
};

export function loadSeasonAccuracy(season: number): SeasonAccuracy | null {
  if (!fs.existsSync(ACCURACY_PATH)) return null;
  const all = JSON.parse(fs.readFileSync(ACCURACY_PATH, "utf8")) as Record<string, SeasonAccuracy>;
  return all[String(season)] ?? null;
}
