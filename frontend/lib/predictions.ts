import fs from "node:fs";
import path from "node:path";

const PREDICTIONS_DIR = path.resolve(process.cwd(), "..", "predictions");

export type Game = {
  id: number | null;
  kickoff: string | null;
  home_team: string;
  away_team: string;
  home_conference: string | null;
  away_conference: string | null;
  venue: string | null;
  neutral_site: boolean;
  conference_game: boolean;
  home_win_probability: number;
  predicted_winner: string;
  confidence: number;
  predicted_margin: number | null;
};

export type ModelMeta = {
  trained_at: string | null;
  market_independent: boolean;
  calibration_method: string;
  holdout_variant: "calibrated" | "uncalibrated";
  holdout_accuracy: number | null;
  holdout_auc: number | null;
  holdout_brier: number | null;
  spread?: {
    trained_at: string | null;
    holdout_mae: number | null;
    holdout_rmse: number | null;
    holdout_sign_accuracy: number | null;
  } | null;
};

export type WeekPrediction = {
  season: number;
  week: number;
  generated_at: string;
  model: ModelMeta;
  games: Game[];
};

export type WeekIndex = {
  season: number;
  week: number;
};

export function loadLatest(): WeekPrediction | null {
  const filePath = path.join(PREDICTIONS_DIR, "latest.json");
  if (!fs.existsSync(filePath)) return null;
  return JSON.parse(fs.readFileSync(filePath, "utf8")) as WeekPrediction;
}

export function loadWeek(season: number, week: number): WeekPrediction | null {
  const filePath = path.join(PREDICTIONS_DIR, String(season), `week${week}.json`);
  if (!fs.existsSync(filePath)) return null;
  return JSON.parse(fs.readFileSync(filePath, "utf8")) as WeekPrediction;
}

export function listWeeks(): WeekIndex[] {
  if (!fs.existsSync(PREDICTIONS_DIR)) return [];
  const entries: WeekIndex[] = [];
  for (const name of fs.readdirSync(PREDICTIONS_DIR)) {
    const seasonPath = path.join(PREDICTIONS_DIR, name);
    if (!fs.statSync(seasonPath).isDirectory()) continue;
    const season = Number(name);
    if (!Number.isFinite(season)) continue;
    for (const file of fs.readdirSync(seasonPath)) {
      const match = file.match(/^week(\d+)\.json$/);
      if (!match) continue;
      entries.push({ season, week: Number(match[1]) });
    }
  }
  return entries.sort((a, b) =>
    a.season !== b.season ? a.season - b.season : a.week - b.week,
  );
}
