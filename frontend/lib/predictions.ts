/**
 * Load prediction JSON files at build time.
 *
 * Predictions live under ../predictions/ in the parent repo. Everything
 * is read synchronously from disk — no runtime fetching, no API server.
 */

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
};

export type ModelMeta = {
  trained_at: string | null;
  market_independent: boolean;
  holdout_accuracy_calibrated: number | null;
  holdout_auc_calibrated: number | null;
  holdout_brier_calibrated: number | null;
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

/**
 * Load the newest prediction file (predictions/latest.json). Returns null if
 * no predictions have been produced yet.
 */
export function loadLatest(): WeekPrediction | null {
  const filePath = path.join(PREDICTIONS_DIR, "latest.json");
  if (!fs.existsSync(filePath)) return null;
  const raw = fs.readFileSync(filePath, "utf8");
  return JSON.parse(raw) as WeekPrediction;
}

/**
 * Load predictions for a specific season and week. Returns null if the file
 * doesn't exist.
 */
export function loadWeek(season: number, week: number): WeekPrediction | null {
  const filePath = path.join(PREDICTIONS_DIR, String(season), `week${week}.json`);
  if (!fs.existsSync(filePath)) return null;
  const raw = fs.readFileSync(filePath, "utf8");
  return JSON.parse(raw) as WeekPrediction;
}

/**
 * Walk the predictions directory and return every {season, week} pair that
 * has a JSON file on disk. Used to pre-generate static routes.
 */
export function listAllWeeks(): WeekIndex[] {
  if (!fs.existsSync(PREDICTIONS_DIR)) return [];
  const entries: WeekIndex[] = [];
  for (const seasonName of fs.readdirSync(PREDICTIONS_DIR)) {
    const seasonPath = path.join(PREDICTIONS_DIR, seasonName);
    if (!fs.statSync(seasonPath).isDirectory()) continue;
    const season = Number(seasonName);
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
