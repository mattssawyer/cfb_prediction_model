import fs from "node:fs";
import path from "node:path";

const MODELS_DIR = path.resolve(process.cwd(), "..", "models");

export type ClassifierMetrics = {
  accuracy: number;
  auc: number;
  brier: number;
  logloss: number;
  n_games: number;
};

export type WinnerTraining = {
  holdout_season: number;
  eval_train_max_season: number;
  n_training_games: number;
  n_features: number;
  best_iteration: number;
  calibration_method: string;
  uncalibrated: ClassifierMetrics;
  calibrated: ClassifierMetrics | null;
};

export type SpreadTraining = {
  holdout_season: number;
  n_training_games: number;
  n_features: number;
  best_iteration: number;
  mae: number;
  rmse: number;
  sign_accuracy: number;
  n_games: number;
};

export type TrainingReport = {
  winner: WinnerTraining | null;
  spread: SpreadTraining | null;
};

function readJson(fileName: string): Record<string, unknown> | null {
  const filePath = path.join(MODELS_DIR, fileName);
  if (!fs.existsSync(filePath)) return null;
  return JSON.parse(fs.readFileSync(filePath, "utf8")) as Record<string, unknown>;
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value != null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function num(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function classifierMetrics(value: unknown): ClassifierMetrics | null {
  const row = asRecord(value);
  if (!row) return null;
  const accuracy = num(row.accuracy);
  const auc = num(row.auc);
  const brier = num(row.brier);
  const logloss = num(row.logloss);
  const n_games = num(row.n_games);
  if (
    accuracy == null ||
    auc == null ||
    brier == null ||
    logloss == null ||
    n_games == null
  ) {
    return null;
  }
  return { accuracy, auc, brier, logloss, n_games };
}

function loadWinner(): WinnerTraining | null {
  const schema = readJson("winner_model_schema.json");
  if (!schema) return null;
  const holdout = asRecord(schema.honest_holdout_metrics);
  const production = asRecord(schema.production_model_metrics);
  const calibration = asRecord(schema.calibration);
  const uncalibrated = classifierMetrics(holdout?.uncalibrated);
  if (!holdout || !production || !uncalibrated) return null;
  return {
    holdout_season: num(production.holdout_season) ?? 0,
    eval_train_max_season: num(production.eval_train_max_season) ?? 0,
    n_training_games: num(production.n_training_games) ?? 0,
    n_features: num(production.n_features) ?? 0,
    best_iteration: num(production.best_iteration) ?? 0,
    calibration_method:
      typeof calibration?.method === "string" ? calibration.method : "none",
    uncalibrated,
    calibrated: classifierMetrics(holdout.calibrated),
  };
}

function loadSpread(): SpreadTraining | null {
  const schema = readJson("spread_model_schema.json");
  if (!schema) return null;
  const holdout = asRecord(schema.honest_holdout_metrics);
  const production = asRecord(schema.production_model_metrics);
  const mae = num(holdout?.mae);
  const rmse = num(holdout?.rmse);
  const sign_accuracy = num(holdout?.sign_accuracy);
  const n_games = num(holdout?.n_games);
  if (!holdout || !production || mae == null || rmse == null || sign_accuracy == null || n_games == null) {
    return null;
  }
  return {
    holdout_season: num(production.holdout_season) ?? 0,
    n_training_games: num(production.n_training_games) ?? 0,
    n_features: num(production.n_features) ?? 0,
    best_iteration: num(production.best_iteration) ?? 0,
    mae,
    rmse,
    sign_accuracy,
    n_games,
  };
}

export function loadTraining(): TrainingReport {
  return { winner: loadWinner(), spread: loadSpread() };
}

export function pct(n: number): string {
  return `${(n * 100).toFixed(1)}%`;
}

export function fixed(n: number, digits: number): string {
  return n.toFixed(digits);
}
