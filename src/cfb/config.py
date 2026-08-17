"""Central configuration: paths, hyperparameters, split boundaries.

All the "magic constants" that used to be scattered across winner_model.py,
weekly_predictor.py, sunday_automation.py, and update_2025_weekly.py live here.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
PREDICTIONS_DIR = PROJECT_ROOT / "predictions"

MODEL_PATH = MODELS_DIR / "winner_model.txt"
SCHEMA_PATH = MODELS_DIR / "winner_model_schema.json"
SPREAD_MODEL_PATH = MODELS_DIR / "spread_model.txt"
SPREAD_MODEL_SCHEMA_PATH = MODELS_DIR / "spread_model_schema.json"
TRAINING_MATRIX_PATH = DATA_DIR / "training_matrix.parquet"

HISTORICAL_START_YEAR = 2015
HISTORICAL_END_YEAR = 2025

# Time-based split. Train on seasons <= TRAIN_MAX_SEASON, early-stop on
# CALIB_SEASON, fit isotonic calibration on that same fold, then score honest
# out-of-sample metrics on HOLDOUT_SEASON. The production booster is finally
# retrained on seasons <= PROD_MAX_SEASON with the frozen best_iteration.
TRAIN_MAX_SEASON = 2023
CALIB_SEASON = 2024
HOLDOUT_SEASON = 2025
PROD_MAX_SEASON = 2025

# Booster hyperparameters. Anything the sweep can tune goes here so that
# ``scripts/tune.py`` and ``scripts/train.py`` share a single source of truth.
# Best values found via 200-trial Optuna TPE sweep on 2015-2023 train,
# 2024 early-stop -> 2025 honest holdout: acc=71.6%, AUC=0.759, Brier=0.194.
LGBM_PARAMS = dict(
    objective="binary",
    metric="binary_logloss",
    random_state=42,
    n_jobs=-1,
    verbose=-1,
    max_bin=255,
    force_row_wise=True,
    learning_rate=0.056072749794376806,
    num_leaves=21,
    max_depth=9,
    min_child_samples=98,
    reg_alpha=0.0028731908510668063,
    reg_lambda=0.0005432473018269719,
    colsample_bytree=0.7396501743797512,
    subsample=0.791398526387835,
    subsample_freq=4,
    min_split_gain=0.14182019178660907,
)

LGBM_SPREAD_PARAMS = dict(
    objective="mae",          # L1; blowouts hurt less than L2
    metric="l1",
    random_state=42,
    n_jobs=-1,
    verbose=-1,
    max_bin=255,
    force_row_wise=True,
    learning_rate=0.056,      # start from the winner sweep; retune later
    num_leaves=21,
    max_depth=9,
    min_child_samples=98,
    reg_alpha=0.00287,
    reg_lambda=0.00054,
    colsample_bytree=0.74,
    subsample=0.791,
    subsample_freq=4,
    min_split_gain=0.142,
)

# Pregame market / probability features. Available for ~95% of completed games but only
# ~12% of upcoming games (CFBD publishes them mid-week after lines close). At inference
# for a Sunday-run predictor most of the current-week slate has no line, so a model
# that leans on them collapses to near-50/50 predictions. Dropping them costs ~1 pp of
# honest-holdout accuracy but yields sharp, consistent predictions for the full slate.
DROP_MARKET_FEATURES = [
    "spread_line",
    "homeWinProbability",
    "awayWinProbability",
    "win_probability_differential",
    "spread",
]

DROP_COLUMNS = ["home_win", "point_differential", "season", "week"]

# Numeric features we refuse to zero-fill silently at inference. Elo is the
# only truly essential input for FBS teams; the in-season stat columns can
# legitimately be zero for week-1 games (that's how they're seen in training).
ESSENTIAL_NUMERIC_FEATURES = [
    "homePregameElo",
    "awayPregameElo",
]


def load_api_key() -> str:
    """Load CFBD_API_KEY from the environment (via .env if present)."""
    load_dotenv(override=True)
    key = os.getenv("CFBD_API_KEY")
    if not key:
        raise RuntimeError(
            "CFBD_API_KEY is not set. Add it to .env or export it in your shell."
        )
    return key


def detect_current_season(today: datetime | None = None) -> int:
    """Best-effort guess of the current CFB season year.

    The CFB regular season runs Aug-Dec. From Aug onward we return the current
    calendar year; before Aug we assume the user is still looking at the prior
    completed season.
    """
    now = today or datetime.utcnow()
    return now.year if now.month >= 8 else now.year - 1


def ensure_dirs() -> None:
    """Create output directories that must exist for pipeline runs."""
    for path in (DATA_DIR, MODELS_DIR, PREDICTIONS_DIR):
        path.mkdir(parents=True, exist_ok=True)
