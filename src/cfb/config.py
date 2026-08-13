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
TRAINING_MATRIX_PATH = DATA_DIR / "training_matrix.parquet"

HISTORICAL_START_YEAR = 2015
HISTORICAL_END_YEAR = 2024

TRAIN_MAX_SEASON = 2022
CALIB_SEASON = 2023
HOLDOUT_SEASON = 2024
PROD_MAX_SEASON = 2024

LGBM_PARAMS = dict(
    objective="binary",
    random_state=42,
    max_bin=255,
    n_jobs=-1,
    max_depth=10,
    num_leaves=90,
    reg_alpha=0.3848493186606046,
    reg_lambda=0.2829952798614409,
    subsample=0.9820795722130331,
    learning_rate=0.05,
    verbose=-1,
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

ESSENTIAL_NUMERIC_FEATURES = [
    "homePregameElo",
    "awayPregameElo",
    "home_offensivePPA",
    "away_offensivePPA",
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
