"""Winner-prediction model: train, load, and predict.

Training happens in three stages with proper time-based splits:

Stage 1a: train evaluation model on ``season <= TRAIN_MAX_SEASON`` (2022),
  use ``season == CALIB_SEASON`` (2023) for early-stopping. Capture best_iteration.

Stage 1b: fit an isotonic ``IsotonicRegression`` on the Stage 1a booster's
  predictions over the 2023 fold. Serialize its X/Y thresholds into the schema.

Stage 1c: score the Stage 1a model on ``season == HOLDOUT_SEASON`` (2024)
  uncalibrated and calibrated. These are the honest generalization numbers.

Stage 2 (production): retrain on ``season <= PROD_MAX_SEASON`` (2024) with
  ``n_estimators = best_iteration``. This is the deployed booster.

Deployed artifacts (under ``models/``):
  winner_model.txt          -- LightGBM booster (portable text format)
  winner_model_schema.json  -- feature schema, categorical values, calibration
                                lookup, honest holdout metrics.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
)

from .config import (
    CALIB_SEASON,
    DROP_COLUMNS,
    DROP_MARKET_FEATURES,
    HOLDOUT_SEASON,
    LGBM_PARAMS,
    LGBM_SPREAD_PARAMS,
    MODEL_PATH,
    MODELS_DIR,
    PROD_MAX_SEASON,
    SCHEMA_PATH,
    SPREAD_MODEL_PATH,
    SPREAD_MODEL_SCHEMA_PATH,
    TRAIN_MAX_SEASON,
    TRAINING_MATRIX_PATH,
    ensure_dirs,
)
from .data import CFBDataLoader
from .features import _align_to_schema, build_week_features


# =============================================================================
# Training
# =============================================================================
def train_spread(training_matrix: pd.DataFrame | None = None) -> dict:
    """Train the spread-prediction model end-to-end.

    ``training_matrix`` may be an in-memory DataFrame (used by tests). If None
    the transformed parquet at ``TRAINING_MATRIX_PATH`` is loaded.
    """
    ensure_dirs()

    if training_matrix is None:
        print(f"Loading training matrix from {TRAINING_MATRIX_PATH}")
        training_matrix = pd.read_parquet(TRAINING_MATRIX_PATH)
    print(f"  {len(training_matrix)} rows, {training_matrix.shape[1]} columns")

    train_mask = training_matrix["season"] <= TRAIN_MAX_SEASON
    calib_mask = training_matrix["season"] == CALIB_SEASON
    holdout_mask = training_matrix["season"] == HOLDOUT_SEASON

    X_train, y_train = _split_features_target(training_matrix, train_mask, target="point_differential")
    X_calib, y_calib = _split_features_target(training_matrix, calib_mask, target="point_differential")
    X_holdout, y_holdout = _split_features_target(training_matrix, holdout_mask, target="point_differential")

    for name, X in (("train", X_train), ("calibration", X_calib), ("holdout", X_holdout)):
        if len(X) == 0:
            raise RuntimeError(
                f"{name} split is empty; training matrix must cover all needed seasons."
            )

    categorical_features = X_train.select_dtypes(include=["category"]).columns.tolist()
    print(
        f"  train={len(X_train)}, calib={len(X_calib)}, holdout={len(X_holdout)} | "
        f"features={X_train.shape[1]}, categoricals={len(categorical_features)}"
    )
    if categorical_features:
        print(f"  categorical: {categorical_features}")

    # ----- Stage 1a -----
    print(f"\nStage 1a: train season <= {TRAIN_MAX_SEASON}, early-stop on {CALIB_SEASON}")
    eval_model = lgb.LGBMRegressor(**LGBM_SPREAD_PARAMS, n_estimators=3000)
    eval_model.fit(
        X_train,
        y_train,
        eval_set=[(X_calib, y_calib)],
        eval_metric="l1",
        categorical_feature=categorical_features if categorical_features else "auto",
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )
    best_iteration = int(eval_model.best_iteration_ or eval_model.n_estimators_)
    print(f"Stage 1a: best_iteration = {best_iteration}")

    # ----- Stage 1b: holdout MAE/RMSE -----
    pred = eval_model.predict(X_holdout, num_iteration=best_iteration)
    metrics = {
        "mae": float(mean_absolute_error(y_holdout, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_holdout, pred))),
        "sign_accuracy": float(np.mean(np.sign(pred) == np.sign(y_holdout))),
        "n_games": int(len(y_holdout)),
    }
    _print_metrics(f"honest {HOLDOUT_SEASON} holdout (margin)", metrics)

    # ----- Stage 2: production retrain -----
    print(f"\nStage 2: train production on season <= {PROD_MAX_SEASON}, n_estimators={best_iteration}")
    prod_mask = training_matrix["season"] <= PROD_MAX_SEASON
    X_prod, y_prod = _split_features_target(training_matrix, prod_mask, target="point_differential")

    prod_categoricals = X_prod.select_dtypes(include=["category"]).columns.tolist()
    if set(prod_categoricals) != set(categorical_features):
        raise RuntimeError(
            "Categorical column mismatch between stages: "
            f"eval={sorted(categorical_features)}, prod={sorted(prod_categoricals)}"
        )

    print(f"  training rows: {len(X_prod)} | features: {X_prod.shape[1]}")
    prod_model = lgb.LGBMRegressor(**LGBM_SPREAD_PARAMS, n_estimators=best_iteration)
    prod_model.fit(
        X_prod,
        y_prod,
        categorical_feature=categorical_features if categorical_features else "auto",
    )

    # ----- Persist -----
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    prod_model.booster_.save_model(str(SPREAD_MODEL_PATH))
    print(f"\nSaved booster to {SPREAD_MODEL_PATH}")

    categorical_values = {
        col: [v for v in X_prod[col].cat.categories.tolist() if pd.notna(v)]
        for col in categorical_features
    }
    schema = {
        "task": "regression",
        "target": "point_differential",
        "feature_columns": X_prod.columns.tolist(),
        "categorical_features": categorical_features,
        "categorical_values": categorical_values,
        "excluded_features": DROP_MARKET_FEATURES,
        "market_independent": True,
        "honest_holdout_metrics": metrics,
        "production_model_metrics": {
            "n_training_games": int(len(X_prod)),
            "n_features": int(X_prod.shape[1]),
            "best_iteration": best_iteration,
            "train_max_season": PROD_MAX_SEASON,
            "eval_train_max_season": TRAIN_MAX_SEASON,
            "holdout_season": HOLDOUT_SEASON,
        },
        "lightgbm_version": lgb.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(SPREAD_MODEL_SCHEMA_PATH, "w") as fh:
        json.dump(schema, fh, indent=2, default=str)
    print(f"Saved schema to {SPREAD_MODEL_SCHEMA_PATH}")

    print("\nTop 10 features by importance:")
    importance_df = pd.DataFrame({
        "feature": X_prod.columns,
        "importance": prod_model.feature_importances_,
    }).sort_values("importance", ascending=False)
    for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
        print(f"  {i:>2}. {row['feature']:<45} {row['importance']:>7.1f}")

    return {
        "best_iteration": best_iteration,
        "honest_holdout_metrics": metrics,
        "model_path": str(SPREAD_MODEL_PATH),
        "schema_path": str(SPREAD_MODEL_SCHEMA_PATH),
    }

def train(training_matrix: pd.DataFrame | None = None) -> dict:
    """Train the winner-prediction model end-to-end.

    ``training_matrix`` may be an in-memory DataFrame (used by tests). If None
    the transformed parquet at ``TRAINING_MATRIX_PATH`` is loaded.
    """
    ensure_dirs()

    if training_matrix is None:
        print(f"Loading training matrix from {TRAINING_MATRIX_PATH}")
        training_matrix = pd.read_parquet(TRAINING_MATRIX_PATH)
    print(f"  {len(training_matrix)} rows, {training_matrix.shape[1]} columns")

    train_mask = training_matrix["season"] <= TRAIN_MAX_SEASON
    calib_mask = training_matrix["season"] == CALIB_SEASON
    holdout_mask = training_matrix["season"] == HOLDOUT_SEASON

    X_train, y_train = _split_features_target(training_matrix, train_mask)
    X_calib, y_calib = _split_features_target(training_matrix, calib_mask)
    X_holdout, y_holdout = _split_features_target(training_matrix, holdout_mask)

    for name, X in (("train", X_train), ("calibration", X_calib), ("holdout", X_holdout)):
        if len(X) == 0:
            raise RuntimeError(
                f"{name} split is empty; training matrix must cover all needed seasons."
            )

    categorical_features = X_train.select_dtypes(include=["category"]).columns.tolist()
    print(
        f"  train={len(X_train)}, calib={len(X_calib)}, holdout={len(X_holdout)} | "
        f"features={X_train.shape[1]}, categoricals={len(categorical_features)}"
    )
    if categorical_features:
        print(f"  categorical: {categorical_features}")

    # ----- Stage 1a -----
    print(f"\nStage 1a: train season <= {TRAIN_MAX_SEASON}, early-stop on {CALIB_SEASON}")
    eval_model = lgb.LGBMClassifier(**LGBM_PARAMS, n_estimators=3000)
    eval_model.fit(
        X_train,
        y_train,
        eval_set=[(X_calib, y_calib)],
        eval_metric="binary_logloss",
        categorical_feature=categorical_features if categorical_features else "auto",
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )
    best_iteration = int(eval_model.best_iteration_ or eval_model.n_estimators_)
    print(f"Stage 1a: best_iteration = {best_iteration}")

    # ----- Stage 1b: isotonic calibration -----
    print(f"\nStage 1b: fit isotonic calibrator on season {CALIB_SEASON}")
    raw_calib_probs = eval_model.predict_proba(X_calib, num_iteration=best_iteration)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(raw_calib_probs, y_calib.values)

    x_thresholds = np.asarray(calibrator.X_thresholds_, dtype=float).tolist()
    y_thresholds = np.asarray(calibrator.y_thresholds_, dtype=float).tolist()
    print(f"  isotonic map: {len(x_thresholds)} knots")

    cal_on_calib_probs = np.interp(raw_calib_probs, x_thresholds, y_thresholds)
    raw_calib_brier = brier_score_loss(y_calib.values, raw_calib_probs)
    cal_calib_brier = brier_score_loss(y_calib.values, cal_on_calib_probs)
    print(f"  calibration Brier: raw={raw_calib_brier:.4f} -> calibrated={cal_calib_brier:.4f}")

    # ----- Stage 1c: honest holdout metrics -----
    print(f"\nStage 1c: score honest holdout (season {HOLDOUT_SEASON})")
    raw_holdout_probs = eval_model.predict_proba(X_holdout, num_iteration=best_iteration)[:, 1]
    cal_holdout_probs = np.clip(np.interp(raw_holdout_probs, x_thresholds, y_thresholds), 0.0, 1.0)

    uncalibrated_metrics = _compute_metrics(y_holdout.values, raw_holdout_probs)
    calibrated_metrics = _compute_metrics(y_holdout.values, cal_holdout_probs)
    _print_metrics(f"honest {HOLDOUT_SEASON} holdout (uncalibrated)", uncalibrated_metrics)
    _print_metrics(f"honest {HOLDOUT_SEASON} holdout (calibrated)", calibrated_metrics)

    # Only ship calibration if it actually helps the honest holdout Brier.
    # When it doesn't (e.g. the calibration fold is a poor representative of
    # the holdout, as can happen across regime shifts in college football),
    # we serialize the identity mapping so ``apply_calibration`` becomes a
    # no-op, but keep the metrics for both variants in the schema.
    ship_calibration = calibrated_metrics["brier"] <= uncalibrated_metrics["brier"]
    if not ship_calibration:
        print(
            f"\nCalibration hurts holdout Brier "
            f"({uncalibrated_metrics['brier']:.4f} raw -> "
            f"{calibrated_metrics['brier']:.4f} calibrated). "
            "Disabling calibration in the schema; using raw booster probs."
        )

    # ----- Stage 2: production retrain -----
    print(f"\nStage 2: train production on season <= {PROD_MAX_SEASON}, n_estimators={best_iteration}")
    prod_mask = training_matrix["season"] <= PROD_MAX_SEASON
    X_prod, y_prod = _split_features_target(training_matrix, prod_mask)

    prod_categoricals = X_prod.select_dtypes(include=["category"]).columns.tolist()
    if set(prod_categoricals) != set(categorical_features):
        raise RuntimeError(
            "Categorical column mismatch between stages: "
            f"eval={sorted(categorical_features)}, prod={sorted(prod_categoricals)}"
        )

    print(f"  training rows: {len(X_prod)} | features: {X_prod.shape[1]}")
    prod_model = lgb.LGBMClassifier(**LGBM_PARAMS, n_estimators=best_iteration)
    prod_model.fit(
        X_prod,
        y_prod,
        categorical_feature=categorical_features if categorical_features else "auto",
    )

    # ----- Persist -----
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    prod_model.booster_.save_model(str(MODEL_PATH))
    print(f"\nSaved booster to {MODEL_PATH}")

    categorical_values = {
        col: [v for v in X_prod[col].cat.categories.tolist() if pd.notna(v)]
        for col in categorical_features
    }
    schema = {
        "feature_columns": X_prod.columns.tolist(),
        "categorical_features": categorical_features,
        "categorical_values": categorical_values,
        "excluded_features": DROP_MARKET_FEATURES,
        "market_independent": True,
        "honest_holdout_metrics": {
            "uncalibrated": uncalibrated_metrics,
            "calibrated": calibrated_metrics,
        },
        "calibration": (
            {
                "method": "isotonic",
                "fitted_on_season": CALIB_SEASON,
                "n_calibration_games": int(len(y_calib)),
                "X_thresholds": x_thresholds,
                "y_thresholds": y_thresholds,
            }
            if ship_calibration
            else {
                "method": "none",
                "fitted_on_season": CALIB_SEASON,
                "n_calibration_games": int(len(y_calib)),
                "reason": "calibrated_brier_exceeded_uncalibrated_on_holdout",
            }
        ),
        "production_model_metrics": {
            "n_training_games": int(len(X_prod)),
            "n_features": int(X_prod.shape[1]),
            "best_iteration": best_iteration,
            "train_max_season": PROD_MAX_SEASON,
            "eval_train_max_season": TRAIN_MAX_SEASON,
            "holdout_season": HOLDOUT_SEASON,
        },
        "lightgbm_version": lgb.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(SCHEMA_PATH, "w") as fh:
        json.dump(schema, fh, indent=2, default=str)
    print(f"Saved schema to {SCHEMA_PATH}")

    print("\nTop 10 features by importance:")
    importance_df = pd.DataFrame({
        "feature": X_prod.columns,
        "importance": prod_model.feature_importances_,
    }).sort_values("importance", ascending=False)
    for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
        print(f"  {i:>2}. {row['feature']:<45} {row['importance']:>7.1f}")

    return {
        "best_iteration": best_iteration,
        "honest_holdout_metrics": {
            "uncalibrated": uncalibrated_metrics,
            "calibrated": calibrated_metrics,
        },
        "model_path": str(MODEL_PATH),
        "schema_path": str(SCHEMA_PATH),
    }


def _split_features_target(df: pd.DataFrame, mask: pd.Series, target: str="home_win") -> tuple[pd.DataFrame, pd.Series]:
    subset = df[mask]
    drop_cols = DROP_COLUMNS + [c for c in DROP_MARKET_FEATURES if c in subset.columns]
    X = subset.drop(columns=drop_cols)
    y = subset[target]
    if target == "home_win":
        y = y.astype(int)
    else: 
        y = pd.to_numeric(y, errors="coerce")
    
    valid = y.notna()
    return X.loc[valid], y.loc[valid]


def _compute_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray) -> dict:
    y_pred = (y_pred_proba >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auc": float(roc_auc_score(y_true, y_pred_proba)),
        "brier": float(brier_score_loss(y_true, y_pred_proba)),
        "logloss": float(log_loss(y_true, y_pred_proba)),
        "n_games": int(len(y_true)),
    }


def _print_metrics(label: str, metrics: dict) -> None:
    print(f"  {label}:")
    for key, value in metrics.items():
        if key == "n_games":
            print(f"    {key}: {value}")
        else:
            print(f"    {key}: {value:.4f}")


# =============================================================================
# Loading
# =============================================================================
def load(
    model_path: Path | None = None,
    schema_path: Path | None = None,
) -> tuple[lgb.Booster, dict]:
    """Load the deployed booster + schema, raising if either is missing."""
    model_path = model_path or MODEL_PATH
    schema_path = schema_path or SCHEMA_PATH

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained booster not found at {model_path}. Run `python scripts/train.py`."
        )
    if not schema_path.exists():
        raise FileNotFoundError(
            f"Schema not found at {schema_path}. Retrain to produce the schema."
        )

    booster = lgb.Booster(model_file=str(model_path))
    with open(schema_path) as fh:
        schema = json.load(fh)

    required = {"feature_columns", "categorical_features", "categorical_values"}
    missing = required - set(schema.keys())
    if missing:
        raise ValueError(f"Schema is missing required keys: {missing}")

    return booster, schema


def _predict_margins(X: pd.DataFrame) -> tuple[np.ndarray | None, dict | None]:
    """Run the spread booster on the same feature matrix, if artifacts exist."""
    if not SPREAD_MODEL_PATH.exists() or not SPREAD_MODEL_SCHEMA_PATH.exists():
        print("Spread model not found; skipping predicted_margin.")
        return None, None

    spread_booster, spread_schema = load(SPREAD_MODEL_PATH, SPREAD_MODEL_SCHEMA_PATH)
    X_spread = X
    expected = spread_schema.get("feature_columns") or []
    if list(X.columns) != expected:
        X_spread = _align_to_schema(X, spread_schema)

    margins = np.asarray(spread_booster.predict(X_spread), dtype=float)
    holdout = spread_schema.get("honest_holdout_metrics") or {}
    meta = {
        "trained_at": spread_schema.get("trained_at"),
        "holdout_mae": _optional_float(holdout.get("mae")),
        "holdout_rmse": _optional_float(holdout.get("rmse")),
        "holdout_sign_accuracy": _optional_float(holdout.get("sign_accuracy")),
    }
    mae = meta["holdout_mae"]
    print(f"Spread holdout MAE={mae:.2f}" if mae is not None else "Spread model loaded")
    return margins, meta


def apply_calibration(raw_probs: np.ndarray, schema: dict) -> np.ndarray:
    """Apply the isotonic calibration lookup baked into the schema, if any."""
    cal = schema.get("calibration") if isinstance(schema, dict) else None
    if not cal:
        return np.asarray(raw_probs, dtype=float)

    method = cal.get("method")
    xs = cal.get("X_thresholds")
    ys = cal.get("y_thresholds")
    if method == "none":
        return np.asarray(raw_probs, dtype=float)
    if method != "isotonic" or not xs or not ys or len(xs) != len(ys):
        print(
            f"WARNING: calibration block unusable "
            f"(method={method!r}, n_x={len(xs) if xs else 0}, n_y={len(ys) if ys else 0}); "
            "returning raw probabilities."
        )
        return np.asarray(raw_probs, dtype=float)

    xs_arr = np.asarray(xs, dtype=float)
    ys_arr = np.asarray(ys, dtype=float)
    calibrated = np.interp(np.asarray(raw_probs, dtype=float), xs_arr, ys_arr)
    return np.clip(calibrated, 0.0, 1.0)


# =============================================================================
# High-level prediction workflow
# =============================================================================
def predict_week(
    season: int,
    week: int,
    refresh_data: bool = True,
    loader: CFBDataLoader | None = None,
) -> dict:
    """End-to-end: load model, build features, predict, and return the JSON contract.

    Returns the dict that will be serialized to
    ``predictions/{season}/week{N}.json``. See README for the exact schema.
    """
    print(f"\n=== Predict season {season} week {week} ===")

    booster, schema = load()
    _print_model_meta(schema)

    loader = loader or CFBDataLoader()
    X, week_games = build_week_features(loader, season, week, schema, refresh_data=refresh_data)

    if X.empty:
        return _empty_prediction(season, week, schema)

    raw_probs = booster.predict(X)
    calibrated = apply_calibration(raw_probs, schema)

    margins, spread_meta = _predict_margins(X)

    games = []
    n = len(calibrated)
    for i, (_, game) in enumerate(week_games.head(n).iterrows()):
        home_prob = float(calibrated[i])
        away_prob = 1.0 - home_prob
        home_team = game["homeTeam"]
        away_team = game["awayTeam"]
        predicted_winner = home_team if home_prob >= 0.5 else away_team
        games.append({
            "id": _optional_int(game.get("id")),
            "kickoff": _iso_date(game.get("startDate")),
            "home_team": str(home_team),
            "away_team": str(away_team),
            "home_conference": _optional_str(game.get("homeConference")),
            "away_conference": _optional_str(game.get("awayConference")),
            "venue": _optional_str(game.get("venue")),
            "neutral_site": bool(game.get("neutralSite", False)),
            "conference_game": bool(game.get("conferenceGame", False)),
            "home_win_probability": round(home_prob, 4),
            "predicted_winner": predicted_winner,
            "confidence": round(max(home_prob, away_prob), 4),
            "predicted_margin": (
                round(float(margins[i]), 1) if margins is not None else None
            ),
            # CFBD home-team line: negative = home favorite. Same convention as Vegas.
            "vegas_spread": _optional_float(game.get("spread_line")),
        })

    games.sort(key=lambda g: (g["kickoff"] or "", -g["confidence"]))

    payload = {
        "season": season,
        "week": week,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": {**_model_meta(schema), "spread": spread_meta},
        "games": games,
    }
    print(f"Produced {len(games)} predictions")
    return payload


def _effective_holdout_metrics(schema: dict) -> tuple[dict | None, str]:
    """Return (metrics_dict, variant_name) that reflect what the model ships.

    When calibration is disabled in the schema (method != "isotonic"), the
    raw-booster metrics are the honest read of holdout quality.
    """
    holdout = schema.get("honest_holdout_metrics", {}) or {}
    calibration = schema.get("calibration", {}) or {}
    if calibration.get("method") == "isotonic":
        variant = "calibrated"
    else:
        variant = "uncalibrated"
    metrics = holdout.get(variant) if isinstance(holdout.get(variant), dict) else holdout
    return metrics, variant


def _print_model_meta(schema: dict) -> None:
    metrics, variant = _effective_holdout_metrics(schema)
    if metrics:
        print(
            f"Model {variant} holdout: acc={metrics.get('accuracy', float('nan')):.4f} "
            f"AUC={metrics.get('auc', float('nan')):.4f} Brier={metrics.get('brier', float('nan')):.4f}"
        )


def _model_meta(schema: dict) -> dict:
    metrics, variant = _effective_holdout_metrics(schema)
    return {
        "trained_at": schema.get("trained_at"),
        "market_independent": bool(schema.get("market_independent", False)),
        "calibration_method": (schema.get("calibration") or {}).get("method", "none"),
        "holdout_variant": variant,
        "holdout_accuracy": _optional_float(metrics.get("accuracy") if metrics else None),
        "holdout_auc": _optional_float(metrics.get("auc") if metrics else None),
        "holdout_brier": _optional_float(metrics.get("brier") if metrics else None),
    }


def _empty_prediction(season: int, week: int, schema: dict) -> dict:
    return {
        "season": season,
        "week": week,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": _model_meta(schema),
        "games": [],
    }


def _iso_date(value) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, str):
        return value
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _optional_str(value) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return str(value)


def _optional_int(value) -> int | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value) -> float | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        n = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(n):
        return None
    return n
