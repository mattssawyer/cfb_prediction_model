"""Hyperparameter sweep for the winner-prediction LightGBM booster.

Uses Optuna's TPE sampler to minimize binary log-loss on the calibration
fold (``CALIB_SEASON``) after early-stopping on that fold. The training
matrix must already exist -- run ``scripts/train.py`` once first to
materialize ``data/training_matrix.parquet``.

Discipline: the honest holdout (``HOLDOUT_SEASON``) is NOT touched during
the sweep. It is evaluated exactly once, at the end, using the best
configuration, to give an unbiased read on generalization.

Usage:
    python scripts/tune.py                      # 200 trials, default seed
    python scripts/tune.py --n-trials 400       # more thorough
    python scripts/tune.py --write params.json  # dump best params to disk
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

# Make ``src/cfb`` importable when running directly (uv/venv scenarios).
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cfb.config import (  # noqa: E402
    CALIB_SEASON,
    HOLDOUT_SEASON,
    TRAIN_MAX_SEASON,
    TRAINING_MATRIX_PATH,
)
from cfb.model import _compute_metrics, _split_features_target  # noqa: E402


# Fixed booster settings we don't want the sampler to touch. ``objective``
# and ``metric`` define the task; ``verbose=-1`` silences per-iteration
# LGBM output; ``max_bin`` and ``force_row_wise`` are stable defaults.
FIXED_PARAMS: dict[str, object] = {
    "objective": "binary",
    "metric": "binary_logloss",
    "verbose": -1,
    "n_jobs": -1,
    "max_bin": 255,
    "force_row_wise": True,
}


def suggest_params(trial: optuna.Trial) -> dict[str, object]:
    """Search space. Ranges chosen for a ~10k-row tabular classifier."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "subsample_freq": trial.suggest_int("subsample_freq", 1, 7),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.5),
    }


def train_and_score(
    params: dict[str, object],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_calib: pd.DataFrame,
    y_calib: pd.Series,
    categorical_features: list[str],
    n_estimators_cap: int = 3000,
    seed: int = 42,
) -> tuple[float, int, np.ndarray]:
    """Fit one booster with early stopping. Returns (calib_logloss, best_iteration, calib_probs)."""
    booster = lgb.LGBMClassifier(
        **FIXED_PARAMS,
        **params,
        n_estimators=n_estimators_cap,
        random_state=seed,
    )
    booster.fit(
        X_train,
        y_train,
        eval_set=[(X_calib, y_calib)],
        eval_metric="binary_logloss",
        categorical_feature=categorical_features if categorical_features else "auto",
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    best_iter = int(booster.best_iteration_ or booster.n_estimators_)
    calib_probs = booster.predict_proba(X_calib, num_iteration=best_iter)[:, 1]
    from sklearn.metrics import log_loss

    return float(log_loss(y_calib.values, calib_probs)), best_iter, calib_probs


def make_objective(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_calib: pd.DataFrame,
    y_calib: pd.Series,
    categorical_features: list[str],
):
    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        calib_ll, best_iter, _ = train_and_score(
            params, X_train, y_train, X_calib, y_calib, categorical_features
        )
        trial.set_user_attr("best_iteration", best_iter)
        return calib_ll

    return objective


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trials", type=int, default=200, help="Number of Optuna trials")
    parser.add_argument("--seed", type=int, default=42, help="Sampler seed")
    parser.add_argument(
        "--write",
        type=Path,
        default=None,
        help="Optional path to write best params as JSON (does not modify config.py).",
    )
    args = parser.parse_args()

    if not TRAINING_MATRIX_PATH.exists():
        print(
            f"Training matrix not found at {TRAINING_MATRIX_PATH}. "
            "Run `python scripts/train.py` once before tuning."
        )
        return 1

    print(f"Loading training matrix from {TRAINING_MATRIX_PATH}")
    matrix = pd.read_parquet(TRAINING_MATRIX_PATH)
    print(f"  {len(matrix)} rows, {matrix.shape[1]} columns")

    train_mask = matrix["season"] <= TRAIN_MAX_SEASON
    calib_mask = matrix["season"] == CALIB_SEASON
    holdout_mask = matrix["season"] == HOLDOUT_SEASON

    X_train, y_train = _split_features_target(matrix, train_mask)
    X_calib, y_calib = _split_features_target(matrix, calib_mask)
    X_holdout, y_holdout = _split_features_target(matrix, holdout_mask)

    categorical_features = X_train.select_dtypes(include=["category"]).columns.tolist()
    print(
        f"  train={len(X_train)} (<= {TRAIN_MAX_SEASON}), "
        f"calib={len(X_calib)} ({CALIB_SEASON}), "
        f"holdout={len(X_holdout)} ({HOLDOUT_SEASON})"
    )
    print(f"  features={X_train.shape[1]}, categoricals={len(categorical_features)}")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.seed, multivariate=True),
        study_name="cfb-winner-lgbm",
    )

    print(f"\nRunning {args.n_trials} Optuna trials (TPE, seed={args.seed})")
    t0 = time.time()
    study.optimize(
        make_objective(X_train, y_train, X_calib, y_calib, categorical_features),
        n_trials=args.n_trials,
        show_progress_bar=False,
    )
    elapsed = time.time() - t0
    print(f"Sweep completed in {elapsed:.1f}s ({elapsed / args.n_trials:.2f}s/trial avg)")

    # ---- Leaderboard ----
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    completed.sort(key=lambda t: t.value)

    print(f"\nTop 5 trials by calib log-loss:")
    print(f"  {'#':>4} {'logloss':>9} {'best_iter':>10}   params")
    for t in completed[:5]:
        param_summary = ", ".join(
            f"{k}={_format_param(v)}" for k, v in t.params.items()
        )
        best_iter = t.user_attrs.get("best_iteration", "?")
        print(f"  {t.number:>4} {t.value:>9.4f} {best_iter:>10}   {param_summary}")

    baseline_trial = completed[0]
    best_params = baseline_trial.params
    best_iter = baseline_trial.user_attrs["best_iteration"]

    print(f"\nBest calib log-loss: {baseline_trial.value:.4f}")

    # ---- Honest holdout evaluation (ONE-SHOT) ----
    print(
        f"\nEvaluating best config on honest holdout (season {HOLDOUT_SEASON}). "
        "This uses the same train->calib fit; holdout is untouched during search."
    )
    _, refit_best_iter, _ = train_and_score(
        best_params,
        X_train,
        y_train,
        X_calib,
        y_calib,
        categorical_features,
    )
    booster = lgb.LGBMClassifier(
        **FIXED_PARAMS,
        **best_params,
        n_estimators=refit_best_iter,
        random_state=args.seed,
    )
    booster.fit(
        X_train,
        y_train,
        categorical_feature=categorical_features if categorical_features else "auto",
    )
    holdout_probs = booster.predict_proba(X_holdout)[:, 1]
    holdout_metrics = _compute_metrics(y_holdout.values, holdout_probs)

    print(f"  honest {HOLDOUT_SEASON} holdout (uncalibrated, tuned params):")
    for k, v in holdout_metrics.items():
        if k == "n_games":
            print(f"    {k}: {v}")
        else:
            print(f"    {k}: {v:.4f}")

    print(
        "\nBaseline comparison (from schema.json produced by scripts/train.py with "
        "current config.LGBM_PARAMS): use those numbers to decide whether these "
        "params meaningfully improve holdout Brier/AUC/logloss before applying."
    )

    # ---- Write params ----
    output = {
        "best_params": best_params,
        "best_iteration": refit_best_iter,
        "calib_logloss": baseline_trial.value,
        "holdout_metrics": holdout_metrics,
        "n_trials": args.n_trials,
        "seed": args.seed,
        "train_max_season": TRAIN_MAX_SEASON,
        "calib_season": CALIB_SEASON,
        "holdout_season": HOLDOUT_SEASON,
    }
    if args.write:
        args.write.write_text(json.dumps(output, indent=2))
        print(f"\nWrote best-params report to {args.write}")

    print("\nBest params (paste into src/cfb/config.py::LGBM_PARAMS if applying):")
    print(json.dumps(best_params, indent=2))

    return 0


def _format_param(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.3g}"
    return str(value)


if __name__ == "__main__":
    sys.exit(main())
