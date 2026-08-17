# cfb-predictor

Weekly college football win-probability predictions, produced by a LightGBM
model served as static JSON to a Next.js frontend.

- **Backend**: Python package `cfb` (`src/cfb/`) with three CLI entry points
  (`cfb-train`, `cfb-pipeline`, `cfb-tune`).
- **Frontend**: Next.js 16 App Router (`frontend/`), fully static export.
- **Weekly refresh**: GitHub Actions cron (`.github/workflows/weekly.yml`).

## Repo layout

```
.
├── src/cfb/                     # Python package
│   ├── config.py                # paths, hyperparams, split boundaries
│   ├── data.py                  # unified CFBD loader (year-agnostic)
│   ├── features.py              # training matrix + per-week feature vector
│   └── model.py                 # train + load + predict_week
├── scripts/
│   ├── train.py                 # cfb-train: rebuild dataset + retrain
│   ├── pipeline.py              # cfb-pipeline: refresh + predict + write JSON
│   └── tune.py                  # cfb-tune: Optuna hyperparameter sweep
├── models/
│   ├── winner_model.txt         # LightGBM booster (portable text)
│   └── winner_model_schema.json # feature schema + calibration + metrics
├── data/                        # parquet cache (gitignored)
├── predictions/                 # weekly JSON outputs (committed)
│   ├── latest.json
│   └── {season}/week{N}.json
├── frontend/                    # Next.js app
├── .github/workflows/weekly.yml # Sunday cron
└── pyproject.toml
```

## The JSON contract

Every weekly run produces one file at `predictions/{season}/week{N}.json`
plus a copy at `predictions/latest.json`:

```json
{
  "season": 2026,
  "week": 1,
  "generated_at": "2026-08-13T03:47:00Z",
  "model": {
    "trained_at": "2026-08-13T03:47:00Z",
    "market_independent": true,
    "calibration_method": "none",
    "holdout_variant": "uncalibrated",
    "holdout_accuracy": 0.7156,
    "holdout_auc": 0.7589,
    "holdout_brier": 0.1942
  },
  "games": [
    {
      "id": 401752722,
      "kickoff": "2026-08-29T23:00:00Z",
      "home_team": "USC",
      "away_team": "San José State",
      "home_conference": "Big Ten",
      "away_conference": "Mountain West",
      "venue": "United Airlines Field at the Los Angeles Memorial Coliseum",
      "neutral_site": false,
      "conference_game": false,
      "home_win_probability": 0.867,
      "predicted_winner": "USC",
      "confidence": 0.867
    }
  ]
}
```

`calibration_method` is `"isotonic"` when the isotonic calibrator was shipped
and `"none"` when training auto-disabled it (see the Model section).
`holdout_variant` tells the frontend which metrics were actually reported —
`calibrated` or `uncalibrated` — so the numbers always describe the booster
the app is running.

The frontend reads these files at build time and pre-renders every route as
static HTML. There is no backend server.

## Local setup

Requires Python 3.11+ and Node 20+.

```bash
# One-time setup
cp .env.example .env                    # then paste your CFBD_API_KEY
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Predict the next unplayed week
python scripts/pipeline.py               # auto-detects season and week
python scripts/pipeline.py --season 2026 --week 1 --force

# Retrain (only needed after an offseason with new historical data)
python scripts/train.py

# Re-tune hyperparameters (optional; sweep takes ~6 min)
pip install -e ".[tune]"
python scripts/tune.py --n-trials 200 --write models/tuned_params.json
```

Get a CFBD API key at <https://collegefootballdata.com/key>.

## Model

- Binary classifier: LightGBM (`objective="binary"`), currently 109 trees.
- Time-based split (configured for the 2026 season):
  1. Train on `season <= 2023`, early-stop on `season == 2024`.
  2. Fit isotonic calibration on the 2024 fold. If calibrated Brier is worse
     than uncalibrated on the honest holdout, calibration is auto-disabled
     and the schema records `method = "none"`.
  3. Score honest holdout on `season == 2025` (both variants).
  4. Retrain production on `season <= 2025` with the frozen `best_iteration`.
- Pregame market features (`spread`, `homeWinProbability`, …) are dropped from
  training by default. They're only available for ~12% of upcoming games, and
  including them makes the model collapse to near-50/50 for the rest.
- 325 features per game after all joins.
- Hyperparameters found via a 200-trial Optuna TPE sweep on the same
  train → early-stop split; rerun with `python scripts/tune.py --n-trials 200`.

### Honest holdout metrics

The honest holdout is the 2025 season — the model never sees it during
training or hyperparameter search, so these numbers are an unbiased estimate
of what the shipped booster will do on future games.

| Metric | Value |
|---|---|
| Accuracy | **71.6%** |
| AUC | 0.759 |
| Brier score | 0.194 |
| Log-loss | 0.570 |
| Games | 844 |

Calibration hurt Brier on 2025 (raw 0.194 → calibrated 0.199), so it is
currently disabled and the app reports uncalibrated probabilities.

## Anatomy of a weekly run

```
python scripts/pipeline.py
  ├─ CFBDataLoader.load_games(season, refresh=True)
  ├─ CFBDataLoader.load_batch_weekly_stats / advanced / talent / lines / winprob
  ├─ CFBDataLoader.create_ml_features_dataset(week_games)  # N games × ~336 raw cols
  ├─ features.transform_for_lightgbm(...)                  # → ~59-334 cols, categoricals aligned
  ├─ features._align_to_schema(...)                        # → 325 cols exactly matching schema
  ├─ booster.predict(X)                                    # raw probabilities
  ├─ model.apply_calibration(raw)                          # no-op when method="none"
  └─ write predictions/{season}/week{N}.json + predictions/latest.json
```

End-to-end takes ~2 minutes cold (first CFBD refresh of the season) and
<2 seconds warm.
