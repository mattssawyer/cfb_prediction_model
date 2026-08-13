# cfb-predictor

Weekly college football win-probability predictions, produced by a LightGBM
model served as static JSON to a Next.js frontend.

- **Backend**: Python package `cfb` (`src/cfb/`) with two CLI entry points.
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
│   └── pipeline.py              # cfb-pipeline: refresh + predict + write JSON
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
  "season": 2025,
  "week": 4,
  "generated_at": "2025-09-14T08:00:00Z",
  "model": {
    "trained_at": "2026-08-13T02:52:02Z",
    "market_independent": true,
    "holdout_accuracy_calibrated": 0.6949,
    "holdout_auc_calibrated": 0.7650,
    "holdout_brier_calibrated": 0.1939
  },
  "games": [
    {
      "id": 401628495,
      "kickoff": "2025-09-20T19:00:00Z",
      "home_team": "Oregon",
      "away_team": "Oregon State",
      "home_conference": "Big Ten",
      "away_conference": "Pac-12",
      "venue": "Autzen Stadium",
      "neutral_site": false,
      "conference_game": false,
      "home_win_probability": 0.85,
      "predicted_winner": "Oregon",
      "confidence": 0.85
    }
  ]
}
```

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
python scripts/pipeline.py --season 2025 --week 4 --force

# Retrain (only needed after an offseason with new historical data)
python scripts/train.py
```

Get a CFBD API key at <https://collegefootballdata.com/key>.

### Frontend dev

```bash
cd frontend
npm install
npm run dev       # http://localhost:3000
npm run build     # produces frontend/out/ (static export)
```

The frontend reads `../predictions/*.json` at build time, so run the pipeline
first if you want real data.

## Model

- Binary classifier: LightGBM (`objective="binary"`).
- Three-stage time-based split:
  1. Train on `season <= 2022`, early-stop on `season == 2023`.
  2. Fit isotonic calibration on 2023 raw probabilities.
  3. Score honest holdout on `season == 2024` (uncalibrated + calibrated).
  4. Retrain production on `season <= 2024` with the frozen `best_iteration`.
- Pregame market features (`spread`, `homeWinProbability`, …) are dropped from
  training by default. They're only available for ~12% of upcoming games, and
  including them makes the model collapse to near-50/50 for the rest.
- ~334 features per game after all joins.
- Honest 2024 calibrated holdout: 69.5% accuracy, 0.194 Brier score.

## Automating weekly runs (GitHub Actions)

1. Push this repo to GitHub.
2. Settings → Secrets and variables → Actions → New repository secret:
   - Name: `CFBD_API_KEY`
   - Value: your CFBD API key.
3. The workflow runs every Sunday 08:00 UTC. To trigger it manually, go to
   Actions → Weekly Predictions → Run workflow (with optional season/week
   overrides).

Every run:
- Refreshes CFBD data for the current season.
- Builds features and runs the model.
- Commits `predictions/{season}/week{N}.json` back to `main`.
- Vercel (or whatever the frontend host is) auto-redeploys on the commit.

## Deploying the frontend to Vercel

1. Push the repo to GitHub.
2. On Vercel: New Project → import the repo.
3. Root directory: `frontend`.
4. Framework preset: Next.js (auto-detected).
5. Deploy.

Every push to `main` — including the weekly bot commit — triggers a redeploy.

## Anatomy of a weekly run

```
python scripts/pipeline.py
  ├─ CFBDataLoader.load_games(2025, refresh=True)
  ├─ CFBDataLoader.load_batch_weekly_stats / advanced / talent / lines / winprob
  ├─ CFBDataLoader.create_ml_features_dataset(week_games)  # 50 games x 336 features
  ├─ features.transform_for_lightgbm(...)                  # → 328 cols, categoricals aligned
  ├─ features._align_to_schema(...)                        # → 334 cols exactly matching schema
  ├─ booster.predict(X)                                    # raw probabilities
  ├─ model.apply_calibration(raw)                          # isotonic lookup
  └─ write predictions/2025/week{N}.json + predictions/latest.json
```

End-to-end takes ~5s cold (with CFBD refresh) and <1s warm.
