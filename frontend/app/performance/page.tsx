import { fixed, loadTraining, pct } from "@/lib/training";

function MetricList({
  rows,
}: {
  rows: { label: string; value: string }[];
}) {
  return (
    <dl className="mt-4 divide-y divide-dotted divide-hairline border-y border-dotted border-hairline">
      {rows.map((row) => (
        <div
          key={row.label}
          className="flex items-baseline justify-between gap-4 py-2.5"
        >
          <dt className="font-mono text-[10px] uppercase tracking-[0.12em] text-ink-muted">
            {row.label}
          </dt>
          <dd className="font-mono text-lg tabular-nums leading-none text-ink">
            {row.value}
          </dd>
        </div>
      ))}
    </dl>
  );
}

export default function PerformancePage() {
  const { winner, spread } = loadTraining();

  return (
    <>
      <div className="border-b-2 border-ink pb-8">
        <h1 className="mt-3 text-5xl leading-none font-bold tracking-tight text-ink uppercase">
          Performance
        </h1>
      </div>
      <div className="w-full max-w-2xl">
        <h2 className="mt-8 text-xl leading-none font-bold tracking-tight text-ink uppercase">
          2026 Season Performance
        </h2>
        <p className="mt-3 text-ink">
          The accuracy, AUC, MAE, and RMSE of the models will be updated weekly as the season progresses.
        </p>
        <h2 className="mt-8 text-xl leading-none font-bold tracking-tight text-ink uppercase">
          Training
        </h2>
        <p className="mt-3 text-ink">
          Both models are scored on an honest holdout: train on seasons through{" "}
          {winner?.eval_train_max_season ?? 2023}, early-stop on the next
          season, then measure {winner?.holdout_season ?? 2025} — a year the
          booster never sees while trees or hyperparameters are chosen. The
          production models are then retrained on every season through{" "}
          {winner?.holdout_season ?? 2025}.
        </p>

        {winner ? (
          <>
            <h3 className="mt-10 font-mono text-xs uppercase tracking-[0.15em] text-ink-muted">
              Winner
            </h3>
            <p className="mt-2 text-sm text-ink-muted">
              LightGBM classifier on {winner.uncalibrated.n_games.toLocaleString()}{" "}
              {winner.holdout_season} games. Production fit uses{" "}
              {winner.n_training_games.toLocaleString()} games, {winner.n_features}{" "}
              features, {winner.best_iteration} trees.
            </p>
            <MetricList
              rows={[
                { label: "Accuracy", value: pct(winner.uncalibrated.accuracy) },
                { label: "AUC", value: fixed(winner.uncalibrated.auc, 3) },
                { label: "Brier", value: fixed(winner.uncalibrated.brier, 3) },
                { label: "Log-loss", value: fixed(winner.uncalibrated.logloss, 3) },
              ]}
            />
            {winner.calibration_method !== "isotonic" && winner.calibrated ? (
              <p className="mt-3 text-sm text-ink-muted">
                Isotonic calibration on 2024 raised holdout Brier from{" "}
                {fixed(winner.uncalibrated.brier, 3)} to{" "}
                {fixed(winner.calibrated.brier, 3)}, so the shipped model uses
                raw probabilities.
              </p>
            ) : null}
          </>
        ) : (
          <p className="mt-6 text-sm text-ink-muted">
            Winner holdout metrics are not on disk yet. Retrain to generate{" "}
            <code className="bg-canvas-dim px-1 font-mono text-ink">
              models/winner_model_schema.json
            </code>
            .
          </p>
        )}

        {spread ? (
          <>
            <h3 className="mt-10 font-mono text-xs uppercase tracking-[0.15em] text-ink-muted">
              Spread
            </h3>
            <p className="mt-2 text-sm text-ink-muted">
              LightGBM regressor predicting home minus away, scored on the same{" "}
              {spread.n_games.toLocaleString()} {spread.holdout_season} games.
              MAE and RMSE are in points.
            </p>
            <MetricList
              rows={[
                { label: "MAE", value: fixed(spread.mae, 1) },
                { label: "RMSE", value: fixed(spread.rmse, 1) },
                { label: "Sign accuracy", value: pct(spread.sign_accuracy) },
              ]}
            />
          </>
        ) : (
          <p className="mt-6 text-sm text-ink-muted">
            Spread holdout metrics are not on disk yet. Retrain to generate{" "}
            <code className="bg-canvas-dim px-1 font-mono text-ink">
              models/spread_model_schema.json
            </code>
            .
          </p>
        )}
      </div>
    </>
  );
}
