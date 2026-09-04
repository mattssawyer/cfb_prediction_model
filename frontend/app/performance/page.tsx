import { loadSeasonAccuracy } from "@/lib/accuracy";
import { loadLatest } from "@/lib/predictions";
import { fixed, loadTraining, pct } from "@/lib/training";

function pctOrDash(n: number | null): string {
  return n == null ? "—" : pct(n);
}

function fixedOrDash(n: number | null, digits: number): string {
  return n == null ? "—" : fixed(n, digits);
}

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
  const latest = loadLatest();
  const accuracy = latest ? loadSeasonAccuracy(latest.season) : null;
  const season = latest?.season ?? new Date().getFullYear();

  return (
    <>
      <div className="border-b-2 border-ink pb-8">
        <h1 className="mt-3 text-5xl leading-none font-bold tracking-tight text-ink uppercase">
          Performance
        </h1>
      </div>
      <div className="w-full max-w-2xl">
        <h2 className="mt-8 text-xl leading-none font-bold tracking-tight text-ink uppercase">
          {season} Season Performance
        </h2>
        {accuracy && accuracy.season_to_date.games_graded > 0 ? (
          <>
            <p className="mt-3 text-ink">
              Graded against final scores as games are played, updated every
              Sunday alongside the new predictions. ATS is whether the model
              picked the covering side of the Vegas home line.
            </p>
            <MetricList
              rows={[
                { label: "Games graded", value: String(accuracy.season_to_date.games_graded) },
                { label: "Winner accuracy", value: pctOrDash(accuracy.season_to_date.binary_accuracy) },
                { label: "Brier", value: fixedOrDash(accuracy.season_to_date.brier, 3) },
                { label: "Spread MAE", value: fixedOrDash(accuracy.season_to_date.spread_mae, 1) },
                { label: "Spread RMSE", value: fixedOrDash(accuracy.season_to_date.spread_rmse, 1) },
                {
                  label: "Spread sign accuracy",
                  value: pctOrDash(accuracy.season_to_date.spread_sign_accuracy),
                },
                {
                  label: "ATS vs Vegas",
                  value: pctOrDash(accuracy.season_to_date.ats_accuracy ?? null),
                },
                {
                  label: "ATS games",
                  value: String(accuracy.season_to_date.ats_games_graded ?? 0),
                },
              ]}
            />
            {accuracy.weeks.length > 1 ? (
              <>
                <h3 className="mt-10 font-mono text-xs uppercase tracking-[0.15em] text-ink-muted">
                  By week
                </h3>
                <dl className="mt-4 divide-y divide-dotted divide-hairline border-y border-dotted border-hairline">
                  {accuracy.weeks.map((w) => (
                    <div
                      key={w.week}
                      className="flex items-baseline justify-between gap-4 py-2.5"
                    >
                      <dt className="font-mono text-[10px] uppercase tracking-[0.12em] text-ink-muted">
                        Week {w.week} ({w.games_graded} games)
                      </dt>
                      <dd className="font-mono text-sm tabular-nums leading-none text-ink">
                        {pctOrDash(w.binary_accuracy)} win · {fixedOrDash(w.spread_mae, 1)} spread MAE
                        {w.ats_accuracy != null ? ` · ${pctOrDash(w.ats_accuracy)} ATS` : ""}
                      </dd>
                    </div>
                  ))}
                </dl>
              </>
            ) : null}
          </>
        ) : (
          <p className="mt-3 text-ink">
            No games have finished yet this season. Live accuracy, Brier, and
            spread error will appear here once results roll in and the
            pipeline grades them.
          </p>
        )}
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
