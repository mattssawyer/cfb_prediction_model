import Link from "next/link";
import { loadSeasonAccuracy } from "@/lib/accuracy";
import { loadLatest } from "@/lib/predictions";
import { fixed, loadTraining, pct } from "@/lib/training";

function pctOrDash(n: number | null | undefined): string {
  return n == null ? "—" : pct(n);
}

function fixedOrDash(n: number | null, digits: number): string {
  return n == null ? "—" : fixed(n, digits);
}

function MetricTable({ rows }: { rows: { label: string; value: string; note?: string }[] }) {
  return (
    <div className="mt-4 max-w-md rounded-xl border border-line bg-panel px-5 py-1">
      <table className="w-full text-sm">
        <tbody>
          {rows.map((row) => (
            <tr key={row.label} className="border-b border-line last:border-b-0">
              <th scope="row" className="py-2 pr-4 text-left font-normal text-muted">
                {row.label}
                {row.note ? <span className="block text-xs">{row.note}</span> : null}
              </th>
              <td className="py-2 text-right font-semibold text-ink">{row.value}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function SectionHeading({ children }: { children: React.ReactNode }) {
  return <h2 className="mt-12 text-xl font-bold">{children}</h2>;
}

export default function PerformancePage() {
  const { winner, spread } = loadTraining();
  const latest = loadLatest();
  const accuracy = latest ? loadSeasonAccuracy(latest.season) : null;
  const season = latest?.season ?? new Date().getFullYear();
  const total = accuracy?.season_to_date;

  return (
    <div className="max-w-3xl">
      <h1 className="text-3xl font-extrabold tracking-tight">
        Track record
      </h1>

      {accuracy && total && total.games_graded > 0 ? (
        <>
          <p className="mt-2 text-lg text-ink">
            In {season}, the model has picked the winner in{" "}
            <strong className="font-semibold">{pctOrDash(total.binary_accuracy)}</strong> of{" "}
            {total.games_graded} finished games
            {total.ats_accuracy != null ? (
              <>
                {" "}and is{" "}
                <strong className="font-semibold">{pctOrDash(total.ats_accuracy)}</strong>{" "}
                against the Vegas line
              </>
            ) : null}
            . Its spread is off by {fixedOrDash(total.spread_mae, 1)} points on
            an average game.
          </p>
          <p className="mt-2 text-sm text-muted">
            Games are graded against final scores every Sunday, when the next
            week&apos;s picks go up.
          </p>

          <SectionHeading>Week by week</SectionHeading>
          <div className="mt-4 overflow-x-auto rounded-xl border border-line bg-panel px-5 pt-3">
            <table className="w-full min-w-[32rem] text-sm">
              <thead>
                <tr className="border-b border-line text-left text-xs text-muted">
                  <th scope="col" className="pb-2 font-normal">Week</th>
                  <th scope="col" className="pb-2 text-right font-normal">Games</th>
                  <th scope="col" className="pb-2 text-right font-normal">Winners picked</th>
                  <th scope="col" className="pb-2 text-right font-normal">Against the line</th>
                  <th scope="col" className="pb-2 text-right font-normal">Avg. spread miss</th>
                </tr>
              </thead>
              <tbody>
                {accuracy.weeks.map((w) => (
                  <tr key={w.week} className="border-b border-line">
                    <td className="py-2">
                      <Link
                        href={`/${season}/week/${w.week}`}
                        className="font-semibold text-ink hover:underline"
                      >
                        Week {w.week}
                      </Link>
                    </td>
                    <td className="py-2 text-right text-muted">{w.games_graded}</td>
                    <td className="py-2 text-right font-semibold">{pctOrDash(w.binary_accuracy)}</td>
                    <td className="py-2 text-right font-semibold">{pctOrDash(w.ats_accuracy)}</td>
                    <td className="py-2 text-right">{fixedOrDash(w.spread_mae, 1)} pts</td>
                  </tr>
                ))}
              </tbody>
              <tfoot>
                <tr className="border-t-2 border-line">
                  <th scope="row" className="py-2 text-left font-semibold">Season</th>
                  <td className="py-2 text-right text-muted">{total.games_graded}</td>
                  <td className="py-2 text-right font-semibold">{pctOrDash(total.binary_accuracy)}</td>
                  <td className="py-2 text-right font-semibold">{pctOrDash(total.ats_accuracy)}</td>
                  <td className="py-2 text-right">{fixedOrDash(total.spread_mae, 1)} pts</td>
                </tr>
              </tfoot>
            </table>
          </div>
          <MetricTable
            rows={[
              { label: "Brier score", note: "Lower is better; 0.25 is a coin flip", value: fixedOrDash(total.brier, 3) },
              { label: "Spread RMSE", value: `${fixedOrDash(total.spread_rmse, 1)} pts` },
              { label: "Spread picked the right winner", value: pctOrDash(total.spread_sign_accuracy) },
            ]}
          />
        </>
      ) : (
        <p className="mt-2 text-lg text-ink">
          No {season} games have finished yet. Results show up here after the
          first Saturday of games is graded.
        </p>
      )}

      <SectionHeading>Before the season</SectionHeading>
      <p className="mt-3 text-ink">
        Both models were tested on a season they never saw. They trained on
        seasons through {winner?.eval_train_max_season ?? 2023}, tuned on the
        next one, and were scored on {winner?.holdout_season ?? 2025}. The
        versions making picks now were then retrained on every season through{" "}
        {winner?.holdout_season ?? 2025}.
      </p>

      {winner ? (
        <>
          <h3 className="mt-8 text-base font-bold">Who wins</h3>
          <p className="mt-1 text-sm text-muted">
            LightGBM classifier, scored on {winner.uncalibrated.n_games.toLocaleString()}{" "}
            {winner.holdout_season} games. It uses {winner.n_features} features
            and {winner.best_iteration} trees, trained on{" "}
            {winner.n_training_games.toLocaleString()} games.
          </p>
          <MetricTable
            rows={[
              { label: "Accuracy", value: pct(winner.uncalibrated.accuracy) },
              { label: "AUC", value: fixed(winner.uncalibrated.auc, 3) },
              { label: "Brier score", value: fixed(winner.uncalibrated.brier, 3) },
              { label: "Log loss", value: fixed(winner.uncalibrated.logloss, 3) },
            ]}
          />
          {winner.calibration_method !== "isotonic" && winner.calibrated ? (
            <p className="mt-3 text-sm text-muted">
              Isotonic calibration made the holdout Brier score worse (
              {fixed(winner.uncalibrated.brier, 3)} to{" "}
              {fixed(winner.calibrated.brier, 3)}), so the model uses its raw
              probabilities.
            </p>
          ) : null}
        </>
      ) : (
        <p className="mt-6 text-sm text-muted">
          Winner model metrics are missing. Retrain to generate{" "}
          <code className="bg-page px-1 text-ink">models/winner_model_schema.json</code>.
        </p>
      )}

      {spread ? (
        <>
          <h3 className="mt-8 text-base font-bold">By how much</h3>
          <p className="mt-1 text-sm text-muted">
            LightGBM regressor predicting the home team&apos;s margin, scored on
            the same {spread.n_games.toLocaleString()} {spread.holdout_season} games.
          </p>
          <MetricTable
            rows={[
              { label: "Average miss", value: `${fixed(spread.mae, 1)} pts` },
              { label: "RMSE", value: `${fixed(spread.rmse, 1)} pts` },
              { label: "Picked the right winner", value: pct(spread.sign_accuracy) },
            ]}
          />
        </>
      ) : (
        <p className="mt-6 text-sm text-muted">
          Spread model metrics are missing. Retrain to generate{" "}
          <code className="bg-page px-1 text-ink">models/spread_model_schema.json</code>.
        </p>
      )}
    </div>
  );
}
