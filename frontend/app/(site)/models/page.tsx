export default function ModelsPage() {
  return (
    <article className="max-w-2xl">
      <h1 className="text-3xl font-extrabold tracking-tight">
        How it works
      </h1>
      <p className="mt-2 text-lg text-ink">
        Two models make each week&apos;s picks. One predicts who wins, and the
        other predicts by how many points.
      </p>

      <h2 className="mt-10 text-xl font-bold">Who wins</h2>
      <p className="mt-2 leading-relaxed text-ink">
        A LightGBM classifier gives each team&apos;s chance of winning. That&apos;s
        the win chance on the picks page.
      </p>

      <h2 className="mt-8 text-xl font-bold">By how much</h2>
      <p className="mt-2 leading-relaxed text-ink">
        A LightGBM regressor predicts the final margin. That&apos;s the model
        line, and comparing it with the Vegas line decides which side the
        model takes against the spread.
      </p>

      <h2 className="mt-8 text-xl font-bold">What goes in</h2>
      <p className="mt-2 leading-relaxed text-ink">
        Rankings, roster talent, opponent-adjusted efficiency, and each
        team&apos;s stats so far this season. Both models are trained on the
        2015 to 2025 seasons and get fresh data every week. Early in the
        season there isn&apos;t much to go on, so the first few weeks are the
        least accurate.
      </p>

      <h2 className="mt-8 text-xl font-bold">When they disagree</h2>
      <p className="mt-2 leading-relaxed text-ink">
        The two models are trained separately, so once in a while a team gets
        the better win chance while the other team is favored on the model
        line. It happens in close games, where both numbers are near even.
      </p>

      <p className="mt-10 text-muted">
        The code is on{" "}
        <a
          href="https://github.com/mattssawyer/cfb_prediction_model"
          className="font-medium text-ink hover:underline"
        >
          GitHub
        </a>
        .
      </p>
    </article>
  );
}
