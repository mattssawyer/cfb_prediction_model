export default function ModelsPage() {
  return (
    <>
    <div className="border-b-2 border-ink pb-8 w-2/3">
      <h1 className="mt-3 text-5xl leading-none font-bold tracking-tight text-ink uppercase">
        Models
      </h1>
    </div>
    <div className="w-2/3">
      <p className="mt-3 text-ink">
          Two separate models are used to predict 1.) the binary outcome of the game (home vs away win) and 
          2.) the spread of the game week by week. For each prediction, the models take into account data such as 
          team rankings, talent, adjusted efficiency metrics, and raw stats to date this season. Because this data is accumulated throughout 
          the season, the model will not be very accurate in the first few weeks.
          Both models use LightGBM, the first being a classifier and the second being a regressor.
          Because two separate mdoels are used, they will not always agree. This is why some games might have one team favored 
          in points in a game they are projected to lose. Because the models are so similar, however, this will not happen very often.
          They are both trained on the same data from the past 10 seasons (2015-2025), and updated weekly with the latest data. 
      </p>
      <p className="mt-3 text-ink">
        Future work will include experimenting with more advanced models as well as tracking the 
        the models' performance over time.
      </p>
      <p className="mt-3 text-ink">
        The source code for this project is available on <a href='https://github.com/mattssawyer/cfb_prediction_model' className="text-blue-500 hover:text-blue-600">GitHub</a>.
      </p>
    </div>
    </>
  );
}
