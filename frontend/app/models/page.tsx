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
          The winner prediction model is a LightGBM classifier trained to predict the binary outcome of home team win or away team win. 
          It is trained on the 2015-2025 seasons, and updated weekly with the latest data. 
          It was trained on features such as team rankings, talent, adjusted efficiency metrics, game stats, and more. 
          Because many of the features come from games within the season, the model will not be very accurate in the first few weeks.
      </p>
      <p className="mt-3 text-ink">
        Future work will include adding more models, including a game spread regression model to predict the final point differential of each game. 
        Additionally, experimentation with more advanced models, such as neural networks, will be explored.
      </p>
      <p className="mt-3 text-ink">
        The source code for this project is available on <a href='https://github.com/mattssawyer/cfb_prediction_model' className="text-blue-500 hover:text-blue-600">GitHub</a>.
      </p>
    </div>
    </>
  );
}
