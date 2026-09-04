"""Feature engineering used at both train and inference time.

Two public entry points:

- ``build_training_matrix()`` -- rebuilds the whole training-time parquet from
  the parquet caches under ``data/``. Called by ``scripts/train.py``.

- ``build_week_features(loader, season, week, schema)`` -- builds the feature
  matrix for a single upcoming week, aligned to the trained model's schema.
  Called by ``cfb.model.predict_week()``.

The underlying transformation (``transform_for_lightgbm``) operates on a pandas
DataFrame directly -- no temp-file dance -- which is the single most important
simplification compared to the legacy ``preprocessing/transform_data.py``.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .config import (
    ESSENTIAL_NUMERIC_FEATURES,
    HISTORICAL_END_YEAR,
    HISTORICAL_START_YEAR,
    TRAINING_MATRIX_PATH,
)
from .data import CFBDataLoader


# ---------------------------------------------------------------------------
# transform_for_lightgbm: raw ml_dataset -> LightGBM-ready feature matrix
# ---------------------------------------------------------------------------
_ZERO_FILL_COLUMNS: Sequence[str] = (
    "stats_through_week", "home_usage", "away_usage",
    "home_receivingUsage", "away_receivingUsage",
    # Passes
    "home_netPassingYards", "home_netPassingYardsOpponent", "home_passAttempts",
    "home_passAttemptsOpponent", "home_passCompletions", "away_passCompletions",
    "away_passCompletionsOpponent", "away_passesIntercepted",
    "away_passesInterceptedOpponent", "away_passingTDs", "away_passingTDsOpponent",
    "home_passCompletionsOpponent", "home_passingTDs", "home_passingTDsOpponent",
    "home_passesIntercepted", "home_passesInterceptedOpponent",
    "away_netPassingYards", "away_netPassingYardsOpponent", "away_passAttempts",
    "away_passAttemptsOpponent",
    # Rushing
    "home_rushingAttempts", "home_rushingAttemptsOpponent", "home_rushingYards",
    "home_rushingYardsOpponent", "home_rushingTDs", "home_rushingTDsOpponent",
    "away_rushingAttempts", "away_rushingAttemptsOpponent", "away_rushingYards",
    "away_rushingYardsOpponent", "away_rushingTDs", "away_rushingTDsOpponent",
    # Yards
    "away_totalYardsOpponent", "away_totalYards", "home_totalYardsOpponent",
    "home_totalYards",
    # Turnovers
    "home_turnovers", "home_turnoversOpponent", "away_turnovers", "away_turnoversOpponent",
    # Fumbles
    "home_fumblesLost", "home_fumblesLostOpponent", "home_fumblesRecovered",
    "home_fumblesRecoveredOpponent", "away_fumblesLost", "away_fumblesLostOpponent",
    "away_fumblesRecovered", "away_fumblesRecoveredOpponent",
    # Interceptions
    "home_interceptions", "home_interceptionYards", "home_interceptionTDs",
    "away_interceptions", "away_interceptionYards", "away_interceptionTDs",
    "home_interceptionsOpponent", "home_interceptionYardsOpponent",
    "home_interceptionTDsOpponent", "away_interceptionsOpponent",
    "away_interceptionYardsOpponent", "away_interceptionTDsOpponent",
    # Punt / kick returns
    "home_puntReturns", "home_puntReturnYards", "home_puntReturnTDs",
    "away_puntReturns", "away_puntReturnYards", "away_puntReturnTDs",
    "home_puntReturnsOpponent", "home_puntReturnYardsOpponent",
    "home_puntReturnTDsOpponent", "away_puntReturnsOpponent",
    "away_puntReturnYardsOpponent", "away_puntReturnTDsOpponent",
    "home_kickReturns", "home_kickReturnYards", "home_kickReturnTDs",
    "away_kickReturns", "away_kickReturnYards", "away_kickReturnTDs",
    "home_kickReturnsOpponent", "home_kickReturnYardsOpponent",
    "home_kickReturnTDsOpponent", "away_kickReturnsOpponent",
    "away_kickReturnYardsOpponent", "away_kickReturnTDsOpponent",
    # Penalties
    "home_penalties", "home_penaltyYards", "away_penalties", "away_penaltyYards",
    "home_penaltiesOpponent", "home_penaltyYardsOpponent",
    "away_penaltiesOpponent", "away_penaltyYardsOpponent",
    # Sacks
    "home_sacks", "home_sackYards", "away_sacks", "away_sackYards",
    "home_sacksOpponent", "home_sackYardsOpponent",
    "away_sacksOpponent", "away_sackYardsOpponent",
    # TFL
    "home_tacklesForLoss", "home_tacklesForLossYards", "away_tacklesForLoss",
    "away_tacklesForLossYards", "home_tacklesForLossOpponent",
    "home_tacklesForLossYardsOpponent", "away_tacklesForLossOpponent",
    "away_tacklesForLossYardsOpponent",
    # Fourth-down conversions
    "home_fourthDownConversions", "home_fourthDowns",
    "away_fourthDownConversions", "away_fourthDowns",
    "home_fourthDownConversionsOpponent", "home_fourthDownsOpponent",
    "away_fourthDownConversionsOpponent", "away_fourthDownsOpponent",
    # Two-point conversions
    "home_twoPointConversions", "home_twoPointConversionAttempts",
    "away_twoPointConversions", "away_twoPointConversionAttempts",
    "home_twoPointConversionsOpponent", "home_twoPointConversionAttemptsOpponent",
    "away_twoPointConversionsOpponent", "away_twoPointConversionAttemptsOpponent",
    # Safeties
    "home_safeties", "away_safeties", "home_safetiesOpponent", "away_safetiesOpponent",
    # Punts / punt returns
    "home_kicksPunted", "home_puntYards", "away_kicksPunted", "away_puntYards",
    "home_kicksPuntedOpponent", "home_puntYardsOpponent",
    "away_kicksPuntedOpponent", "away_puntYardsOpponent",
    # Field goals
    "home_fieldGoalAttempts", "home_fieldGoalReturns", "home_fieldGoalReturnYards",
    "home_fieldGoalReturnTDs", "away_fieldGoalAttempts", "away_fieldGoalReturns",
    "away_fieldGoalReturnYards", "away_fieldGoalReturnTDs",
    "home_fieldGoalAttemptsOpponent", "home_fieldGoalReturnsOpponent",
    "home_fieldGoalReturnYardsOpponent", "home_fieldGoalReturnTDOpponent",
    "away_fieldGoalAttemptsOpponent", "away_fieldGoalReturnsOpponent",
    "away_fieldGoalReturnYardsOpponent", "away_fieldGoalReturnTDOpponent",
    # Downs
    "home_firstDowns", "home_firstDownsOpponent", "away_firstDowns", "away_firstDownsOpponent",
    "home_thirdDownConversions", "home_thirdDownConversionsOpponent",
    "away_thirdDownConversions", "away_thirdDownConversionsOpponent",
    "home_thirdDowns", "home_thirdDownsOpponent",
    "away_thirdDowns", "away_thirdDownsOpponent",
    "home_games", "home_gamesOpponent", "away_games", "away_gamesOpponent",
    "home_possessionTime", "away_possessionTime",
    "home_possessionTimeOpponent", "away_possessionTimeOpponent",
    # Advanced stats (scalar copies of dict-derived features)
    "home_explosiveness", "away_explosiveness",
    "home_explosivenessOpponent", "away_explosivenessOpponent",
    "home_efficiency", "away_efficiency",
    "home_efficiencyOpponent", "away_efficiencyOpponent",
    "home_fieldPosition", "away_fieldPosition",
    "home_fieldPositionOpponent", "away_fieldPositionOpponent",
    "home_havoc", "away_havoc", "home_havocOpponent", "away_havocOpponent",
    "home_punting", "away_punting", "home_puntingOpponent", "away_puntingOpponent",
    "home_returning", "away_returning",
    "home_returningOpponent", "away_returningOpponent",
    "home_rushing", "away_rushing", "home_rushingOpponent", "away_rushingOpponent",
    "home_passing", "away_passing", "home_passingOpponent", "away_passingOpponent",
    "home_standardDowns", "away_standardDowns",
    "home_standardDownsOpponent", "away_standardDownsOpponent",
    "home_passingDowns", "away_passingDowns",
    "home_passingDownsOpponent", "away_passingDownsOpponent",
    "home_offense", "away_offense", "home_offenseOpponent", "away_offenseOpponent",
    "home_defense", "away_defense", "home_defenseOpponent", "away_defenseOpponent",
    "home_specialTeams", "away_specialTeams",
    "home_specialTeamsOpponent", "away_specialTeamsOpponent",
    "home_overall", "away_overall", "home_overallOpponent", "away_overallOpponent",
    # Advanced stats - PPA / SuccessRate variants
    "home_offensivePPA", "away_offensivePPA",
    "home_offensivePPAOpponent", "away_offensivePPAOpponent",
    "home_defensivePPA", "away_defensivePPA",
    "home_defensivePPAOpponent", "away_defensivePPAOpponent",
    "home_offensiveSuccessRate", "away_offensiveSuccessRate",
    "home_offensiveSuccessRateOpponent", "away_offensiveSuccessRateOpponent",
    "home_defensiveSuccessRate", "away_defensiveSuccessRate",
    "home_defensiveSuccessRateOpponent", "away_defensiveSuccessRateOpponent",
    "home_offensiveExplosiveness", "away_offensiveExplosiveness",
    "home_offensiveExplosivenessOpponent", "away_offensiveExplosivenessOpponent",
    "home_defensiveExplosiveness", "away_defensiveExplosiveness",
    "home_defensiveExplosivenessOpponent", "away_defensiveExplosivenessOpponent",
    "home_offensiveFieldPosition", "away_offensiveFieldPosition",
    "home_offensiveFieldPositionOpponent", "away_offensiveFieldPositionOpponent",
    "home_defensiveFieldPosition", "away_defensiveFieldPosition",
    "home_defensiveFieldPositionOpponent", "away_defensiveFieldPositionOpponent",
    "home_offensiveHavoc", "away_offensiveHavoc",
    "home_offensiveHavocOpponent", "away_offensiveHavocOpponent",
    "home_defensiveHavoc", "away_defensiveHavoc",
    "home_defensiveHavocOpponent", "away_defensiveHavocOpponent",
    "home_offensivePunting", "away_offensivePunting",
    "home_offensivePuntingOpponent", "away_offensivePuntingOpponent",
    "home_defensivePunting", "away_defensivePunting",
    "home_defensivePuntingOpponent", "away_defensivePuntingOpponent",
    "home_offensiveReturning", "away_offensiveReturning",
    "home_offensiveReturningOpponent", "away_offensiveReturningOpponent",
    "home_defensiveReturning", "away_defensiveReturning",
    "home_defensiveReturningOpponent", "away_defensiveReturningOpponent",
    "home_offensiveRushing", "away_offensiveRushing",
    "home_offensiveRushingOpponent", "away_offensiveRushingOpponent",
    "home_defensiveRushing", "away_defensiveRushing",
    "home_defensiveRushingOpponent", "away_defensiveRushingOpponent",
    "home_offensivePassing", "away_offensivePassing",
    "home_offensivePassingOpponent", "away_offensivePassingOpponent",
    "home_defensivePassing", "away_defensivePassing",
    "home_defensivePassingOpponent", "away_defensivePassingOpponent",
    "home_offensiveStandardDowns", "away_offensiveStandardDowns",
    "home_offensiveStandardDownsOpponent", "away_offensiveStandardDownsOpponent",
    "home_defensiveStandardDowns", "away_defensiveStandardDowns",
    "home_defensiveStandardDownsOpponent", "away_defensiveStandardDownsOpponent",
    "home_offensivePassingDowns", "away_offensivePassingDowns",
    "home_offensivePassingDownsOpponent", "away_offensivePassingDownsOpponent",
    "home_defensivePassingDowns", "away_defensivePassingDowns",
    "home_defensivePassingDownsOpponent", "away_defensivePassingDownsOpponent",
)

# Pregame market / probability columns whose NaN we deliberately preserve.
# Zero-filling these tricks LightGBM into learning "value == 0 means <outcome>",
# which fires spuriously at inference for the ~90% of week-N games missing lines.
_KEEP_AS_NAN = frozenset({
    "homeWinProbability",
    "awayWinProbability",
    "win_probability_differential",
    "spread",
    "spread_line",
})

_ADVANCED_STATS_PATTERNS = (
    "explosiveness", "efficiency", "fieldposition", "havoc", "punting", "returning",
    "rushing", "passing", "standarddowns", "passingdowns", "offense", "defense",
    "specialteams", "overall", "ppa", "successrate",
)

_ADVANCED_CATEGORICAL_PATTERNS = ("team", "conference", "division", "poll", "rank")


def transform_for_lightgbm(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the training-time transform to a raw ML DataFrame.

    Returns a DataFrame where:
      - FCS / Division II / III rows are filtered out
      - target and target-adjacent columns (home_win, point_differential) are
        preserved as NaN where unavailable
      - stat columns are numerically-typed, zero-filled where appropriate, but
        market/probability columns keep their NaNs (see ``_KEEP_AS_NAN``)
      - categorical columns are pandas ``category`` dtype so LightGBM can pick
        them up via ``categorical_feature=`` natively.
    """
    df = df.copy()
    print(f"transform_for_lightgbm: input {df.shape[0]} rows x {df.shape[1]} columns")

    # Filter out FCS / Division II / III games.
    if "homeClassification" in df.columns and "awayClassification" in df.columns:
        df["homeClassification"] = df["homeClassification"].fillna("")
        df["awayClassification"] = df["awayClassification"].fillna("")
        for pattern in ("fcs", "iii", "ii"):
            df = df[~df["homeClassification"].str.contains(pattern, case=False, na=False)]
            df = df[~df["awayClassification"].str.contains(pattern, case=False, na=False)]

    # Drop columns that are never predictive (or leak the answer).
    drop_cols = [
        "attendance", "homeClassification", "awayClassification", "seasonType",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Zero-fill known statistical columns.
    zero_cols = [c for c in _ZERO_FILL_COLUMNS if c in df.columns]
    for col in zero_cols:
        df[col] = df[col].fillna(0)

    # Zero-fill any additional advanced-stats columns matched by pattern.
    for col in df.columns:
        if col in zero_cols:
            continue
        lower = col.lower()
        if any(pat in lower for pat in _ADVANCED_STATS_PATTERNS):
            if df[col].isna().any():
                df[col] = df[col].fillna(0)

    # Extract scalar features from any remaining dict-valued advanced-stats cols.
    for col in ("offense", "defense", "offense_advanced", "defense_advanced"):
        if col in df.columns:
            df = _extract_dict_metrics(df, col)

    # Booleans -> int (LightGBM can handle bool but this is more portable).
    for col in df.select_dtypes(include=["bool"]).columns:
        df[col] = df[col].astype(int)

    # Object columns -> category (with a few complex columns dropped outright).
    drop_complex = {"lines", "homeLineScores", "awayLineScores", "highlights"}
    for col in df.select_dtypes(include=["object"]).columns.tolist():
        if col in drop_complex:
            df = df.drop(columns=[col])
            continue
        df[col] = df[col].astype("category")

    # Some categorical-flavoured columns can still be object-typed post-flatten
    # (e.g. ``home_conference``). Convert those too.
    for col in df.columns:
        if df[col].dtype != "object":
            continue
        lower = col.lower()
        if any(pat in lower for pat in _ADVANCED_CATEGORICAL_PATTERNS):
            df[col] = df[col].astype("category")

    # Drop rows with nulls in truly essential columns.
    essential = [c for c in ("season", "week", "homeTeam", "awayTeam",
                             "homePregameElo", "awayPregameElo") if c in df.columns]
    if essential:
        before = len(df)
        df = df.dropna(subset=essential)
        after = len(df)
        if before != after:
            print(f"  dropped {before - after} rows with null essential values")

    # Fill remaining nulls per-column policy.
    for col in df.columns:
        if col in _KEEP_AS_NAN:
            continue
        if col in ("homePoints", "awayPoints", "home_win", "point_differential"):
            continue
        if str(df[col].dtype) == "category":
            continue
        if df[col].dtype in ("float64", "int64"):
            if df[col].isna().any():
                df[col] = df[col].fillna(0)
        else:
            if df[col].isna().any():
                df[col] = df[col].fillna("unknown")

    # Anything still non-numeric AND non-category becomes numeric via coerce.
    non_numeric = [
        col
        for col in df.select_dtypes(exclude=[np.number]).columns
        if str(df[col].dtype) != "category"
    ]
    for col in non_numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    print(f"transform_for_lightgbm: output {df.shape[0]} rows x {df.shape[1]} columns")
    return df


# ---------------------------------------------------------------------------
# Training-time matrix builder
# ---------------------------------------------------------------------------
def build_training_matrix(
    start_year: int = HISTORICAL_START_YEAR,
    end_year: int = HISTORICAL_END_YEAR,
    write: bool = True,
) -> pd.DataFrame:
    """Build the transformed training-time matrix from cached parquets.

    Concatenates ``data/games/games_{year}.parquet`` for every season in the
    range, joins on features via ``CFBDataLoader.create_ml_features_dataset``,
    runs the LightGBM transform, and writes ``data/training_matrix.parquet``
    when ``write=True``.
    """
    loader = CFBDataLoader()

    frames = []
    for year in range(start_year, end_year + 1):
        frames.append(loader.load_games(year))
    games_df = pd.concat(frames, ignore_index=True)
    print(f"Concatenated per-season games parquets: {len(games_df)} rows")

    home_pts = pd.to_numeric(games_df["homePoints"], errors="coerce")
    away_pts = pd.to_numeric(games_df["awayPoints"], errors="coerce")
    filtered = games_df[
        games_df["homeConference"].notna()
        & games_df["awayConference"].notna()
        & home_pts.notna()
        & away_pts.notna()
    ]
    print(f"After FBS filter: {len(filtered)} rows")

    ml_df = loader.create_ml_features_dataset(games_df=filtered)

    # Drop non-feature columns (leakage / identifiers).
    non_feature = [
        "homePostgameWinProbability", "awayPostgameWinProbability", "id",
        "homeId", "awayId", "homeLineScores", "awayLineScores", "venueId",
        "completed", "startTimeTBD", "venue", "notes", "homePostgameElo",
        "awayPostgameElo", "startDate", "highlights", "homeTeam", "awayTeam",
        "homePoints", "awayPoints", "excitementIndex",
    ]
    ml_df = ml_df.drop(columns=[c for c in non_feature if c in ml_df.columns], errors="ignore")

    # Filter FCS classifications early (transform will too, but this keeps the
    # intermediate parquet clean if we ever inspect it).
    if "homeClassification" in ml_df.columns:
        for pattern in ("fcs", "iii", "ii"):
            ml_df = ml_df[~ml_df["homeClassification"].str.contains(pattern, case=False, na=False)]
            ml_df = ml_df[~ml_df["awayClassification"].str.contains(pattern, case=False, na=False)]

    transformed = transform_for_lightgbm(ml_df)

    if write:
        TRAINING_MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)
        transformed.to_parquet(TRAINING_MATRIX_PATH)
        print(f"Wrote training matrix to {TRAINING_MATRIX_PATH}")

    return transformed


# ---------------------------------------------------------------------------
# Inference-time per-week feature builder
# ---------------------------------------------------------------------------
def build_week_features(
    loader: CFBDataLoader,
    season: int,
    week: int,
    schema: dict,
    refresh_data: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the model-ready feature matrix for one week of upcoming games.

    Returns ``(X, week_games)`` where:
      - ``X`` is the DataFrame with exactly ``schema['feature_columns']`` in
        the exact order the model was trained on, with categoricals aligned to
        the trained categories.
      - ``week_games`` is the filtered games DataFrame (FBS-vs-FBS,
        uncompleted, matching the input week) with metadata for JSON output.
    """
    print(f"Building features for season {season} week {week}...")

    if refresh_data:
        # Refresh current-season endpoints. Historical data stays cached.
        _refresh_current_season(loader, season)

    games = loader.load_games(season, refresh=False)
    week_games = games[
        (games["week"] == week)
        & (games["completed"] != True)  # noqa: E712
        & (games["homeClassification"] == "fbs")
        & (games["awayClassification"] == "fbs")
    ].copy()

    if week_games.empty:
        print(f"  no uncompleted FBS games found for season {season} week {week}")
        return pd.DataFrame(), week_games

    print(f"  found {len(week_games)} FBS games")
    week_games["season"] = season

    ml_df = loader.create_ml_features_dataset(games_df=week_games)
    if "spread_line" in ml_df.columns and "id" in ml_df.columns:
        spread_by_id = {
            int(gid): line
            for gid, line in zip(ml_df["id"], ml_df["spread_line"])
            if pd.notna(gid)
        }
        week_games["spread_line"] = week_games["id"].map(spread_by_id)
    transformed = transform_for_lightgbm(ml_df)

    X = _align_to_schema(transformed, schema)
    print(f"  aligned to schema: {X.shape[0]} games x {X.shape[1]} features")
    return X, week_games


def _refresh_current_season(loader: CFBDataLoader, season: int) -> None:
    """Refresh the CFBD endpoints that change during a live season."""
    print(f"Refreshing current-season data for {season}...")
    loader.load_games(season, refresh=True)
    loader.load_rankings(season, refresh=True)
    loader.load_team_talent(season, refresh=True)

    current_week = loader.get_current_week(season, refresh=False)
    loader.load_batch_weekly_stats(season, refresh=True, max_week=max(current_week, 1))
    loader.load_advanced_weekly_stats(season, refresh=True, max_week=max(current_week, 1))

    # These may or may not be populated for the current season (returning
    # production is offseason, betting lines and win probs come mid-week).
    loader.load_betting_lines(season, refresh=True)
    loader.load_win_probability(season, refresh=True)


def _align_to_schema(transformed: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """Coerce ``transformed`` into the exact column-set/order/dtype the model expects."""
    feature_columns: list[str] = schema["feature_columns"]
    categorical_features: list[str] = schema["categorical_features"]
    categorical_values: dict[str, list] = schema["categorical_values"]

    present = set(transformed.columns)
    missing = [c for c in feature_columns if c not in present]
    extras = [
        c for c in transformed.columns
        if c not in feature_columns
        and c not in ("season", "week", "home_win", "point_differential")
    ]

    if missing:
        missing_categorical = [c for c in missing if c in categorical_features]
        missing_essential = [c for c in missing if c in ESSENTIAL_NUMERIC_FEATURES]
        if missing_essential:
            raise RuntimeError(
                "Refusing to predict: essential training features are missing.\n"
                f"  Missing essential numeric: {missing_essential}\n"
                f"  Total missing: {len(missing)}"
            )
        print(f"  filling {len(missing)} missing features (0.0 numeric, NaN categorical)")
        fill_df = pd.DataFrame(
            {c: 0.0 for c in missing if c not in missing_categorical},
            index=transformed.index,
        )
        for c in missing_categorical:
            fill_df[c] = pd.Series([pd.NA] * len(transformed), index=transformed.index)
        transformed = pd.concat([transformed, fill_df], axis=1)

    if extras:
        transformed = transformed.drop(columns=extras)

    # Align categorical columns to the trained categories. Unseen values become
    # NaN, which LightGBM handles as "missing" — matching training behaviour.
    for col, cats in categorical_values.items():
        if col not in transformed.columns:
            continue
        series = transformed[col]
        if str(series.dtype) == "category":
            series = series.astype(object)
        aligned = pd.Categorical(series, categories=cats)
        unseen_mask = pd.isna(aligned) & series.notna()
        n_unseen = int(unseen_mask.sum())
        if n_unseen:
            samples = sorted({str(v) for v in series[unseen_mask].unique()})[:5]
            print(
                f"  categorical '{col}': {n_unseen} rows had unseen values "
                f"{samples}{'...' if len(samples) == 5 else ''}"
            )
        transformed[col] = aligned

    return transformed[feature_columns]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
_OFFENSE_DEFENSE_KEYS = (
    ("explosiveness", "explosiveness"),
    ("ppa", "ppa"),
    ("successRate", "successRate"),
    ("drives", "drives"),
    ("plays", "plays"),
    ("lineYards", "lineYards"),
    ("stuffRate", "stuffRate"),
    ("powerSuccess", "powerSuccess"),
    ("pointsPerOpportunity", "pointsPerOpportunity"),
)

_OFFENSE_DEFENSE_NESTED = (
    ("fieldPosition", "averagePredictedPoints", "fieldPosition_avgPredictedPoints"),
    ("fieldPosition", "averageStart", "fieldPosition_avgStart"),
    ("havoc", "total", "havoc_total"),
    ("havoc", "db", "havoc_db"),
    ("havoc", "frontSeven", "havoc_frontSeven"),
    ("rushingPlays", "ppa", "rushingPlays_ppa"),
    ("rushingPlays", "successRate", "rushingPlays_successRate"),
    ("passingPlays", "ppa", "passingPlays_ppa"),
    ("passingPlays", "successRate", "passingPlays_successRate"),
    ("standardDowns", "ppa", "standardDowns_ppa"),
    ("standardDowns", "successRate", "standardDowns_successRate"),
    ("passingDowns", "ppa", "passingDowns_ppa"),
    ("passingDowns", "successRate", "passingDowns_successRate"),
)


def _extract_dict_metrics(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Explode dict-valued ``col`` into scalar {col}_metric columns, then drop it."""
    for source_key, dest_suffix in _OFFENSE_DEFENSE_KEYS:
        df[f"{col}_{dest_suffix}"] = df[col].apply(
            lambda x, k=source_key: x.get(k, 0) if isinstance(x, dict) else 0
        )
    for parent, child, dest in _OFFENSE_DEFENSE_NESTED:
        df[f"{col}_{dest}"] = df[col].apply(
            lambda x, p=parent, c=child: (
                x.get(p, {}).get(c, 0) if isinstance(x, dict) else 0
            )
        )
    return df.drop(columns=[col])
