"""Unified CFBD data loader with on-disk parquet caching.

This module replaces the split ``preprocessing/data_loader.py`` (historical) +
``update_2025_weekly.py`` (current-season) pair. Everything is year-agnostic:
seasons are always passed as parameters, and the ``refresh=`` flag controls
whether the current-season parquet is re-fetched from CFBD.

Cache layout under ``config.DATA_DIR`` (default ``data/``):

    games/games_{year}.parquet
    rankings/rankings_{year}.parquet
    team_talent/team_talent_{year}.parquet
    betting_lines/betting_lines_{year}.parquet
    win_probability/win_probability_{year}.parquet
    batch_weekly_stats/batch_weekly_stats_{year}.parquet
    advanced_weekly_stats/advanced_weekly_stats_{year}.parquet
    returning_production/returning_production_{year}.parquet
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

import cfbd
import numpy as np
import pandas as pd

from .config import DATA_DIR, HISTORICAL_END_YEAR, HISTORICAL_START_YEAR, load_api_key


DATA_TYPES = (
    "games",
    "rankings",
    "team_talent",
    "betting_lines",
    "win_probability",
    "batch_weekly_stats",
    "advanced_weekly_stats",
    "returning_production",
)


class CFBDataLoader:
    """CFBD API client with a parquet cache.

    Instantiate once per process. All ``load_*`` methods take a ``year`` and a
    ``refresh`` flag; when ``refresh=False`` and the parquet exists it is read
    from disk, otherwise a fresh copy is fetched and written.
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        self.api_key = load_api_key()
        self.configuration = cfbd.Configuration(access_token=self.api_key)
        self.data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.call_count = 0

        self.subfolders = {name: self.data_dir / name for name in DATA_TYPES}
        for path in self.subfolders.values():
            path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Small utilities
    # ------------------------------------------------------------------
    def _rate_limit(self) -> None:
        time.sleep(0.1)
        self.call_count += 1

    def _cache_path(self, kind: str, year: int) -> Path:
        return self.subfolders[kind] / f"{kind}_{year}.parquet"

    def _read_cache(self, kind: str, year: int) -> pd.DataFrame | None:
        path = self._cache_path(kind, year)
        if not path.exists():
            return None
        return pd.read_parquet(path)

    def _write_cache(self, kind: str, year: int, df: pd.DataFrame) -> None:
        df.to_parquet(self._cache_path(kind, year))

    # ------------------------------------------------------------------
    # Per-endpoint loaders
    # ------------------------------------------------------------------
    def load_games(self, year: int, refresh: bool = False) -> pd.DataFrame:
        if not refresh:
            cached = self._read_cache("games", year)
            if cached is not None:
                print(f"  games {year}: cached ({len(cached)} rows)")
                return cached

        print(f"  games {year}: fetching from CFBD (call #{self.call_count + 1})")
        with cfbd.ApiClient(self.configuration) as api_client:
            games_api = cfbd.GamesApi(api_client)
            games = games_api.get_games(year=year)
            self._rate_limit()

        df = pd.DataFrame.from_records([g.to_dict() for g in games])
        self._write_cache("games", year, df)
        return df

    def load_rankings(self, year: int, refresh: bool = False) -> pd.DataFrame:
        if not refresh:
            cached = self._read_cache("rankings", year)
            if cached is not None:
                print(f"  rankings {year}: cached ({len(cached)} rows)")
                return cached

        print(f"  rankings {year}: fetching from CFBD (call #{self.call_count + 1})")
        with cfbd.ApiClient(self.configuration) as api_client:
            rankings_api = cfbd.RankingsApi(api_client)
            try:
                rankings = rankings_api.get_rankings(year=year)
            except Exception as exc:
                print(f"    rankings {year}: fetch failed ({exc})")
                return pd.DataFrame()
            self._rate_limit()

        if not rankings:
            return pd.DataFrame()
        df = pd.DataFrame.from_records([r.to_dict() for r in rankings])
        self._write_cache("rankings", year, df)
        return df

    def load_team_talent(self, year: int, refresh: bool = False) -> pd.DataFrame:
        if not refresh:
            cached = self._read_cache("team_talent", year)
            if cached is not None:
                print(f"  team_talent {year}: cached ({len(cached)} rows)")
                return cached

        print(f"  team_talent {year}: fetching from CFBD (call #{self.call_count + 1})")
        with cfbd.ApiClient(self.configuration) as api_client:
            teams_api = cfbd.TeamsApi(api_client)
            try:
                talent = teams_api.get_talent(year=year)
            except Exception as exc:
                print(f"    team_talent {year}: fetch failed ({exc})")
                return pd.DataFrame()
            self._rate_limit()

        if not talent:
            return pd.DataFrame()
        df = pd.DataFrame.from_records([t.to_dict() for t in talent])
        self._write_cache("team_talent", year, df)
        return df

    def load_betting_lines(self, year: int, refresh: bool = False) -> pd.DataFrame:
        if not refresh:
            cached = self._read_cache("betting_lines", year)
            if cached is not None:
                print(f"  betting_lines {year}: cached ({len(cached)} rows)")
                return cached

        print(f"  betting_lines {year}: fetching from CFBD (call #{self.call_count + 1})")
        with cfbd.ApiClient(self.configuration) as api_client:
            betting_api = cfbd.BettingApi(api_client)
            try:
                lines = betting_api.get_lines(year=year)
            except Exception as exc:
                print(f"    betting_lines {year}: fetch failed ({exc})")
                return pd.DataFrame()
            self._rate_limit()

        if not lines:
            return pd.DataFrame()
        df = pd.DataFrame.from_records([l.to_dict() for l in lines])
        self._write_cache("betting_lines", year, df)
        return df

    def load_win_probability(self, year: int, refresh: bool = False) -> pd.DataFrame:
        if not refresh:
            cached = self._read_cache("win_probability", year)
            if cached is not None:
                print(f"  win_probability {year}: cached ({len(cached)} rows)")
                return cached

        print(f"  win_probability {year}: fetching from CFBD (call #{self.call_count + 1})")
        with cfbd.ApiClient(self.configuration) as api_client:
            metrics_api = cfbd.MetricsApi(api_client)
            try:
                probs = metrics_api.get_pregame_win_probabilities(year=year)
            except Exception as exc:
                print(f"    win_probability {year}: fetch failed ({exc})")
                return pd.DataFrame()
            self._rate_limit()

        if not probs:
            return pd.DataFrame()
        df = pd.DataFrame.from_records([p.to_dict() for p in probs])
        self._write_cache("win_probability", year, df)
        return df

    def load_batch_weekly_stats(
        self,
        year: int,
        refresh: bool = False,
        max_week: int = 15,
    ) -> pd.DataFrame:
        """Cumulative team stats (long -> wide pivot) for each week 1..max_week."""
        if not refresh:
            cached = self._read_cache("batch_weekly_stats", year)
            if cached is not None:
                print(f"  batch_weekly_stats {year}: cached ({len(cached)} rows)")
                return cached

        print(
            f"  batch_weekly_stats {year}: fetching {max_week} weeks from CFBD "
            f"(call #{self.call_count + 1}...)"
        )
        weekly = []
        with cfbd.ApiClient(self.configuration) as api_client:
            stats_api = cfbd.StatsApi(api_client)
            for week in range(1, max_week + 1):
                try:
                    stats = stats_api.get_team_stats(
                        year=year, start_week=1, end_week=week
                    )
                    self._rate_limit()
                except Exception as exc:
                    print(f"    week {week}: fetch failed ({exc})")
                    continue
                if not stats:
                    continue
                long_df = pd.DataFrame.from_records([s.to_dict() for s in stats])
                long_df["stats_through_week"] = week
                long_df["year"] = year
                wide_df = long_df.pivot_table(
                    index=["team", "conference", "stats_through_week", "year"],
                    columns="statName",
                    values="statValue",
                    aggfunc="first",
                ).reset_index()
                wide_df.columns.name = None
                weekly.append(wide_df)

        if not weekly:
            return pd.DataFrame()
        df = pd.concat(weekly, ignore_index=True)
        self._write_cache("batch_weekly_stats", year, df)
        return df

    def load_advanced_weekly_stats(
        self,
        year: int,
        refresh: bool = False,
        max_week: int = 15,
    ) -> pd.DataFrame:
        """Advanced team stats per single week (weeks 1..max_week)."""
        if not refresh:
            cached = self._read_cache("advanced_weekly_stats", year)
            if cached is not None:
                print(f"  advanced_weekly_stats {year}: cached ({len(cached)} rows)")
                return cached

        print(
            f"  advanced_weekly_stats {year}: fetching {max_week} weeks from CFBD "
            f"(call #{self.call_count + 1}...)"
        )
        weekly = []
        with cfbd.ApiClient(self.configuration) as api_client:
            stats_api = cfbd.StatsApi(api_client)
            for week in range(1, max_week + 1):
                try:
                    stats = stats_api.get_advanced_season_stats(
                        year=year, start_week=week, end_week=week
                    )
                    self._rate_limit()
                except Exception as exc:
                    print(f"    week {week}: fetch failed ({exc})")
                    continue
                if not stats:
                    continue
                week_df = pd.DataFrame.from_records([s.to_dict() for s in stats])
                week_df["week"] = week
                week_df["year"] = year
                weekly.append(week_df)

        if not weekly:
            return pd.DataFrame()
        df = pd.concat(weekly, ignore_index=True)
        self._write_cache("advanced_weekly_stats", year, df)
        return df

    def load_returning_production(self, year: int, refresh: bool = False) -> pd.DataFrame:
        if not refresh:
            cached = self._read_cache("returning_production", year)
            if cached is not None:
                print(f"  returning_production {year}: cached ({len(cached)} rows)")
                return cached

        print(
            f"  returning_production {year}: fetching from CFBD "
            f"(call #{self.call_count + 1})"
        )
        with cfbd.ApiClient(self.configuration) as api_client:
            players_api = cfbd.PlayersApi(api_client)
            try:
                stats = players_api.get_returning_production(year=year)
            except Exception as exc:
                print(f"    returning_production {year}: fetch failed ({exc})")
                return pd.DataFrame()
            self._rate_limit()

        if not stats:
            return pd.DataFrame()

        records = []
        for s in stats:
            record = s.to_dict()
            for key, value in list(record.items()):
                if value is None:
                    if "percent" in key.lower() or "rate" in key.lower():
                        record[key] = 0.0
                    elif isinstance(value, (int, float)):
                        record[key] = 0
                    else:
                        record[key] = ""
            records.append(record)

        df = pd.DataFrame.from_records(records)
        self._write_cache("returning_production", year, df)
        return df

    # ------------------------------------------------------------------
    # Bulk operations
    # ------------------------------------------------------------------
    def load_season(
        self,
        year: int,
        refresh: bool = False,
        max_week: int = 15,
        data_types: Iterable[str] | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Load every relevant endpoint for a single season, with caching."""
        if data_types is None:
            data_types = DATA_TYPES

        results: dict[str, pd.DataFrame] = {}
        for kind in data_types:
            try:
                if kind == "games":
                    results[kind] = self.load_games(year, refresh)
                elif kind == "rankings":
                    results[kind] = self.load_rankings(year, refresh)
                elif kind == "team_talent":
                    results[kind] = self.load_team_talent(year, refresh)
                elif kind == "betting_lines":
                    results[kind] = self.load_betting_lines(year, refresh)
                elif kind == "win_probability":
                    results[kind] = self.load_win_probability(year, refresh)
                elif kind == "batch_weekly_stats":
                    results[kind] = self.load_batch_weekly_stats(year, refresh, max_week)
                elif kind == "advanced_weekly_stats":
                    results[kind] = self.load_advanced_weekly_stats(year, refresh, max_week)
                elif kind == "returning_production":
                    results[kind] = self.load_returning_production(year, refresh)
                else:
                    print(f"  unknown data type: {kind}")
            except Exception as exc:
                print(f"  {kind} {year}: unexpected error ({exc})")
                results[kind] = pd.DataFrame()
        return results

    def load_seasons(
        self,
        start_year: int = HISTORICAL_START_YEAR,
        end_year: int = HISTORICAL_END_YEAR,
        refresh: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """Load and concatenate every endpoint over an inclusive year range."""
        all_data: dict[str, list[pd.DataFrame]] = {kind: [] for kind in DATA_TYPES}
        for year in range(start_year, end_year + 1):
            print(f"\n-- season {year} --")
            year_data = self.load_season(year, refresh=refresh)
            for kind, df in year_data.items():
                if not df.empty:
                    all_data[kind].append(df)

        combined: dict[str, pd.DataFrame] = {}
        for kind, frames in all_data.items():
            if frames:
                combined[kind] = pd.concat(frames, ignore_index=True)
        return combined

    def get_current_week(self, year: int, refresh: bool = True) -> int:
        """Return the next week to predict.

        By default we refresh the games parquet so ``completed`` is up to date.
        The returned week is (max completed week) + 1, or 1 if no games have
        completed yet.
        """
        games = self.load_games(year, refresh=refresh)
        completed = games[games["completed"] == True]  # noqa: E712
        if completed.empty:
            return 1
        return int(completed["week"].max()) + 1

    # ------------------------------------------------------------------
    # ML feature-matrix construction
    # ------------------------------------------------------------------
    def create_ml_features_dataset(
        self,
        games_df: pd.DataFrame,
        force_refresh_stats: bool = False,
    ) -> pd.DataFrame:
        """Join games with pre-game team statistics and other features.

        For each game, this adds:
        - Home / away team stats through the week BEFORE the game (avoiding leakage)
        - Team talent composite
        - Betting lines spread
        - Advanced weekly stats
        - Returning production
        - Pregame win probability + spread from CFBD
        """
        start_year = int(games_df["season"].min())
        end_year = int(games_df["season"].max())

        print(
            f"Building weekly stats for seasons {start_year}-{end_year} "
            f"(~{(end_year - start_year + 1) * 15} calls if cold cache)"
        )

        weekly_frames = []
        for year in range(start_year, end_year + 1):
            df = self.load_batch_weekly_stats(year, refresh=force_refresh_stats)
            if not df.empty:
                weekly_frames.append(df)

        ml_df = games_df.copy()
        if weekly_frames:
            weekly_stats = pd.concat(weekly_frames, ignore_index=True)
            print(f"Joining {len(games_df)} games with weekly team statistics...")

            # Stats "through week N-1" avoids leakage on games happening in week N.
            ml_df["home_stats_week"] = ml_df["week"] - 1
            ml_df["away_stats_week"] = ml_df["week"] - 1

            home_stats = _prefix_team_stats(weekly_stats, "home")
            away_stats = _prefix_team_stats(weekly_stats, "away")

            ml_df = ml_df.merge(
                home_stats,
                left_on=["season", "homeTeam", "home_stats_week"],
                right_on=["year", "homeTeam", "stats_through_week"],
                how="left",
                suffixes=("", "_home_dup"),
            )
            ml_df = ml_df.merge(
                away_stats,
                left_on=["season", "awayTeam", "away_stats_week"],
                right_on=["year", "awayTeam", "stats_through_week"],
                how="left",
                suffixes=("", "_away_dup"),
            )
            dup_cols = [c for c in ml_df.columns if c.endswith("_dup")]
            ml_df = ml_df.drop(columns=dup_cols + ["home_stats_week", "away_stats_week"], errors="ignore")
        else:
            # No in-season stats yet (early-season prediction). The transform
            # zero-fills the missing per-team stat columns downstream, matching
            # how week-1 rows are handled during training.
            print(f"  no weekly stats available for {start_year}-{end_year}; skipping join")

        ml_df = self._add_team_talent_features(ml_df, start_year, end_year)
        ml_df = self._add_betting_lines_features(ml_df)
        ml_df = self._add_advanced_weekly_stats(ml_df, start_year, end_year)
        ml_df = self._add_returning_production_features(ml_df, start_year, end_year)
        ml_df = self._add_win_probability_features(ml_df)

        # Target variables: home_win and point_differential (only for completed games)
        if "homePoints" in ml_df.columns and "awayPoints" in ml_df.columns:
            mask = ml_df["homePoints"].notna() & ml_df["awayPoints"].notna()
            ml_df["home_win"] = np.nan
            ml_df["point_differential"] = np.nan
            ml_df.loc[mask, "home_win"] = (
                ml_df.loc[mask, "homePoints"] > ml_df.loc[mask, "awayPoints"]
            ).astype(int)
            ml_df.loc[mask, "point_differential"] = (
                ml_df.loc[mask, "homePoints"] - ml_df.loc[mask, "awayPoints"]
            )
        else:
            ml_df["home_win"] = np.nan
            ml_df["point_differential"] = np.nan

        # De-duplicate columns (some merges create name collisions).
        if ml_df.columns.duplicated().any():
            dup_names = ml_df.columns[ml_df.columns.duplicated()].tolist()
            print(f"  removing duplicate columns: {dup_names}")
            ml_df = ml_df.loc[:, ~ml_df.columns.duplicated()]

        print(f"Built ML dataset: {len(ml_df)} games x {len(ml_df.columns)} features")
        return ml_df

    # ------------------------------------------------------------------
    # Feature-augmentation helpers (moved from data_loader.py)
    # ------------------------------------------------------------------
    def _add_team_talent_features(
        self, ml_df: pd.DataFrame, start_year: int, end_year: int
    ) -> pd.DataFrame:
        talent_frames = []
        for year in range(start_year, end_year + 1):
            df = self.load_team_talent(year)
            if not df.empty:
                talent_frames.append(df)

        if not talent_frames:
            print("  team_talent: no data available, skipping")
            return ml_df

        talent_df = pd.concat(talent_frames, ignore_index=True)

        ml_df = ml_df.merge(
            talent_df[["year", "team", "talent"]],
            left_on=["season", "homeTeam"],
            right_on=["year", "team"],
            how="left",
        )
        ml_df = ml_df.rename(columns={"talent": "home_talent"}).drop(
            ["year", "team"], axis=1, errors="ignore"
        )

        ml_df = ml_df.merge(
            talent_df[["year", "team", "talent"]],
            left_on=["season", "awayTeam"],
            right_on=["year", "team"],
            how="left",
        )
        ml_df = ml_df.rename(columns={"talent": "away_talent"}).drop(
            ["year", "team"], axis=1, errors="ignore"
        )

        # Fill missing per-year with the year median so early-season upsets don't NaN-out.
        for year in ml_df["season"].unique():
            year_median = talent_df.loc[talent_df["year"] == year, "talent"].median()
            year_mask = ml_df["season"] == year
            ml_df.loc[year_mask & ml_df["home_talent"].isna(), "home_talent"] = year_median
            ml_df.loc[year_mask & ml_df["away_talent"].isna(), "away_talent"] = year_median

        ml_df["talent_differential"] = ml_df["home_talent"] - ml_df["away_talent"]
        print("  team_talent: added home_talent, away_talent, talent_differential")
        return ml_df

    def _add_betting_lines_features(self, ml_df: pd.DataFrame) -> pd.DataFrame:
        seasons = sorted(ml_df["season"].unique())
        betting_frames = []
        for year in seasons:
            df = self.load_betting_lines(int(year))
            if not df.empty:
                betting_frames.append(df)

        if not betting_frames:
            print("  betting_lines: no data available, skipping")
            return ml_df

        betting_df = pd.concat(betting_frames, ignore_index=True)
        ml_df = ml_df.merge(
            betting_df[["season", "homeTeam", "awayTeam", "lines"]],
            on=["season", "homeTeam", "awayTeam"],
            how="left",
            suffixes=("", "_betting"),
        )
        ml_df["spread_line"] = ml_df["lines"].apply(_extract_spread_from_lines)
        coverage = ml_df["spread_line"].notna().sum()
        print(
            f"  betting_lines: spread_line coverage {coverage}/{len(ml_df)} "
            f"({coverage / len(ml_df) * 100:.1f}%)"
        )
        return ml_df

    def _add_advanced_weekly_stats(
        self, ml_df: pd.DataFrame, start_year: int, end_year: int
    ) -> pd.DataFrame:
        frames = []
        for year in range(start_year, end_year + 1):
            df = self.load_advanced_weekly_stats(year)
            if not df.empty:
                frames.append(df)

        if not frames:
            print("  advanced_weekly_stats: no data available, skipping")
            return ml_df

        adv_df = pd.concat(frames, ignore_index=True)

        # Flatten offense/defense dict columns into scalar features.
        for col in ("offense", "defense"):
            if col in adv_df.columns:
                flat = _flatten_dict_column(adv_df, col, col)
                adv_df = pd.concat([adv_df, flat], axis=1).drop(col, axis=1)

        home_adv = _prefix_team_advanced_stats(adv_df, "home")
        away_adv = _prefix_team_advanced_stats(adv_df, "away")

        ml_df["home_advanced_stats_week"] = ml_df["week"] - 1
        ml_df["away_advanced_stats_week"] = ml_df["week"] - 1

        ml_df = ml_df.merge(
            home_adv,
            left_on=["season", "homeTeam", "home_advanced_stats_week"],
            right_on=["year", "homeTeam", "week"],
            how="left",
            suffixes=("", "_home_advanced_dup"),
        )
        ml_df = ml_df.merge(
            away_adv,
            left_on=["season", "awayTeam", "away_advanced_stats_week"],
            right_on=["year", "awayTeam", "week"],
            how="left",
            suffixes=("", "_away_advanced_dup"),
        )
        dup_cols = [c for c in ml_df.columns if c.endswith("_dup")]
        ml_df = ml_df.drop(
            columns=dup_cols + ["home_advanced_stats_week", "away_advanced_stats_week"],
            errors="ignore",
        )
        print("  advanced_weekly_stats: joined")
        return ml_df

    def _add_returning_production_features(
        self, ml_df: pd.DataFrame, start_year: int, end_year: int
    ) -> pd.DataFrame:
        frames = []
        for year in range(start_year, end_year + 1):
            df = self.load_returning_production(year)
            if not df.empty:
                frames.append(df)

        if not frames:
            print("  returning_production: no data available, skipping")
            return ml_df

        ret_df = pd.concat(frames, ignore_index=True)
        ret_cols = [c for c in ret_df.columns if c not in ("season", "team")]
        join_cols = ["season", "team"] + ret_cols

        ml_df = ml_df.merge(
            ret_df[join_cols],
            left_on=["season", "homeTeam"],
            right_on=["season", "team"],
            how="left",
        )
        for col in ret_cols:
            if col in ml_df.columns:
                ml_df = ml_df.rename(columns={col: f"home_{col}"})
        ml_df = ml_df.drop(["team"], axis=1, errors="ignore")

        ml_df = ml_df.merge(
            ret_df[join_cols],
            left_on=["season", "awayTeam"],
            right_on=["season", "team"],
            how="left",
        )
        for col in ret_cols:
            if col in ml_df.columns:
                ml_df = ml_df.rename(columns={col: f"away_{col}"})
        ml_df = ml_df.drop(["team"], axis=1, errors="ignore")

        print("  returning_production: joined for home and away teams")
        return ml_df

    def _add_win_probability_features(self, ml_df: pd.DataFrame) -> pd.DataFrame:
        seasons = sorted(ml_df["season"].unique())
        frames = []
        for year in seasons:
            df = self.load_win_probability(int(year))
            if not df.empty:
                frames.append(df)

        if not frames:
            print("  win_probability: no data available, skipping")
            return ml_df

        wp_df = pd.concat(frames, ignore_index=True)
        ml_df = ml_df.merge(
            wp_df[["season", "homeTeam", "awayTeam", "homeWinProbability", "spread"]],
            on=["season", "homeTeam", "awayTeam"],
            how="left",
        )
        if "homeWinProbability" in ml_df.columns:
            ml_df["awayWinProbability"] = 1 - ml_df["homeWinProbability"]
            ml_df["win_probability_differential"] = (
                ml_df["homeWinProbability"] - ml_df["awayWinProbability"]
            )
            coverage = ml_df["homeWinProbability"].notna().sum()
            print(
                f"  win_probability: coverage {coverage}/{len(ml_df)} "
                f"({coverage / len(ml_df) * 100:.1f}%)"
            )
        return ml_df


# ----------------------------------------------------------------------
# Module-level helpers (kept out of the class for testability)
# ----------------------------------------------------------------------
def _prefix_team_stats(weekly_stats: pd.DataFrame, side: str) -> pd.DataFrame:
    """Rename ``team`` -> ``{side}Team`` and prefix all stat columns with ``{side}_``."""
    df = weekly_stats.copy()
    if "team" in df.columns:
        df = df.rename(columns={"team": f"{side}Team"})
    stat_cols = [
        c for c in df.columns if c not in ("year", "stats_through_week", f"{side}Team")
    ]
    df = df.rename(columns={c: f"{side}_{c}" for c in stat_cols})
    return df


def _prefix_team_advanced_stats(adv: pd.DataFrame, side: str) -> pd.DataFrame:
    df = adv.copy()
    if "team" in df.columns:
        df = df.rename(columns={"team": f"{side}Team"})
    stat_cols = [c for c in df.columns if c not in ("year", "week", f"{side}Team")]
    df = df.rename(columns={c: f"{side}_{c}" for c in stat_cols})
    return df


def _extract_spread_from_lines(lines_data) -> float | None:
    """Pull the first provider's spread out of the raw ``lines`` column."""
    try:
        if lines_data is None:
            return None
        if hasattr(lines_data, "tolist"):
            lines_list = lines_data.tolist()
        else:
            lines_list = lines_data
        if not lines_list:
            return None
        return lines_list[0].get("spread")
    except Exception:
        return None


def _flatten_dict_column(df: pd.DataFrame, col_name: str, prefix: str) -> pd.DataFrame:
    """Flatten one dict-valued column into scalar columns keyed by nested paths."""

    def flatten_nested(d: dict, parent: str = "") -> dict:
        items: list[tuple[str, object]] = []
        for k, v in d.items():
            key = f"{parent}_{k}" if parent else k
            if isinstance(v, dict):
                items.extend(flatten_nested(v, key).items())
            elif isinstance(v, (int, float, type(None))):
                items.append((key, v))
        return dict(items)

    rows = []
    for idx, val in df[col_name].items():
        row: dict[str, object] = {"idx": idx}
        if isinstance(val, dict):
            for key, value in flatten_nested(val).items():
                row[f"{prefix}_{key}"] = value
        rows.append(row)

    out = pd.DataFrame(rows).set_index("idx")
    return out
