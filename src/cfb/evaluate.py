"""Grade past predictions against actual game results.

Reads every ``predictions/{season}/week*.json`` file, matches each game's
CFBD ``id`` against that season's games, and annotates completed games with
the actual score/winner/margin and whether the model called it. Also
rebuilds ``predictions/accuracy.json``, a season-to-date and per-week
accuracy rollup consumed by the frontend performance page.

Grading is per-game and idempotent: a week file can hold a mix of graded and
ungraded games (some may not have kicked off yet), and re-running against
already-graded games is a no-op unless the underlying CFBD result changed.
Safe to call on every pipeline run.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pandas as pd

from .config import ACCURACY_PATH, PREDICTIONS_DIR
from .data import CFBDataLoader

_WEEK_FILE_RE = re.compile(r"^week(\d+)\.json$")

_GRADED_FIELDS = (
    "actual_home_score",
    "actual_away_score",
    "actual_winner",
    "actual_margin",
    "correct",
    "spread_error",
)

_UNGRADED = {field: None for field in _GRADED_FIELDS}


def grade_all(loader: CFBDataLoader | None = None) -> dict:
    """Grade every ``predictions/{season}/week*.json`` against CFBD results.

    Returns the ``accuracy.json`` payload that was written.
    """
    loader = loader or CFBDataLoader()
    accuracy: dict = {}
    if ACCURACY_PATH.exists():
        with open(ACCURACY_PATH) as fh:
            accuracy = json.load(fh)

    latest_path = PREDICTIONS_DIR / "latest.json"
    latest_key = _latest_key(latest_path)

    for season_dir in sorted(p for p in PREDICTIONS_DIR.iterdir() if p.is_dir()):
        if not season_dir.name.isdigit():
            continue
        season = int(season_dir.name)

        results = _results_by_id(loader.load_games(season, refresh=True))

        all_graded: list[dict] = []
        week_summaries: list[dict] = []
        for week_path in sorted(season_dir.glob("week*.json"), key=_week_number):
            graded_games, week_num = _grade_week_file(
                week_path, results, latest_path, latest_key
            )
            if not graded_games:
                continue
            all_graded.extend(graded_games)
            week_summaries.append({"week": week_num, **_summarize(graded_games)})

        if week_summaries:
            accuracy[str(season)] = {
                "season_to_date": _summarize(all_graded),
                "weeks": week_summaries,
            }
            print(
                f"  season {season}: graded {len(all_graded)} games "
                f"across {len(week_summaries)} week(s)"
            )

    with open(ACCURACY_PATH, "w") as fh:
        json.dump(accuracy, fh, indent=2, default=str)
    print(f"Wrote {ACCURACY_PATH}")
    return accuracy


# =============================================================================
# Per-week grading
# =============================================================================
def _grade_week_file(
    week_path: Path,
    results: dict[int, dict],
    latest_path: Path,
    latest_key: tuple[int, int] | None,
) -> tuple[list[dict], int]:
    with open(week_path) as fh:
        payload = json.load(fh)

    season = payload.get("season")
    week_num = payload.get("week")
    changed = False
    graded_games: list[dict] = []

    for game in payload.get("games", []):
        before = {field: game.get(field) for field in _GRADED_FIELDS}
        result = results.get(game.get("id"))
        after = _grade_game(game, result)
        if after != before:
            changed = True
        game.update(after)
        if after["correct"] is not None or after["actual_winner"] is not None:
            graded_games.append(game)

    if changed:
        with open(week_path, "w") as fh:
            json.dump(payload, fh, indent=2, default=str)
        if latest_key == (season, week_num):
            with open(latest_path, "w") as fh:
                json.dump(payload, fh, indent=2, default=str)

    return graded_games, week_num


def _grade_game(game: dict, result: dict | None) -> dict:
    if result is None or not result["completed"]:
        return dict(_UNGRADED)
    home_pts, away_pts = result["home_points"], result["away_points"]
    if home_pts is None or away_pts is None:
        return dict(_UNGRADED)

    actual_margin = home_pts - away_pts
    if home_pts > away_pts:
        actual_winner = game["home_team"]
    elif away_pts > home_pts:
        actual_winner = game["away_team"]
    else:
        actual_winner = None  # tie; unreachable under current CFB OT rules

    correct = (game["predicted_winner"] == actual_winner) if actual_winner is not None else None

    predicted_margin = game.get("predicted_margin")
    spread_error = (
        round(float(predicted_margin - actual_margin), 1) if predicted_margin is not None else None
    )

    return {
        "actual_home_score": int(home_pts),
        "actual_away_score": int(away_pts),
        "actual_winner": actual_winner,
        "actual_margin": round(float(actual_margin), 1),
        "correct": correct,
        "spread_error": spread_error,
    }


# =============================================================================
# Aggregation
# =============================================================================
def _summarize(graded_games: list[dict]) -> dict:
    decided = [g for g in graded_games if g["correct"] is not None]
    spread_rows = [g for g in graded_games if g.get("spread_error") is not None]
    sign_rows = [g for g in spread_rows if g["actual_margin"] != 0]

    binary_accuracy = _mean(g["correct"] for g in decided) if decided else None
    brier = (
        _mean(
            (g["home_win_probability"] - (1.0 if g["actual_winner"] == g["home_team"] else 0.0)) ** 2
            for g in decided
        )
        if decided
        else None
    )
    spread_mae = _mean(abs(g["spread_error"]) for g in spread_rows) if spread_rows else None
    spread_rmse = (
        math.sqrt(_mean(g["spread_error"] ** 2 for g in spread_rows)) if spread_rows else None
    )
    spread_sign_accuracy = (
        _mean((g["predicted_margin"] > 0) == (g["actual_margin"] > 0) for g in sign_rows)
        if sign_rows
        else None
    )

    return {
        "games_graded": len(graded_games),
        "binary_accuracy": _round(binary_accuracy, 4),
        "brier": _round(brier, 4),
        "spread_games_graded": len(spread_rows),
        "spread_mae": _round(spread_mae, 2),
        "spread_rmse": _round(spread_rmse, 2),
        "spread_sign_accuracy": _round(spread_sign_accuracy, 4),
    }


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values)


def _round(value: float | None, ndigits: int) -> float | None:
    return round(value, ndigits) if value is not None else None


# =============================================================================
# CFBD results lookup
# =============================================================================
def _results_by_id(games_df: pd.DataFrame) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for _, row in games_df.iterrows():
        gid = row.get("id")
        if gid is None or (isinstance(gid, float) and math.isnan(gid)):
            continue
        home_pts = row.get("homePoints")
        away_pts = row.get("awayPoints")
        out[int(gid)] = {
            "completed": bool(row.get("completed")) if pd.notna(row.get("completed")) else False,
            "home_points": None if pd.isna(home_pts) else float(home_pts),
            "away_points": None if pd.isna(away_pts) else float(away_pts),
        }
    return out


def _latest_key(latest_path: Path) -> tuple[int, int] | None:
    if not latest_path.exists():
        return None
    with open(latest_path) as fh:
        payload = json.load(fh)
    season, week = payload.get("season"), payload.get("week")
    return (season, week) if season is not None and week is not None else None


def _week_number(path: Path) -> int:
    match = _WEEK_FILE_RE.match(path.name)
    return int(match.group(1)) if match else -1


if __name__ == "__main__":
    grade_all()
