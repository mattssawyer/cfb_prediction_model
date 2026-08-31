"""Weekly prediction pipeline.

Refreshes current-season CFBD data, builds features, runs the model, and
writes:
  predictions/{season}/week{N}.json
  predictions/latest.json  (copy of the newest week just produced)

Usage:
    python scripts/pipeline.py                 # auto-detect season and next week
    python scripts/pipeline.py --season 2025 --week 4
    python scripts/pipeline.py --season 2025 --week 4 --force  # overwrite
    python scripts/pipeline.py --no-refresh    # skip CFBD refresh (use cache)

Exit codes:
    0  success
    1  no games for the requested week / unrecoverable failure
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

# Allow running as a script without installing the package.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from cfb.config import PREDICTIONS_DIR, detect_current_season, ensure_dirs
from cfb.data import CFBDataLoader
from cfb.evaluate import grade_all
from cfb.model import predict_week


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=None, help="Season year (default: auto)")
    parser.add_argument("--week", type=int, default=None, help="Week number (default: next unplayed)")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing predictions JSON for this week.",
    )
    parser.add_argument(
        "--no-refresh",
        action="store_true",
        help="Skip re-fetching current-season CFBD data; use cache only.",
    )
    return parser.parse_args()


def resolve_target(season: int | None, week: int | None) -> tuple[int, int]:
    """Fill in defaults for --season / --week using the current calendar + CFBD."""
    season = season or detect_current_season()
    if week is None:
        loader = CFBDataLoader()
        week = loader.get_current_week(season, refresh=True)
    return season, week


def output_path(season: int, week: int) -> Path:
    return PREDICTIONS_DIR / str(season) / f"week{week}.json"


def write_prediction(payload: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    latest = PREDICTIONS_DIR / "latest.json"
    shutil.copyfile(out_path, latest)


def main() -> int:
    args = parse_args()
    ensure_dirs()

    print("\n=== Grading past predictions ===")
    grade_all()

    season, week = resolve_target(args.season, args.week)
    out = output_path(season, week)

    if out.exists() and not args.force:
        print(f"Predictions already exist: {out}. Pass --force to overwrite.")
        return 0

    payload = predict_week(season, week, refresh_data=not args.no_refresh)

    if not payload["games"]:
        print(f"No games to predict for season {season} week {week}. Nothing written.")
        return 1

    write_prediction(payload, out)
    print(f"\nWrote {out}")
    print(f"Wrote {PREDICTIONS_DIR / 'latest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
