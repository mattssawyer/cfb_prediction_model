import json

from cfb.evaluate import _grade_game, _grade_week_file, _results_by_id, _summarize
import pandas as pd


def _game(**overrides):
    base = {
        "id": 1,
        "home_team": "Home U",
        "away_team": "Away U",
        "home_win_probability": 0.7,
        "predicted_winner": "Home U",
        "predicted_margin": 10.0,
    }
    base.update(overrides)
    return base


def test_grade_game_ungraded_when_not_completed():
    result = {"completed": False, "home_points": None, "away_points": None}
    assert _grade_game(_game(), result) == {
        "actual_home_score": None,
        "actual_away_score": None,
        "actual_winner": None,
        "actual_margin": None,
        "correct": None,
        "spread_error": None,
    }


def test_grade_game_ungraded_when_no_result():
    assert _grade_game(_game(), None)["correct"] is None


def test_grade_game_home_win_correct_call():
    result = {"completed": True, "home_points": 28.0, "away_points": 14.0}
    graded = _grade_game(_game(), result)
    assert graded["actual_winner"] == "Home U"
    assert graded["actual_margin"] == 14.0
    assert graded["correct"] is True
    assert graded["spread_error"] == -4.0  # predicted 10, actual 14


def test_grade_game_away_win_wrong_call():
    result = {"completed": True, "home_points": 10.0, "away_points": 24.0}
    graded = _grade_game(_game(), result)
    assert graded["actual_winner"] == "Away U"
    assert graded["actual_margin"] == -14.0
    assert graded["correct"] is False
    assert graded["spread_error"] == 24.0  # predicted 10, actual -14


def test_grade_game_no_predicted_margin_leaves_spread_error_none():
    result = {"completed": True, "home_points": 28.0, "away_points": 14.0}
    graded = _grade_game(_game(predicted_margin=None), result)
    assert graded["correct"] is True
    assert graded["spread_error"] is None


def test_results_by_id_treats_missing_scores_as_ungraded():
    df = pd.DataFrame(
        [
            {"id": 1, "completed": True, "homePoints": 21.0, "awayPoints": 14.0},
            {"id": 2, "completed": False, "homePoints": float("nan"), "awayPoints": float("nan")},
        ]
    )
    results = _results_by_id(df)
    assert results[1] == {"completed": True, "home_points": 21.0, "away_points": 14.0}
    assert results[2]["completed"] is False


def test_summarize_aggregates_binary_and_spread_metrics():
    graded_games = [
        _game(id=1, home_win_probability=0.7, predicted_margin=10.0)
        | {"actual_winner": "Home U", "actual_margin": 10.0, "correct": True, "spread_error": 0.0},
        _game(id=2, home_win_probability=0.4, predicted_margin=-3.0)
        | {"actual_winner": "Home U", "actual_margin": 6.0, "correct": False, "spread_error": -9.0},
    ]
    summary = _summarize(graded_games)
    assert summary["games_graded"] == 2
    assert summary["binary_accuracy"] == 0.5
    assert summary["spread_games_graded"] == 2
    assert summary["spread_mae"] == 4.5
    # game 2 predicted a home loss (-3) but home actually won (+6): sign miss
    assert summary["spread_sign_accuracy"] == 0.5


def test_grade_week_file_partial_week_leaves_unplayed_games_ungraded(tmp_path):
    week_path = tmp_path / "week1.json"
    payload = {
        "season": 2026,
        "week": 1,
        "games": [_game(id=1), _game(id=2, predicted_winner="Away U")],
    }
    week_path.write_text(json.dumps(payload))

    results = {1: {"completed": True, "home_points": 28.0, "away_points": 14.0}}
    graded_games, week_num = _grade_week_file(week_path, results, tmp_path / "latest.json", None)

    assert week_num == 1
    assert len(graded_games) == 1
    assert graded_games[0]["id"] == 1

    on_disk = json.loads(week_path.read_text())
    assert on_disk["games"][0]["correct"] is True
    assert on_disk["games"][1]["correct"] is None


def test_grade_week_file_updates_latest_when_it_matches(tmp_path):
    week_path = tmp_path / "week1.json"
    latest_path = tmp_path / "latest.json"
    payload = {"season": 2026, "week": 1, "games": [_game(id=1)]}
    week_path.write_text(json.dumps(payload))
    latest_path.write_text(json.dumps(payload))

    results = {1: {"completed": True, "home_points": 28.0, "away_points": 14.0}}
    _grade_week_file(week_path, results, latest_path, latest_key=(2026, 1))

    latest_payload = json.loads(latest_path.read_text())
    assert latest_payload["games"][0]["correct"] is True


def test_grade_week_file_does_not_touch_latest_for_other_weeks(tmp_path):
    week_path = tmp_path / "week1.json"
    latest_path = tmp_path / "latest.json"
    payload = {"season": 2026, "week": 1, "games": [_game(id=1)]}
    week_path.write_text(json.dumps(payload))
    latest_payload = {"season": 2026, "week": 2, "games": []}
    latest_path.write_text(json.dumps(latest_payload))

    results = {1: {"completed": True, "home_points": 28.0, "away_points": 14.0}}
    _grade_week_file(week_path, results, latest_path, latest_key=(2026, 2))

    assert json.loads(latest_path.read_text()) == latest_payload
