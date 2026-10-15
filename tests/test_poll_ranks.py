import pandas as pd

from cfb.model import poll_ranks


def _poll(name, schools):
    return {"poll": name, "ranks": [{"school": s, "rank": i + 1} for i, s in enumerate(schools)]}


RANKINGS = pd.DataFrame(
    [
        {"seasonType": "regular", "week": 1, "polls": [_poll("Coaches Poll", ["Ohio State", "Oregon"])]},
        {
            "seasonType": "regular",
            "week": 3,
            "polls": [_poll("Coaches Poll", ["Oregon", "Texas"]), _poll("AP Top 25", ["Texas", "Oregon"])],
        },
        {"seasonType": "postseason", "week": 1, "polls": [_poll("AP Top 25", ["Georgia"])]},
    ]
)


def test_uses_latest_poll_at_or_before_week_and_prefers_ap():
    assert poll_ranks(RANKINGS, 5) == ("AP Top 25", {"Texas": 1, "Oregon": 2})


def test_falls_back_to_coaches_poll():
    assert poll_ranks(RANKINGS, 2) == ("Coaches Poll", {"Ohio State": 1, "Oregon": 2})


def test_no_poll_yet():
    assert poll_ranks(RANKINGS, 0) == (None, {})
    assert poll_ranks(pd.DataFrame(), 4) == (None, {})
