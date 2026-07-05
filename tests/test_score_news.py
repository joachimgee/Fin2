"""score_news aggregation tests — pure function, no FinBERT, no network."""

from __future__ import annotations

from datetime import UTC, datetime

from scripts.score_news import cap_by_day

D1 = datetime(2024, 1, 2, tzinfo=UTC)
D2 = datetime(2024, 1, 3, tzinfo=UTC)


def _at(day: datetime, hour: int) -> datetime:
    return day.replace(hour=hour)


def test_groups_by_utc_calendar_date() -> None:
    news = [(_at(D1, 9), "a"), (_at(D1, 15), "b"), (_at(D2, 10), "c")]
    assert cap_by_day(news, max_per_day=10) == {D1: ["a", "b"], D2: ["c"]}


def test_caps_chronologically_first_n() -> None:
    news = [(_at(D1, 9), "a"), (_at(D1, 10), "b"), (_at(D1, 11), "c")]
    assert cap_by_day(news, max_per_day=2) == {D1: ["a", "b"]}  # "c" never scored


def test_days_ordered_and_empty_input_safe() -> None:
    news = [(_at(D2, 9), "late"), (_at(D1, 9), "early")]
    assert list(cap_by_day(news, max_per_day=5)) == [D1, D2]
    assert cap_by_day([], max_per_day=5) == {}
