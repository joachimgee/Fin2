"""build_sentiment_fn tests — lagged, strictly-before, bounded lookback."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
from scripts.run_backtest import build_sentiment_fn


def _frame(rows: list[tuple[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {"date": [d for d, _ in rows], "score": [s for _, s in rows], "n_headlines": 5}
    )


def _fn(rows: list[tuple[str, float]], lookback: int = 3):
    return build_sentiment_fn({"SPY": _frame(rows)}, lookback)


def _ts(day: str) -> datetime:
    return datetime.fromisoformat(f"{day}T21:00:00+00:00").astimezone(UTC)


def test_uses_most_recent_day_strictly_before_bar() -> None:
    fn = _fn([("2024-01-02", -0.5), ("2024-01-03", 0.4)])
    assert fn("SPY", _ts("2024-01-04")) == 0.4  # yesterday, not today
    assert fn("SPY", _ts("2024-01-03")) == -0.5  # SAME-day score never used


def test_weekend_gap_covered_by_lookback() -> None:
    fn = _fn([("2024-01-05", -0.6)])  # Friday news
    assert fn("SPY", _ts("2024-01-08")) == -0.6  # Monday bar sees Friday


def test_stale_news_beyond_lookback_is_neutral() -> None:
    fn = _fn([("2024-01-02", -0.9)], lookback=3)
    assert fn("SPY", _ts("2024-01-08")) == 0.0  # 6 days old — no veto


def test_unknown_symbol_and_empty_frame_neutral() -> None:
    fn = build_sentiment_fn({"SPY": _frame([]), "": _frame([])}, 3)
    assert fn("SPY", _ts("2024-01-04")) == 0.0
    assert fn("AAPL", _ts("2024-01-04")) == 0.0
