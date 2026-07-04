"""Phase 1 — DuckDB storage tests (docs/plan/PHASE_1_DATA.md)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from src.data.models import Bar
from src.data.storage import BarStorage


def _bars(symbol: str = "SPY", n: int = 3) -> list[Bar]:
    return [
        Bar(
            symbol=symbol,
            timestamp=datetime(2024, 1, 2 + i, 21, 0, tzinfo=UTC),
            open=100.0 + i,
            high=101.0 + i,
            low=99.0 + i,
            close=100.5 + i,
            volume=1_000 * (i + 1),
            vwap=100.2 + i,
        )
        for i in range(n)
    ]


def test_insert_then_get_roundtrip(tmp_db_path: Path) -> None:
    storage = BarStorage(tmp_db_path)
    assert storage.insert_bars(_bars()) == 3
    df = storage.get_bars(
        "SPY",
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 1, 10, tzinfo=UTC),
    )
    assert len(df) == 3
    assert df["close"].tolist() == [100.5, 101.5, 102.5]


def test_insert_same_bar_twice_one_row(tmp_db_path: Path) -> None:
    storage = BarStorage(tmp_db_path)
    storage.insert_bars(_bars(n=1))
    storage.insert_bars(_bars(n=1))  # idempotent re-sync
    df = storage.get_bars(
        "SPY",
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 1, 10, tzinfo=UTC),
    )
    assert len(df) == 1


def test_get_bars_sorted_ascending(tmp_db_path: Path) -> None:
    storage = BarStorage(tmp_db_path)
    storage.insert_bars(list(reversed(_bars())))  # inserted out of order
    df = storage.get_bars(
        "SPY",
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 1, 10, tzinfo=UTC),
    )
    timestamps = df["timestamp"].tolist()
    assert timestamps == sorted(timestamps)


def test_get_bars_filters_symbol_and_range(tmp_db_path: Path) -> None:
    storage = BarStorage(tmp_db_path)
    storage.insert_bars(_bars("SPY") + _bars("TSLA"))
    df = storage.get_bars(
        "SPY",
        datetime(2024, 1, 3, tzinfo=UTC),  # excludes the Jan 2 bar
        datetime(2024, 1, 10, tzinfo=UTC),
    )
    assert set(df["symbol"]) == {"SPY"}
    assert len(df) == 2


def test_sentiment_roundtrip_and_idempotent_upsert(tmp_db_path: Path) -> None:
    storage = BarStorage(tmp_db_path)
    day1 = datetime(2024, 1, 2, tzinfo=UTC)
    day2 = datetime(2024, 1, 3, tzinfo=UTC)
    assert storage.insert_daily_sentiment([("SPY", day1, 0.4, 12), ("SPY", day2, -0.2, 3)]) == 2
    storage.insert_daily_sentiment([("SPY", day1, 0.5, 15)])  # re-score overwrites
    frame = storage.get_daily_sentiment("SPY")
    assert len(frame) == 2
    assert frame["score"].tolist() == [0.5, -0.2]
    assert frame["n_headlines"].tolist() == [15, 3]
    assert storage.get_daily_sentiment("AAPL").empty
    assert storage.insert_daily_sentiment([]) == 0
