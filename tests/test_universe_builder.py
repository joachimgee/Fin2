"""Mechanical universe selection tests — discriminative hand-built fixture."""

from __future__ import annotations

import pandas as pd
from src.data.universe_builder import select_liquid_universe


def _bars(symbol: str, days: int, close: float, volume: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": symbol,
            "timestamp": pd.date_range("2020-07-01", periods=days, freq="B", tz="UTC"),
            "close": close,
            "volume": volume,
        }
    )


def _fixture() -> pd.DataFrame:
    return pd.concat(
        [
            _bars("BIG", days=60, close=100.0, volume=1_000_000),  # $100M/day — rank 1
            _bars("MID", days=60, close=50.0, volume=400_000),  # $20M/day  — rank 2
            _bars("SML", days=60, close=20.0, volume=100_000),  # $2M/day   — rank 3
            _bars("PNY", days=60, close=2.0, volume=90_000_000),  # huge $ but penny — excluded
            _bars("NEW", days=10, close=80.0, volume=5_000_000),  # listed mid-window — excluded
        ]
    )


def test_ranked_by_median_dollar_volume() -> None:
    assert select_liquid_universe(_fixture(), min_price=5.0, min_days=55, top_n=3) == [
        "BIG",
        "MID",
        "SML",
    ]


def test_top_n_truncates_in_rank_order() -> None:
    assert select_liquid_universe(_fixture(), min_price=5.0, min_days=55, top_n=2) == [
        "BIG",
        "MID",
    ]


def test_penny_stock_excluded_despite_dollar_volume() -> None:
    # PNY trades $180M/day — more than BIG — but median close 2.0 < min_price
    universe = select_liquid_universe(_fixture(), min_price=5.0, min_days=55, top_n=5)
    assert "PNY" not in universe


def test_short_history_excluded() -> None:
    universe = select_liquid_universe(_fixture(), min_price=5.0, min_days=55, top_n=5)
    assert "NEW" not in universe


def test_input_not_mutated() -> None:
    frame = _fixture()
    before = frame.copy()
    select_liquid_universe(frame, min_price=5.0, min_days=55, top_n=3)
    pd.testing.assert_frame_equal(frame, before)
