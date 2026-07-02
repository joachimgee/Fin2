"""Phase 1 — bar validation tests (docs/plan/PHASE_1_DATA.md)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from src.data.models import Bar, validate_bar
from src.shared.exceptions import DataValidationError


def _bar(**overrides: Any) -> Bar:
    fields: dict[str, Any] = {
        "symbol": "SPY",
        "timestamp": datetime(2024, 1, 2, 21, 0, tzinfo=UTC),
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.5,
        "volume": 1_000,
        "vwap": 100.2,
    }
    fields.update(overrides)
    return Bar(**fields)


def test_valid_bar_passes() -> None:
    bar = _bar()
    assert validate_bar(bar) is bar


def test_nan_price_raises() -> None:
    with pytest.raises(DataValidationError, match="close"):
        validate_bar(_bar(close=float("nan")))


def test_negative_price_raises() -> None:
    with pytest.raises(DataValidationError, match="open"):
        validate_bar(_bar(open=-1.0))


def test_low_above_open_raises() -> None:
    # low must be <= min(open, close); here low > open
    with pytest.raises(DataValidationError, match="low"):
        validate_bar(_bar(low=100.2))


def test_close_above_high_raises() -> None:
    with pytest.raises(DataValidationError, match="high"):
        validate_bar(_bar(close=102.0))


def test_naive_timestamp_raises() -> None:
    with pytest.raises(DataValidationError, match="timezone"):
        validate_bar(_bar(timestamp=datetime(2024, 1, 2, 21, 0)))


def test_zero_volume_ok() -> None:
    bar = _bar(volume=0)
    assert validate_bar(bar) is bar
