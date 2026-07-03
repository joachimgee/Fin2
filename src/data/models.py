"""Canonical market-data models. Every module speaks Bar — SDK types never leak
past the module that produced them (alpaca-py types stay in execution/,
polygon types stay in data/polygon_client.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

from src.shared.exceptions import DataValidationError


@dataclass(frozen=True, slots=True)
class Bar:
    """One OHLCV bar. Immutable — bars are facts, never mutated after creation."""

    symbol: str
    timestamp: datetime  # timezone-aware UTC, bar CLOSE time
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: float | None = None


def _fail(bar: Bar, reason: str) -> None:
    raise DataValidationError(f"{bar.symbol}@{bar.timestamp}: {reason}")


def validate_bar(bar: Bar) -> Bar:
    """Fail-fast validation gate. Every bar entering the system passes here once.

    One corrupt price that propagates produces a wrong position size silently;
    crashing immediately is the cheaper failure (root CLAUDE.md <context>).
    Pure: returns the same frozen Bar, mutates nothing.
    """
    prices: dict[str, float] = {
        "open": bar.open,
        "high": bar.high,
        "low": bar.low,
        "close": bar.close,
    }
    if bar.vwap is not None:
        prices["vwap"] = bar.vwap

    for name, value in prices.items():
        if not math.isfinite(value):
            _fail(bar, f"{name} is not finite: {value!r}")
    for name, value in prices.items():
        if value <= 0:
            _fail(bar, f"{name} must be > 0, got {value}")
    if bar.volume < 0:
        _fail(bar, f"volume must be >= 0, got {bar.volume}")
    if bar.low > min(bar.open, bar.close):
        _fail(bar, f"low {bar.low} above open/close")
    if max(bar.open, bar.close) > bar.high:
        _fail(bar, f"open/close above high {bar.high}")
    if bar.timestamp.tzinfo is None:
        _fail(bar, "timestamp must be timezone-aware (UTC)")
    return bar
