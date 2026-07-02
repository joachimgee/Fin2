"""Canonical market-data models. Every module speaks Bar — SDK types never leak
past the module that produced them (alpaca-py types stay in execution/,
polygon types stay in data/polygon_client.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


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


def validate_bar(bar: Bar) -> Bar:
    """Fail-fast validation gate. Every bar entering the system passes here once.

    TODO(Phase 1): implement checks, raise DataValidationError on any failure:
      - no NaN/inf in any price field
      - all prices > 0, volume >= 0
      - low <= open/close <= high
      - timestamp is timezone-aware (UTC)
    Rationale: one corrupt price that propagates produces a wrong position size
    silently. Crash immediately instead (fail-fast, root CLAUDE.md <context>).
    """
    raise NotImplementedError("Phase 1 — bar validation")
