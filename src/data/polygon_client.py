"""Polygon.io historical data sync — the ONLY module that talks to Polygon.

Backtest data comes from Polygon exclusively (never yfinance: it excludes
delisted stocks, so failed companies vanish and strategies look artificially
good — survivorship bias, root CLAUDE.md <trading_rules>).
"""

from __future__ import annotations

from datetime import datetime

from src.data.models import Bar


class PolygonClient:
    """Async client for Polygon.io aggregates.

    TODO(Phase 1):
      - __init__ takes api_key (from require_env("POLYGON_API_KEY") at call site).
      - fetch_bars(symbol, start, end, timeframe) -> list[Bar]
        * always request adjusted=True (splits/dividends)
        * run every bar through data.models.validate_bar before returning
        * httpx async, retry with exponential backoff on 429/5xx
    """

    def __init__(self, api_key: str) -> None:
        raise NotImplementedError("Phase 1 — Polygon sync")

    async def fetch_bars(
        self, symbol: str, start: datetime, end: datetime, timeframe: str = "1Day"
    ) -> list[Bar]:
        raise NotImplementedError("Phase 1")
