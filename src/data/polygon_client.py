"""Polygon.io historical data sync — the ONLY module that talks to Polygon.

Backtest data comes from Polygon exclusively (never yfinance: it excludes
delisted stocks, so failed companies vanish and strategies look artificially
good — survivorship bias, root CLAUDE.md <trading_rules>).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

import httpx

from src.data.models import Bar, validate_bar
from src.shared.exceptions import ConfigError

log = logging.getLogger(__name__)

_BASE_URL = "https://api.polygon.io"
_MAX_TRIES = 5
_TIMEFRAMES: dict[str, tuple[str, str]] = {
    "1Min": ("1", "minute"),
    "1Hour": ("1", "hour"),
    "1Day": ("1", "day"),
}


class PolygonClient:
    """Async client for Polygon.io aggregates.

    Always requests adjusted=True — unadjusted data corrupts every backtest.
    Retries 429/5xx with exponential backoff; any other 4xx raises immediately.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = _BASE_URL,
        backoff_initial_s: float = 1.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._backoff_initial_s = backoff_initial_s
        self._transport = transport

    async def fetch_bars(
        self, symbol: str, start: datetime, end: datetime, timeframe: str = "1Day"
    ) -> list[Bar]:
        if timeframe not in _TIMEFRAMES:
            raise ConfigError(f"unsupported timeframe: {timeframe!r}")
        multiplier, span = _TIMEFRAMES[timeframe]
        url = f"/v2/aggs/ticker/{symbol}/range/{multiplier}/{span}/{start:%Y-%m-%d}/{end:%Y-%m-%d}"
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": "50000",
            "apiKey": self._api_key,
        }
        payload = await self._get_with_retry(url, params)
        results = payload.get("results") or []
        bars = [
            validate_bar(
                Bar(
                    symbol=symbol,
                    # Polygon "t" is MILLISECONDS since epoch
                    timestamp=datetime.fromtimestamp(row["t"] / 1000.0, tz=UTC),
                    open=float(row["o"]),
                    high=float(row["h"]),
                    low=float(row["l"]),
                    close=float(row["c"]),
                    volume=int(row["v"]),
                    vwap=float(row["vw"]) if "vw" in row else None,
                )
            )
            for row in results
        ]
        log.info(
            "polygon_bars_fetched",
            extra={"symbol": symbol, "timeframe": timeframe, "count": len(bars)},
        )
        return bars

    async def _get_with_retry(self, url: str, params: dict[str, str]) -> dict[str, Any]:
        response: httpx.Response | None = None
        async with httpx.AsyncClient(
            base_url=self._base_url, transport=self._transport, timeout=30.0
        ) as client:
            for attempt in range(_MAX_TRIES):
                response = await client.get(url, params=params)
                if response.status_code == 200:
                    data: dict[str, Any] = response.json()
                    return data
                if response.status_code != 429 and response.status_code < 500:
                    response.raise_for_status()  # non-retryable 4xx
                if attempt < _MAX_TRIES - 1:
                    delay = self._backoff_initial_s * 2**attempt
                    log.warning(
                        "polygon_retry",
                        extra={"status": response.status_code, "attempt": attempt, "delay": delay},
                    )
                    await asyncio.sleep(delay)
        assert response is not None
        response.raise_for_status()  # retries exhausted on 429/5xx — always raises
        raise AssertionError("unreachable")
