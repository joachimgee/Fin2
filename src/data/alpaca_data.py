"""Alpaca Market Data (historical daily bars) — the FREE long-history source.

Raw REST via httpx: the dependency graph reserves the alpaca-py SDK for
execution/; this module talks to data.alpaca.markets (read-only market data,
no order capability) directly. See ARCHITECTURE.md ADR-008: sanctioned for
FIXED research universes only — full-market screening stays on Polygon.

Caveats (documented, deliberate): free plan serves the IEX feed — volumes are
a fraction of the consolidated tape (consistently biased), and daily closes
can differ marginally from the official auction close. adjustment=all gives
split+dividend adjusted prices.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

import httpx

from src.data.models import Bar, validate_bar

log = logging.getLogger(__name__)

_BASE_URL = "https://data.alpaca.markets"
_MAX_TRIES = 5
_PAGE_LIMIT = 10_000


class AlpacaDataClient:
    """Async client for /v2/stocks/{symbol}/bars with pagination and backoff."""

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        base_url: str = _BASE_URL,
        backoff_initial_s: float = 1.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._headers = {"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": secret_key}
        self._base_url = base_url
        self._backoff_initial_s = backoff_initial_s
        self._transport = transport

    async def fetch_bars(self, symbol: str, start: datetime, end: datetime) -> list[Bar]:
        bars: list[Bar] = []
        page_token: str | None = None
        async with httpx.AsyncClient(
            base_url=self._base_url,
            headers=self._headers,
            transport=self._transport,
            timeout=30.0,
        ) as client:
            while True:
                params: dict[str, str] = {
                    "timeframe": "1Day",
                    "start": f"{start:%Y-%m-%d}",
                    "end": f"{end:%Y-%m-%d}",
                    "adjustment": "all",  # split + dividend adjusted — non-negotiable
                    "feed": "iex",
                    "limit": str(_PAGE_LIMIT),
                }
                if page_token is not None:
                    params["page_token"] = page_token
                payload = await self._get_with_retry(client, f"/v2/stocks/{symbol}/bars", params)
                for row in payload.get("bars") or []:
                    bars.append(
                        validate_bar(
                            Bar(
                                symbol=symbol,
                                timestamp=datetime.fromisoformat(
                                    str(row["t"]).replace("Z", "+00:00")
                                ).astimezone(UTC),
                                open=float(row["o"]),
                                high=float(row["h"]),
                                low=float(row["l"]),
                                close=float(row["c"]),
                                volume=int(row["v"]),
                                vwap=float(row["vw"]) if "vw" in row else None,
                            )
                        )
                    )
                page_token = payload.get("next_page_token")
                if not page_token:
                    break
        log.info("alpaca_bars_fetched", extra={"symbol": symbol, "count": len(bars)})
        return bars

    async def _get_with_retry(
        self, client: httpx.AsyncClient, url: str, params: dict[str, str]
    ) -> dict[str, Any]:
        response: httpx.Response | None = None
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
                    "alpaca_data_retry",
                    extra={"status": response.status_code, "attempt": attempt, "delay": delay},
                )
                await asyncio.sleep(delay)
        assert response is not None
        response.raise_for_status()  # retries exhausted — always raises here
        raise AssertionError("unreachable")
