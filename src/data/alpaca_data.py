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
import re
from datetime import UTC, datetime
from typing import Any

import httpx

from src.data.models import Bar, validate_bar
from src.shared.exceptions import DataValidationError

log = logging.getLogger(__name__)

_BASE_URL = "https://data.alpaca.markets"
_TRADING_BASE_URL = "https://paper-api.alpaca.markets"  # read-only /v2/assets metadata
_MAX_TRIES = 5
_PAGE_LIMIT = 10_000
_PLAIN_SYMBOL = re.compile(r"^[A-Z]{1,5}$")  # skips units/warrants/preferreds (BRK.A, FOO.WS)


class AlpacaDataClient:
    """Async client for /v2/stocks bars endpoints with pagination and backoff."""

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        base_url: str = _BASE_URL,
        trading_base_url: str = _TRADING_BASE_URL,
        backoff_initial_s: float = 1.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._headers = {"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": secret_key}
        self._base_url = base_url
        self._trading_base_url = trading_base_url
        self._backoff_initial_s = backoff_initial_s
        self._transport = transport

    async def fetch_bars(self, symbol: str, start: datetime, end: datetime) -> list[Bar]:
        bars: list[Bar] = []
        page_token: str | None = None
        async with self._client(self._base_url) as client:
            while True:
                params = self._bar_params(start, end, page_token)
                payload = await self._get_with_retry(client, f"/v2/stocks/{symbol}/bars", params)
                for row in payload.get("bars") or []:
                    bars.append(validate_bar(_bar_from_row(symbol, row)))
                page_token = payload.get("next_page_token")
                if not page_token:
                    break
        log.info("alpaca_bars_fetched", extra={"symbol": symbol, "count": len(bars)})
        return bars

    async def fetch_bars_multi(
        self, symbols: list[str], start: datetime, end: datetime, skip_invalid: bool = False
    ) -> dict[str, list[Bar]]:
        """Batch endpoint /v2/stocks/bars?symbols=A,B,... — one call for many
        symbols (universe scans). skip_invalid drops bars that fail validation
        instead of raising: a broad market scan must not die on one bad row of
        an obscure ticker; backtest data syncs stay strict via fetch_bars."""
        bars: dict[str, list[Bar]] = {symbol: [] for symbol in symbols}
        skipped = 0
        page_token: str | None = None
        async with self._client(self._base_url) as client:
            while True:
                params = self._bar_params(start, end, page_token)
                params["symbols"] = ",".join(symbols)
                payload = await self._get_with_retry(client, "/v2/stocks/bars", params)
                for symbol, rows in (payload.get("bars") or {}).items():
                    for row in rows or []:
                        try:
                            bars.setdefault(symbol, []).append(
                                validate_bar(_bar_from_row(symbol, row))
                            )
                        except DataValidationError:
                            if not skip_invalid:
                                raise
                            skipped += 1
                page_token = payload.get("next_page_token")
                if not page_token:
                    break
        log.info(
            "alpaca_multi_bars_fetched",
            extra={"symbols": len(symbols), "skipped_invalid": skipped},
        )
        return bars

    async def fetch_active_symbols(self, exchanges: list[str]) -> list[str]:
        """Active, tradable US equities on the given exchanges — plain symbols
        only. Trading API metadata (read-only); today's active list, so any
        universe built from it carries residual delisting survivorship —
        callers must document that."""
        params = {"status": "active", "asset_class": "us_equity"}
        async with self._client(self._trading_base_url) as client:
            payload = await self._get_with_retry(client, "/v2/assets", params)
        wanted = set(exchanges)
        symbols = sorted(
            str(asset["symbol"])
            for asset in payload
            if asset.get("exchange") in wanted
            and asset.get("tradable")
            and _PLAIN_SYMBOL.match(str(asset["symbol"]))
        )
        log.info("alpaca_assets_fetched", extra={"count": len(symbols)})
        return symbols

    # --- internals ------------------------------------------------------------------

    def _client(self, base_url: str) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=base_url, headers=self._headers, transport=self._transport, timeout=30.0
        )

    @staticmethod
    def _bar_params(start: datetime, end: datetime, page_token: str | None) -> dict[str, str]:
        params = {
            "timeframe": "1Day",
            "start": f"{start:%Y-%m-%d}",
            "end": f"{end:%Y-%m-%d}",
            "adjustment": "all",  # split + dividend adjusted — non-negotiable
            "feed": "iex",
            "limit": str(_PAGE_LIMIT),
        }
        if page_token is not None:
            params["page_token"] = page_token
        return params

    async def _get_with_retry(self, client: httpx.AsyncClient, url: str, params: Any) -> Any:
        response: httpx.Response | None = None
        for attempt in range(_MAX_TRIES):
            response = await client.get(url, params=params)
            if response.status_code == 200:
                return response.json()
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


def _bar_from_row(symbol: str, row: dict[str, Any]) -> Bar:
    return Bar(
        symbol=symbol,
        timestamp=datetime.fromisoformat(str(row["t"]).replace("Z", "+00:00")).astimezone(UTC),
        open=float(row["o"]),
        high=float(row["h"]),
        low=float(row["l"]),
        close=float(row["c"]),
        volume=int(row["v"]),
        vwap=float(row["vw"]) if "vw" in row else None,
    )
