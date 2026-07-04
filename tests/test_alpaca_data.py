"""Alpaca Market Data client tests, zero network (mirrors test_polygon_client.py).

ADR-008: free long-history source for FIXED research universes only.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
from src.data.alpaca_data import AlpacaDataClient
from src.shared.exceptions import DataValidationError

START = datetime(2016, 1, 1, tzinfo=UTC)
END = datetime(2016, 1, 31, tzinfo=UTC)


def _row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "t": "2016-01-04T05:00:00Z",
        "o": 100.0,
        "h": 101.0,
        "l": 99.0,
        "c": 100.5,
        "v": 1000,
        "vw": 100.2,
    }
    row.update(overrides)
    return row


def _payload(*rows: dict[str, Any], next_page_token: str | None = None) -> dict[str, Any]:
    return {"bars": list(rows), "next_page_token": next_page_token}


class _Recorder:
    """Mock transport handler recording every request, scripted responses."""

    def __init__(self, responses: list[httpx.Response]) -> None:
        self.requests: list[httpx.Request] = []
        self._responses = responses

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self._responses[min(len(self.requests), len(self._responses)) - 1]


def _client(recorder: _Recorder) -> AlpacaDataClient:
    return AlpacaDataClient(
        api_key="test-key",
        secret_key="test-secret",
        backoff_initial_s=0.0,  # no real sleeping in tests
        transport=httpx.MockTransport(recorder),
    )


async def test_maps_alpaca_fields_to_bar() -> None:
    recorder = _Recorder([httpx.Response(200, json=_payload(_row()))])
    bars = await _client(recorder).fetch_bars("SPY", START, END)
    assert len(bars) == 1
    bar = bars[0]
    assert bar.symbol == "SPY"
    assert bar.timestamp == datetime(2016, 1, 4, 5, 0, tzinfo=UTC)  # ISO 'Z' -> UTC
    assert (bar.open, bar.high, bar.low, bar.close) == (100.0, 101.0, 99.0, 100.5)
    assert bar.volume == 1000
    assert bar.vwap == 100.2


async def test_requests_adjustment_all_iex_and_auth_headers() -> None:
    recorder = _Recorder([httpx.Response(200, json=_payload(_row()))])
    await _client(recorder).fetch_bars("SPY", START, END)
    request = recorder.requests[0]
    assert request.url.params["adjustment"] == "all"  # split+dividend adjusted
    assert request.url.params["feed"] == "iex"
    assert request.url.params["timeframe"] == "1Day"
    assert request.url.params["start"] == "2016-01-01"
    assert request.url.params["end"] == "2016-01-31"
    assert "/v2/stocks/SPY/bars" in str(request.url)
    assert request.headers["APCA-API-KEY-ID"] == "test-key"
    assert request.headers["APCA-API-SECRET-KEY"] == "test-secret"


async def test_paginates_via_next_page_token() -> None:
    recorder = _Recorder(
        [
            httpx.Response(200, json=_payload(_row(), next_page_token="tok-1")),
            httpx.Response(200, json=_payload(_row(t="2016-01-05T05:00:00Z"))),
        ]
    )
    bars = await _client(recorder).fetch_bars("SPY", START, END)
    assert len(bars) == 2
    assert len(recorder.requests) == 2
    assert "page_token" not in recorder.requests[0].url.params
    assert recorder.requests[1].url.params["page_token"] == "tok-1"


async def test_retries_on_429_then_succeeds() -> None:
    recorder = _Recorder(
        [
            httpx.Response(429, json={}),
            httpx.Response(429, json={}),
            httpx.Response(200, json=_payload(_row())),
        ]
    )
    bars = await _client(recorder).fetch_bars("SPY", START, END)
    assert len(bars) == 1
    assert len(recorder.requests) == 3


async def test_4xx_raises_no_retry() -> None:
    recorder = _Recorder([httpx.Response(403, json={})])
    with pytest.raises(httpx.HTTPStatusError):
        await _client(recorder).fetch_bars("SPY", START, END)
    assert len(recorder.requests) == 1  # no retry on non-retryable 4xx


async def test_invalid_bar_from_api_raises() -> None:
    recorder = _Recorder([httpx.Response(200, json=_payload(_row(c=-5.0)))])
    with pytest.raises(DataValidationError, match="close"):
        await _client(recorder).fetch_bars("SPY", START, END)


async def test_empty_bars_page_returns_no_bars() -> None:
    recorder = _Recorder([httpx.Response(200, json={"bars": None, "next_page_token": None})])
    bars = await _client(recorder).fetch_bars("SPY", START, END)
    assert bars == []
