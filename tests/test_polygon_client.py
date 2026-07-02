"""Phase 1 — Polygon client tests, zero network (docs/plan/PHASE_1_DATA.md)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
from src.data.polygon_client import PolygonClient
from src.shared.exceptions import DataValidationError

START = datetime(2024, 1, 1, tzinfo=UTC)
END = datetime(2024, 1, 31, tzinfo=UTC)


def _payload(**row_overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        # 1704229200000 ms == 2024-01-02T21:00:00Z (exact mapping asserted below)
        "t": 1704229200000,
        "o": 100.0,
        "h": 101.0,
        "l": 99.0,
        "c": 100.5,
        "v": 1000,
        "vw": 100.2,
    }
    row.update(row_overrides)
    return {"status": "OK", "results": [row]}


class _Recorder:
    """Mock transport handler recording every request, scripted responses."""

    def __init__(self, responses: list[httpx.Response]) -> None:
        self.requests: list[httpx.Request] = []
        self._responses = responses

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self._responses[min(len(self.requests), len(self._responses)) - 1]


def _client(recorder: _Recorder) -> PolygonClient:
    return PolygonClient(
        api_key="test-key",
        backoff_initial_s=0.0,  # no real sleeping in tests
        transport=httpx.MockTransport(recorder),
    )


async def test_maps_polygon_fields_to_bar() -> None:
    recorder = _Recorder([httpx.Response(200, json=_payload())])
    bars = await _client(recorder).fetch_bars("SPY", START, END)
    assert len(bars) == 1
    bar = bars[0]
    assert bar.symbol == "SPY"
    assert bar.timestamp == datetime(2024, 1, 2, 21, 0, tzinfo=UTC)  # ms epoch, not s
    assert (bar.open, bar.high, bar.low, bar.close) == (100.0, 101.0, 99.0, 100.5)
    assert bar.volume == 1000
    assert bar.vwap == 100.2


async def test_requests_adjusted_true() -> None:
    recorder = _Recorder([httpx.Response(200, json=_payload())])
    await _client(recorder).fetch_bars("SPY", START, END)
    request = recorder.requests[0]
    assert request.url.params["adjusted"] == "true"
    assert "/range/1/day/2024-01-01/2024-01-31" in str(request.url)


async def test_retries_on_429_then_succeeds() -> None:
    recorder = _Recorder(
        [
            httpx.Response(429, json={}),
            httpx.Response(429, json={}),
            httpx.Response(200, json=_payload()),
        ]
    )
    bars = await _client(recorder).fetch_bars("SPY", START, END)
    assert len(bars) == 1
    assert len(recorder.requests) == 3


async def test_invalid_bar_from_api_raises() -> None:
    recorder = _Recorder([httpx.Response(200, json=_payload(c=-5.0))])
    with pytest.raises(DataValidationError, match="close"):
        await _client(recorder).fetch_bars("SPY", START, END)


async def test_4xx_raises_no_retry() -> None:
    recorder = _Recorder([httpx.Response(403, json={})])
    with pytest.raises(httpx.HTTPStatusError):
        await _client(recorder).fetch_bars("SPY", START, END)
    assert len(recorder.requests) == 1  # no retry on non-retryable 4xx
