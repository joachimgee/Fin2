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


# --- fetch_bars_multi (universe scans) --------------------------------------------


def _multi_payload(
    bars_by_symbol: dict[str, list[dict[str, Any]]], next_page_token: str | None = None
) -> dict[str, Any]:
    return {"bars": bars_by_symbol, "next_page_token": next_page_token}


async def test_multi_maps_symbols_and_sends_symbols_param() -> None:
    recorder = _Recorder(
        [
            httpx.Response(
                200,
                json=_multi_payload({"AAA": [_row()], "BBB": [_row(c=50.0, o=50.0, l=49.0)]}),
            )
        ]
    )
    bars = await _client(recorder).fetch_bars_multi(["AAA", "BBB"], START, END)
    assert set(bars) == {"AAA", "BBB"}
    assert bars["AAA"][0].symbol == "AAA"
    assert bars["BBB"][0].close == 50.0
    request = recorder.requests[0]
    assert request.url.params["symbols"] == "AAA,BBB"
    assert request.url.params["adjustment"] == "all"
    assert str(request.url).split("?")[0].endswith("/v2/stocks/bars")


async def test_multi_paginates_and_merges_pages() -> None:
    recorder = _Recorder(
        [
            httpx.Response(200, json=_multi_payload({"AAA": [_row()]}, next_page_token="tok")),
            httpx.Response(200, json=_multi_payload({"AAA": [_row(t="2016-01-05T05:00:00Z")]})),
        ]
    )
    bars = await _client(recorder).fetch_bars_multi(["AAA"], START, END)
    assert len(bars["AAA"]) == 2
    assert recorder.requests[1].url.params["page_token"] == "tok"


async def test_multi_skip_invalid_drops_bad_rows_only() -> None:
    payload = _multi_payload({"AAA": [_row(c=-5.0), _row()], "BBB": [_row()]})
    recorder = _Recorder([httpx.Response(200, json=payload)])
    bars = await _client(recorder).fetch_bars_multi(["AAA", "BBB"], START, END, skip_invalid=True)
    assert len(bars["AAA"]) == 1  # bad row dropped, good row kept
    assert len(bars["BBB"]) == 1


async def test_multi_strict_raises_on_invalid_bar() -> None:
    payload = _multi_payload({"AAA": [_row(c=-5.0)]})
    recorder = _Recorder([httpx.Response(200, json=payload)])
    with pytest.raises(DataValidationError):
        await _client(recorder).fetch_bars_multi(["AAA"], START, END)


# --- fetch_news (Benzinga headlines) ----------------------------------------------


def _news_payload(*items: dict[str, Any], next_page_token: str | None = None) -> dict[str, Any]:
    return {"news": list(items), "next_page_token": next_page_token}


def _news_item(headline: str, created_at: str = "2016-01-04T14:30:00Z") -> dict[str, Any]:
    return {"headline": headline, "created_at": created_at}


async def test_news_maps_headline_and_timestamp() -> None:
    recorder = _Recorder([httpx.Response(200, json=_news_payload(_news_item("Apple up")))])
    news = await _client(recorder).fetch_news("AAPL", START, END)
    assert news == [(datetime(2016, 1, 4, 14, 30, tzinfo=UTC), "Apple up")]
    request = recorder.requests[0]
    assert request.url.params["symbols"] == "AAPL"
    assert request.url.params["include_content"] == "false"
    assert request.url.params["sort"] == "asc"
    assert "/v1beta1/news" in str(request.url)


async def test_news_paginates() -> None:
    recorder = _Recorder(
        [
            httpx.Response(200, json=_news_payload(_news_item("one"), next_page_token="t")),
            httpx.Response(200, json=_news_payload(_news_item("two"))),
        ]
    )
    news = await _client(recorder).fetch_news("AAPL", START, END)
    assert [h for _, h in news] == ["one", "two"]
    assert recorder.requests[1].url.params["page_token"] == "t"


# --- fetch_active_symbols (trading API metadata) ----------------------------------


def _asset(symbol: str, exchange: str = "NYSE", tradable: bool = True) -> dict[str, Any]:
    return {"symbol": symbol, "exchange": exchange, "tradable": tradable}


async def test_active_symbols_filters_exchange_tradable_and_plain_symbols() -> None:
    recorder = _Recorder(
        [
            httpx.Response(
                200,
                json=[
                    _asset("AAA"),
                    _asset("BBB", exchange="NASDAQ"),
                    _asset("OTC1", exchange="OTC"),  # wrong exchange
                    _asset("CCC", tradable=False),  # not tradable
                    _asset("BRK.A"),  # dotted class share — excluded
                    _asset("TOOLONGX"),  # > 5 chars — excluded
                ],
            )
        ]
    )
    symbols = await _client(recorder).fetch_active_symbols(["NYSE", "NASDAQ"])
    assert symbols == ["AAA", "BBB"]
    request = recorder.requests[0]
    assert request.url.params["status"] == "active"
    assert request.url.params["asset_class"] == "us_equity"
    assert request.headers["APCA-API-KEY-ID"] == "test-key"
