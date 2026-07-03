"""Phase 7 — Telegram alert tests, MockTransport only (PHASE_7_MONITORING.md)."""

from __future__ import annotations

import asyncio
import json
import logging
from types import SimpleNamespace

import httpx
import pytest
from src.monitoring.alerts import make_alert_dispatcher, send_telegram_alert
from src.monitoring.report import build_daily_report


class _Recorder:
    def __init__(self, status: int = 200, error: Exception | None = None) -> None:
        self.requests: list[httpx.Request] = []
        self._status = status
        self._error = error

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self._error is not None:
            raise self._error
        return httpx.Response(self._status, json={"ok": True})


async def test_posts_expected_payload() -> None:
    recorder = _Recorder()
    await send_telegram_alert(
        "breaker tripped", "TOKEN", "CHAT", 5.0, transport=httpx.MockTransport(recorder)
    )
    request = recorder.requests[0]
    assert "/botTOKEN/sendMessage" in str(request.url)
    body = json.loads(request.content)
    assert body == {"chat_id": "CHAT", "text": "breaker tripped"}


async def test_http_error_swallowed_and_logged(caplog: pytest.LogCaptureFixture) -> None:
    recorder = _Recorder(status=500)
    with caplog.at_level(logging.ERROR):
        await send_telegram_alert("m", "T", "C", 5.0, transport=httpx.MockTransport(recorder))
    assert any(r.message == "telegram_alert_failed" for r in caplog.records)


async def test_connect_error_swallowed() -> None:
    recorder = _Recorder(error=httpx.ConnectError("telegram unreachable"))
    # must return quietly — an alerting outage never touches the trading path
    await send_telegram_alert("m", "T", "C", 5.0, transport=httpx.MockTransport(recorder))


def test_dispatcher_in_sync_context() -> None:
    recorder = _Recorder()
    dispatch = make_alert_dispatcher("T", "C", 5.0, transport=httpx.MockTransport(recorder))
    dispatch("sync alert")  # CircuitBreaker.on_trip shape: plain callable
    assert len(recorder.requests) == 1


async def test_dispatcher_in_async_context_fire_and_forget() -> None:
    recorder = _Recorder()
    dispatch = make_alert_dispatcher("T", "C", 5.0, transport=httpx.MockTransport(recorder))
    dispatch("async alert")
    await asyncio.sleep(0.05)  # let the scheduled task run
    assert len(recorder.requests) == 1


def test_daily_report_contains_key_figures() -> None:
    tracker = SimpleNamespace(
        equity=101_500.0,
        total_exposure=lambda: 42_000.0,
        position_values=lambda: {"AAPL": 30_000.0, "MSFT": 12_000.0},
    )
    report = build_daily_report(tracker, [110.0, -40.0, 25.0])
    assert "101,500.00" in report
    assert "42,000.00" in report
    assert "open positions: 2" in report
    assert "trades closed: 3" in report
    assert "realized pnl: +95.00" in report
    assert "win rate: 67%" in report
    assert "AAPL" in report and "MSFT" in report
