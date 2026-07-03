"""Phase 4 — StreamManager tests, mocked SDK + fake Redis (PHASE_4_EXECUTION.md)."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest
import src.execution.stream_manager as sm_mod
from src.execution.stream_manager import StreamManager


class FakeRedis:
    def __init__(self) -> None:
        self.published: list[tuple[str, str]] = []

    async def publish(self, channel: str, payload: str) -> None:
        self.published.append((channel, payload))


class Harness:
    def __init__(self, base_config: dict[str, Any]) -> None:
        self.redis = FakeRedis()
        self.fills: list[dict[str, Any]] = []
        self.alerts: list[str] = []
        self.resyncs = 0

        async def resync() -> None:
            self.resyncs += 1

        self.sm = StreamManager(
            "k",
            "s",
            paper=True,
            redis_client=self.redis,
            on_fill=self.fills.append,
            resync=resync,
            alert=self.alerts.append,
            config=base_config,
        )


def _update(event: str, side: str = "buy", qty: float = 5, price: float = 101.0) -> Any:
    return SimpleNamespace(
        event=event, qty=qty, price=price, order=SimpleNamespace(symbol="AAPL", side=side)
    )


@pytest.fixture
def harness(base_config: dict[str, Any]) -> Harness:
    return Harness(base_config)


async def test_fill_forwarded_to_on_fill(harness: Harness) -> None:
    await harness.sm._on_trade_update(_update("fill"))
    assert harness.fills == [{"symbol": "AAPL", "side": "buy", "qty": 5.0, "price": 101.0}]


async def test_partial_fill_applied_and_warned(
    harness: Harness, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        await harness.sm._on_trade_update(_update("partial_fill"))
    assert len(harness.fills) == 1
    assert any(r.message == "order_partial_fill" for r in caplog.records)


async def test_rejected_triggers_alert(harness: Harness) -> None:
    await harness.sm._on_trade_update(_update("rejected"))
    assert harness.alerts and "AAPL" in harness.alerts[0]
    assert harness.fills == []


async def test_unknown_event_logs_unhandled(
    harness: Harness, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        await harness.sm._on_trade_update(_update("mystery_event"))
    assert any(r.message == "unhandled_trade_event" for r in caplog.records)
    assert harness.fills == []  # explicitly NOT treated as a fill


async def test_bar_published_to_redis_channel(harness: Harness) -> None:
    bar = SimpleNamespace(
        symbol="AAPL",
        timestamp=datetime(2026, 1, 5, 21, tzinfo=UTC),
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        volume=1000,
        vwap=100.2,
    )
    await harness.sm._on_bar(bar)
    channel, payload = harness.redis.published[0]
    assert channel == "channel:bars:AAPL"
    assert '"close": 100.5' in payload


def test_backoff_sequence_doubles_and_caps(harness: Harness) -> None:
    delays = [harness.sm._backoff_delay(a) for a in range(8)]
    assert delays == [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 60.0, 60.0]


async def test_resync_before_every_reconnect(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Failing stream: resync must run on every attempt, delays must grow."""
    delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)
        if len(delays) >= 3:
            raise RuntimeError("stop-test")

    async def failing_runner() -> None:
        raise ConnectionError("ws dropped")

    monkeypatch.setattr(sm_mod.asyncio, "sleep", fake_sleep)
    with pytest.raises(RuntimeError, match="stop-test"):
        await harness.sm._run_with_reconnect(failing_runner, "test")
    assert harness.resyncs == 3  # sync before consuming events, every single time
    assert delays == [1.0, 2.0, 4.0]  # instant reconnects would risk a ban


async def test_start_twice_raises(harness: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeStream:
        instances = 0

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            FakeStream.instances += 1

        def subscribe_trade_updates(self, handler: Any) -> None:
            pass

        def subscribe_bars(self, handler: Any, *symbols: str) -> None:
            pass

        async def _run_forever(self) -> None:
            return None

    async def noop_loop(runner: Any, name: str) -> None:
        return None

    monkeypatch.setattr(sm_mod, "TradingStream", FakeStream)
    monkeypatch.setattr(sm_mod, "StockDataStream", FakeStream)
    monkeypatch.setattr(harness.sm, "_run_with_reconnect", noop_loop)
    await harness.sm.start(["AAPL"])
    assert FakeStream.instances == 2  # exactly one trading + one data stream
    with pytest.raises(RuntimeError, match="already started"):
        await harness.sm.start(["AAPL"])
