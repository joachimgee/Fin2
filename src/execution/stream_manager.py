"""StreamManager — owns the single WebSocket connection per Alpaca endpoint.

Alpaca allows ONE WebSocket per endpoint per account: a second connection
silently kills the first. This class is the sole consumer; start() refuses to
run twice. Bars fan out over Redis pub/sub (channel:bars:{symbol}) so N
strategies subscribe without touching Alpaca.

Reconnect: exponential backoff from stream.reconnect_backoff_initial_s doubling
to reconnect_backoff_cap_s — never instant (rate-limit ban during outages).
Before consuming ANY event (startup and every reconnect) the injected resync
callback runs: positions + account -> tracker.sync_from_api.

Callbacks stay fast: parse, hand off (injected on_fill / Redis publish), return.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

from alpaca.data.live import StockDataStream
from alpaca.trading.stream import TradingStream

log = logging.getLogger(__name__)

FillHandler = Callable[[dict[str, Any]], None]  # composition wires RiskManager.on_fill
ResyncHook = Callable[[], Awaitable[None]]  # positions/account -> tracker.sync_from_api
AlertHook = Callable[[str], None]  # composition wires monitoring's dispatch_alert


class StreamManager:
    def __init__(
        self,
        api_key: str,
        secret_key: str,
        paper: bool,
        redis_client: Any,
        on_fill: FillHandler,
        resync: ResyncHook,
        alert: AlertHook,
        config: dict[str, Any],
    ) -> None:
        stream_cfg = config["stream"]
        self._backoff_initial = float(stream_cfg["reconnect_backoff_initial_s"])
        self._backoff_cap = float(stream_cfg["reconnect_backoff_cap_s"])
        self._api_key = api_key
        self._secret_key = secret_key
        self._paper = paper
        self._redis = redis_client
        self._on_fill = on_fill
        self._resync = resync
        self._alert = alert
        self._started = False

    async def start(self, symbols: list[str]) -> None:
        if self._started:
            raise RuntimeError(
                "StreamManager already started — 1 WebSocket per Alpaca endpoint; "
                "a second connection silently kills the first"
            )
        self._started = True
        trading = TradingStream(self._api_key, self._secret_key, paper=self._paper)
        trading.subscribe_trade_updates(self._on_trade_update)
        data = StockDataStream(self._api_key, self._secret_key)
        data.subscribe_bars(self._on_bar, *symbols)
        await asyncio.gather(
            self._run_with_reconnect(trading._run_forever, "trading"),
            self._run_with_reconnect(data._run_forever, "data"),
        )

    # --- reconnect loop ---------------------------------------------------------

    def _backoff_delay(self, attempt: int) -> float:
        return float(min(self._backoff_initial * 2**attempt, self._backoff_cap))

    async def _run_with_reconnect(self, runner: Callable[[], Awaitable[Any]], name: str) -> None:
        attempt = 0
        while True:
            connected_at: float | None = None
            try:
                await self._resync()  # positions BEFORE any event — every (re)connect
                connected_at = time.monotonic()
                await runner()
                log.warning("stream_ended", extra={"stream": name})
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # the reconnect loop must survive any stream/transport error class
                log.error("stream_error", extra={"stream": name, "error": str(exc)})
            if connected_at is not None and time.monotonic() - connected_at > self._backoff_cap:
                attempt = 0  # ran healthily for a while — restart backoff from initial
            delay = self._backoff_delay(attempt)
            attempt += 1
            log.warning(
                "stream_reconnect_scheduled",
                extra={"stream": name, "delay_s": delay, "attempt": attempt},
            )
            await asyncio.sleep(delay)

    # --- event handlers (fast: parse, hand off, return) ---------------------------

    async def _on_trade_update(self, data: Any) -> None:
        event = str(data.event)
        order = data.order
        side = getattr(order.side, "value", str(order.side))
        if event in ("fill", "partial_fill"):
            fill = {
                "symbol": order.symbol,
                "side": side,
                "qty": float(data.qty),
                "price": float(data.price),
            }
            self._on_fill(fill)
            if event == "fill":
                log.info("order_filled", extra=fill)
            else:
                log.warning("order_partial_fill", extra=fill)
        elif event in ("canceled", "expired"):
            log.info("order_terminal", extra={"event": event, "symbol": order.symbol})
        elif event in ("new", "pending_new"):
            log.debug("order_accepted", extra={"event": event, "symbol": order.symbol})
        elif event == "rejected":
            log.error("order_rejected_by_broker", extra={"symbol": order.symbol})
            self._alert(f"order rejected by broker: {order.symbol}")
        else:
            log.warning(
                "unhandled_trade_event",
                extra={"event": event, "symbol": getattr(order, "symbol", None)},
            )

    async def _on_bar(self, bar: Any) -> None:
        payload = json.dumps(
            {
                "symbol": bar.symbol,
                "timestamp": bar.timestamp.isoformat(),
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": int(bar.volume),
                "vwap": float(bar.vwap) if bar.vwap is not None else None,
            }
        )
        await self._redis.publish(f"channel:bars:{bar.symbol}", payload)
