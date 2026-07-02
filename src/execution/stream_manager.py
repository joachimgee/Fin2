"""StreamManager — owns the single WebSocket connection per Alpaca endpoint.

Alpaca allows ONE WebSocket per endpoint per account: a second connection
silently kills the first. This process is therefore the sole consumer and
fans events out over Redis pub/sub (channel:bars:{symbol}, channel:trade_updates)
so multiple strategies can subscribe without touching Alpaca.

Reconnect: exponential backoff 1s -> 2s -> 4s ... capped at 60s. Never instant
(rate-limit ban during Alpaca outages). After EVERY reconnection, before
accepting any signal: get_positions + get_account -> tracker.sync_from_api().
"""

from __future__ import annotations


class StreamManager:
    """TODO(Phase 4):
    - run TradingStream (trade_updates) + StockDataStream (bars/quotes).
    - handle every TradingStream event explicitly, no silent catch-all:
        "fill"               -> tracker.on_fill + log order_filled
        "partial_fill"       -> partial update + log WARNING
        "canceled"|"expired" -> log INFO
        "new"|"pending_new"  -> log DEBUG
        "rejected"           -> log ERROR + Telegram alert
        _                    -> log WARNING unhandled_trade_event
    - callbacks stay < 5 ms: publish to Redis and return, no blocking logic.
    """

    async def start(self, symbols: list[str]) -> None:
        raise NotImplementedError("Phase 4 — stream manager")
