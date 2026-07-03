"""ExposureTracker — in-memory source of truth for validate().

Updated from FILL EVENTS pushed by the execution stream — never by polling
the REST API (polling lags fills; validate() would approve against stale
exposure). REST re-sync happens exactly twice: at system startup, and after
every WebSocket reconnection (sync_from_api).

Thread-safe: all public methods take the internal threading.Lock — the
trading stream callback and validate() run on different threads.

Equity between API syncs is tracked as sync equity + realized P&L since the
sync; sync_from_api replaces it with the broker's authoritative number.
"""

from __future__ import annotations

import threading
from typing import Any


class ExposureTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        # symbol -> {"qty": signed, "avg_price": entry, "last_price": mark}
        self._positions: dict[str, dict[str, float]] = {}
        self._equity = 0.0

    # --- writes ---------------------------------------------------------------

    def on_fill(self, event: dict[str, Any]) -> float:
        """Apply a fill event {symbol, side, qty, price}; return realized P&L
        (0.0 for opening/increasing fills)."""
        symbol = event["symbol"]
        qty = float(event["qty"])
        price = float(event["price"])
        signed = qty if event["side"] == "buy" else -qty
        with self._lock:
            pos = self._positions.get(symbol, {"qty": 0.0, "avg_price": 0.0, "last_price": price})
            realized = self._apply_fill(pos, signed, qty, price)
            pos["last_price"] = price
            if pos["qty"] == 0.0:
                self._positions.pop(symbol, None)
            else:
                self._positions[symbol] = pos
            self._equity += realized
            return realized

    @staticmethod
    def _apply_fill(pos: dict[str, float], signed: float, qty: float, price: float) -> float:
        old_qty = pos["qty"]
        new_qty = old_qty + signed
        if old_qty * signed >= 0:  # opening or increasing — weighted average entry
            if new_qty != 0:
                pos["avg_price"] = (abs(old_qty) * pos["avg_price"] + qty * price) / abs(new_qty)
            pos["qty"] = new_qty
            return 0.0
        closing_qty = min(qty, abs(old_qty))  # reducing, closing, or crossing zero
        direction = 1.0 if old_qty > 0 else -1.0
        realized = (price - pos["avg_price"]) * closing_qty * direction
        if abs(signed) > abs(old_qty):  # crossed zero — remainder opens at fill price
            pos["avg_price"] = price
        pos["qty"] = new_qty
        return realized

    def sync_from_api(self, positions: list[dict[str, Any]], equity: float) -> None:
        """Full state replacement — startup and post-reconnect only."""
        with self._lock:
            self._positions = {}
            for p in positions:
                qty = float(p["qty"])
                if qty == 0.0:
                    continue
                avg = float(p["avg_entry_price"])
                value = p.get("market_value")
                last = abs(float(value)) / abs(qty) if value is not None else avg
                self._positions[p["symbol"]] = {"qty": qty, "avg_price": avg, "last_price": last}
            self._equity = equity

    # --- reads for validate() — all O(positions), zero I/O ---------------------

    @property
    def equity(self) -> float:
        with self._lock:
            return self._equity

    def position_qty(self, symbol: str) -> float:
        with self._lock:
            pos = self._positions.get(symbol)
            return pos["qty"] if pos else 0.0

    def position_value(self, symbol: str) -> float:
        with self._lock:
            pos = self._positions.get(symbol)
            return abs(pos["qty"]) * pos["last_price"] if pos else 0.0

    def position_values(self) -> dict[str, float]:
        with self._lock:
            return {s: abs(p["qty"]) * p["last_price"] for s, p in self._positions.items()}

    def total_exposure(self) -> float:
        return sum(self.position_values().values())
