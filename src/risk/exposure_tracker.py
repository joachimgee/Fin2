"""ExposureTracker — in-memory source of truth for validate().

Updated from FILL EVENTS pushed by the execution stream — never by polling
the REST API (polling lags fills; validate() would approve against stale
exposure). REST re-sync happens exactly twice: at system startup, and after
every WebSocket reconnection (sync_from_api).

Thread-safe: all public methods take the internal threading.Lock — the
trading stream callback and validate() run on different threads.
"""

from __future__ import annotations

from typing import Any


class ExposureTracker:
    """TODO(Phase 3):
    - on_fill(event): update position qty/avg price, realized P&L.
    - sync_from_api(positions, equity): full state replace (startup/reconnect).
    - Read API for validate(): position_value(symbol), total_exposure(),
      sector_exposure(sector), equity — all lock-protected, all < 1 ms.
    """

    def on_fill(self, event: dict[str, Any]) -> None:
        raise NotImplementedError("Phase 3 — exposure tracking")

    def sync_from_api(self, positions: list[dict[str, Any]], equity: float) -> None:
        raise NotImplementedError("Phase 3")
