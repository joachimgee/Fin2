"""Phase 3 — exposure tracker tests (docs/plan/PHASE_3_RISK.md)."""

from __future__ import annotations

import threading

import pytest
from src.risk.exposure_tracker import ExposureTracker


def _fill(symbol: str = "AAPL", side: str = "buy", qty: float = 10, price: float = 100.0) -> dict:
    return {"symbol": symbol, "side": side, "qty": qty, "price": price}


def test_on_fill_updates_position() -> None:
    tracker = ExposureTracker()
    tracker.on_fill(_fill(qty=10, price=100.0))
    tracker.on_fill(_fill(qty=10, price=110.0))  # increase -> weighted avg 105
    assert tracker.position_qty("AAPL") == 20
    assert tracker.position_value("AAPL") == pytest.approx(20 * 110.0)  # marked at last fill


def test_reducing_fill_realizes_pnl() -> None:
    tracker = ExposureTracker()
    tracker.on_fill(_fill(qty=10, price=100.0))
    realized = tracker.on_fill(_fill(side="sell", qty=4, price=110.0))
    assert realized == pytest.approx(4 * 10.0)  # (110-100) * 4
    assert tracker.position_qty("AAPL") == 6
    assert tracker.equity == pytest.approx(40.0)  # realized pnl accrues to equity


def test_closing_fill_removes_position() -> None:
    tracker = ExposureTracker()
    tracker.on_fill(_fill(qty=10, price=100.0))
    tracker.on_fill(_fill(side="sell", qty=10, price=90.0))
    assert tracker.position_qty("AAPL") == 0
    assert tracker.total_exposure() == 0.0


def test_sync_replaces_all_state() -> None:
    tracker = ExposureTracker()
    tracker.on_fill(_fill(symbol="TSLA", qty=5, price=200.0))
    tracker.sync_from_api(
        [{"symbol": "AAPL", "qty": 10, "avg_entry_price": 150.0, "market_value": 1600.0}],
        equity=50_000.0,
    )
    assert tracker.position_qty("TSLA") == 0  # gone — full replacement
    assert tracker.position_value("AAPL") == pytest.approx(1600.0)
    assert tracker.equity == 50_000.0


def test_concurrent_fills_consistent() -> None:
    tracker = ExposureTracker()
    n_threads, n_fills = 8, 200

    def hammer() -> None:
        for _ in range(n_fills):
            tracker.on_fill(_fill(qty=1, price=100.0))

    threads = [threading.Thread(target=hammer) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert tracker.position_qty("AAPL") == n_threads * n_fills
