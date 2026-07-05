"""Phase 6 — the Liskov proof (docs/plan/PHASE_6_BACKTEST_WFO.md).

One strategy class, the FULL real risk stack, two drivers:
  A) BacktestEngine
  B) an inline loop implementing the documented live sequence
     (fills -> on_trade_update/on_fill -> on_bar -> validate -> submit)
Identical orders and identical equity curves, or the parity claim is false.
"""

from __future__ import annotations

from typing import Any

import pytest
from src.backtest.engine import BacktestEngine, SimulatedBroker
from src.data.models import Bar
from src.risk.circuit_breaker import CircuitBreaker
from src.risk.exposure_tracker import ExposureTracker
from src.risk.manager import RiskManager

from tests._helpers import BuyLowSellHigh, NullGen, bars_frame, frictionless

_PRICES = [(100.0, 100.0), (100.0, 95.0), (96.0, 100.0), (101.0, 106.0), (107.0, 107.0)]
_STATS = {"win_rate": 0.6, "avg_win": 2.0, "avg_loss": 1.0}


def _real_stack(
    config: dict[str, Any],
) -> tuple[BuyLowSellHigh, RiskManager, ExposureTracker, SimulatedBroker]:
    tracker = ExposureTracker()
    breaker = CircuitBreaker(config["risk"]["circuit_breakers"], on_trip=lambda r, v: None)
    risk = RiskManager(config, tracker, breaker, _STATS)
    return BuyLowSellHigh(config, NullGen()), risk, tracker, SimulatedBroker(config)


async def test_liskov_same_orders_backtest_vs_live_loop(base_config: dict[str, Any]) -> None:
    config = frictionless(base_config, capital=100_000.0)

    # Path A — the engine
    strategy_a, risk_a, tracker_a, broker_a = _real_stack(config)
    results_a = await BacktestEngine(
        strategy_a, risk_a, tracker_a, broker_a, bars_frame(_PRICES), config
    ).run()

    # Path B — the documented live sequence, written out
    strategy_b, risk_b, tracker_b, broker_b = _real_stack(config)
    tracker_b.sync_from_api(
        await broker_b.get_positions(), (await broker_b.get_account())["equity"]
    )
    orders_b: list[dict[str, Any]] = []
    equity_b: list[float] = []
    current_day = None
    for row in bars_frame(_PRICES).itertuples():
        bar = Bar("SPY", row.timestamp, row.open, row.high, row.low, row.close, int(row.volume))
        if current_day != bar.timestamp:  # day boundary first — as the engine does
            current_day = bar.timestamp
            risk_b.on_new_day(broker_b.equity())
        for fill in broker_b.fill_at_open(bar):
            strategy_b.on_trade_update(fill)
            risk_b.on_fill(fill)
        intent = strategy_b.on_bar(bar)
        if intent is not None:
            result = risk_b.validate(intent)
            if result.approved:
                order = {"symbol": intent.symbol, "side": intent.side, "qty": result.adjusted_qty}
                await broker_b.submit_order(order)
                orders_b.append({"timestamp": bar.timestamp, **order})
        broker_b.mark(bar)
        equity_b.append(broker_b.equity())

    assert len(results_a["orders"]) == len(orders_b) > 0  # the test actually traded
    for order_a, order_b in zip(results_a["orders"], orders_b, strict=True):
        assert order_a["timestamp"] == order_b["timestamp"]
        assert order_a["side"] == order_b["side"]
        assert order_a["qty"] == pytest.approx(order_b["qty"])
    assert list(results_a["equity_curve"]) == pytest.approx(equity_b)


async def test_kelly_sizing_flows_through_both_paths(base_config: dict[str, Any]) -> None:
    """The entry qty must be risk-sized (Kelly+caps), not the strategy's 10."""
    config = frictionless(base_config, capital=100_000.0)
    strategy, risk, tracker, broker = _real_stack(config)
    results = await BacktestEngine(
        strategy, risk, tracker, broker, bars_frame(_PRICES), config
    ).run()
    entry = results["orders"][0]
    # equity 100k, quarter-Kelly .1 -> 10k, per-trade cap 2% -> 2k at ref 95
    assert entry["qty"] == pytest.approx(2_000.0 / 95.0)
    assert entry["qty"] != 10.0
