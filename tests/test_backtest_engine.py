"""Phase 6 — backtest engine tests (docs/plan/PHASE_6_BACKTEST_WFO.md).

Hand-computed scenario (frictionless, capital 10k):
  t0 o100 c100 | t1 o100 c95 (buy signal) | t2 o96 c100 (fill @96)
  t3 o101 c106 (sell signal)              | t4 o107 c107 (fill @107)
  cash: 10000 -> 9040 (buy 10@96) -> 10110 (sell 10@107); realized = +110
  equity: [10000, 10000, 9040+10*100=10040, 9040+10*106=10100, 10110]
"""

from __future__ import annotations

import copy
from typing import Any

import pytest
from src.backtest.engine import BacktestEngine, SimulatedBroker

from tests._helpers import (
    BuyLowSellHigh,
    HaltedRisk,
    NullGen,
    PassThroughRisk,
    bars_frame,
    frictionless,
)

_PRICES = [(100.0, 100.0), (100.0, 95.0), (96.0, 100.0), (101.0, 106.0), (107.0, 107.0)]


def _engine(config: dict[str, Any], risk: PassThroughRisk) -> tuple[BacktestEngine, BuyLowSellHigh]:
    strategy = BuyLowSellHigh(config, NullGen())
    broker = SimulatedBroker(config)
    engine = BacktestEngine(strategy, risk, risk.tracker, broker, bars_frame(_PRICES), config)  # type: ignore[arg-type]
    return engine, strategy


async def test_fill_at_next_bar_open_not_signal_bar(base_config: dict[str, Any]) -> None:
    engine, strategy = _engine(frictionless(base_config), PassThroughRisk())
    await engine.run()
    # signal came at t1 (close 95); the fill must be at t2's OPEN (96), never 95
    assert strategy.fills_seen[0]["price"] == pytest.approx(96.0)


async def test_equity_curve_hand_computed(base_config: dict[str, Any]) -> None:
    engine, _ = _engine(frictionless(base_config), PassThroughRisk())
    results = await engine.run()
    assert list(results["equity_curve"]) == pytest.approx([10000, 10000, 10040, 10100, 10110])
    assert results["n_trades"] == 1
    assert results["profit_factor"] == float("inf")  # single winning round-trip (+110)
    assert results["total_return"] == pytest.approx(0.011)


async def test_slippage_and_commission_applied(base_config: dict[str, Any]) -> None:
    config = copy.deepcopy(base_config)
    config["backtest"].update(
        {"initial_capital": 10_000, "slippage_bps": 100, "commission_per_share": 1.0}
    )
    engine, strategy = _engine(config, PassThroughRisk())
    results = await engine.run()
    assert strategy.fills_seen[0]["price"] == pytest.approx(96.0 * 1.01)  # buy pays up
    assert strategy.fills_seen[1]["price"] == pytest.approx(107.0 * 0.99)  # sell receives less
    assert results["total_commission"] == pytest.approx(20.0)  # 10 shares x $1, both legs


async def test_win_loss_tallies_hand_computed(base_config: dict[str, Any]) -> None:
    """Round-trip 1: buy@96 sell@107 -> +110 (win). Round-trip 2: buy@98,
    sell signal at c106 but next open gaps to 92 -> (92-98)*10 = -60 (loss)."""
    prices = [
        (100.0, 100.0),
        (100.0, 95.0),  # buy signal
        (96.0, 100.0),  # fill @96
        (101.0, 106.0),  # sell signal
        (107.0, 107.0),  # fill @107 -> +110
        (100.0, 95.0),  # buy signal
        (98.0, 100.0),  # fill @98
        (101.0, 106.0),  # sell signal
        (92.0, 92.0),  # fill @92 -> -60
    ]
    config = frictionless(base_config)
    risk = PassThroughRisk()
    strategy = BuyLowSellHigh(config, NullGen())
    engine = BacktestEngine(
        strategy, risk, risk.tracker, SimulatedBroker(config), bars_frame(prices), config
    )  # type: ignore[arg-type]
    results = await engine.run()
    assert (results["n_wins"], results["n_losses"]) == (1, 1)
    assert results["gross_win"] == pytest.approx(110.0)
    assert results["gross_loss"] == pytest.approx(60.0)


async def test_rejected_intent_never_fills(base_config: dict[str, Any]) -> None:
    engine, strategy = _engine(frictionless(base_config), HaltedRisk())
    results = await engine.run()
    assert strategy.fills_seen == []
    assert results["orders"] == []
    assert list(results["equity_curve"]) == pytest.approx([10000.0] * 5)


async def test_fill_event_shape_matches_live_contract(base_config: dict[str, Any]) -> None:
    engine, strategy = _engine(frictionless(base_config), PassThroughRisk())
    await engine.run()
    # exact keys the live TradingStream handler produces — strategies can't
    # tell the engines apart
    assert all(set(f) == {"symbol", "side", "qty", "price"} for f in strategy.fills_seen)


async def test_trade_start_lead_in_warms_up_without_trading(base_config: dict[str, Any]) -> None:
    """Bars before trade_start feed the strategy but can never trade, and are
    excluded from the equity curve — the WFO warmup lead-in contract."""
    config = frictionless(base_config)
    strategy = BuyLowSellHigh(config, NullGen())
    risk = PassThroughRisk()
    bars = bars_frame(_PRICES)
    trade_start = bars["timestamp"].iloc[3]  # t1's buy signal falls in the lead-in
    engine = BacktestEngine(
        strategy,
        risk,  # type: ignore[arg-type]
        risk.tracker,
        SimulatedBroker(config),
        bars,
        config,
        trade_start=trade_start,
    )
    results = await engine.run()
    assert strategy.fills_seen == []  # the lead-in signal was discarded, never filled
    assert len(results["equity_curve"]) == 2  # only the evaluation span is measured
    assert list(results["equity_curve"]) == pytest.approx([10000.0, 10000.0])


async def test_same_strategy_instance_reusable_after_reset(base_config: dict[str, Any]) -> None:
    config = frictionless(base_config)
    strategy = BuyLowSellHigh(config, NullGen())
    first_risk = PassThroughRisk()
    engine_one = BacktestEngine(
        strategy,
        first_risk,
        first_risk.tracker,
        SimulatedBroker(config),
        bars_frame(_PRICES),
        config,  # type: ignore[arg-type]
    )
    first = await engine_one.run()
    strategy.reset()
    second_risk = PassThroughRisk()
    engine_two = BacktestEngine(
        strategy,
        second_risk,
        second_risk.tracker,
        SimulatedBroker(config),
        bars_frame(_PRICES),
        config,  # type: ignore[arg-type]
    )
    second = await engine_two.run()
    assert list(first["equity_curve"]) == pytest.approx(list(second["equity_curve"]))
