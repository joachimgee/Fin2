"""Event-driven backtest engine — replays bars through the SAME
strategy.on_bar() -> risk.validate() -> broker.submit_order() path as live.

SimulatedBroker implements shared.interfaces.AbstractBrokerClient (ADR-001):
backtest/ never imports execution/. Orders fill at the NEXT bar's open —
same-bar fills are execution-side lookahead — with slippage and commission
from config. Fill events use the exact dict shape the live TradingStream
produces, so strategies and risk cannot tell the two engines apart.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.backtest.metrics import max_drawdown, profit_factor, sharpe_ratio
from src.data.models import Bar
from src.risk.exposure_tracker import ExposureTracker
from src.risk.manager import RiskManager
from src.shared.interfaces import AbstractBrokerClient
from src.strategies.base import AbstractStrategy

log = logging.getLogger(__name__)


class SimulatedBroker(AbstractBrokerClient):
    def __init__(self, config: dict[str, Any]) -> None:
        backtest_cfg = config["backtest"]
        self._slippage = float(backtest_cfg["slippage_bps"]) / 10_000.0
        self._commission_per_share = float(backtest_cfg["commission_per_share"])
        self._cash = float(backtest_cfg["initial_capital"])
        self._pending: list[dict[str, Any]] = []
        self._positions: dict[str, dict[str, float]] = {}  # {"qty", "avg_price"}
        self._last_price: dict[str, float] = {}
        self._next_id = 0
        self.total_commission = 0.0

    # --- AbstractBrokerClient ---------------------------------------------------

    async def submit_order(self, request: Any) -> str:
        self._next_id += 1
        order_id = f"sim-{self._next_id}"
        self._pending.append({"order_id": order_id, **request})
        return order_id

    async def cancel_order(self, order_id: str) -> None:
        self._pending = [o for o in self._pending if o["order_id"] != order_id]

    async def get_positions(self) -> list[dict[str, Any]]:
        return [
            {
                "symbol": symbol,
                "qty": pos["qty"],
                "avg_entry_price": pos["avg_price"],
                "market_value": pos["qty"] * self._last_price.get(symbol, pos["avg_price"]),
            }
            for symbol, pos in self._positions.items()
            if pos["qty"] != 0.0
        ]

    async def get_account(self) -> dict[str, Any]:
        return {"equity": self.equity(), "cash": self._cash}

    # --- simulation drivers (called by the engine only) ---------------------------

    def fill_at_open(self, bar: Bar) -> list[dict[str, Any]]:
        """Fill pending orders for this symbol at THIS bar's open (orders were
        submitted on a PREVIOUS bar — next-bar-open semantics, no lookahead)."""
        fills: list[dict[str, Any]] = []
        remaining: list[dict[str, Any]] = []
        for order in self._pending:
            if order["symbol"] != bar.symbol:
                remaining.append(order)
                continue
            side, qty = str(order["side"]), float(order["qty"])
            slip = 1.0 + self._slippage if side == "buy" else 1.0 - self._slippage
            price = bar.open * slip
            self._settle(bar.symbol, side, qty, price)
            # exact live TradingStream fill shape — parity is structural
            fills.append({"symbol": bar.symbol, "side": side, "qty": qty, "price": price})
        self._pending = remaining
        return fills

    def _settle(self, symbol: str, side: str, qty: float, price: float) -> None:
        commission = qty * self._commission_per_share
        self.total_commission += commission
        signed = qty if side == "buy" else -qty
        self._cash -= signed * price + commission
        pos = self._positions.setdefault(symbol, {"qty": 0.0, "avg_price": 0.0})
        new_qty = pos["qty"] + signed
        if pos["qty"] * signed >= 0 and new_qty != 0:  # opening/increasing
            pos["avg_price"] = (abs(pos["qty"]) * pos["avg_price"] + qty * price) / abs(new_qty)
        pos["qty"] = new_qty

    def mark(self, bar: Bar) -> None:
        self._last_price[bar.symbol] = bar.close

    def equity(self) -> float:
        held = sum(
            pos["qty"] * self._last_price.get(symbol, pos["avg_price"])
            for symbol, pos in self._positions.items()
        )
        return self._cash + held


class BacktestEngine:
    """Drives bars chronologically through the identical live sequence:
    sync -> fills -> on_trade_update/on_fill -> on_bar -> validate -> submit."""

    def __init__(
        self,
        strategy: AbstractStrategy,
        risk_manager: RiskManager,
        tracker: ExposureTracker,
        broker: SimulatedBroker,
        bars: pd.DataFrame,  # columns: symbol, timestamp, open, high, low, close, volume
        config: dict[str, Any],
        trade_start: Any | None = None,
    ) -> None:
        """trade_start: bars before this timestamp are a WARMUP LEAD-IN — the
        strategy sees them (indicator buffers fill) but every intent is
        discarded and they are excluded from the equity curve/metrics. Lets
        WFO evaluate windows shorter than the strategy warmup using only
        PAST bars (never future ones)."""
        self._strategy = strategy
        self._risk = risk_manager
        self._tracker = tracker
        self._broker = broker
        self._bars = bars.sort_values("timestamp")
        self._periods_per_year = int(config["backtest"]["periods_per_year"])
        self._trade_start = trade_start

    async def run(self) -> dict[str, Any]:
        # startup sync — same sequence as the live entrypoint
        self._tracker.sync_from_api(
            await self._broker.get_positions(), (await self._broker.get_account())["equity"]
        )
        equity_curve: list[float] = []
        timestamps: list[Any] = []
        pnls: list[float] = []
        orders: list[dict[str, Any]] = []
        current_day: Any = None
        for row in self._bars.itertuples():
            bar = Bar(
                symbol=str(row.symbol),
                timestamp=row.timestamp,
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                volume=int(row.volume),
            )
            if current_day != bar.timestamp:  # same live sequence: day boundary first
                current_day = bar.timestamp
                self._risk.on_new_day(self._broker.equity())
            in_lead_in = self._trade_start is not None and bar.timestamp < self._trade_start
            for fill in self._broker.fill_at_open(bar):
                self._strategy.on_trade_update(fill)
                realized = self._risk.on_fill(fill)
                if realized != 0.0:
                    pnls.append(realized)
            intent = self._strategy.on_bar(bar)
            if in_lead_in:
                intent = None  # warmup only — no trading before the evaluation span
            if intent is not None:
                result = self._risk.validate(intent)  # the mandatory gate — as live
                if result.approved:
                    order = {
                        "symbol": intent.symbol,
                        "side": intent.side,
                        "qty": result.adjusted_qty,  # never intent.qty
                    }
                    await self._broker.submit_order(order)
                    orders.append({"timestamp": bar.timestamp, **order})
            self._broker.mark(bar)
            if not in_lead_in:  # metrics cover the evaluation span only
                # ONE equity point per timestamp (multi-symbol frames carry N
                # rows per day — per-row points would distort Sharpe scaling)
                equity = self._broker.equity()
                if timestamps and timestamps[-1] == bar.timestamp:
                    equity_curve[-1] = equity
                else:
                    equity_curve.append(equity)
                    timestamps.append(bar.timestamp)
        return self._results(pd.Series(equity_curve, index=timestamps), pnls, orders)

    def _results(
        self, equity: pd.Series, pnls: list[float], orders: list[dict[str, Any]]
    ) -> dict[str, Any]:
        returns = equity.pct_change()
        total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0) if len(equity) else 0.0
        wins = [p for p in pnls if p > 0.0]
        losses = [p for p in pnls if p < 0.0]
        results = {
            "sharpe": sharpe_ratio(returns, self._periods_per_year),
            "max_drawdown": max_drawdown(equity),
            "profit_factor": profit_factor(pnls),
            "n_trades": len(pnls),
            # raw win/loss tallies — WFO aggregates them across OOS windows
            # into the Kelly inputs (strategy.stats: win_rate/avg_win/avg_loss)
            "n_wins": len(wins),
            "n_losses": len(losses),
            "gross_win": float(sum(wins)),
            "gross_loss": float(abs(sum(losses))),
            "total_return": total_return,
            "total_commission": self._broker.total_commission,
            "equity_curve": equity,
            "orders": orders,
        }
        log.info(
            "backtest_completed",
            extra={k: v for k, v in results.items() if k not in ("equity_curve", "orders")},
        )
        return results
