"""Event-driven backtest engine — replays DuckDB bars through the SAME
strategy.on_bar() -> risk.validate() -> broker.submit_order() path as live.

The broker here is SimulatedBroker(AbstractBrokerClient): fills modeled with
spread + commission + slippage. Because both engines run identical strategy
and risk code, backtest/live parity is structural, not aspirational.
"""

from __future__ import annotations

from typing import Any

from src.shared.interfaces import AbstractBrokerClient


class SimulatedBroker(AbstractBrokerClient):
    """TODO(Phase 6): fill simulation with configurable slippage/commission model."""

    async def submit_order(self, request: Any) -> str:
        raise NotImplementedError("Phase 6 — backtest engine")

    async def cancel_order(self, order_id: str) -> None:
        raise NotImplementedError("Phase 6")

    async def get_positions(self) -> list[dict[str, Any]]:
        raise NotImplementedError("Phase 6")

    async def get_account(self) -> dict[str, Any]:
        raise NotImplementedError("Phase 6")


class BacktestEngine:
    """TODO(Phase 6): bar replay loop, equity curve, trade ledger, metrics
    (Sharpe, max drawdown, profit factor) — computed on OOS segments only."""

    def run(self) -> dict[str, Any]:
        raise NotImplementedError("Phase 6")
