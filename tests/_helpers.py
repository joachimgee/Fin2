"""Shared test doubles for the backtest/parity test files (not collected)."""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

import pandas as pd
from src.risk.exposure_tracker import ExposureTracker
from src.strategies.base import AbstractStrategy, OrderIntent


class BuyLowSellHigh(AbstractStrategy):
    """Deterministic price-rule strategy: buy 10 when close <= 95, exit when
    close >= 105. Exists to drive engines — no features, no ML."""

    def __init__(self, config: dict[str, Any], signal_generator: Any) -> None:
        super().__init__(config, signal_generator)
        self._pos = 0.0
        self.fills_seen: list[dict[str, Any]] = []

    @property
    def universe(self) -> list[str]:
        return ["SPY"]

    @property
    def is_ready(self) -> bool:
        return True

    def on_bar(self, bar: Any) -> OrderIntent | None:
        if self._pos == 0.0 and bar.close <= 95:
            return OrderIntent("SPY", "buy", 10.0, 1.0, "test", bar.close)
        if self._pos > 0.0 and bar.close >= 105:
            return OrderIntent("SPY", "sell", self._pos, -1.0, "test", bar.close)
        return None

    def on_trade_update(self, update: dict[str, Any]) -> None:
        self.fills_seen.append(update)
        signed = float(update["qty"]) if update["side"] == "buy" else -float(update["qty"])
        self._pos += signed

    def reset(self) -> None:
        self._pos = 0.0
        self.fills_seen = []


class NullGen:
    def generate(self, features: Any) -> float:
        return 0.0


class PassThroughRisk:
    """Approves intents as-asked; forwards fills to a real tracker."""

    def __init__(self) -> None:
        self.tracker = ExposureTracker()

    def validate(self, intent: OrderIntent) -> Any:
        return SimpleNamespace(approved=True, adjusted_qty=intent.qty, reason="ok")

    def on_fill(self, fill: dict[str, Any]) -> float:
        return self.tracker.on_fill(fill)


class HaltedRisk(PassThroughRisk):
    def validate(self, intent: OrderIntent) -> Any:
        return SimpleNamespace(approved=False, adjusted_qty=0.0, reason="circuit_breaker")


def bars_frame(prices: list[tuple[float, float]]) -> pd.DataFrame:
    """(open, close) pairs -> engine-shaped bars DataFrame for SPY."""
    timestamps = pd.date_range("2024-01-01", periods=len(prices), freq="B", tz="UTC")
    return pd.DataFrame(
        [
            {
                "symbol": "SPY",
                "timestamp": timestamps[i],
                "open": open_,
                "high": max(open_, close) + 1.0,
                "low": min(open_, close) - 1.0,
                "close": close,
                "volume": 1_000,
            }
            for i, (open_, close) in enumerate(prices)
        ]
    )


def frictionless(base_config: dict[str, Any], capital: float = 10_000.0) -> dict[str, Any]:
    """Config copy with zero slippage/commission for hand-computed math."""
    config = copy.deepcopy(base_config)
    config["backtest"].update(
        {"initial_capital": capital, "slippage_bps": 0, "commission_per_share": 0}
    )
    return config
