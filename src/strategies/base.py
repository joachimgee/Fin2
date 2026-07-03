"""AbstractStrategy — the contract both engines (backtest and live) run unchanged.

Interface is intentionally minimal (Interface Segregation, root CLAUDE.md):
on_bar(), on_trade_update(), is_ready, universe, reset(). Nothing else.
Strategies never talk to a broker: they emit OrderIntent objects; the engine
routes them through RiskManager.validate() and then to the broker client.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Literal

from src.data.models import Bar
from src.shared.interfaces import SignalGenerator


@dataclass(frozen=True, slots=True)
class OrderIntent:
    """What a strategy WANTS to do. Not an order — becomes one only after
    RiskManager.validate() approves (and possibly reduces) it.

    Defined here because strategies produce it; risk/ imports it from here
    (the one sanctioned risk -> strategies/base dependency).
    """

    symbol: str
    side: Literal["buy", "sell"]
    qty: float
    signal_strength: float  # [-1.0, 1.0], from a SignalGenerator
    strategy_id: str
    reference_price: float  # close that generated the signal — risk sizes off it
    order_type: Literal["market", "limit"] = "market"
    limit_price: float | None = None


class AbstractStrategy(abc.ABC):
    """Liskov contract: any subclass drops into either engine unchanged.

    Signal generators arrive via constructor injection typed against the
    shared SignalGenerator Protocol — strategies never import src/signals/.
    """

    def __init__(self, config: dict[str, Any], signal_generator: SignalGenerator) -> None:
        self._config = config
        self._signal_generator = signal_generator

    @property
    @abc.abstractmethod
    def universe(self) -> list[str]:
        """Symbols this strategy trades (from YAML config, never hardcoded)."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def is_ready(self) -> bool:
        """False until enough bars accumulated for feature computation (warmup)."""
        raise NotImplementedError

    @abc.abstractmethod
    def on_bar(self, bar: Bar) -> OrderIntent | None:
        """Process one bar; return an OrderIntent or None.

        Must be pure computation: no I/O, no network, deterministic given state.
        Identical code path in backtest and live — that is the whole point.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def on_trade_update(self, update: dict[str, Any]) -> None:
        """React to fill/cancel/reject events (update internal position state)."""
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self) -> None:
        """Clear all internal state (fresh backtest window / restart)."""
        raise NotImplementedError
