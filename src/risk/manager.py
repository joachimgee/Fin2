"""RiskManager — every order passes validate() or it doesn't exist.

validate() is synchronous, < 1 ms, zero network calls, zero DB reads: it only
consults in-memory state (ExposureTracker, CircuitBreaker). When in doubt,
reject — a missed trade is recoverable, a blown account is not.

Sizing interpretation (documented deliberately): without stop-distance
modeling yet, the risk-per-trade cap bounds the NEW trade's notional at
max_risk_per_trade_pct * equity — conservative by construction. Refine when
bracket orders land. Correlations are precomputed offline and injected —
validate() never computes them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

from src.risk.circuit_breaker import CircuitBreaker
from src.risk.exposure_tracker import ExposureTracker
from src.risk.kelly import kelly_fraction
from src.strategies.base import OrderIntent

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Outcome of validate(). Execution MUST use adjusted_qty, never intent.qty —
    steps 3 and 4 of the pipeline may have reduced the quantity."""

    approved: bool
    adjusted_qty: float
    reason: str  # "ok" or the first failed check's name


class CorrelationProvider(Protocol):
    """Precomputed pairwise correlations; None means unknown (treated as 0)."""

    def get(self, symbol_a: str, symbol_b: str) -> float | None: ...


def _reject(reason: str, intent: OrderIntent) -> ValidationResult:
    log.info(
        "order_rejected",
        extra={"reason": reason, "symbol": intent.symbol, "strategy_id": intent.strategy_id},
    )
    return ValidationResult(False, 0.0, reason)


class RiskManager:
    def __init__(
        self,
        config: dict[str, Any],
        tracker: ExposureTracker,
        breaker: CircuitBreaker,
        strategy_stats: dict[str, float],
        correlation_provider: CorrelationProvider | None = None,
        sector_map: dict[str, str] | None = None,
    ) -> None:
        risk_cfg = config["risk"]
        self._kelly_frac = float(risk_cfg["kelly_fraction"])
        self._per_trade_cap = float(risk_cfg["max_risk_per_trade_pct"])
        self._position_cap = float(risk_cfg["max_position_pct"])
        self._total_cap = float(risk_cfg["max_total_exposure_pct"])
        self._sector_cap = float(risk_cfg["max_sector_exposure_pct"])
        self._max_corr = float(risk_cfg["max_correlation"])
        self._tracker = tracker
        self._breaker = breaker
        self._stats = strategy_stats  # win_rate/avg_win/avg_loss from WFO results
        self._corr = correlation_provider
        self._sector_map = sector_map or {}

    def validate(self, intent: OrderIntent) -> ValidationResult:
        """The 7-step pipeline, in this exact order. First failure returns."""
        if self._breaker.trading_halted:  # 1
            return _reject("circuit_breaker", intent)
        if self._is_reducing(intent):  # exits shrink risk — caps don't apply
            reduced = min(intent.qty, abs(self._tracker.position_qty(intent.symbol)))
            return ValidationResult(True, reduced, "ok")
        equity = self._tracker.equity
        notional = self._kelly_notional(intent, equity)  # 2
        if notional <= 0:
            return _reject("kelly_zero", intent)
        notional = min(notional, equity * self._per_trade_cap)  # 3 — reduce
        capacity = equity * self._position_cap - self._tracker.position_value(intent.symbol)
        notional = min(notional, capacity)  # 4 — reduce
        if notional <= 0:
            return _reject("position_cap", intent)
        if self._tracker.total_exposure() + notional > equity * self._total_cap:  # 5
            return _reject("total_exposure", intent)
        if self._breaches_sector_cap(intent.symbol, notional, equity):  # 6
            return _reject("sector_cap", intent)
        if self._breaches_correlation(intent.symbol):  # 7
            return _reject("correlation", intent)
        return ValidationResult(True, notional / intent.reference_price, "ok")

    def on_fill(self, event: dict[str, Any]) -> float:
        """Forward a fill to the tracker and breakers; return realized P&L
        (0.0 for opening fills). Called by both engines on every fill."""
        realized = self._tracker.on_fill(event)
        if realized != 0.0:
            self._breaker.on_trade_closed(realized)
        self._breaker.on_equity_update(self._tracker.equity)
        return realized

    # --- pipeline steps ---------------------------------------------------------

    def _is_reducing(self, intent: OrderIntent) -> bool:
        existing = self._tracker.position_qty(intent.symbol)
        opposite = (intent.side == "sell" and existing > 0) or (
            intent.side == "buy" and existing < 0
        )
        return opposite and intent.qty <= abs(existing)

    def _kelly_notional(self, intent: OrderIntent, equity: float) -> float:
        fraction = kelly_fraction(
            self._stats["win_rate"],
            self._stats["avg_win"],
            self._stats["avg_loss"],
            self._kelly_frac,
        )
        return equity * fraction * abs(intent.signal_strength)

    def _breaches_sector_cap(self, symbol: str, notional: float, equity: float) -> bool:
        sector = self._sector_map.get(symbol)
        if sector is None:
            return False  # unmapped symbol: no sector information to enforce
        sector_value = sum(
            value
            for sym, value in self._tracker.position_values().items()
            if self._sector_map.get(sym) == sector
        )
        return sector_value + notional > equity * self._sector_cap

    def _breaches_correlation(self, symbol: str) -> bool:
        if self._corr is None:
            return False
        for other in self._tracker.position_values():
            if other == symbol:
                continue
            corr = self._corr.get(symbol, other)
            if corr is not None and abs(corr) > self._max_corr:
                return True
        return False
