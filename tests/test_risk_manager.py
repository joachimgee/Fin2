"""Phase 3 — validate() pipeline tests (docs/plan/PHASE_3_RISK.md).

Standard setup: equity 100k, stats win_rate .6 / payoff 2 -> quarter-Kelly
fraction .1, so a full-strength signal targets 10k notional; the 2%
risk-per-trade cap then reduces it to 2k (20 shares at 100).
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest
from src.risk.circuit_breaker import CircuitBreaker
from src.risk.exposure_tracker import ExposureTracker
from src.risk.manager import RiskManager
from src.strategies.base import OrderIntent

_STATS = {"win_rate": 0.6, "avg_win": 2.0, "avg_loss": 1.0}


class _FixedCorr:
    def __init__(self, value: float | None) -> None:
        self._value = value

    def get(self, symbol_a: str, symbol_b: str) -> float | None:
        return self._value


def _intent(symbol: str = "AAPL", side: str = "buy", strength: float = 1.0) -> OrderIntent:
    return OrderIntent(
        symbol=symbol,
        side=side,  # type: ignore[arg-type]
        qty=50.0,
        signal_strength=strength,
        strategy_id="test",
        reference_price=100.0,
    )


def _manager(
    base_config: dict[str, Any],
    positions: list[dict[str, Any]] | None = None,
    equity: float = 100_000.0,
    stats: dict[str, float] = _STATS,
    corr: _FixedCorr | None = None,
    sector_map: dict[str, str] | None = None,
    halted: bool = False,
) -> RiskManager:
    tracker = ExposureTracker()
    tracker.sync_from_api(positions or [], equity)
    breaker = CircuitBreaker(base_config["risk"]["circuit_breakers"], on_trip=lambda r, v: None)
    if halted:
        breaker.on_equity_update(equity * 2)  # set a high peak
        breaker.on_equity_update(equity)  # -50% -> trip max_drawdown
    return RiskManager(base_config, tracker, breaker, stats, corr, sector_map)


def test_halted_rejected_before_anything_else(base_config: dict[str, Any]) -> None:
    # stats would also fail (kelly_zero) — the reason must still be the breaker
    manager = _manager(base_config, halted=True, stats={"win_rate": 0, "avg_win": 0, "avg_loss": 0})
    result = manager.validate(_intent())
    assert not result.approved
    assert result.reason == "circuit_breaker"


def test_kelly_zero_rejects(base_config: dict[str, Any]) -> None:
    result = _manager(base_config, stats={"win_rate": 0.3, "avg_win": 1, "avg_loss": 1}).validate(
        _intent()
    )
    assert not result.approved
    assert result.reason == "kelly_zero"


def test_caps_reduce_not_reject(base_config: dict[str, Any]) -> None:
    result = _manager(base_config).validate(_intent())
    assert result.approved
    # kelly wanted 10k (100 shares); 2% per-trade cap reduced to 2k = 20 shares
    assert result.adjusted_qty == pytest.approx(20.0)


def test_result_carries_adjusted_qty_not_intent_qty(base_config: dict[str, Any]) -> None:
    intent = _intent()
    result = _manager(base_config).validate(intent)
    assert result.adjusted_qty != intent.qty


def test_position_cap_zero_capacity_rejects(base_config: dict[str, Any]) -> None:
    positions = [{"symbol": "AAPL", "qty": 100, "avg_entry_price": 100.0}]  # already 10%
    result = _manager(base_config, positions=positions).validate(_intent())
    assert not result.approved
    assert result.reason == "position_cap"


def test_exposure_cap_rejects(base_config: dict[str, Any]) -> None:
    positions = [{"symbol": "XLE", "qty": 790, "avg_entry_price": 100.0}]  # 79% exposure
    result = _manager(base_config, positions=positions).validate(_intent())
    assert not result.approved
    assert result.reason == "total_exposure"  # 79k + 2k > 80k — reject, not reduce


def test_sector_cap_rejects(base_config: dict[str, Any]) -> None:
    positions = [{"symbol": "MSFT", "qty": 240, "avg_entry_price": 100.0}]  # 24k tech
    result = _manager(
        base_config, positions=positions, sector_map={"AAPL": "tech", "MSFT": "tech"}
    ).validate(_intent())
    assert not result.approved
    assert result.reason == "sector_cap"  # 24k + 2k > 25k


def test_correlation_above_threshold_rejects(base_config: dict[str, Any]) -> None:
    positions = [{"symbol": "MSFT", "qty": 50, "avg_entry_price": 100.0}]
    result = _manager(base_config, positions=positions, corr=_FixedCorr(0.9)).validate(_intent())
    assert not result.approved
    assert result.reason == "correlation"


def test_unknown_correlation_passes(base_config: dict[str, Any]) -> None:
    positions = [{"symbol": "MSFT", "qty": 50, "avg_entry_price": 100.0}]
    result = _manager(base_config, positions=positions, corr=_FixedCorr(None)).validate(_intent())
    assert result.approved


def test_reducing_order_bypasses_caps(base_config: dict[str, Any]) -> None:
    """A sell that shrinks an over-exposed book must NOT be blocked by caps."""
    positions = [{"symbol": "AAPL", "qty": 850, "avg_entry_price": 100.0}]  # 85% > all caps
    result = _manager(base_config, positions=positions).validate(_intent(side="sell"))
    assert result.approved
    assert result.adjusted_qty == pytest.approx(50.0)  # the requested exit size


def test_validate_is_synchronous(base_config: dict[str, Any]) -> None:
    assert not inspect.iscoroutinefunction(RiskManager.validate)


def test_on_fill_feeds_breaker_losses(base_config: dict[str, Any]) -> None:
    manager = _manager(base_config)
    for _ in range(5):  # five losing round-trips -> consecutive_losses breaker
        manager.on_fill({"symbol": "T", "side": "buy", "qty": 1, "price": 100.0})
        manager.on_fill({"symbol": "T", "side": "sell", "qty": 1, "price": 99.0})
    assert not manager.validate(_intent()).approved
