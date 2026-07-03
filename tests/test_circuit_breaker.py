"""Phase 3 — circuit breaker tests (docs/plan/PHASE_3_RISK.md)."""

from __future__ import annotations

import pytest
from src.risk.circuit_breaker import CircuitBreaker

_THRESHOLDS = {"daily_loss_pct": 0.03, "max_drawdown_pct": 0.15, "consecutive_losses": 5}


@pytest.fixture
def trips() -> list[tuple[str, dict[str, float]]]:
    return []


@pytest.fixture
def breaker(trips: list[tuple[str, dict[str, float]]]) -> CircuitBreaker:
    return CircuitBreaker(
        _THRESHOLDS, on_trip=lambda reason, values: trips.append((reason, values))
    )


def test_daily_loss_trips(breaker: CircuitBreaker) -> None:
    breaker.start_of_day(100_000.0)
    breaker.on_equity_update(97_100.0)  # -2.9% — still fine
    assert not breaker.trading_halted
    breaker.on_equity_update(96_900.0)  # -3.1% — trip
    assert breaker.trading_halted


def test_drawdown_trips(breaker: CircuitBreaker) -> None:
    breaker.on_equity_update(100_000.0)  # peak
    breaker.on_equity_update(85_100.0)  # -14.9% — still fine
    assert not breaker.trading_halted
    breaker.on_equity_update(84_900.0)  # -15.1% below peak — trip
    assert breaker.trading_halted


def test_five_consecutive_losses_trip(breaker: CircuitBreaker) -> None:
    for _ in range(4):
        breaker.on_trade_closed(-10.0)
    assert not breaker.trading_halted
    breaker.on_trade_closed(-10.0)
    assert breaker.trading_halted


def test_win_resets_loss_streak(breaker: CircuitBreaker) -> None:
    for _ in range(4):
        breaker.on_trade_closed(-10.0)
    breaker.on_trade_closed(5.0)  # streak broken
    for _ in range(4):
        breaker.on_trade_closed(-10.0)
    assert not breaker.trading_halted


def test_no_auto_reset_on_new_day(breaker: CircuitBreaker) -> None:
    breaker.start_of_day(100_000.0)
    breaker.on_equity_update(90_000.0)
    assert breaker.trading_halted
    breaker.start_of_day(90_000.0)  # a new session must NOT clear the halt
    breaker.on_equity_update(95_000.0)  # nor a recovery
    assert breaker.trading_halted


def test_manual_reset_restores(breaker: CircuitBreaker) -> None:
    for _ in range(5):
        breaker.on_trade_closed(-10.0)
    assert breaker.trading_halted
    breaker.reset_circuit_breaker()
    assert not breaker.trading_halted
    breaker.on_trade_closed(-10.0)  # streak was cleared too — one loss is fine
    assert not breaker.trading_halted


def test_trip_calls_injected_callback(
    breaker: CircuitBreaker, trips: list[tuple[str, dict[str, float]]]
) -> None:
    for _ in range(5):
        breaker.on_trade_closed(-10.0)
    assert [reason for reason, _ in trips] == ["consecutive_losses"]
    assert trips[0][1]["loss_streak"] == 5.0
