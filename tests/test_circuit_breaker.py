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


def _losing_day(breaker: CircuitBreaker, n_trades: int = 1) -> None:
    for _ in range(n_trades):
        breaker.on_trade_closed(-10.0)
    breaker.start_of_day(100_000.0)  # boundary settles the finished day


def test_five_consecutive_losing_days_trip(breaker: CircuitBreaker) -> None:
    for _ in range(4):
        _losing_day(breaker)
    assert not breaker.trading_halted
    _losing_day(breaker)
    assert breaker.trading_halted


def test_many_losing_fills_in_one_day_do_not_trip(breaker: CircuitBreaker) -> None:
    # a rotating portfolio can realize many losses in ONE red day — that is
    # the daily_loss breaker's territory, never a streak of days
    _losing_day(breaker, n_trades=20)
    assert not breaker.trading_halted


def test_positive_day_resets_streak(breaker: CircuitBreaker) -> None:
    for _ in range(4):
        _losing_day(breaker)
    breaker.on_trade_closed(-10.0)
    breaker.on_trade_closed(60.0)  # net positive day
    breaker.start_of_day(100_000.0)
    for _ in range(4):
        _losing_day(breaker)
    assert not breaker.trading_halted


def test_day_without_trades_keeps_streak_frozen(breaker: CircuitBreaker) -> None:
    for _ in range(4):
        _losing_day(breaker)
    breaker.start_of_day(100_000.0)  # idle day: no information, no reset
    _losing_day(breaker)  # 5th losing day -> trip
    assert breaker.trading_halted


def test_no_auto_reset_on_new_day(breaker: CircuitBreaker) -> None:
    breaker.start_of_day(100_000.0)
    breaker.on_equity_update(90_000.0)
    assert breaker.trading_halted
    breaker.start_of_day(90_000.0)  # a new session must NOT clear the halt
    breaker.on_equity_update(95_000.0)  # nor a recovery
    assert breaker.trading_halted


def test_manual_reset_restores(breaker: CircuitBreaker) -> None:
    for _ in range(5):
        _losing_day(breaker)
    assert breaker.trading_halted
    breaker.reset_circuit_breaker()
    assert not breaker.trading_halted
    _losing_day(breaker)  # streak was cleared too — one losing day is fine
    assert not breaker.trading_halted


def test_trip_calls_injected_callback(
    breaker: CircuitBreaker, trips: list[tuple[str, dict[str, float]]]
) -> None:
    for _ in range(5):
        _losing_day(breaker)
    assert [reason for reason, _ in trips] == ["consecutive_losses"]
    assert trips[0][1]["loss_streak"] == 5.0
