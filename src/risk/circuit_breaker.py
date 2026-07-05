"""Three independent circuit breakers. Any one halts ALL new orders.

  daily_loss_pct:     P&L below -X% of start-of-day equity
  max_drawdown_pct:   equity below (1 - X%) of peak equity
  consecutive_losses: N consecutive losing DAYS (sum of realized P&L of the
                      day < 0; days without closed trades don't count).
                      Day-based, not fill-based: a rotating multi-position
                      portfolio can realize 5 losing fills in ONE red day —
                      that is the daily_loss breaker's job, not a streak.
(thresholds from YAML config risk.circuit_breakers)

When tripped: halt, log CRITICAL, dispatch the injected alert callback.
Open orders are NOT canceled (they may be protecting positions). There is
no auto-reset of any kind — reset_circuit_breaker() is a manual, human call.

Alert dispatch is an injected callable because risk/ may not import
monitoring/ (dependency graph); the composition root wires it to Telegram.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

log = logging.getLogger(__name__)

OnTrip = Callable[[str, dict[str, float]], None]


class CircuitBreaker:
    def __init__(self, thresholds: dict[str, Any], on_trip: OnTrip) -> None:
        self._daily_loss_pct = float(thresholds["daily_loss_pct"])
        self._max_drawdown_pct = float(thresholds["max_drawdown_pct"])
        self._max_consecutive_losses = int(thresholds["consecutive_losses"])
        self._on_trip = on_trip
        self._halted = False
        self._start_of_day_equity: float | None = None
        self._peak_equity: float | None = None
        self._loss_streak = 0
        self._day_realized = 0.0
        self._day_trade_count = 0

    @property
    def trading_halted(self) -> bool:
        return self._halted

    def start_of_day(self, equity: float) -> None:
        """Day boundary: settle the FINISHED day's realized P&L into the
        losing-days streak, then open the new session. Never resets a trip."""
        self._settle_day()
        self._start_of_day_equity = equity
        self._peak_equity = max(self._peak_equity or equity, equity)

    def on_equity_update(self, equity: float) -> None:
        self._peak_equity = max(self._peak_equity or equity, equity)
        drawdown_floor = (1.0 - self._max_drawdown_pct) * self._peak_equity
        if equity < drawdown_floor:
            self._trip("max_drawdown", {"equity": equity, "peak_equity": self._peak_equity})
            return
        if self._start_of_day_equity is not None:
            daily_floor = (1.0 - self._daily_loss_pct) * self._start_of_day_equity
            if equity < daily_floor:
                self._trip(
                    "daily_loss",
                    {"equity": equity, "start_of_day_equity": self._start_of_day_equity},
                )

    def on_trade_closed(self, pnl: float) -> None:
        """Accumulate the day's realized P&L; the streak is judged per DAY
        at the next start_of_day() boundary."""
        self._day_realized += pnl
        self._day_trade_count += 1

    def reset_circuit_breaker(self) -> None:
        """The ONLY way back to trading. Called by a human decision, never code."""
        log.warning("circuit_breaker_manually_reset", extra={"was_halted": self._halted})
        self._halted = False
        self._loss_streak = 0
        self._day_realized = 0.0
        self._day_trade_count = 0

    def _settle_day(self) -> None:
        if self._day_trade_count == 0:
            return  # a day without closed trades carries no streak information
        if self._day_realized < 0.0:
            self._loss_streak += 1
            if self._loss_streak >= self._max_consecutive_losses:
                self._trip("consecutive_losses", {"loss_streak": float(self._loss_streak)})
        else:
            self._loss_streak = 0
        self._day_realized = 0.0
        self._day_trade_count = 0

    def _trip(self, reason: str, values: dict[str, float]) -> None:
        if self._halted:
            return
        self._halted = True
        log.critical("circuit_breaker_tripped", extra={"reason": reason, **values})
        self._on_trip(reason, values)
