"""Three independent circuit breakers. Any one halts ALL new orders.

  daily_loss_pct:     P&L < -3% of start-of-day equity
  max_drawdown_pct:   equity < 85% of peak equity
  consecutive_losses: 5 losing trades in a row
(thresholds from YAML config; values above are defaults)

When tripped (src/risk/CLAUDE.md):
  DO:   set halted, log CRITICAL circuit_breaker_tripped {reason, values}, Telegram alert.
  DON'T: auto-cancel open orders (they may be protecting positions).
  DON'T: auto-reset — reset_circuit_breaker() is a manual, human decision.
"""

from __future__ import annotations


class CircuitBreaker:
    """TODO(Phase 3):
      - on_fill/on_equity_update feed the three checks after every event.
      - trading_halted: bool property read by RiskManager.validate() step 1.
      - reset_circuit_breaker(): the ONLY way back to trading. Manual call.
        No time-based, condition-based, or startup-based auto-reset anywhere.
    """

    @property
    def trading_halted(self) -> bool:
        raise NotImplementedError("Phase 3 — circuit breakers")
