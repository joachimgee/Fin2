"""RiskManager — every order passes validate() or it doesn't exist.

validate() is synchronous, < 1 ms, zero network calls, zero DB reads: it only
consults in-memory state (ExposureTracker, CircuitBreaker). When in doubt,
reject — a missed trade is recoverable, a blown account is not.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.strategies.base import OrderIntent


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Outcome of validate(). Execution MUST use adjusted_qty, never intent.qty —
    steps 3 and 4 of the pipeline may have reduced the quantity."""

    approved: bool
    adjusted_qty: float
    reason: str  # "ok" or the first failed check's name


class RiskManager:
    """TODO(Phase 3): implement validate() in this EXACT order — first failure
    rejects immediately, stop evaluating (src/risk/CLAUDE.md):

      1. circuit breaker active?   -> reject if trading_halted
      2. Kelly sizing              -> adjusted_qty from signal_strength (kelly.py)
      3. risk-per-trade cap  2%    -> REDUCE qty (don't reject)
      4. position cap       10%    -> REDUCE qty (don't reject)
      5. total exposure cap 80%    -> REJECT if breached even after reduction
      6. sector cap         25%    -> REJECT if sector would breach
      7. correlation guard  0.80   -> REJECT if > 0.80 vs any existing position

    All caps come from YAML config (config/base.yaml risk: section) — the
    numbers above are the defaults, never hardcoded here.
    Also owns on_fill(): forwards fills to ExposureTracker + CircuitBreaker.
    """

    def validate(self, intent: OrderIntent) -> ValidationResult:
        raise NotImplementedError("Phase 3 — risk validation pipeline")
