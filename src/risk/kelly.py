"""Kelly position sizing. Default quarter-Kelly; hard maximum half-Kelly.

Full Kelly is catastrophic in practice because edge estimates are noisy —
estimation error compounds into drawdowns no account survives.
"""

from __future__ import annotations


def kelly_fraction(
    win_rate: float, avg_win: float, avg_loss: float, fraction: float = 0.25
) -> float:
    """Return the capital fraction to allocate, in [0.0, fraction * full_kelly].

    TODO(Phase 3): implement with the guards (src/risk/CLAUDE.md <kelly>):
      - raise InvalidKellyFractionError if fraction > 0.50 (hard cap: half-Kelly)
      - return 0.0 (do not trade) if edge <= 0 or avg_loss <= 0
      - full_kelly = win_rate - (1 - win_rate) / (avg_win / avg_loss)
      - return max(0.0, full_kelly * fraction)
    """
    raise NotImplementedError("Phase 3 — Kelly sizing")
