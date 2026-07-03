"""Kelly position sizing. Default quarter-Kelly; hard maximum half-Kelly.

Full Kelly is catastrophic in practice because edge estimates are noisy —
estimation error compounds into drawdowns no account survives.
"""

from __future__ import annotations

from src.shared.exceptions import InvalidKellyFractionError

_HARD_CAP = 0.50  # half-Kelly — src/risk/CLAUDE.md <kelly>


def kelly_fraction(
    win_rate: float, avg_win: float, avg_loss: float, fraction: float = 0.25
) -> float:
    """Capital fraction to allocate: full_kelly * fraction, floored at 0.

    Guards first, math second:
      - fraction above half-Kelly is a configuration error, not a request
      - degenerate stats (no wins/losses measured) mean no edge: do not trade
      - negative edge: do not trade
    """
    if fraction > _HARD_CAP:
        raise InvalidKellyFractionError(
            f"kelly fraction {fraction} exceeds hard cap {_HARD_CAP} (half-Kelly)"
        )
    if avg_win <= 0 or avg_loss <= 0:
        return 0.0
    payoff = avg_win / avg_loss
    full_kelly = win_rate - (1.0 - win_rate) / payoff
    if full_kelly <= 0:
        return 0.0
    return full_kelly * fraction
