"""Training labels — triple-barrier method (López de Prado, AFML ch. 3).

Port provenance: Fin v1 (models/metalabel.py) used this scheme; per the port
protocol the logic is RE-DERIVED from first principles and validated against
hand-computed fixtures in tests/test_labels.py — never copied.

Labels look FORWARD by design — legitimate in training only, never as features.
The label answers the question the strategy actually trades: "will the
take-profit be touched before the stop-loss within the horizon?"
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ewma_volatility(close: pd.Series, span: int) -> pd.Series:
    """EWMA std of close-to-close returns. NaN until `span` observations —
    an unreliable early estimate must not silently set barrier widths."""
    return close.pct_change().ewm(span=span, min_periods=span).std()


def triple_barrier_labels(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volatility: pd.Series,
    pt_mult: float,
    sl_mult: float,
    horizon_bars: int,
) -> pd.Series:
    """1.0 if the take-profit barrier close_t*(1 + pt_mult*vol_t) is touched
    before the stop-loss close_t*(1 - sl_mult*vol_t) within horizon_bars;
    0.0 on stop-loss touch or horizon expiry; NaN when undecidable (no
    volatility estimate, or the series ends before a barrier or the expiry).

    Barrier widths use only information available at t (vol_t); the touch
    scan starts at t+1 (entry is at the close of t). If both barriers fall
    inside one bar, the stop wins — conservative, worst case first.
    """
    n = len(close)
    highs, lows = high.to_numpy(dtype=float), low.to_numpy(dtype=float)
    closes, vols = close.to_numpy(dtype=float), volatility.to_numpy(dtype=float)
    labels = np.full(n, np.nan)
    for t in range(n):
        if not np.isfinite(vols[t]) or vols[t] <= 0.0:
            continue  # no usable volatility -> no barriers -> undecidable
        upper = closes[t] * (1.0 + pt_mult * vols[t])
        lower = closes[t] * (1.0 - sl_mult * vols[t])
        for j in range(t + 1, min(t + horizon_bars, n - 1) + 1):
            if lows[j] <= lower:
                labels[t] = 0.0  # stop-loss first (or same-bar ambiguity)
                break
            if highs[j] >= upper:
                labels[t] = 1.0  # take-profit first
                break
        else:
            if t + horizon_bars <= n - 1:
                labels[t] = 0.0  # full horizon elapsed, no touch: expiry
            # else: truncated before expiry and undecided -> stays NaN
    return pd.Series(labels, index=close.index, dtype="float64")
