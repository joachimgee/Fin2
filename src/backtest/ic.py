"""Information Coefficient — is there predictive signal, before any backtest?

The sharpest cheap instrument (alphalens-style): the cross-sectional rank
correlation between a factor at date T and the forward return, averaged over
dates. A factor with no edge gives mean IC ≈ 0; a real one gives a small
positive IC with a t-stat that survives. Used to decide whether an idea earns
a full WFO — the same "measure before you build" discipline as the sentiment
A/B and the vol-target probe.

Port provenance: finbot backtest/ic_reporting.py (inspiration inventory);
re-derived. Forward-return alignment is the caller's responsibility and must
be lag-correct (factor known at decision time strictly precedes the return
window) — this module never shifts, so it cannot hide lookahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def cross_sectional_ic(factor: pd.DataFrame, forward_return: pd.DataFrame) -> pd.Series:
    """Per-date Spearman rank correlation between factor and forward return.

    Both are wide frames (index=date, columns=symbol), pre-aligned by the
    caller. Dates with < 3 jointly-present names yield NaN (rank correlation
    undefined on 2 points).
    """
    aligned = factor.reindex_like(forward_return)
    out: dict[pd.Timestamp, float] = {}
    for date in forward_return.index:
        f = aligned.loc[date]
        r = forward_return.loc[date]
        mask = f.notna() & r.notna()
        if int(mask.sum()) < 3:
            out[date] = float("nan")
            continue
        out[date] = float(f[mask].rank().corr(r[mask].rank()))
    return pd.Series(out, name="ic")


def ic_summary(ic: pd.Series) -> dict[str, float]:
    """Mean IC, its std, the t-stat (mean/SE) and hit-rate (fraction > 0).
    A |t-stat| > ~2 is the usual bar for "the mean IC is not zero"."""
    clean = ic.dropna()
    n = len(clean)
    if n < 3:
        return {"mean": 0.0, "std": 0.0, "t_stat": 0.0, "hit_rate": 0.0, "n_dates": float(n)}
    std = float(clean.std(ddof=1))
    t_stat = float(clean.mean() / (std / np.sqrt(n))) if std > 0 else 0.0
    return {
        "mean": float(clean.mean()),
        "std": std,
        "t_stat": t_stat,
        "hit_rate": float((clean > 0).mean()),
        "n_dates": float(n),
    }
