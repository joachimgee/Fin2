"""Long-short paper portfolio — measure the short-leg premium before building it.

Adding real short selling touches the engine, the risk stack and the strategy
(borrow costs, negative exposure, margin) — a large change to working code. The
canonical academic estimator of the short-term reversal premium (Lehmann 1990,
Jegadeesh 1990, Nagel 2012) is a paper portfolio computed directly on the
return frames: each rebalance, LONG the biggest losers and SHORT the biggest
winners, equal weight, hold one period, net of honest transaction + borrow
costs. That answers "is the long-short premium there, net of costs?" without
risking a line of production code — measure before you build.

Rows are assumed NON-OVERLAPPING (the caller passes rebalance dates only), so
each period's return is an independent draw and the Sharpe is not autocorrelation-
inflated. Signal convention: HIGHER signal = more "winner" (e.g. past return),
so the reversal bet longs the low tail and shorts the high tail.
"""

from __future__ import annotations

import pandas as pd


def long_short_periodic_returns(
    signal: pd.DataFrame,
    forward_return: pd.DataFrame,
    top_frac: float,
    round_trip_bps: float,
    borrow_bps_annual: float,
    holding_years: float,
) -> pd.Series:
    """One net long-short return per (non-overlapping) rebalance date.

    top_frac: fraction of the cross-section in each leg (e.g. 0.1 = deciles).
    round_trip_bps: per-period entry+exit cost on the gross book (both legs).
    borrow_bps_annual / holding_years: short-leg financing for the holding span.
    Reversal: long the bottom-`top_frac` by signal, short the top-`top_frac`.
    """
    aligned = signal.reindex_like(forward_return)
    cost = round_trip_bps / 1e4 + borrow_bps_annual / 1e4 * holding_years
    out: dict[pd.Timestamp, float] = {}
    for date in forward_return.index:
        s = aligned.loc[date]
        r = forward_return.loc[date]
        mask = s.notna() & r.notna()
        s, r = s[mask], r[mask]
        n = int(top_frac * len(s))
        if n < 1:
            continue
        order = s.to_numpy().argsort()  # ascending: losers first, winners last
        losers = r.to_numpy()[order[:n]]
        winners = r.to_numpy()[order[-n:]]
        gross = float(losers.mean() - winners.mean())  # long losers, short winners
        out[date] = gross - cost
    return pd.Series(out, name="long_short_return")
