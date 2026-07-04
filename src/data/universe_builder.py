"""Mechanical universe selection — the survivorship-selection-bias fix.

Ranks symbols by liquidity measured in a window at the START of the research
period, using ONLY information available then: no hand-picking of names known
(in hindsight) to have performed. Residual bias remains — symbols delisted
since are absent from the data source entirely — and must be documented
wherever the resulting universe is used.

IEX volumes are a fraction of the consolidated tape, so dollar volume is used
for RANKING only, never against absolute thresholds.
"""

from __future__ import annotations

import pandas as pd


def select_liquid_universe(
    bars: pd.DataFrame, min_price: float, min_days: int, top_n: int
) -> list[str]:
    """Top-n symbols by median daily dollar volume over the given bars.

    bars: columns symbol/timestamp/close/volume covering the ranking window.
    Excluded: symbols with < min_days bars (illiquid or not yet listed) and
    symbols whose median close < min_price (penny-stock microstructure).
    Returns symbols in rank order (most liquid first).
    """
    ranked = (
        bars.assign(dollar_volume=bars["close"] * bars["volume"])
        .groupby("symbol")
        .agg(
            n_days=("timestamp", "nunique"),
            median_close=("close", "median"),
            median_dollar_volume=("dollar_volume", "median"),
        )
    )
    eligible = ranked[(ranked["n_days"] >= min_days) & (ranked["median_close"] >= min_price)]
    ordered = eligible.sort_values("median_dollar_volume", ascending=False)
    return [str(s) for s in ordered.index[:top_n]]
