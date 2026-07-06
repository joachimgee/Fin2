"""Volatility targeting — the highest-leverage portfolio-allocation lever.

Moreira & Muir (2017): scaling exposure inversely to recent realized
volatility raises the Sharpe of many factors, because high-vol periods tend
to carry lower risk-adjusted returns. Nagel (2012) finds the OPPOSITE for
short-term reversal (its premium rises with volatility), so applying this to
our reversion family is a genuine, decisive test, not a foregone win.

Scale applied on day t uses only volatility estimated from returns <= t-1
(lagged rolling std, the same shift chokepoint as features) — no lookahead.
The TARGET is the input series' own full-sample annualized vol, so average
leverage is ~1 and any Sharpe change is pure vol TIMING, not a static leverage
bump (Sharpe is invariant to constant leverage — the honest comparison).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def vol_target_returns(
    returns: pd.Series,
    periods_per_year: int,
    vol_window: int,
    max_leverage: float,
) -> pd.Series:
    """Return the vol-targeted return series (same index, NaN-free rows only).

    leverage_t = clip(target_vol / realized_vol_{t-1}, 0, max_leverage), with
    realized_vol_{t-1} the trailing std over vol_window periods ending at t-1.
    Rows before the window has filled are dropped (no leverage estimate).
    """
    clean = returns.dropna()
    if len(clean) <= vol_window:
        return clean.iloc[0:0]
    target_daily = float(clean.std(ddof=1))  # sets scale only; Sharpe-invariant
    # lagged trailing vol: value at t uses returns in [t-vol_window, t-1]
    trailing = clean.rolling(vol_window).std(ddof=1).shift(1)
    leverage = (target_daily / trailing).clip(upper=max_leverage)
    scaled = (clean * leverage).dropna()
    _ = periods_per_year  # annualization cancels in the target/realized ratio
    return scaled


def realized_annual_vol(returns: pd.Series, periods_per_year: int) -> float:
    """Annualized realized volatility of a return series (ddof=1)."""
    clean = returns.dropna()
    if len(clean) < 2:
        return 0.0
    return float(clean.std(ddof=1) * np.sqrt(periods_per_year))
