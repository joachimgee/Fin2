"""Backtest performance metrics — pure functions.

Formulas re-derived per the port protocol (inspiration inventory from Fin v1
backtest/metrics.py); cross-checked against quantstats in tests.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def sharpe_ratio(returns: pd.Series, periods_per_year: int) -> float:
    """Annualized Sharpe (risk-free = 0): mean/std(ddof=1) * sqrt(periods)."""
    clean = returns.dropna()
    if len(clean) < 2:
        return 0.0
    std = float(clean.std(ddof=1))
    if std == 0.0:
        return 0.0
    return float(clean.mean() / std * np.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> float:
    """Peak-to-trough loss as a POSITIVE fraction (0.15 == -15% from peak)."""
    if equity.empty:
        return 0.0
    drawdown = equity / equity.cummax() - 1.0
    return float(-drawdown.min())


def profit_factor(pnls: Sequence[float]) -> float:
    """Gross profit / gross loss. No losing trades -> inf; no trades -> 0."""
    gains = sum(p for p in pnls if p > 0)
    losses = -sum(p for p in pnls if p < 0)
    if losses == 0.0:
        return float("inf") if gains > 0 else 0.0
    return gains / losses
