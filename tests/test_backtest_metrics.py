"""Phase 6 — metrics tests + quantstats cross-check (PHASE_6_BACKTEST_WFO.md)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.backtest.metrics import max_drawdown, profit_factor, sharpe_ratio

try:  # heavy import (matplotlib); skip the cross-check if unavailable
    import quantstats

    _HAS_QS = True
except Exception:
    _HAS_QS = False


def test_sharpe_hand_computed() -> None:
    """returns [.01, -.01, .02]: mean=.0066667, std(ddof=1)=.0152753
    daily sharpe = .43644; annualized = .43644 * sqrt(252) = 6.9282"""
    r = pd.Series([0.01, -0.01, 0.02])
    assert sharpe_ratio(r, 252) == pytest.approx(6.9282, rel=1e-4)


def test_sharpe_degenerate_cases() -> None:
    assert sharpe_ratio(pd.Series([0.01]), 252) == 0.0  # too short
    assert sharpe_ratio(pd.Series([0.01, 0.01, 0.01]), 252) == 0.0  # zero variance


def test_max_drawdown_hand_computed() -> None:
    # peak 120 -> trough 90: 1 - 90/120 = 0.25, reported as positive magnitude
    assert max_drawdown(pd.Series([100.0, 120.0, 90.0, 130.0])) == pytest.approx(0.25)


def test_max_drawdown_monotonic_is_zero() -> None:
    assert max_drawdown(pd.Series([100.0, 110.0, 120.0])) == 0.0


def test_profit_factor_hand_computed() -> None:
    assert profit_factor([10.0, -5.0, 20.0, -10.0]) == pytest.approx(2.0)  # 30/15
    assert profit_factor([10.0, 5.0]) == float("inf")
    assert profit_factor([]) == 0.0


@pytest.mark.skipif(not _HAS_QS, reason="quantstats reference library not importable")
def test_metrics_match_reference_library() -> None:
    """Port protocol 2b: same input through quantstats, agreement to 1e-9."""
    rng = np.random.default_rng(7)
    returns = pd.Series(
        rng.normal(0.0005, 0.01, 252), index=pd.date_range("2024-01-01", periods=252, freq="B")
    )
    equity = (1.0 + returns).cumprod()
    np.testing.assert_allclose(
        sharpe_ratio(returns, 252), float(quantstats.stats.sharpe(returns)), rtol=1e-9
    )
    np.testing.assert_allclose(
        max_drawdown(equity), -float(quantstats.stats.max_drawdown(returns)), rtol=1e-9
    )
