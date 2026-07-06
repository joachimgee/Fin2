"""Volatility-targeting tests — lookahead safety and scale invariance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.backtest.bootstrap import _stationary_resample_index  # noqa: F401 (shared rng style)
from src.backtest.metrics import sharpe_ratio
from src.backtest.vol_target import realized_annual_vol, vol_target_returns

PPY = 252


def _series(n: int, seed: int, scale: float = 0.01) -> pd.Series:
    return pd.Series(np.random.default_rng(seed).normal(0.0005, scale, n))


def test_constant_vol_input_leaves_returns_essentially_unchanged() -> None:
    # IID returns have ~constant trailing vol -> leverage ~1 -> Sharpe ~ same
    r = _series(2000, seed=1)
    scaled = vol_target_returns(r, PPY, vol_window=20, max_leverage=3.0)
    assert sharpe_ratio(scaled, PPY) == pytest.approx(
        sharpe_ratio(r.iloc[-len(scaled) :], PPY), abs=0.25
    )


def test_sharpe_is_truncation_invariant_no_lookahead() -> None:
    # The leverage timing is causal (lagged vol); the full-sample target is a
    # constant scalar that cancels in the Sharpe. So the Sharpe of a prefix
    # must NOT depend on future data — the operational anti-lookahead property.
    r = _series(400, seed=2)
    full = vol_target_returns(r, PPY, vol_window=20, max_leverage=3.0)
    k = 250
    prefix_from_full = full.iloc[: k - 20]  # scaled rows within the first k inputs
    prefix_recomputed = vol_target_returns(r.iloc[:k], PPY, vol_window=20, max_leverage=3.0)
    n = min(len(prefix_from_full), len(prefix_recomputed))
    assert sharpe_ratio(prefix_from_full.iloc[:n], PPY) == pytest.approx(
        sharpe_ratio(prefix_recomputed.iloc[:n], PPY), rel=1e-9
    )


def test_leverage_capped_at_max() -> None:
    # a burst of tiny returns after high vol -> uncapped leverage would explode
    r = pd.concat([_series(60, seed=3, scale=0.05), pd.Series([1e-6] * 40)], ignore_index=True)
    scaled = vol_target_returns(r, PPY, vol_window=20, max_leverage=2.0)
    # implied leverage = scaled/original never exceeds the cap (within fp tolerance)
    orig = r.dropna().iloc[-len(scaled) :].to_numpy()
    lev = np.abs(scaled.to_numpy()[orig != 0] / orig[orig != 0])
    assert lev.max() <= 2.0 + 1e-9


def test_short_series_returns_empty() -> None:
    assert vol_target_returns(_series(10, seed=4), PPY, vol_window=20, max_leverage=3.0).empty


def test_input_not_mutated() -> None:
    r = _series(100, seed=5)
    before = r.copy()
    vol_target_returns(r, PPY, vol_window=20, max_leverage=3.0)
    pd.testing.assert_series_equal(r, before)


def test_realized_annual_vol_scales_with_sqrt_periods() -> None:
    r = _series(5000, seed=6, scale=0.01)
    assert realized_annual_vol(r, PPY) == pytest.approx(0.01 * np.sqrt(PPY), rel=0.1)
    assert realized_annual_vol(pd.Series([0.01]), PPY) == 0.0
