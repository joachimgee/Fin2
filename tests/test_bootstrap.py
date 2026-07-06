"""Stationary-bootstrap Sharpe interval tests (port protocol: fixtures +
reference cross-check against Lo 2002 analytic SE)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.backtest.bootstrap import _stationary_resample_index, bootstrap_sharpe
from src.backtest.metrics import sharpe_ratio

PPY = 252


def _gaussian(n: int, mean: float, std: float, seed: int) -> pd.Series:
    return pd.Series(np.random.default_rng(seed).normal(mean, std, n))


def test_point_matches_the_single_sharpe_definition() -> None:
    r = _gaussian(500, 0.0005, 0.01, seed=1)
    result = bootstrap_sharpe(r, PPY, threshold=1.5, n_resamples=200)
    assert result.point == sharpe_ratio(r, PPY)  # no second Sharpe formula


def test_reproducible_with_seed() -> None:
    r = _gaussian(400, 0.0004, 0.01, seed=2)
    a = bootstrap_sharpe(r, PPY, threshold=1.5, n_resamples=500, seed=7)
    b = bootstrap_sharpe(r, PPY, threshold=1.5, n_resamples=500, seed=7)
    assert a.percentiles == b.percentiles and a.p_at_least == b.p_at_least


def test_percentiles_ordered_and_bracket_point() -> None:
    r = _gaussian(750, 0.0006, 0.01, seed=3)
    res = bootstrap_sharpe(r, PPY, threshold=1.5, n_resamples=2000)
    pcts = [res.percentiles[q] for q in (5, 25, 50, 75, 95)]
    assert pcts == sorted(pcts)
    assert res.percentiles[5] <= res.point <= res.percentiles[95]


def test_p_at_least_monotonic_in_threshold() -> None:
    r = _gaussian(600, 0.0008, 0.01, seed=4)
    low = bootstrap_sharpe(r, PPY, threshold=0.5, n_resamples=1500)
    high = bootstrap_sharpe(r, PPY, threshold=2.5, n_resamples=1500)
    assert low.p_at_least >= high.p_at_least  # harder bar -> fewer pass


def test_degenerate_short_series_is_zero_not_error() -> None:
    res = bootstrap_sharpe(pd.Series([0.01]), PPY, threshold=1.5)
    assert res.point == 0.0 and res.p_at_least == 0.0
    assert res.n_returns == 1


def test_constant_returns_zero_variance_safe() -> None:
    res = bootstrap_sharpe(pd.Series([0.001] * 50), PPY, threshold=1.5, n_resamples=100)
    assert res.point == 0.0 and res.p_at_least == 0.0


def test_mean_block_below_one_rejected() -> None:
    with pytest.raises(ValueError, match="mean_block"):
        bootstrap_sharpe(_gaussian(100, 0.0, 0.01, seed=5), PPY, threshold=1.5, mean_block=0.5)


# --- index construction properties ------------------------------------------------


def test_mean_block_one_gives_unit_length_blocks() -> None:
    # p = 1/mean_block = 1 -> geometric(1) is always 1 -> pure IID resample
    rng = np.random.default_rng(0)
    idx = _stationary_resample_index(50, p=1.0, rng=rng)
    assert len(idx) == 50 and idx.min() >= 0 and idx.max() < 50


def test_empirical_mean_block_length_matches_target() -> None:
    rng = np.random.default_rng(0)
    target = 10.0
    lengths = [int(rng.geometric(1.0 / target)) for _ in range(20000)]
    assert np.mean(lengths) == pytest.approx(target, rel=0.05)


# --- reference cross-checks -------------------------------------------------------


def test_iid_bootstrap_matches_lo_2002_analytic_se() -> None:
    """Lo (2002): for IID returns the per-period Sharpe SE is
    sqrt((1 + 0.5*SR^2)/T). With periods_per_year=1 (per-period Sharpe) and
    mean_block=1 (IID), the bootstrap std of the Sharpe must match it."""
    n = 2000
    r = _gaussian(n, 0.05, 1.0, seed=11)  # SR ~ 0.05 per period
    rng = np.random.default_rng(0)
    sharpes = [
        sharpe_ratio(pd.Series(r.to_numpy()[_stationary_resample_index(n, 1.0, rng)]), 1)
        for _ in range(4000)
    ]
    sr = sharpe_ratio(r, 1)
    lo_se = np.sqrt((1.0 + 0.5 * sr**2) / n)
    assert np.std(sharpes) == pytest.approx(lo_se, rel=0.15)


def test_block_bootstrap_wider_than_iid_for_autocorrelated_returns() -> None:
    """THE reason for the re-derivation: on positively autocorrelated returns
    the stationary bootstrap must report a WIDER Sharpe interval than IID —
    IID would falsely tighten it by ignoring serial dependence."""
    rng = np.random.default_rng(21)
    n = 1500
    noise = rng.normal(0.0, 0.01, n)
    ar = np.empty(n)
    ar[0] = noise[0]
    for t in range(1, n):  # AR(1), phi=0.6 -> strong volatility persistence
        ar[t] = 0.6 * ar[t - 1] + noise[t]
    returns = pd.Series(ar + 0.0005)
    iid = bootstrap_sharpe(returns, PPY, threshold=1.5, n_resamples=2000, mean_block=1.0)
    block = bootstrap_sharpe(returns, PPY, threshold=1.5, n_resamples=2000, mean_block=20.0)
    iid_width = iid.percentiles[95] - iid.percentiles[5]
    block_width = block.percentiles[95] - block.percentiles[5]
    assert block_width > iid_width * 1.1  # meaningfully wider, not a rounding blip
