"""IC tests — hand-computed Spearman fixtures and the null property."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.backtest.ic import cross_sectional_ic, ic_summary

_DATES = pd.date_range("2024-01-02", periods=2, freq="B", tz="UTC")
_SYMS = ["A", "B", "C"]


def _wide(rows: list[list[float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, index=_DATES, columns=_SYMS)


def test_perfect_rank_agreement_is_plus_one() -> None:
    factor = _wide([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0]])
    fwd = _wide([[10.0, 20.0, 30.0], [30.0, 10.0, 20.0]])  # same ranking both dates
    ic = cross_sectional_ic(factor, fwd)
    assert ic.tolist() == [1.0, 1.0]


def test_perfect_rank_reversal_is_minus_one() -> None:
    factor = _wide([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    fwd = _wide([[30.0, 20.0, 10.0], [30.0, 20.0, 10.0]])
    assert cross_sectional_ic(factor, fwd).tolist() == [-1.0, -1.0]


def test_fewer_than_three_present_names_is_nan() -> None:
    factor = _wide([[1.0, 2.0, np.nan], [1.0, 2.0, 3.0]])  # date0: 2 names, date1: 3
    fwd = _wide([[10.0, 20.0, 30.0], [10.0, 20.0, 30.0]])
    ic = cross_sectional_ic(factor, fwd)
    assert np.isnan(ic.iloc[0])  # 2 present -> rank corr undefined
    assert ic.iloc[1] == pytest.approx(1.0)


def test_random_factor_has_mean_ic_near_zero() -> None:
    rng = np.random.default_rng(0)
    dates = pd.date_range("2020-01-01", periods=400, freq="B", tz="UTC")
    syms = [f"S{i}" for i in range(30)]
    factor = pd.DataFrame(rng.normal(size=(400, 30)), index=dates, columns=syms)
    fwd = pd.DataFrame(rng.normal(size=(400, 30)), index=dates, columns=syms)  # independent
    summary = ic_summary(cross_sectional_ic(factor, fwd))
    assert abs(summary["mean"]) < 0.02  # no relationship -> IC ~ 0
    assert abs(summary["t_stat"]) < 2.5  # and not significant


def test_summary_t_stat_and_hit_rate_hand_computed() -> None:
    ic = pd.Series([0.1, 0.2, 0.3, -0.1, float("nan")])
    s = ic_summary(ic)
    clean = np.array([0.1, 0.2, 0.3, -0.1])
    assert s["mean"] == pytest.approx(clean.mean())
    assert s["hit_rate"] == pytest.approx(0.75)  # 3 of 4 positive
    assert s["n_dates"] == 4.0
    assert s["t_stat"] == pytest.approx(clean.mean() / (clean.std(ddof=1) / np.sqrt(4)), rel=1e-9)


def test_short_series_is_safe_zeros() -> None:
    assert ic_summary(pd.Series([0.1, 0.2]))["t_stat"] == 0.0
