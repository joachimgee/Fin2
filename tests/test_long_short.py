"""Long-short paper-portfolio tests — hand-computed spread and cost math."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.backtest.long_short import long_short_periodic_returns

_DATES = pd.date_range("2024-01-02", periods=2, freq="B", tz="UTC")
_SYMS = ["A", "B", "C", "D"]


def _wide(rows: list[list[float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, index=_DATES, columns=_SYMS)


def test_reversal_spread_hand_computed_zero_cost() -> None:
    # signal (past return): A,B losers (-2,-1), C,D winners (+1,+2).
    # top_frac 0.25 -> n=1 each leg: long A (biggest loser), short D (biggest winner).
    # fwd: A reverts +0.05, D reverts -0.03 -> spread = 0.05 - (-0.03) = 0.08.
    signal = _wide([[-2.0, -1.0, 1.0, 2.0], [-2.0, -1.0, 1.0, 2.0]])
    fwd = _wide([[0.05, 0.01, -0.01, -0.03], [0.05, 0.01, -0.01, -0.03]])
    out = long_short_periodic_returns(signal, fwd, 0.25, 0.0, 0.0, 0.0)
    assert out.tolist() == pytest.approx([0.08, 0.08])


def test_costs_subtracted_exactly() -> None:
    signal = _wide([[-2.0, -1.0, 1.0, 2.0]] * 2)
    fwd = _wide([[0.05, 0.01, -0.01, -0.03]] * 2)
    # round_trip 20bps + borrow 100bps/yr * 0.02yr = 0.002 + 0.01*0.02 = 0.0022
    out = long_short_periodic_returns(signal, fwd, 0.25, 20.0, 100.0, 0.02)
    assert out.iloc[0] == pytest.approx(0.08 - 0.0022)


def test_two_names_per_leg_averages() -> None:
    # top_frac 0.5 -> n=2: long {A,B}, short {C,D}
    signal = _wide([[-2.0, -1.0, 1.0, 2.0]] * 2)
    fwd = _wide([[0.06, 0.02, -0.02, -0.04]] * 2)
    out = long_short_periodic_returns(signal, fwd, 0.5, 0.0, 0.0, 0.0)
    long_leg = (0.06 + 0.02) / 2
    short_leg = (-0.02 + -0.04) / 2
    assert out.iloc[0] == pytest.approx(long_leg - short_leg)


def test_nan_cells_excluded_from_ranking() -> None:
    signal = _wide([[-2.0, -1.0, 1.0, np.nan], [-2.0, -1.0, 1.0, 2.0]])
    fwd = _wide([[0.05, 0.01, -0.01, np.nan], [0.05, 0.01, -0.01, -0.03]])
    out = long_short_periodic_returns(signal, fwd, 0.5, 0.0, 0.0, 0.0)
    # date0: only A,B,C valid -> n=1: long A (0.05), short C (-0.01) -> 0.06
    assert out.iloc[0] == pytest.approx(0.06)


def test_random_signal_zero_cost_spread_near_zero() -> None:
    rng = np.random.default_rng(0)
    dates = pd.date_range("2020-01-01", periods=300, freq="W", tz="UTC")
    syms = [f"S{i}" for i in range(40)]
    signal = pd.DataFrame(rng.normal(size=(300, 40)), index=dates, columns=syms)
    fwd = pd.DataFrame(rng.normal(0, 0.02, size=(300, 40)), index=dates, columns=syms)
    out = long_short_periodic_returns(signal, fwd, 0.1, 0.0, 0.0, 0.0)
    assert abs(out.mean()) < 0.002  # no signal -> no spread
