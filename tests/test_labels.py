"""Triple-barrier label tests — hand-computed fixtures (port protocol step 2).

With constant volatility 0.02, pt_mult 1.5, sl_mult 1.0 and entry close 100:
upper barrier = 100 * (1 + 1.5*0.02) = 103, lower = 100 * (1 - 1.0*0.02) = 98.
Every expected value below is derived from those numbers by hand.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.signals.labels import ewma_volatility, triple_barrier_labels

PT, SL, HORIZON, VOL = 1.5, 1.0, 3, 0.02


def _frame(highs: list[float], lows: list[float], closes: list[float]) -> pd.DataFrame:
    index = pd.date_range("2024-01-02", periods=len(closes), freq="B", tz="UTC")
    return pd.DataFrame({"high": highs, "low": lows, "close": closes}, index=index)


def _labels(frame: pd.DataFrame, vol: float | list[float] = VOL) -> pd.Series:
    volatility = pd.Series(vol, index=frame.index, dtype="float64")
    return triple_barrier_labels(
        frame["high"], frame["low"], frame["close"], volatility, PT, SL, HORIZON
    )


def test_take_profit_touched_first_is_1() -> None:
    # t=0: upper 103, lower 98. Bar 1 stays inside; bar 2 high 103.5 >= 103.
    frame = _frame(
        highs=[100.5, 102.0, 103.5, 104.0, 104.0],
        lows=[99.5, 99.0, 101.0, 102.0, 102.0],
        closes=[100.0, 101.0, 102.5, 103.0, 103.0],
    )
    assert _labels(frame).iloc[0] == 1.0


def test_stop_loss_touched_first_is_0() -> None:
    # t=0: bar 1 low 97.5 <= 98 -> stop, even though bar 2 would hit 103.
    frame = _frame(
        highs=[100.5, 100.0, 104.0, 104.0, 104.0],
        lows=[99.5, 97.5, 101.0, 102.0, 102.0],
        closes=[100.0, 99.0, 103.5, 103.0, 103.0],
    )
    assert _labels(frame).iloc[0] == 0.0


def test_both_barriers_same_bar_is_0_conservative() -> None:
    # Bar 1 spans 97..104: touches both 98 and 103 -> stop wins (worst case).
    frame = _frame(
        highs=[100.5, 104.0, 100.0, 100.0, 100.0],
        lows=[99.5, 97.0, 99.0, 99.0, 99.0],
        closes=[100.0, 100.0, 99.5, 99.5, 99.5],
    )
    assert _labels(frame).iloc[0] == 0.0


def test_no_touch_full_horizon_is_expiry_0() -> None:
    # Bars 1..3 stay strictly inside (98, 103) -> vertical barrier: 0.
    frame = _frame(
        highs=[100.5, 101.0, 101.0, 101.0, 101.0],
        lows=[99.5, 99.0, 99.0, 99.0, 99.0],
        closes=[100.0, 100.0, 100.0, 100.0, 100.0],
    )
    assert _labels(frame).iloc[0] == 0.0


def test_no_touch_truncated_horizon_is_nan() -> None:
    # t=2 has only 2 future bars (< HORIZON=3), none touching -> undecidable.
    frame = _frame(
        highs=[100.5, 101.0, 101.0, 101.0, 101.0],
        lows=[99.5, 99.0, 99.0, 99.0, 99.0],
        closes=[100.0, 100.0, 100.0, 100.0, 100.0],
    )
    assert np.isnan(_labels(frame).iloc[2])
    assert np.isnan(_labels(frame).iloc[4])  # last row: no future bars at all


def test_touch_decides_even_with_truncated_horizon() -> None:
    # t=3 (close 100) has a single future bar, but it hits 103 -> decided 1.
    frame = _frame(
        highs=[100.5, 100.5, 100.5, 100.5, 103.5],
        lows=[99.5, 99.5, 99.5, 99.5, 101.0],
        closes=[100.0, 100.0, 100.0, 100.0, 103.0],
    )
    assert _labels(frame).iloc[3] == 1.0


def test_missing_or_zero_volatility_is_nan() -> None:
    frame = _frame(
        highs=[100.5, 104.0, 104.0, 104.0, 104.0],
        lows=[99.5, 101.0, 101.0, 101.0, 101.0],
        closes=[100.0, 103.5, 103.0, 103.0, 103.0],
    )
    vol: list[float] = [float("nan"), 0.0, VOL, VOL, VOL]
    labels = _labels(frame, vol)
    assert np.isnan(labels.iloc[0])  # NaN vol -> no barriers
    assert np.isnan(labels.iloc[1])  # zero vol -> degenerate barriers refused


@pytest.mark.parametrize("cut", [10, 25, 40])
def test_truncation_never_flips_a_decided_label(cut: int) -> None:
    rng = np.random.default_rng(7)
    n = 60
    closes = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.015, n))
    frame = _frame(highs=list(closes * 1.01), lows=list(closes * 0.99), closes=list(closes))
    full = _labels(frame)
    short = _labels(frame.iloc[:cut])
    for t in range(cut):
        if not np.isnan(short.iloc[t]):
            assert short.iloc[t] == full.iloc[t]  # decided labels are final


def test_inputs_not_mutated() -> None:
    frame = _frame(
        highs=[100.5, 104.0, 104.0, 104.0, 104.0],
        lows=[99.5, 101.0, 101.0, 101.0, 101.0],
        closes=[100.0, 103.5, 103.0, 103.0, 103.0],
    )
    volatility = pd.Series(VOL, index=frame.index, dtype="float64")
    before = frame.copy()
    triple_barrier_labels(frame["high"], frame["low"], frame["close"], volatility, PT, SL, HORIZON)
    pd.testing.assert_frame_equal(frame, before)
    assert (volatility == VOL).all()


def test_ewma_volatility_nan_until_span_then_tracks_dispersion() -> None:
    index = pd.date_range("2024-01-02", periods=40, freq="B", tz="UTC")
    calm = pd.Series(100.0 * 1.001 ** np.arange(40), index=index)
    rng = np.random.default_rng(11)
    wild = pd.Series(100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.03, 40)), index=index)
    vol_calm, vol_wild = ewma_volatility(calm, span=10), ewma_volatility(wild, span=10)
    assert vol_calm.iloc[:10].isna().all()  # returns[0] is NaN + min_periods=span
    assert vol_calm.iloc[10:].notna().all()
    assert vol_calm.iloc[-1] == pytest.approx(0.0, abs=1e-12)  # constant returns
    assert vol_wild.iloc[-1] > 0.01  # 3% daily noise -> vol of that order
