"""Phase 2 — feature tests (docs/plan/PHASE_2_SIGNALS.md).

Includes THE critical test: truncation invariance over ALL columns — any
feature reading the future fails it. Columns are discovered from the output,
never hardcoded, so every future ported indicator is covered automatically.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd
import pytest
from src.shared.exceptions import ConfigError, DataValidationError
from src.signals import features as feat

try:  # ta may be missing or numpy-incompatible in some envs — skip, never fail
    import ta

    _HAS_TA = True
except Exception:
    _HAS_TA = False


@pytest.fixture
def feature_frame(sample_bars: pd.DataFrame, base_config: dict[str, Any]) -> pd.DataFrame:
    return feat.compute_features(sample_bars, base_config)


# --- lookahead safety ---------------------------------------------------------


def test_no_lookahead_truncation_invariance(
    sample_bars: pd.DataFrame, base_config: dict[str, Any], feature_frame: pd.DataFrame
) -> None:
    """Feature row at t must be identical whether or not the future exists."""
    for t in (80, 150, 299):
        truncated = feat.compute_features(sample_bars.iloc[:t], base_config)
        for col in feature_frame.columns:  # ALL columns, discovered not hardcoded
            full_val = feature_frame[col].iloc[t - 1]
            trunc_val = truncated[col].iloc[-1]
            both_nan = np.isnan(full_val) and np.isnan(trunc_val)
            assert both_nan or full_val == pytest.approx(trunc_val, rel=1e-12), (
                f"{col} at t={t - 1} changed when future data was removed"
            )


def test_shift_applied_one_bar(sample_bars: pd.DataFrame, feature_frame: pd.DataFrame) -> None:
    """log_ret_1 at T is the raw return at T-1 (uses close[T-2..T-1] only)."""
    close = sample_bars["close"]
    for k in (100, 200):
        expected = np.log(close.iloc[k - 1]) - np.log(close.iloc[k - 2])
        assert feature_frame["log_ret_1"].iloc[k] == pytest.approx(expected, rel=1e-12)


# --- NaN policy and purity ----------------------------------------------------


def test_warmup_rows_nan_and_tail_complete(feature_frame: pd.DataFrame) -> None:
    assert feature_frame.iloc[0].isna().all()  # shift makes row 0 all-NaN
    assert feature_frame.iloc[-1].notna().all()  # past warmup everything is filled


def test_unexpected_nan_raises(sample_bars: pd.DataFrame, base_config: dict[str, Any]) -> None:
    corrupt = sample_bars.copy()
    corrupt.iloc[150, corrupt.columns.get_loc("close")] = np.nan
    with pytest.raises(DataValidationError, match="non-finite feature"):
        feat.compute_features(corrupt, base_config)


def test_input_not_mutated(sample_bars: pd.DataFrame, base_config: dict[str, Any]) -> None:
    snapshot = sample_bars.copy(deep=True)
    feat.compute_features(sample_bars, base_config)
    pd.testing.assert_frame_equal(sample_bars, snapshot)


def test_missing_input_column_raises(
    sample_bars: pd.DataFrame, base_config: dict[str, Any]
) -> None:
    with pytest.raises(ConfigError, match="volume"):
        feat.compute_features(sample_bars.drop(columns=["volume"]), base_config)


# --- determinism of the contract ------------------------------------------------


def test_column_order_stable(feature_frame: pd.DataFrame) -> None:
    """This exact order becomes features.json — changing it breaks artifacts."""
    assert list(feature_frame.columns) == [
        "log_ret_1",
        "log_ret_5",
        "log_ret_20",
        "vol_short",
        "vol_long",
        "parkinson_vol",
        "garman_klass_vol",
        "sma_ratio",
        "ema_ratio",
        "ma_ratio_fast_slow",
        "macd_hist_pct",
        "zscore",
        "bb_pctb",
        "bb_bandwidth",
        "cci",
        "rsi",
        "stoch_k",
        "stoch_d",
        "roc",
        "mfi",
        "volume_ratio",
        "volume_zscore",
        "range_pct",
        "close_in_range",
    ]


def test_windows_come_from_config(
    sample_bars: pd.DataFrame, base_config: dict[str, Any], feature_frame: pd.DataFrame
) -> None:
    modified = copy.deepcopy(base_config)
    modified["signals"]["features"]["rsi_window"] = 5
    other = feat.compute_features(sample_bars, modified)
    assert not np.allclose(feature_frame["rsi"].iloc[-30:], other["rsi"].iloc[-30:], equal_nan=True)


# --- port protocol step 2a: hand-computed fixtures ------------------------------


def test_rsi_hand_computed() -> None:
    """Wilder RSI, window=2, close=[100,102,101,103].
    diffs=[_,+2,-1,+2] -> gains=[_,2,0,2], losses=[_,0,1,0]
    ewm(alpha=.5, adjust=False): avg_gain=[_,2,1,1.5], avg_loss=[_,0,.5,.25]
    t2: RS=1/.5=2   -> RSI=100-100/3 = 66.6667
    t3: RS=1.5/.25=6 -> RSI=100-100/7 = 85.7143
    """
    out = feat.rsi(pd.Series([100.0, 102.0, 101.0, 103.0]), window=2)
    assert np.isnan(out.iloc[0]) and np.isnan(out.iloc[1])
    assert out.iloc[2] == pytest.approx(200.0 / 3.0)
    assert out.iloc[3] == pytest.approx(600.0 / 7.0)


def test_stoch_hand_computed() -> None:
    """%K window=2, smooth=1.
    t1: hh=103, ll=99  -> 100*(102-99)/4 = 75
    t2: hh=103, ll=100 -> 100*(101-100)/3 = 33.3333
    t3: hh=104, ll=100 -> 100*(103-100)/4 = 75
    """
    high = pd.Series([101.0, 103.0, 102.0, 104.0])
    low = pd.Series([99.0, 101.0, 100.0, 102.0])
    close = pd.Series([100.0, 102.0, 101.0, 103.0])
    k, d = feat.stochastic(high, low, close, window=2, smooth=1)
    assert np.isnan(k.iloc[0])
    assert k.iloc[1] == pytest.approx(75.0)
    assert k.iloc[2] == pytest.approx(100.0 / 3.0)
    assert k.iloc[3] == pytest.approx(75.0)
    pd.testing.assert_series_equal(d, k)  # smooth=1 is the identity


# --- port protocol step 2b: reference library cross-check -----------------------


@pytest.mark.skipif(not _HAS_TA, reason="ta reference library not importable")
def test_indicator_values_match_reference_library(sample_bars: pd.DataFrame) -> None:
    """Ours vs the ta library on identical input, last 50 of 300 bars.

    Exact-formula indicators must agree to 1e-9; EWM-seeded ones (rsi, atr,
    macd) differ only by the decayed seed after 250+ bars — 1e-4 is ample.
    """
    high, low = sample_bars["high"], sample_bars["low"]
    close, volume = sample_bars["close"], sample_bars["volume"].astype("float64")

    pctb, _ = feat.bollinger(close, 20, 2.0)
    _, _, macd_hist = feat.macd_lines(close, 12, 26, 9)
    stoch_k, _ = feat.stochastic(high, low, close, 14, 3)
    cases = {
        "roc": (100.0 * (close / close.shift(10) - 1.0), ta.momentum.roc(close, window=10), 1e-9),
        "stoch_k": (stoch_k, ta.momentum.stoch(high, low, close, window=14), 1e-9),
        "cci": (feat.cci(high, low, close, 20), ta.trend.cci(high, low, close, window=20), 1e-9),
        "mfi": (
            feat.mfi(high, low, close, volume, 14),
            ta.volume.money_flow_index(high, low, close, volume, window=14),
            1e-9,
        ),
        "bb_pctb": (pctb, ta.volatility.bollinger_pband(close, window=20, window_dev=2), 1e-9),
        "rsi": (feat.rsi(close, 14), ta.momentum.rsi(close, window=14), 1e-4),
        "atr": (
            feat.atr(high, low, close, 14),
            ta.volatility.average_true_range(high, low, close, window=14),
            1e-4,
        ),
        "macd_hist": (
            macd_hist,
            ta.trend.macd_diff(close, window_slow=26, window_fast=12, window_sign=9),
            1e-4,
        ),
    }
    for name, (ours, theirs, rtol) in cases.items():
        np.testing.assert_allclose(
            ours.iloc[-50:].to_numpy(),
            theirs.iloc[-50:].to_numpy(),
            rtol=rtol,
            atol=1e-8,
            err_msg=f"{name} disagrees with the ta reference implementation",
        )
