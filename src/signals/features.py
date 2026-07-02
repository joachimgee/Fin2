"""Feature engineering. THE lookahead-bias chokepoint of the codebase.

Design: every indicator is computed RAW at time t (using data up to t) by a
pure helper, then compute_features() applies ONE .shift(1) to the whole
frame — a single chokepoint instead of 24 scattered shifts, so forgetting one
is structurally impossible. tests/test_features.py proves it per column via
truncation invariance.

All features are stationary transforms (returns, ratios, z-scores, bounded
oscillators) — never raw prices (src/signals/CLAUDE.md <stationarity>).

Indicator formulas re-derived per the port protocol (docs/plan/CODING_PLAN.md);
inspiration inventory from Fin v1 feature_engine.py, WITHOUT its defects
(no NaN->0.0 fills, no unshifted features, no input mutation).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.shared.exceptions import ConfigError, DataValidationError

_REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}


# --- pure indicator helpers (raw, unshifted — compute_features shifts) -------


def rsi(close: pd.Series, window: int) -> pd.Series:
    """Wilder RSI: EMA(alpha=1/window) of gains/losses (matches the ta lib)."""
    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = losses.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    return 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)


def macd_lines(
    close: pd.Series, fast: int, slow: int, signal: int
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Raw MACD line, signal line, histogram (price-scaled — normalize before use)."""
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return macd, sig, macd - sig


def bollinger(close: pd.Series, window: int, n_std: float) -> tuple[pd.Series, pd.Series]:
    """(%B, bandwidth). ddof=0 to match the reference implementation."""
    sma = close.rolling(window).mean()
    std = close.rolling(window).std(ddof=0)
    upper = sma + n_std * std
    lower = sma - n_std * std
    pctb = (close - lower) / (upper - lower)
    bandwidth = (upper - lower) / sma
    return pctb, bandwidth


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    """Wilder ATR: EMA(alpha=1/window) of true range."""
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(
        axis=1
    )
    return tr.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def stochastic(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int, smooth: int
) -> tuple[pd.Series, pd.Series]:
    """(%K, %D). Zero-range windows (all prices equal) are defined as 0.5 mid-range."""
    lowest = low.rolling(window).min()
    highest = high.rolling(window).max()
    span = highest - lowest
    k = (100.0 * (close - lowest) / span.where(span != 0, 1.0)).where(span != 0, 50.0)
    d = k.rolling(smooth).mean()
    return k, d


def cci(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    """Commodity Channel Index: (TP - SMA(TP)) / (0.015 * mean |TP - SMA|)."""
    tp = (high + low + close) / 3.0
    sma_tp = tp.rolling(window).mean()
    mean_dev = tp.rolling(window).apply(lambda x: float(np.mean(np.abs(x - x.mean()))), raw=True)
    return (tp - sma_tp) / (0.015 * mean_dev)


def mfi(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int
) -> pd.Series:
    """Money Flow Index in [0, 100]. All-positive-flow windows are 100 by definition."""
    tp = (high + low + close) / 3.0
    flow = tp * volume
    direction = tp.diff()
    pos = flow.where(direction > 0, 0.0).rolling(window).sum()
    neg = flow.where(direction < 0, 0.0).rolling(window).sum()
    ratio = pos / neg.where(neg != 0, np.nan)
    out = 100.0 - 100.0 / (1.0 + ratio)
    return out.where(neg != 0, 100.0).where(pos.notna() & neg.notna())


def parkinson_vol(high: pd.Series, low: pd.Series, window: int) -> pd.Series:
    """Parkinson range-based volatility estimator (un-annualized)."""
    hl_sq = np.log(high / low) ** 2
    return np.sqrt(hl_sq.rolling(window).mean() / (4.0 * np.log(2.0)))


def garman_klass_vol(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, window: int
) -> pd.Series:
    """Garman-Klass range-based volatility estimator (un-annualized)."""
    term = 0.5 * np.log(high / low) ** 2 - (2.0 * np.log(2.0) - 1.0) * np.log(close / open_) ** 2
    return np.sqrt(term.rolling(window).mean().clip(lower=0.0))


# --- feature blocks -----------------------------------------------------------


def _return_features(df: pd.DataFrame, p: dict[str, Any]) -> dict[str, pd.Series]:
    log_close = np.log(df["close"])
    out = {f"log_ret_{w}": log_close.diff(w) for w in p["ret_windows"]}
    ret1 = log_close.diff()
    out["vol_short"] = ret1.rolling(p["vol_short_window"]).std(ddof=0)
    out["vol_long"] = ret1.rolling(p["vol_long_window"]).std(ddof=0)
    out["parkinson_vol"] = parkinson_vol(df["high"], df["low"], p["parkinson_window"])
    out["garman_klass_vol"] = garman_klass_vol(
        df["open"], df["high"], df["low"], df["close"], p["gk_window"]
    )
    return out


def _trend_features(df: pd.DataFrame, p: dict[str, Any]) -> dict[str, pd.Series]:
    close = df["close"]
    sma = close.rolling(p["sma_window"]).mean()
    ema = close.ewm(span=p["ema_window"], adjust=False, min_periods=p["ema_window"]).mean()
    ma_fast = close.rolling(p["ma_fast_window"]).mean()
    ma_slow = close.rolling(p["ma_slow_window"]).mean()
    _, _, hist = macd_lines(close, p["macd_fast"], p["macd_slow"], p["macd_signal"])
    return {
        "sma_ratio": close / sma,
        "ema_ratio": close / ema,
        "ma_ratio_fast_slow": ma_fast / ma_slow,
        "macd_hist_pct": hist / close,  # normalized: raw MACD is price-scaled
    }


def _oscillator_features(df: pd.DataFrame, p: dict[str, Any]) -> dict[str, pd.Series]:
    close = df["close"]
    sma = close.rolling(p["zscore_window"]).mean()
    std = close.rolling(p["zscore_window"]).std(ddof=0)
    pctb, bandwidth = bollinger(close, p["bb_window"], p["bb_std"])
    k, d = stochastic(df["high"], df["low"], close, p["stoch_window"], p["stoch_smooth"])
    return {
        "zscore": (close - sma) / std,
        "bb_pctb": pctb,
        "bb_bandwidth": bandwidth,
        "cci": cci(df["high"], df["low"], close, p["cci_window"]),
        "rsi": rsi(close, p["rsi_window"]),
        "stoch_k": k,
        "stoch_d": d,
        "roc": 100.0 * (close / close.shift(p["roc_window"]) - 1.0),
        "mfi": mfi(df["high"], df["low"], close, df["volume"], p["mfi_window"]),
    }


def _volume_and_range_features(df: pd.DataFrame, p: dict[str, Any]) -> dict[str, pd.Series]:
    volume = df["volume"].astype("float64")
    vol_sma = volume.rolling(p["volume_window"]).mean()
    vol_std = volume.rolling(p["volume_window"]).std(ddof=0)
    span = df["high"] - df["low"]
    # zero-range bar (open=high=low=close): close position defined as 0.5 mid
    close_in_range = ((df["close"] - df["low"]) / span.where(span != 0, 1.0)).where(span != 0, 0.5)
    return {
        "volume_ratio": volume / vol_sma,
        "volume_zscore": (volume - vol_sma) / vol_std,
        "range_pct": span / df["close"],
        "close_in_range": close_in_range,
    }


# --- assembly ------------------------------------------------------------------


def compute_features(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Return the shifted, lookahead-safe feature matrix from an OHLCV frame.

    Column order is deterministic (it becomes features.json). Warmup rows stay
    NaN; any NaN/Inf appearing AFTER warmup raises DataValidationError — never
    silently filled. The input frame is never mutated.
    """
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ConfigError(f"compute_features: input missing columns {sorted(missing)}")
    try:
        p: dict[str, Any] = config["signals"]["features"]
    except KeyError as exc:
        raise ConfigError("compute_features: config lacks signals.features section") from exc

    raw: dict[str, pd.Series] = {}
    raw.update(_return_features(df, p))
    raw.update(_trend_features(df, p))
    raw.update(_oscillator_features(df, p))
    raw.update(_volume_and_range_features(df, p))

    # lookahead-safe: SINGLE shift chokepoint — feature value at T is the raw
    # indicator at T-1, so every column only sees data <= T-1.
    features = pd.DataFrame(raw, index=df.index).shift(1)
    _reject_post_warmup_gaps(features)
    return features


def _reject_post_warmup_gaps(features: pd.DataFrame) -> None:
    """NaN/Inf after the warmup period means corrupt input — fail fast."""
    bad = ~np.isfinite(features.to_numpy(dtype="float64", na_value=np.nan))
    complete = ~np.isnan(features.to_numpy(dtype="float64", na_value=np.nan)).any(axis=1)
    if not complete.any():
        return  # frame shorter than warmup — all rows are legitimate warmup
    first_complete = int(np.argmax(complete))
    if bad[first_complete:].any():
        rows, cols = np.where(bad[first_complete:])
        col = features.columns[cols[0]]
        ts = features.index[first_complete + rows[0]]
        raise DataValidationError(f"non-finite feature after warmup: {col}@{ts}")
