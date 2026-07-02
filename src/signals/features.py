"""Feature engineering. THE lookahead-bias chokepoint of the codebase.

Every feature at time T uses only data from times <= T-1: every rolling/derived
column ends with .shift(1) and carries a `# lookahead-safe:` comment.
Raw prices are never features — only stationary transforms (log returns,
z-scores, momentum ratios, ATR, RSI). See src/signals/CLAUDE.md.
"""

from __future__ import annotations

import pandas as pd


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return the model feature matrix from an OHLCV DataFrame.

    TODO(Phase 2): implement. Reference shape (every line shifted):
        out["log_ret_1"] = np.log(df["close"]).diff().shift(1)       # lookahead-safe: uses close[t-2..t-1]
        out["sma_ratio"] = (df["close"] / df["close"].rolling(20).mean()).shift(1)  # lookahead-safe
        out["rsi_14"]    = compute_rsi(df["close"], 14).shift(1)     # lookahead-safe
        out["vol_20"]    = np.log(df["close"]).diff().rolling(20).std().shift(1)    # lookahead-safe
    Column order here defines features.json order — training and live must match.
    """
    raise NotImplementedError("Phase 2 — feature engineering")
