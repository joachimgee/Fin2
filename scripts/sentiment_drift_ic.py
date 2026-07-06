"""Does a FinBERT news-sentiment shock predict multi-day drift? (PEAD probe)

The orthogonal-signal test: every price/TA signal we tried hit an AUC ~0.51
wall (the most-arbitraged data on Earth). News sentiment is the one ORTHOGONAL
free source we hold (96k symbol-days). We used it as a diluted daily feature
(+0.006 AUC) and a risk veto — never as an EVENT-DRIFT signal.

This measures, cheaply and before any backtest:
  (1) cross-sectional IC of sentiment_T vs forward h-day return, and
  (2) mean forward h-day return by sentiment quintile (the direct PEAD view),
for h in {1,3,5,10}. Lag-correct (skip-1, PEAD convention): sentiment_T is
known at close T; the return is measured from close T+1 to close T+1+h, so the
announcement day's own move is never counted. If the IC is ~0 across horizons,
even our orthogonal data is exhausted on free daily bars — an honest stop.

Usage: python3 -m scripts.sentiment_drift_ic --universe config/universe_mech2020.yaml
"""

from __future__ import annotations

import argparse
import logging
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from src.backtest.ic import cross_sectional_ic, ic_summary
from src.data.storage import BarStorage
from src.monitoring.logging_setup import setup_logging
from src.shared.config import load_config, load_universe

log = logging.getLogger(__name__)

_HORIZONS = (1, 3, 5, 10)
_N_BUCKETS = 5


def build_frames(
    config: dict, universe: list[str], horizon: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Wide (date x symbol) sentiment and skip-1 forward h-day return frames.
    Return at date T = close_{T+1+h}/close_{T+1} - 1 (enter next close, exit h
    later) — uses only data strictly after T. No-news days are absent from the
    sentiment frame (NaN), so the IC is naturally conditioned on news days."""
    storage = BarStorage(Path(config["data"]["db_path"]))
    start = datetime(2015, 1, 1, tzinfo=UTC)
    end = datetime.now(tz=UTC)
    sentiment_cols: dict[str, pd.Series] = {}
    return_cols: dict[str, pd.Series] = {}
    for symbol in universe:
        bars = storage.get_bars(symbol, start, end)
        if bars.empty:
            continue
        close = bars.set_index(bars["timestamp"].dt.normalize())["close"].sort_index()
        fwd = close.shift(-(1 + horizon)) / close.shift(-1) - 1.0
        return_cols[symbol] = fwd
        sent = storage.get_daily_sentiment(symbol)
        if len(sent):
            s = sent.set_index(pd.to_datetime(sent["date"], utc=True))["score"].sort_index()
            sentiment_cols[symbol] = s
    factor = pd.DataFrame(sentiment_cols)
    fwd_ret = pd.DataFrame(return_cols)
    factor = factor.reindex(fwd_ret.index)  # align sentiment onto trading days
    return factor, fwd_ret


def bucket_drift(factor: pd.DataFrame, fwd_ret: pd.DataFrame, n_buckets: int) -> list[float]:
    """Mean forward return per sentiment quantile bucket, pooled over all event
    (date, symbol) cells. Monotone increase = PEAD-style drift with sentiment."""
    f = factor.to_numpy().ravel()
    r = fwd_ret.reindex_like(factor).to_numpy().ravel()
    mask = np.isfinite(f) & np.isfinite(r)
    f, r = f[mask], r[mask]
    if len(f) < n_buckets * 10:
        return []
    # rank-based buckets: robust to the ties that cluster FinBERT daily means
    # (value-quantile edges can collapse a middle bucket to empty)
    ranks = pd.Series(f).rank(method="first").to_numpy()
    idx = np.clip(((ranks - 1) / len(f) * n_buckets).astype(int), 0, n_buckets - 1)
    return [float(r[idx == b].mean()) for b in range(n_buckets)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--universe", default=None)
    args = parser.parse_args()

    config = load_config(Path(args.config))
    setup_logging(str(config["monitoring"]["log_level"]))
    universe = load_universe(args.universe) or list(config["strategy"]["universe"])

    for horizon in _HORIZONS:
        factor, fwd_ret = build_frames(config, universe, horizon)
        summary = ic_summary(cross_sectional_ic(factor, fwd_ret))
        buckets = bucket_drift(factor, fwd_ret, _N_BUCKETS)
        log.info(
            "sentiment_drift_ic",
            extra={
                "horizon_days": horizon,
                "mean_ic": round(summary["mean"], 5),
                "t_stat": round(summary["t_stat"], 2),
                "hit_rate": round(summary["hit_rate"], 3),
                "n_dates": int(summary["n_dates"]),
                "bucket_fwd_returns": [round(b, 5) for b in buckets],
                "bucket_spread_top_minus_bottom": (
                    round(buckets[-1] - buckets[0], 5) if buckets else None
                ),
            },
        )


if __name__ == "__main__":
    main()
