"""Short-leg premium probe: long-short reversal Sharpe, net of costs.

Builds the canonical short-term reversal paper portfolio on the real universe
WITHOUT touching the engine/risk/strategy: on a non-overlapping `lookback`-day
grid, long the biggest losers and short the biggest winners of the past period,
hold one period, net of transaction + borrow costs. Reports the long-short
Sharpe (bootstrap CI) vs the long-only leg, swept over borrow assumptions.

Decides whether the (large) build of real short selling into the engine is
justified: if long-short clears ~1.5 net, build it; if not, build avoided.

Usage: python3 -m scripts.long_short_sharpe --universe config/universe_mech2020.yaml
"""

from __future__ import annotations

import argparse
import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from src.backtest.bootstrap import bootstrap_sharpe
from src.backtest.long_short import long_short_periodic_returns
from src.backtest.metrics import sharpe_ratio
from src.data.storage import BarStorage
from src.monitoring.logging_setup import setup_logging
from src.shared.config import load_config, load_universe

log = logging.getLogger(__name__)

_BORROW_SCENARIOS_BPS = (0.0, 50.0, 100.0, 300.0)  # liquid large-cap .. hard-to-borrow


def wide_close(config: dict, universe: list[str]) -> pd.DataFrame:
    storage = BarStorage(Path(config["data"]["db_path"]))
    start = datetime(2015, 1, 1, tzinfo=UTC)
    end = datetime.now(tz=UTC)
    cols = {}
    for symbol in universe:
        bars = storage.get_bars(symbol, start, end)
        if not bars.empty:
            cols[symbol] = bars.set_index(bars["timestamp"].dt.normalize())["close"].sort_index()
    return pd.DataFrame(cols).sort_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--universe", default=None)
    parser.add_argument("--lookback", type=int, default=5)  # weekly reversal
    parser.add_argument("--top-frac", type=float, default=0.1)  # deciles
    parser.add_argument("--round-trip-bps", type=float, default=20.0)
    args = parser.parse_args()

    config = load_config(Path(args.config))
    setup_logging(str(config["monitoring"]["log_level"]))
    universe = load_universe(args.universe) or list(config["strategy"]["universe"])
    ppy = int(config["backtest"]["periods_per_year"])
    ppy_grid = max(1, round(ppy / args.lookback))  # rebalance periods per year
    holding_years = args.lookback / ppy

    grid = wide_close(config, universe).iloc[:: args.lookback]
    period_ret = grid.pct_change()
    signal = period_ret  # past-period return; reversal longs the low tail
    forward = period_ret.shift(-1)  # next-period return (non-overlapping)

    # long-only leg: biggest losers' forward return, round-trip cost only
    ls_gross = long_short_periodic_returns(signal, forward, args.top_frac, 0.0, 0.0, 0.0)
    long_only = _long_leg(signal, forward, args.top_frac, args.round_trip_bps)
    log.info(
        "long_only_leg",
        extra={
            "point_sharpe": round(sharpe_ratio(long_only, ppy_grid), 3),
            "n_periods": int(long_only.notna().sum()),
        },
    )
    log.info("long_short_gross", extra={"point_sharpe": round(sharpe_ratio(ls_gross, ppy_grid), 3)})

    for borrow in _BORROW_SCENARIOS_BPS:
        net = long_short_periodic_returns(
            signal, forward, args.top_frac, args.round_trip_bps, borrow, holding_years
        )
        interval = bootstrap_sharpe(
            net, ppy_grid, threshold=float(config["wfo"]["min_oos_sharpe"]), mean_block=4
        )
        log.info(
            "long_short_net",
            extra={
                "borrow_bps_annual": borrow,
                "round_trip_bps": args.round_trip_bps,
                "point": round(interval.point, 3),
                "p05": round(interval.percentiles[5], 3),
                "p50": round(interval.percentiles[50], 3),
                "p95": round(interval.percentiles[95], 3),
                "p_at_least_1_5": round(interval.p_at_least, 3),
                "n_periods": interval.n_returns,
            },
        )


def _long_leg(
    signal: pd.DataFrame, forward: pd.DataFrame, top_frac: float, round_trip_bps: float
) -> pd.Series:
    aligned = signal.reindex_like(forward)
    cost = round_trip_bps / 1e4
    out: dict[pd.Timestamp, float] = {}
    for date in forward.index:
        s = aligned.loc[date]
        r = forward.loc[date]
        mask = s.notna() & r.notna()
        s, r = s[mask], r[mask]
        n = int(top_frac * len(s))
        if n < 1:
            continue
        losers = r.to_numpy()[s.to_numpy().argsort()[:n]]
        out[date] = float(losers.mean()) - cost
    return pd.Series(out)


if __name__ == "__main__":
    main()
