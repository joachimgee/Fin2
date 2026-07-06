"""Bootstrap confidence interval for a strategy's OOS Sharpe (Phase A).

Runs ONE backtest of a rule-based reversion strategy over the full span, then
stationary-bootstraps the realized daily returns to put a confidence band
around the annualized Sharpe. Answers the question the point WFO Sharpe leaves
open: is the measured ~1.1 plateau reliably below the min_oos_sharpe gate, or
is the gate inside the sampling noise?

Faithful for the RULE-BASED families (mean_reversion*, xsec*): they fit
nothing, so a single full-span backtest reproduces exactly what the WFO
evaluates window by window. Not meaningful for momentum_lightgbm (per-window
models) — rejected here.

Usage: python3 -m scripts.bootstrap_sharpe --strategy xsec_reversion_sentiment
       --start 2020-11-01 --universe config/universe_mech2020.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from src.backtest.bootstrap import bootstrap_sharpe
from src.backtest.engine import BacktestEngine, SimulatedBroker
from src.data.storage import BarStorage
from src.monitoring.logging_setup import setup_logging
from src.risk.circuit_breaker import CircuitBreaker
from src.risk.exposure_tracker import ExposureTracker
from src.risk.manager import RiskManager
from src.shared.config import load_config, load_universe
from src.shared.exceptions import ConfigError
from src.signals.features import compute_features

from scripts.run_backtest import build_reversion_strategy

log = logging.getLogger(__name__)


async def realized_returns(config: dict, strategy_name: str, bars: pd.DataFrame) -> pd.Series:
    storage = BarStorage(Path(config["data"]["db_path"]))
    sentiment_by_symbol = None
    if strategy_name.endswith("_sentiment"):
        sentiment_by_symbol = {
            s: storage.get_daily_sentiment(s) for s in config["strategy"]["universe"]
        }
        missing = [s for s, f in sentiment_by_symbol.items() if not len(f)]
        if missing:
            raise ConfigError(f"no news_sentiment stored for {missing} — run scripts.score_news")

    def features_fn(frame: pd.DataFrame) -> pd.DataFrame:
        return compute_features(frame, config)

    strategy = build_reversion_strategy(config, strategy_name, features_fn, sentiment_by_symbol)
    tracker = ExposureTracker()
    breaker = CircuitBreaker(config["risk"]["circuit_breakers"], on_trip=lambda r, v: None)
    risk = RiskManager(config, tracker, breaker, dict(config["strategy"]["stats"]))
    engine = BacktestEngine(strategy, risk, tracker, SimulatedBroker(config), bars, config)
    results = await engine.run()
    return results["equity_curve"].pct_change().dropna()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default=f"{datetime.now(tz=UTC):%Y-%m-%d}")
    parser.add_argument("--universe", default=None)
    parser.add_argument("--n-resamples", type=int, default=5000)
    parser.add_argument("--mean-block", type=float, default=10.0)
    args = parser.parse_args()

    if args.strategy == "momentum_lightgbm":
        raise ConfigError("bootstrap_sharpe supports rule-based strategies only (see module doc)")
    config = load_config(Path(args.config))
    setup_logging(str(config["monitoring"]["log_level"]))
    override = load_universe(args.universe)
    if override is not None:
        config["strategy"]["universe"] = override
    storage = BarStorage(Path(config["data"]["db_path"]))
    start = datetime.fromisoformat(args.start).replace(tzinfo=UTC)
    end = datetime.fromisoformat(args.end).replace(tzinfo=UTC)
    bars = pd.concat(
        [storage.get_bars(s, start, end) for s in config["strategy"]["universe"]]
    ).sort_values("timestamp")

    returns = asyncio.run(realized_returns(config, args.strategy, bars))
    interval = bootstrap_sharpe(
        returns,
        int(config["backtest"]["periods_per_year"]),
        threshold=float(config["wfo"]["min_oos_sharpe"]),
        n_resamples=args.n_resamples,
        mean_block=args.mean_block,
    )
    log.info(
        "bootstrap_sharpe",
        extra={
            "strategy": args.strategy,
            "point": round(interval.point, 3),
            "p05": round(interval.percentiles[5], 3),
            "p50": round(interval.percentiles[50], 3),
            "p95": round(interval.percentiles[95], 3),
            "threshold": interval.threshold,
            "p_at_least_threshold": round(interval.p_at_least, 3),
            "n_returns": interval.n_returns,
        },
    )


if __name__ == "__main__":
    main()
