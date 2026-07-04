"""WFO runner (make wfo S=<strategy>): TRUE walk-forward — the model is
retrained inside every window on its in-sample span only, then evaluated
once on the untouched out-of-sample span. Exit code 0 only when
cleared_for_paper (cron/CI-usable).

Multi-symbol correct: windows are sliced over TRADING DAYS (unique
timestamps), not rows, so every symbol contributes evenly; each window gets
a warmup lead-in of the preceding warmup_bars days (strictly past data).
The HMM regime detector is not used inside the WFO: per-window regime
models on ~180 bars are unstable, and hostile_regimes gates entries only.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import UTC, datetime
from itertools import count
from pathlib import Path
from typing import Any

import pandas as pd
from src.backtest.engine import BacktestEngine, SimulatedBroker
from src.backtest.wfo import run_wfo
from src.data.storage import BarStorage
from src.monitoring.logging_setup import setup_logging
from src.risk.circuit_breaker import CircuitBreaker
from src.risk.exposure_tracker import ExposureTracker
from src.risk.manager import RiskManager
from src.shared.config import load_config
from src.shared.exceptions import ConfigError
from src.signals.features import compute_features
from src.signals.lgbm_signal import LightGBMSignalGenerator
from src.signals.zscore_signal import ZScoreSignalGenerator
from src.strategies.base import AbstractStrategy
from src.strategies.mean_reversion import MeanReversionZScore
from src.strategies.momentum_lightgbm import MomentumLightGBM

from scripts.train_lgbm import train_lgbm_artifacts

log = logging.getLogger(__name__)

STRATEGIES = ("momentum_lightgbm", "mean_reversion")


def make_window_runners(
    config: dict[str, Any], all_bars: pd.DataFrame, work_dir: Path, strategy_name: str
) -> tuple[Any, Any]:
    if strategy_name not in STRATEGIES:
        raise ConfigError(f"unknown strategy {strategy_name!r} — one of {STRATEGIES}")
    warmup = int(config["strategy"]["warmup_bars"])
    day_list = sorted(all_bars["timestamp"].unique())
    day_position = {day: i for i, day in enumerate(day_list)}
    window_ids = count()

    def features_fn(frame: pd.DataFrame) -> pd.DataFrame:
        return compute_features(frame, config)

    def bars_with_lead_in(days: pd.DataFrame) -> tuple[pd.DataFrame, Any]:
        """All symbols' bars for the day span, prepended with the warmup_bars
        PRECEDING days (strictly older than the span — point-in-time safe)."""
        span_start, span_end = days["timestamp"].iloc[0], days["timestamp"].iloc[-1]
        lead_start = day_list[max(0, day_position[span_start] - warmup)]
        mask = (all_bars["timestamp"] >= lead_start) & (all_bars["timestamp"] <= span_end)
        return all_bars[mask].sort_values("timestamp"), span_start

    def optimize_window(is_days: pd.DataFrame) -> dict[str, Any]:
        """TRUE WFO step: retrain the model on THIS window's IS bars only.
        mean_reversion is rule-based — nothing is fitted, by design: its WFE
        is then a pure robustness check of fixed YAML parameters."""
        if strategy_name == "mean_reversion":
            return {}
        window_bars, train_start = bars_with_lead_in(is_days)
        artifact_dir = work_dir / f"window_{next(window_ids)}"
        train_lgbm_artifacts(window_bars, config, artifact_dir, train_start=train_start)
        return {"artifact_dir": artifact_dir}

    def build_strategy(params: dict[str, Any]) -> AbstractStrategy:
        if strategy_name == "mean_reversion":
            clip = float(config["strategy"]["mean_reversion"]["zscore_clip"])
            return MeanReversionZScore(config, ZScoreSignalGenerator(clip), features_fn)
        return MomentumLightGBM(
            config, LightGBMSignalGenerator(params["artifact_dir"]), features_fn
        )

    def evaluate_window(days: pd.DataFrame, params: dict[str, Any]) -> dict[str, float]:
        run_bars, trade_start = bars_with_lead_in(days)
        tracker = ExposureTracker()
        breaker = CircuitBreaker(config["risk"]["circuit_breakers"], on_trip=lambda r, v: None)
        risk = RiskManager(config, tracker, breaker, dict(config["strategy"]["stats"]))
        strategy = build_strategy(params)
        engine = BacktestEngine(
            strategy,
            risk,
            tracker,
            SimulatedBroker(config),
            run_bars,
            config,
            trade_start=trade_start,
        )
        results = asyncio.run(engine.run())
        return {
            key: float(results[key])
            for key in ("total_return", "sharpe", "max_drawdown", "n_trades")
        }

    return optimize_window, evaluate_window


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default=f"{datetime.now(tz=UTC):%Y-%m-%d}")
    parser.add_argument("--output-dir", default="backtest_results")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    setup_logging(str(config["monitoring"]["log_level"]))
    storage = BarStorage(Path(config["data"]["db_path"]))
    start = datetime.fromisoformat(args.start).replace(tzinfo=UTC)
    end = datetime.fromisoformat(args.end).replace(tzinfo=UTC)
    all_bars = pd.concat(
        [storage.get_bars(symbol, start, end) for symbol in config["strategy"]["universe"]]
    ).sort_values("timestamp")
    log.info(
        "wfo_data_loaded",
        extra={
            "symbols": int(all_bars["symbol"].nunique()),
            "days": int(all_bars["timestamp"].nunique()),
            "rows": len(all_bars),
        },
    )

    # windows are sliced over trading DAYS — one row per unique timestamp
    days = pd.DataFrame({"timestamp": sorted(all_bars["timestamp"].unique())})
    work_dir = Path(args.output_dir) / f"wfo_models_{datetime.now(tz=UTC):%Y%m%d_%H%M%S}"
    optimize_window, evaluate_window = make_window_runners(
        config, all_bars, work_dir, args.strategy
    )
    results = run_wfo(
        days, config, optimize_window, evaluate_window, args.strategy, Path(args.output_dir)
    )
    for name, gate in results["gates"].items():
        log.info(
            "wfo_gate",
            extra={
                "gate": name,
                "value": gate["value"],
                "threshold": gate["threshold"],
                "passed": gate["passed"],
            },
        )
    log.info("wfo_verdict", extra={"cleared_for_paper": results["cleared_for_paper"]})
    raise SystemExit(0 if results["cleared_for_paper"] else 1)


if __name__ == "__main__":
    main()
