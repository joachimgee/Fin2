"""WFO runner (make wfo S=<strategy>): real engine, real risk stack, gates.

Exit code 0 only when cleared_for_paper — usable directly in CI/cron.
The current strategy has no tunable parameters, so optimize_window returns
{} — the WFO structure (IS-only optimization) is already in place for when
parameters arrive.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import UTC, datetime
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
from src.signals.features import compute_features
from src.signals.lgbm_signal import LightGBMSignalGenerator
from src.signals.regime_hmm import RegimeDetector
from src.strategies.momentum_lightgbm import MomentumLightGBM

log = logging.getLogger(__name__)


def make_window_runners(
    config: dict[str, Any], artifact_dir: Path, hmm_dir: Path | None
) -> tuple[Any, Any]:
    def features_fn(frame: pd.DataFrame) -> pd.DataFrame:
        return compute_features(frame, config)

    def optimize_window(is_bars: pd.DataFrame) -> dict[str, Any]:
        return {}  # no tunable params yet — sees IS only by construction

    def evaluate_window(bars: pd.DataFrame, params: dict[str, Any]) -> dict[str, float]:
        tracker = ExposureTracker()
        breaker = CircuitBreaker(config["risk"]["circuit_breakers"], on_trip=lambda r, v: None)
        risk = RiskManager(config, tracker, breaker, dict(config["strategy"]["stats"]))
        strategy = MomentumLightGBM(
            config,
            LightGBMSignalGenerator(artifact_dir),
            features_fn,
            RegimeDetector(hmm_dir) if hmm_dir is not None else None,
        )
        engine = BacktestEngine(strategy, risk, tracker, SimulatedBroker(config), bars, config)
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
    parser.add_argument("--artifacts", default="models/latest")
    parser.add_argument("--hmm-artifacts", default=None)
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default=f"{datetime.now(tz=UTC):%Y-%m-%d}")
    parser.add_argument("--output-dir", default="backtest_results")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    setup_logging(str(config["monitoring"]["log_level"]))
    symbol = str(config["strategy"]["universe"][0])
    bars = BarStorage(Path(config["data"]["db_path"])).get_bars(
        symbol,
        datetime.fromisoformat(args.start).replace(tzinfo=UTC),
        datetime.fromisoformat(args.end).replace(tzinfo=UTC),
    )
    optimize_window, evaluate_window = make_window_runners(
        config, Path(args.artifacts), Path(args.hmm_artifacts) if args.hmm_artifacts else None
    )
    results = run_wfo(
        bars, config, optimize_window, evaluate_window, args.strategy, Path(args.output_dir)
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
