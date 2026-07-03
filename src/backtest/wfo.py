"""Walk-Forward Optimization — the mandatory gate before ANY paper trading.

Rolling windows: in-sample span = is_oos_ratio x the out-of-sample span,
stepping forward one OOS span per window. Parameters are chosen on IS ONLY;
each OOS segment is touched exactly once, with the params chosen before it.

WFE = mean(OOS return-per-bar) / mean(IS return-per-bar). If WFE < min_wfe
the strategy is ABANDONED — this function runs once and never loops "one
more optimization": more tuning on the same data is how overfitting is
manufactured.

Output: {output_dir}/{strategy}_{ts}.yaml with every gate value and
cleared_for_paper — the paper-trade entrypoint refuses to start without it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd
import yaml

from src.shared.exceptions import ConfigError

log = logging.getLogger(__name__)

# optimize(is_bars) -> params — sees IS data only, by construction
OptimizeWindow = Callable[[pd.DataFrame], dict[str, Any]]
# evaluate(bars, params) -> {"total_return", "sharpe", "max_drawdown", "n_trades"}
EvaluateWindow = Callable[[pd.DataFrame, dict[str, Any]], dict[str, float]]


def make_windows(n_bars: int, n_windows: int, is_oos_ratio: int) -> list[tuple[slice, slice]]:
    """Rolling (IS, OOS) index slices: temporal order, IS ends where its OOS
    starts, consecutive OOS segments are contiguous and never overlap."""
    oos_len = n_bars // (n_windows + is_oos_ratio)
    if oos_len < 1:
        raise ConfigError(f"not enough bars ({n_bars}) for {n_windows} WFO windows")
    is_len = is_oos_ratio * oos_len
    windows = []
    for i in range(n_windows):
        is_start = i * oos_len
        windows.append(
            (
                slice(is_start, is_start + is_len),
                slice(is_start + is_len, is_start + is_len + oos_len),
            )
        )
    return windows


def run_wfo(
    bars: pd.DataFrame,
    config: dict[str, Any],
    optimize_window: OptimizeWindow,
    evaluate_window: EvaluateWindow,
    strategy_name: str,
    output_dir: Path,
) -> dict[str, Any]:
    wfo_cfg = config["wfo"]
    windows = make_windows(len(bars), int(wfo_cfg["min_windows"]), int(wfo_cfg["is_oos_ratio"]))

    is_return_per_bar: list[float] = []
    oos_return_per_bar: list[float] = []
    oos_sharpes: list[float] = []
    oos_drawdowns: list[float] = []
    oos_trades = 0
    for is_slice, oos_slice in windows:
        is_bars, oos_bars = bars.iloc[is_slice], bars.iloc[oos_slice]
        params = optimize_window(is_bars)  # IS only — never shown the OOS segment
        metrics_is = evaluate_window(is_bars, params)
        metrics_oos = evaluate_window(oos_bars, params)
        is_return_per_bar.append(metrics_is["total_return"] / len(is_bars))
        oos_return_per_bar.append(metrics_oos["total_return"] / len(oos_bars))
        oos_sharpes.append(metrics_oos["sharpe"])
        oos_drawdowns.append(metrics_oos["max_drawdown"])
        oos_trades += int(metrics_oos["n_trades"])

    mean_is = mean(is_return_per_bar)
    wfe = mean(oos_return_per_bar) / mean_is if mean_is > 0 else 0.0
    gates = {
        "windows": _gate(len(windows), float(wfo_cfg["min_windows"]), at_least=True),
        "wfe": _gate(wfe, float(wfo_cfg["min_wfe"]), at_least=True),
        "oos_trades": _gate(oos_trades, float(wfo_cfg["min_oos_trades"]), at_least=True),
        "oos_sharpe": _gate(mean(oos_sharpes), float(wfo_cfg["min_oos_sharpe"]), at_least=True),
        "oos_max_drawdown": _gate(
            max(oos_drawdowns), float(wfo_cfg["max_oos_drawdown_pct"]), at_least=False
        ),
    }
    results = {
        "strategy": strategy_name,
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "wfe_metric": "return_per_bar",
        "gates": gates,
        "cleared_for_paper": all(g["passed"] for g in gates.values()),
        "data_source": config["data"]["data_source"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{strategy_name}_{datetime.now(tz=UTC):%Y%m%d_%H%M%S}.yaml"
    out_path.write_text(yaml.safe_dump(results, sort_keys=False), encoding="utf-8")
    log.info(
        "wfo_completed",
        extra={"strategy": strategy_name, "wfe": wfe, "cleared": results["cleared_for_paper"]},
    )
    return results


def _gate(value: float, threshold: float, at_least: bool) -> dict[str, Any]:
    passed = value >= threshold if at_least else value <= threshold
    return {"value": float(value), "threshold": threshold, "passed": bool(passed)}
