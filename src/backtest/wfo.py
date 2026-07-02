"""Walk-Forward Optimization — the mandatory gate before ANY paper trading.

Go/no-go gates (root CLAUDE.md <trading_rules>, thresholds from YAML):
  - >= 5 rolling windows (optimize in-sample, evaluate out-of-sample)
  - WFE (walk-forward efficiency = OOS perf / IS perf) >= 0.50
  - >= 200 OOS trades
  - OOS Sharpe >= 1.5, OOS max drawdown <= 20%
If WFE < 0.50: the strategy is ABANDONED — no further tuning. More tuning on
the same data is how overfitting is manufactured.

Output: backtest_results/{strategy}_{ts}.yaml with cleared_for_paper: true/false.
The paper-trade CLI refuses to start any strategy without cleared_for_paper.
"""

from __future__ import annotations

from typing import Any


def run_wfo(strategy_name: str, config: dict[str, Any]) -> dict[str, Any]:
    """TODO(Phase 6): rolling windows via TimeSeriesSplit semantics (never KFold —
    shuffling destroys temporal ordering), Optuna on IS only, metrics on OOS only,
    write the results YAML and return it."""
    raise NotImplementedError("Phase 6 — walk-forward optimization")
