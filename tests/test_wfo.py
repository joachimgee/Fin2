"""Phase 6 — WFO tests (docs/plan/PHASE_6_BACKTEST_WFO.md)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml
from src.backtest.wfo import make_windows, run_wfo
from src.shared.exceptions import ConfigError


def _bars(n: int) -> pd.DataFrame:
    return pd.DataFrame({"i": range(n)})


def _evaluator(oos_total_return: float) -> Any:
    def evaluate(bars: pd.DataFrame, params: dict[str, Any]) -> dict[str, float]:
        if len(bars) == 300:  # IS window
            return {"total_return": 0.03, "sharpe": 1.0, "max_drawdown": 0.1, "n_trades": 100}
        return {
            "total_return": oos_total_return,
            "sharpe": 2.0,
            "max_drawdown": 0.1,
            "n_trades": 50,
        }

    return evaluate


def test_windows_are_temporal_no_overlap() -> None:
    windows = make_windows(800, 5, 3)
    assert len(windows) == 5
    for is_slice, oos_slice in windows:
        assert is_slice.stop == oos_slice.start  # IS ends exactly where OOS starts
        assert is_slice.stop - is_slice.start == 300
        assert oos_slice.stop - oos_slice.start == 100
    oos_starts = [o.start for _, o in windows]
    assert oos_starts == [300, 400, 500, 600, 700]  # contiguous, never overlapping
    assert windows[-1][1].stop <= 800


def test_not_enough_bars_raises() -> None:
    with pytest.raises(ConfigError, match="not enough bars"):
        make_windows(5, 5, 3)


def test_optimization_never_sees_oos(base_config: dict[str, Any], tmp_path: Path) -> None:
    seen: list[tuple[int, int]] = []

    def optimize(is_bars: pd.DataFrame) -> dict[str, Any]:
        seen.append((int(is_bars["i"].min()), int(is_bars["i"].max())))
        return {}

    run_wfo(_bars(800), base_config, optimize, _evaluator(0.006), "s", tmp_path)
    for k, (low, high) in enumerate(seen):
        assert low == k * 100
        assert high == k * 100 + 299  # strictly before this window's OOS start


def test_all_gates_pass_sets_cleared(base_config: dict[str, Any], tmp_path: Path) -> None:
    # wfe = (0.006/100) / (0.03/300) = 0.6 >= 0.5; trades 5x50=250; sharpe 2; dd .1
    results = run_wfo(_bars(800), base_config, lambda df: {}, _evaluator(0.006), "s", tmp_path)
    assert results["cleared_for_paper"] is True
    assert results["gates"]["wfe"]["value"] == pytest.approx(0.6)
    assert results["gates"]["oos_trades"]["value"] == 250


def test_wfe_below_min_not_cleared(base_config: dict[str, Any], tmp_path: Path) -> None:
    # wfe = (0.001/100) / (0.03/300) = 0.1 < 0.5 -> abandon, no re-tuning loop
    results = run_wfo(_bars(800), base_config, lambda df: {}, _evaluator(0.001), "s", tmp_path)
    assert results["cleared_for_paper"] is False
    assert results["gates"]["wfe"]["passed"] is False


def test_results_yaml_written_with_gates(base_config: dict[str, Any], tmp_path: Path) -> None:
    run_wfo(_bars(800), base_config, lambda df: {}, _evaluator(0.006), "momentum", tmp_path)
    files = list(tmp_path.glob("momentum_*.yaml"))
    assert len(files) == 1
    loaded = yaml.safe_load(files[0].read_text(encoding="utf-8"))
    assert loaded["cleared_for_paper"] is True
    assert loaded["data_source"] == "polygon"
    assert set(loaded["gates"]) == {
        "windows",
        "wfe",
        "oos_trades",
        "oos_sharpe",
        "oos_max_drawdown",
    }
