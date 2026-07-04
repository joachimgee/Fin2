"""Per-window trainer tests — synthetic bars, no network (true-WFO support)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scripts.train_lgbm import train_lgbm_artifacts
from src.signals.features import compute_features
from src.signals.lgbm_signal import LightGBMSignalGenerator


def _bars(symbols: list[str], days: int = 200, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2024-01-02", periods=days, freq="B", tz="UTC")
    rows: list[dict[str, Any]] = []
    for symbol in symbols:
        closes = 100.0 * np.cumprod(1.0 + rng.normal(0.0004, 0.01, days))
        rows.extend(
            {
                "symbol": symbol,
                "timestamp": timestamps[i],
                "open": closes[i] * (1 + rng.normal(0, 0.002)),
                "high": closes[i] * 1.012,
                "low": closes[i] * 0.988,
                "close": closes[i],
                "volume": int(rng.integers(1_000_000, 5_000_000)),
            }
            for i in range(days)
        )
    return pd.DataFrame(rows)


def test_artifacts_written_and_loadable(base_config: dict[str, Any], tmp_path: Path) -> None:
    out = train_lgbm_artifacts(_bars(["AAA", "BBB"]), base_config, tmp_path / "w0")
    for name in ("model.pkl", "scaler.pkl", "features.json", "metadata.yaml"):
        assert (out / name).exists()
    generator = LightGBMSignalGenerator(out)  # the exact production loading path
    features = compute_features(
        _bars(["AAA"]).set_index("timestamp").drop(columns="symbol"), base_config
    )
    signal = generator.generate(features.dropna())
    assert -1.0 <= signal <= 1.0


def test_train_start_excludes_lead_in_rows(base_config: dict[str, Any], tmp_path: Path) -> None:
    bars = _bars(["AAA"], days=220)
    cutoff = sorted(bars["timestamp"].unique())[100]
    out = train_lgbm_artifacts(bars, base_config, tmp_path / "w1", train_start=cutoff)
    metadata = (out / "metadata.yaml").read_text(encoding="utf-8")
    # the earliest training row must be at or after the cutoff, never in the lead-in
    assert str(cutoff.date()) in metadata.split("training_period")[1][:60]


def test_multi_symbol_rows_stacked_in_time_order(
    base_config: dict[str, Any], tmp_path: Path
) -> None:
    single = train_lgbm_artifacts(_bars(["AAA"]), base_config, tmp_path / "s")
    double = train_lgbm_artifacts(_bars(["AAA", "BBB"]), base_config, tmp_path / "d")
    import yaml

    rows_single = yaml.safe_load((single / "metadata.yaml").read_text())["metrics"][
        "n_training_rows"
    ]
    rows_double = yaml.safe_load((double / "metadata.yaml").read_text())["metrics"][
        "n_training_rows"
    ]
    assert rows_double == pytest.approx(2 * rows_single, abs=2)  # both symbols contribute
