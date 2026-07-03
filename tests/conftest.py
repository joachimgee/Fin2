"""Shared pytest fixtures — grown phase by phase (see docs/plan/)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from src.shared.config import load_config

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def sample_bars() -> pd.DataFrame:
    """300 valid OHLCV rows, deterministic (seeded RNG), tz-aware UTC index."""
    rng = np.random.default_rng(42)
    n = 300
    returns = rng.normal(0.0005, 0.01, n)
    close = 100.0 * np.cumprod(1.0 + returns)
    open_ = np.concatenate(([100.0], close[:-1]))
    spread = np.abs(rng.normal(0.0, 0.002, n))
    high = np.maximum(open_, close) * (1.0 + spread)
    low = np.minimum(open_, close) * (1.0 - spread)
    volume = rng.integers(100_000, 5_000_000, n)
    index = pd.date_range("2024-01-02", periods=n, freq="B", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )


@pytest.fixture
def base_config() -> dict[str, Any]:
    return load_config(REPO_ROOT / "config" / "base.yaml")


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> Path:
    return tmp_path / "test.duckdb"
