"""DuckDB storage layer — single source of historical bars for backtest/training.

Polygon.io is contacted ONLY by the sync job (polygon_client.py); everything
downstream reads from DuckDB. Zero repeated API calls during backtests.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from src.data.models import Bar


class BarStorage:
    """Owns the DuckDB connection and the bars table.

    TODO(Phase 1):
      - __init__(db_path: Path): connect, CREATE TABLE IF NOT EXISTS bars
        (symbol, timestamp TIMESTAMPTZ, o/h/l/c DOUBLE, volume BIGINT, vwap DOUBLE,
         PRIMARY KEY (symbol, timestamp))
      - insert_bars(bars: list[Bar]) -> int      # INSERT OR REPLACE, returns count
      - get_bars(symbol, start, end) -> pd.DataFrame  # sorted by timestamp ASC
      - Use polars only if a query result exceeds ~500k rows (root CLAUDE.md).
    """

    def __init__(self, db_path: Path) -> None:
        raise NotImplementedError("Phase 1 — DuckDB storage")

    def insert_bars(self, bars: list[Bar]) -> int:
        raise NotImplementedError("Phase 1")

    def get_bars(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        raise NotImplementedError("Phase 1")
