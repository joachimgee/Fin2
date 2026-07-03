"""DuckDB storage layer — single source of historical bars for backtest/training.

Polygon.io is contacted ONLY by the sync job (polygon_client.py); everything
downstream reads from DuckDB. Zero repeated API calls during backtests.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd

from src.data.models import Bar

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS bars (
    symbol      VARCHAR,
    timestamp   TIMESTAMPTZ,
    open        DOUBLE,
    high        DOUBLE,
    low         DOUBLE,
    close       DOUBLE,
    volume      BIGINT,
    vwap        DOUBLE,
    PRIMARY KEY (symbol, timestamp)
)
"""


class BarStorage:
    """Owns the DuckDB connection and the bars table.

    Inserts are idempotent by design (INSERT OR REPLACE on the primary key):
    a re-sync overwrites, never duplicates. Queries are parameterized only.
    """

    def __init__(self, db_path: Path) -> None:
        self._con = duckdb.connect(str(db_path))
        self._con.execute(_CREATE_TABLE)

    def insert_bars(self, bars: list[Bar]) -> int:
        if not bars:
            return 0
        rows = [
            (b.symbol, b.timestamp, b.open, b.high, b.low, b.close, b.volume, b.vwap) for b in bars
        ]
        self._con.executemany("INSERT OR REPLACE INTO bars VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
        return len(rows)

    def get_bars(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        return self._con.execute(
            "SELECT * FROM bars"
            " WHERE symbol = ? AND timestamp >= ? AND timestamp <= ?"
            " ORDER BY timestamp ASC",
            [symbol, start, end],
        ).df()

    def get_all_bars(self, start: datetime, end: datetime) -> pd.DataFrame:
        """All symbols in one frame — the screener's liquidity stage input."""
        return self._con.execute(
            "SELECT * FROM bars WHERE timestamp >= ? AND timestamp <= ?"
            " ORDER BY symbol ASC, timestamp ASC",
            [start, end],
        ).df()
