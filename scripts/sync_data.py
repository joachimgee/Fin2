"""Market data -> DuckDB sync (make data-sync). The only step that contacts a
data provider; everything downstream reads DuckDB.

Sources: polygon (default, ~2y free history) or alpaca (IEX feed, ~2016+,
ADR-008 — fixed research universes only)."""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path

from src.data.alpaca_data import AlpacaDataClient
from src.data.polygon_client import PolygonClient
from src.data.storage import BarStorage
from src.monitoring.logging_setup import setup_logging
from src.shared.config import load_config, require_env

log = logging.getLogger(__name__)


async def sync(args: argparse.Namespace) -> None:
    config = load_config(Path(args.config))
    setup_logging(str(config["monitoring"]["log_level"]))
    storage = BarStorage(Path(config["data"]["db_path"]))
    start = datetime.fromisoformat(args.start).replace(tzinfo=UTC)
    end = datetime.fromisoformat(args.end).replace(tzinfo=UTC)
    universe = list(config["strategy"]["universe"])
    # Polygon free tier is 5 req/min; Alpaca data allows 200/min — no pause needed.
    pause_s = float(config["data"]["sync_pause_s"]) if args.source == "polygon" else 0.0
    polygon = PolygonClient(require_env("POLYGON_API_KEY")) if args.source == "polygon" else None
    alpaca = (
        AlpacaDataClient(require_env("ALPACA_API_KEY"), require_env("ALPACA_SECRET_KEY"))
        if args.source == "alpaca"
        else None
    )
    for i, symbol in enumerate(universe):
        if polygon is not None:
            bars = await polygon.fetch_bars(symbol, start, end, str(config["data"]["timeframe"]))
        else:
            assert alpaca is not None
            bars = await alpaca.fetch_bars(symbol, start, end)
        rows = storage.insert_bars(bars)
        log.info(
            "data_synced",
            extra={
                "symbol": symbol,
                "rows": rows,
                "start": args.start,
                "end": args.end,
                "source": args.source,
            },
        )
        if pause_s and i < len(universe) - 1:
            await asyncio.sleep(pause_s)  # free-tier rate limit courtesy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default=f"{datetime.now(tz=UTC):%Y-%m-%d}")
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--source", choices=["polygon", "alpaca"], default="polygon")
    asyncio.run(sync(parser.parse_args()))


if __name__ == "__main__":
    main()
