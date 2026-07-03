"""Polygon -> DuckDB data sync (make data-sync). The only step that contacts
Polygon; everything downstream reads DuckDB."""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path

from src.data.polygon_client import PolygonClient
from src.data.storage import BarStorage
from src.monitoring.logging_setup import setup_logging
from src.shared.config import load_config, require_env

log = logging.getLogger(__name__)


async def sync(args: argparse.Namespace) -> None:
    config = load_config(Path(args.config))
    setup_logging(str(config["monitoring"]["log_level"]))
    client = PolygonClient(require_env("POLYGON_API_KEY"))
    storage = BarStorage(Path(config["data"]["db_path"]))
    start = datetime.fromisoformat(args.start).replace(tzinfo=UTC)
    end = datetime.fromisoformat(args.end).replace(tzinfo=UTC)
    for symbol in config["strategy"]["universe"]:
        bars = await client.fetch_bars(symbol, start, end, str(config["data"]["timeframe"]))
        rows = storage.insert_bars(bars)
        log.info(
            "data_synced",
            extra={"symbol": symbol, "rows": rows, "start": args.start, "end": args.end},
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default=f"{datetime.now(tz=UTC):%Y-%m-%d}")
    parser.add_argument("--config", default="config/base.yaml")
    asyncio.run(sync(parser.parse_args()))


if __name__ == "__main__":
    main()
