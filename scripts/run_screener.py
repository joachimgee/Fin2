"""Universe screener runner (weekly cron): 12k Alpaca assets -> universe/{date}.yaml.

Flow: assets via AlpacaBrokerClient.list_assets -> optional grouped-daily
Polygon sync into DuckDB -> pure funnel (static -> liquidity -> rank) ->
point-in-time universe file. A symbol leaving the universe never force-closes
its position — the screener only stops NEW entries.

--dry-run exercises the full funnel on synthetic in-memory inputs (no
network, no DB writes) and logs the per-stage counts.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from src.data.polygon_client import PolygonClient
from src.data.screener import apply_liquidity_filters, apply_static_filters, rank_universe
from src.data.storage import BarStorage
from src.execution.broker import AlpacaBrokerClient
from src.monitoring.logging_setup import setup_logging
from src.shared.config import load_config, require_env

log = logging.getLogger(__name__)


def run_funnel(
    assets: list[dict[str, Any]],
    daily: pd.DataFrame,
    config: dict[str, Any],
    sector_map: dict[str, str],
    as_of: pd.Timestamp,
) -> list[str]:
    screener_cfg = config["screener"]
    static_ok = apply_static_filters(assets, screener_cfg)
    symbols = {a["symbol"] for a in static_ok}
    daily_in_scope = daily[daily["symbol"].isin(symbols)]
    candidates = apply_liquidity_filters(daily_in_scope, screener_cfg, as_of=as_of)
    universe = rank_universe(candidates, sector_map, screener_cfg)
    log.info(
        "screener_funnel_summary",
        extra={
            "assets": len(assets),
            "static": len(static_ok),
            "candidates": len(candidates),
            "universe": len(universe),
        },
    )
    return universe


def write_universe(universe: list[str], config: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_date = datetime.now(tz=UTC)
    out_path = output_dir / f"{run_date:%Y-%m-%d}.yaml"
    out_path.write_text(
        yaml.safe_dump(
            {
                "generated_at": run_date.isoformat(),
                "criteria": config["screener"],
                "symbols": universe,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    log.info("universe_written", extra={"path": str(out_path), "size": len(universe)})
    return out_path


def _synthetic_inputs(config: dict[str, Any]) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Deterministic dry-run fixtures: 40 symbols, half liquid, one OTC."""
    rng = np.random.default_rng(9)
    days = int(config["screener"]["min_history_days"]) + 10
    timestamps = pd.date_range("2024-01-02", periods=days, freq="B", tz="UTC")
    assets: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for i in range(40):
        symbol = f"SYN{i:02d}"
        exchange = "OTC" if i == 0 else "NYSE"
        assets.append(
            {"symbol": symbol, "exchange": exchange, "status": "active", "tradable": True}
        )
        volume = 1_000_000 if i % 2 == 0 else 1_000  # half fail the liquidity gate
        drift = 0.0005 * (i % 7)
        closes = 50.0 * np.cumprod(1.0 + drift + rng.normal(0, 0.005, days))
        rows.extend(
            {
                "symbol": symbol,
                "timestamp": timestamps[j],
                "open": closes[j],
                "high": closes[j] * 1.01,
                "low": closes[j] * 0.99,
                "close": closes[j],
                "volume": volume,
            }
            for j in range(days)
        )
    return assets, pd.DataFrame(rows)


async def screen(args: argparse.Namespace) -> None:
    config = load_config(Path(args.config))
    setup_logging(str(config["monitoring"]["log_level"]))
    if args.dry_run:
        assets, daily = _synthetic_inputs(config)
        universe = run_funnel(assets, daily, config, {}, daily["timestamp"].max())
        log.info("screener_dry_run_completed", extra={"universe": universe})
        return

    broker = AlpacaBrokerClient(
        require_env("ALPACA_API_KEY"), require_env("ALPACA_SECRET_KEY"), paper=True
    )
    assets = await broker.list_assets()
    storage = BarStorage(Path(config["data"]["db_path"]))
    if args.sync_days:
        client = PolygonClient(require_env("POLYGON_API_KEY"))
        day = datetime.now(tz=UTC)
        synced = 0
        while synced < args.sync_days:
            day -= timedelta(days=1)
            if day.weekday() >= 5:  # skip weekends; holidays return empty results
                continue
            storage.insert_bars(await client.fetch_grouped_daily(day))
            synced += 1
    as_of = pd.Timestamp(datetime.now(tz=UTC))
    lookback_days = int(config["screener"]["min_history_days"]) * 2
    daily = storage.get_all_bars(
        datetime.now(tz=UTC) - timedelta(days=lookback_days), datetime.now(tz=UTC)
    )
    universe = run_funnel(assets, daily, config, {}, as_of)
    write_universe(universe, config, Path(args.output_dir))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--output-dir", default="universe")
    parser.add_argument("--sync-days", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    asyncio.run(screen(parser.parse_args()))


if __name__ == "__main__":
    main()
