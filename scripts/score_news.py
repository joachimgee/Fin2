"""Batch-score historical headlines with FinBERT into DuckDB daily sentiment.

Alpaca News API (free, Benzinga, ~2015+) -> FinBERT (local, free) ->
news_sentiment table: one row per (symbol, UTC calendar date) with the mean
signed score. Headlines are attributed to their UTC CALENDAR date and every
feature built from this table goes through the compute_features shift(1)
chokepoint, so the value used at T only contains headlines from <= T-1 —
conservative (up to ~21h stale) and structurally lookahead-safe.

Caps headlines per symbol-day (llm.max_headlines_per_day) — news-magnet days
would otherwise dominate runtime without changing the daily mean much.

Usage: python3 -m scripts.score_news --start 2020-07-01 --end 2026-07-04
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

from src.data.alpaca_data import AlpacaDataClient
from src.data.storage import BarStorage
from src.monitoring.logging_setup import setup_logging
from src.shared.config import load_config, load_universe, require_env
from src.signals.finbert_sentiment import load_finbert, signed_score

log = logging.getLogger(__name__)


def cap_by_day(news: list[tuple[datetime, str]], max_per_day: int) -> dict[datetime, list[str]]:
    """First max_per_day headlines per UTC calendar date, chronological.
    Pure — capping happens BEFORE scoring so capped headlines are never paid
    for (FinBERT inference is the expensive step)."""
    by_day: dict[datetime, list[str]] = defaultdict(list)
    for created_at, headline in news:
        day = datetime(created_at.year, created_at.month, created_at.day, tzinfo=UTC)
        if len(by_day[day]) < max_per_day:
            by_day[day].append(headline)
    return dict(sorted(by_day.items()))


async def score(args: argparse.Namespace) -> None:
    config = load_config(Path(args.config))
    setup_logging(str(config["monitoring"]["log_level"]))
    universe = load_universe(args.universe) or list(config["strategy"]["universe"])
    max_per_day = int(config["llm"]["max_headlines_per_day"])
    classifier = load_finbert(str(config["llm"]["finbert_model"]))
    client = AlpacaDataClient(require_env("ALPACA_API_KEY"), require_env("ALPACA_SECRET_KEY"))
    storage = BarStorage(Path(config["data"]["db_path"]))
    start = datetime.fromisoformat(args.start).replace(tzinfo=UTC)
    end = datetime.fromisoformat(args.end).replace(tzinfo=UTC)

    for symbol in universe:
        news = await client.fetch_news(symbol, start, end)
        capped = cap_by_day(news, max_per_day)
        # unique headlines only — daily market-recap stories repeat verbatim
        headlines = sorted({h for day_headlines in capped.values() for h in day_headlines})
        results = await asyncio.to_thread(classifier, headlines)  # type: ignore[arg-type]
        scores = {h: signed_score(r) for h, r in zip(headlines, results, strict=True)}
        rows = [
            (symbol, day, mean(scores[h] for h in day_headlines), len(day_headlines))
            for day, day_headlines in capped.items()
        ]
        inserted = storage.insert_daily_sentiment(rows)
        log.info(
            "news_scored",
            extra={
                "symbol": symbol,
                "headlines": len(news),
                "scored": len(headlines),
                "days": inserted,
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--universe", default=None, help="YAML universe override file")
    asyncio.run(score(parser.parse_args()))


if __name__ == "__main__":
    main()
