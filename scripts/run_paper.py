"""Paper/live trading entrypoint (make paper-trade S=<strategy>).

Startup sequence — exact order, never shortcut:
  1. setup_logging
  2. clearance check: latest backtest_results/{strategy}_*.yaml must say
     cleared_for_paper: true. No results file = not cleared. NOT skippable.
  3. paper = not execution.live_mode — the ONLY line deciding paper/live (ADR-007)
  4. construct components in dependency order (this file is the composition root)
  5. initial position sync BEFORE any signal
  6. stream + bar-consumer loop; SIGINT/SIGTERM stop consuming, open orders
     are left alone.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anthropic
import redis.asyncio as aioredis
import yaml
from src.data.models import Bar
from src.execution.broker import AlpacaBrokerClient, execute_intent
from src.execution.stream_manager import StreamManager
from src.monitoring.alerts import make_alert_dispatcher
from src.monitoring.logging_setup import setup_logging
from src.risk.circuit_breaker import CircuitBreaker
from src.risk.exposure_tracker import ExposureTracker
from src.risk.manager import RiskManager
from src.shared.config import load_config, require_env
from src.signals.features import compute_features
from src.signals.lgbm_signal import LightGBMSignalGenerator
from src.signals.regime_hmm import RegimeDetector
from src.signals.sentiment_factory import create_sentiment_provider
from src.strategies.momentum_lightgbm import MomentumLightGBM

log = logging.getLogger(__name__)


def load_clearance(results_dir: Path, strategy_name: str) -> dict[str, Any]:
    """Refuse to trade anything WFO has not cleared. There is no override flag."""
    files = sorted(results_dir.glob(f"{strategy_name}_*.yaml"))
    if not files:
        raise SystemExit(
            f"{strategy_name}: no WFO results in {results_dir} — "
            "not cleared for paper (run `make wfo` first)"
        )
    latest = yaml.safe_load(files[-1].read_text(encoding="utf-8"))
    if not latest.get("cleared_for_paper"):
        raise SystemExit(f"{strategy_name}: cleared_for_paper is false — WFO gates not passed")
    log.info("clearance_verified", extra={"strategy": strategy_name, "results": str(files[-1])})
    return dict(latest)


def build_components(
    config: dict[str, Any], artifact_dir: Path, hmm_dir: Path | None
) -> SimpleNamespace:
    """Dependency order: alerts -> tracker -> breaker -> risk -> broker ->
    generators -> strategy -> stream. Credentials are read here and only here."""
    dispatch = make_alert_dispatcher(
        require_env("TELEGRAM_BOT_TOKEN"),
        require_env("TELEGRAM_CHAT_ID"),
        float(config["monitoring"]["alert_timeout_s"]),
    )
    tracker = ExposureTracker()

    def on_trip(reason: str, values: dict[str, float]) -> None:
        dispatch(f"circuit breaker tripped: {reason} {values}")

    breaker = CircuitBreaker(config["risk"]["circuit_breakers"], on_trip=on_trip)
    risk = RiskManager(config, tracker, breaker, dict(config["strategy"]["stats"]))

    paper = not config["execution"]["live_mode"]  # ADR-007 — the only paper/live decision
    api_key, secret_key = require_env("ALPACA_API_KEY"), require_env("ALPACA_SECRET_KEY")
    broker = AlpacaBrokerClient(api_key, secret_key, paper=paper)

    def features_fn(frame: Any) -> Any:
        return compute_features(frame, config)

    strategy = MomentumLightGBM(
        config,
        LightGBMSignalGenerator(artifact_dir),
        features_fn,
        RegimeDetector(hmm_dir) if hmm_dir is not None else None,
    )

    async def resync() -> None:
        positions = await broker.get_positions()
        account = await broker.get_account()
        tracker.sync_from_api(positions, float(account["equity"]))
        log.info("position_sync_completed", extra={"positions": len(positions)})

    def on_fill(fill: dict[str, Any]) -> None:
        strategy.on_trade_update(fill)
        risk.on_fill(fill)

    redis_client = aioredis.from_url(require_env("REDIS_URL"))
    # ANTHROPIC_API_KEY is required ONLY when the provider actually needs it
    anthropic_client = (
        anthropic.AsyncAnthropic(api_key=require_env("ANTHROPIC_API_KEY"))
        if config["llm"]["provider"] == "anthropic"
        else None
    )
    sentiment = create_sentiment_provider(config, redis_client, anthropic_client)
    stream = StreamManager(
        api_key,
        secret_key,
        paper=paper,
        redis_client=redis_client,
        on_fill=on_fill,
        resync=resync,
        alert=dispatch,
        config=config,
    )
    return SimpleNamespace(
        tracker=tracker,
        breaker=breaker,
        risk=risk,
        broker=broker,
        strategy=strategy,
        stream=stream,
        redis=redis_client,
        resync=resync,
        paper=paper,
        sentiment=sentiment,  # consumed by the news pipeline when it lands
    )


async def start_trading(components: SimpleNamespace) -> None:
    """Positions are synced BEFORE the first event is consumed — always."""
    await components.resync()
    await components.stream.start(components.strategy.universe)


async def consume_bars(components: SimpleNamespace) -> None:
    """The live engine loop: Redis bar -> on_bar -> validate -> submit."""
    pubsub = components.redis.pubsub()
    channels = [f"channel:bars:{s}" for s in components.strategy.universe]
    await pubsub.subscribe(*channels)
    async for message in pubsub.listen():
        if message.get("type") != "message":
            continue
        payload = json.loads(message["data"])
        bar = Bar(
            symbol=payload["symbol"],
            timestamp=datetime.fromisoformat(payload["timestamp"]),
            open=float(payload["open"]),
            high=float(payload["high"]),
            low=float(payload["low"]),
            close=float(payload["close"]),
            volume=int(payload["volume"]),
            vwap=payload.get("vwap"),
        )
        intent = components.strategy.on_bar(bar)
        if intent is not None:
            await execute_intent(components.broker, components.risk, intent)


async def run(args: argparse.Namespace) -> None:
    config = load_config(Path(args.config))
    setup_logging(str(config["monitoring"]["log_level"]))
    load_clearance(Path(args.results_dir), args.strategy)  # gate BEFORE construction
    components = build_components(
        config, Path(args.artifacts), Path(args.hmm_artifacts) if args.hmm_artifacts else None
    )
    log.info("paper_session_starting", extra={"paper": components.paper})
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    tasks = [
        asyncio.create_task(start_trading(components)),
        asyncio.create_task(consume_bars(components)),
        asyncio.create_task(stop.wait()),
    ]
    _done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()
    # open orders are deliberately NOT canceled — they may protect positions
    log.info("shutdown_complete", extra={"equity": components.tracker.equity})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--results-dir", default="backtest_results")
    parser.add_argument("--artifacts", default="models/latest")
    parser.add_argument("--hmm-artifacts", default=None)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
