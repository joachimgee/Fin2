"""Phase 8 — composition root tests, all wiring mocked (PHASE_8_ENTRYPOINTS.md)."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
import scripts.run_paper as rp
import yaml


def _write_clearance(results_dir: Path, strategy: str, cleared: bool) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"{strategy}_20260101_000000.yaml").write_text(
        yaml.safe_dump({"cleared_for_paper": cleared}), encoding="utf-8"
    )


class FakeBroker:
    last_paper: ClassVar[bool | None] = None

    def __init__(self, api_key: str, secret_key: str, paper: bool = True) -> None:
        FakeBroker.last_paper = paper

    async def get_positions(self) -> list[dict[str, Any]]:
        return []

    async def get_account(self) -> dict[str, Any]:
        return {"equity": 100_000.0, "cash": 100_000.0}


class FakeStream:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.kwargs = kwargs


@pytest.fixture
def alerts() -> list[str]:
    return []


@pytest.fixture
def _mock_wiring(monkeypatch: pytest.MonkeyPatch, alerts: list[str]) -> None:
    monkeypatch.setattr(rp, "AlpacaBrokerClient", FakeBroker)
    monkeypatch.setattr(rp, "StreamManager", FakeStream)
    monkeypatch.setattr(
        rp, "LightGBMSignalGenerator", lambda path: SimpleNamespace(generate=lambda f: 0.0)
    )
    monkeypatch.setattr(rp, "aioredis", SimpleNamespace(from_url=lambda url: SimpleNamespace()))
    monkeypatch.setattr(rp, "make_alert_dispatcher", lambda *a, **k: alerts.append)
    monkeypatch.setattr(rp, "require_env", lambda name: f"fake-{name}")
    monkeypatch.setattr(rp, "create_sentiment_provider", lambda *a, **k: "sentiment-sentinel")


def _args(tmp_path: Path, strategy: str = "momentum_lightgbm") -> argparse.Namespace:
    return argparse.Namespace(
        strategy=strategy,
        config="config/base.yaml",
        results_dir=str(tmp_path / "backtest_results"),
        artifacts=str(tmp_path / "models"),
        hmm_artifacts=None,
    )


# --- clearance gate --------------------------------------------------------------


async def test_paper_refuses_without_clearance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    built: list[Any] = []
    monkeypatch.setattr(rp, "setup_logging", lambda level: None)  # keep root logger pristine
    monkeypatch.setattr(rp, "build_components", lambda *a, **k: built.append(1))
    with pytest.raises(SystemExit, match="not cleared"):
        await rp.run(_args(tmp_path))
    assert built == []  # nothing was constructed — refused BEFORE wiring


async def test_paper_refuses_when_not_cleared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_clearance(tmp_path / "backtest_results", "momentum_lightgbm", cleared=False)
    built: list[Any] = []
    monkeypatch.setattr(rp, "setup_logging", lambda level: None)  # keep root logger pristine
    monkeypatch.setattr(rp, "build_components", lambda *a, **k: built.append(1))
    with pytest.raises(SystemExit, match="cleared_for_paper is false"):
        await rp.run(_args(tmp_path))
    assert built == []


def test_clearance_accepts_latest_cleared(tmp_path: Path) -> None:
    _write_clearance(tmp_path, "s", cleared=True)
    assert rp.load_clearance(tmp_path, "s")["cleared_for_paper"] is True


# --- wiring ----------------------------------------------------------------------


@pytest.mark.usefixtures("_mock_wiring")
def test_paper_flag_is_not_live_mode(base_config: dict[str, Any], tmp_path: Path) -> None:
    rp.build_components(base_config, tmp_path, None)
    assert FakeBroker.last_paper is True  # live_mode false -> paper
    live = {**base_config, "execution": {**base_config["execution"], "live_mode": True}}
    rp.build_components(live, tmp_path, None)
    assert FakeBroker.last_paper is False  # the ONLY path to real orders


@pytest.mark.usefixtures("_mock_wiring")
def test_breaker_wired_to_alert_dispatch(
    base_config: dict[str, Any], tmp_path: Path, alerts: list[str]
) -> None:
    components = rp.build_components(base_config, tmp_path, None)
    for _ in range(5):
        components.breaker.on_trade_closed(-10.0)  # trip consecutive_losses
    assert len(alerts) == 1
    assert "consecutive_losses" in alerts[0]


@pytest.mark.usefixtures("_mock_wiring")
def test_wiring_smoke_and_fill_path(base_config: dict[str, Any], tmp_path: Path) -> None:
    components = rp.build_components(base_config, tmp_path, None)
    assert components.strategy.universe == base_config["strategy"]["universe"]
    assert components.stream.kwargs["paper"] is True
    assert components.sentiment == "sentiment-sentinel"  # provider wired from config
    # the stream's on_fill hook must reach BOTH the strategy and the tracker
    components.stream.kwargs["on_fill"](
        {"symbol": "SPY", "side": "buy", "qty": 5.0, "price": 100.0}
    )
    assert components.tracker.position_qty("SPY") == 5.0


async def test_sync_runs_before_stream_start() -> None:
    order: list[str] = []

    async def resync() -> None:
        order.append("resync")

    async def start(symbols: list[str]) -> None:
        order.append("stream_start")

    components = SimpleNamespace(
        resync=resync,
        stream=SimpleNamespace(start=start),
        strategy=SimpleNamespace(universe=["SPY"]),
    )
    await rp.start_trading(components)
    assert order == ["resync", "stream_start"]  # positions synced BEFORE any event
