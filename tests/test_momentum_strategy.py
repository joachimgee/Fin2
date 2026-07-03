"""Phase 6 — MomentumLightGBM strategy tests."""

from __future__ import annotations

import copy
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from src.data.models import Bar
from src.strategies.momentum_lightgbm import MomentumLightGBM


class ScriptedGen:
    def __init__(self, values: list[float]) -> None:
        self.values = list(values)

    def generate(self, features: Any) -> float:
        return self.values.pop(0)


def _finite_features(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"f": [0.0] * len(df)}, index=df.index)


def _nan_features(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"f": [np.nan] * len(df)}, index=df.index)


class HostileRegime:
    def current_regime(self, features: pd.DataFrame) -> int:
        return 1


def _config(base_config: dict[str, Any]) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    config["strategy"] = {
        "universe": ["SPY"],
        "signal_threshold": 0.2,
        "warmup_bars": 3,
        "hostile_regimes": [1],
    }
    return config


def _bar(index: int, close: float = 100.0, symbol: str = "SPY") -> Bar:
    return Bar(
        symbol=symbol,
        timestamp=datetime(2024, 1, 1 + index, 21, tzinfo=UTC),
        open=close,
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=1_000,
    )


def _feed(strategy: MomentumLightGBM, count: int) -> None:
    for i in range(count):
        strategy.on_bar(_bar(i))


def test_no_intent_before_warmup(base_config: dict[str, Any]) -> None:
    gen = ScriptedGen([1.0])
    strategy = MomentumLightGBM(_config(base_config), gen, _finite_features)
    assert strategy.on_bar(_bar(0)) is None
    assert strategy.on_bar(_bar(1)) is None
    assert len(gen.values) == 1  # generator never consulted during warmup
    assert not strategy.is_ready


def test_entry_above_threshold(base_config: dict[str, Any]) -> None:
    strategy = MomentumLightGBM(_config(base_config), ScriptedGen([0.5]), _finite_features)
    _feed(strategy, 2)
    intent = strategy.on_bar(_bar(2, close=102.0))
    assert intent is not None
    assert (intent.side, intent.qty) == ("buy", 1.0)  # qty placeholder — risk sizes it
    assert intent.signal_strength == 0.5
    assert intent.reference_price == 102.0
    assert strategy.is_ready


def test_no_entry_below_threshold(base_config: dict[str, Any]) -> None:
    strategy = MomentumLightGBM(_config(base_config), ScriptedGen([0.1]), _finite_features)
    _feed(strategy, 2)
    assert strategy.on_bar(_bar(2)) is None


def test_exit_requests_full_held_quantity(base_config: dict[str, Any]) -> None:
    strategy = MomentumLightGBM(_config(base_config), ScriptedGen([0.5, -0.5]), _finite_features)
    _feed(strategy, 2)
    strategy.on_bar(_bar(2))  # entry intent emitted
    strategy.on_trade_update({"symbol": "SPY", "side": "buy", "qty": 21.0, "price": 100.0})
    exit_intent = strategy.on_bar(_bar(3))
    assert exit_intent is not None
    assert (exit_intent.side, exit_intent.qty) == ("sell", 21.0)


def test_weak_negative_signal_holds(base_config: dict[str, Any]) -> None:
    strategy = MomentumLightGBM(_config(base_config), ScriptedGen([0.5, -0.1]), _finite_features)
    _feed(strategy, 2)
    strategy.on_bar(_bar(2))
    strategy.on_trade_update({"symbol": "SPY", "side": "buy", "qty": 21.0, "price": 100.0})
    assert strategy.on_bar(_bar(3)) is None


def test_hostile_regime_blocks_entry_not_exit(base_config: dict[str, Any]) -> None:
    strategy = MomentumLightGBM(
        _config(base_config), ScriptedGen([0.9, -0.5]), _finite_features, HostileRegime()
    )
    _feed(strategy, 2)
    assert strategy.on_bar(_bar(2)) is None  # entry gated by hostile regime
    strategy.on_trade_update({"symbol": "SPY", "side": "buy", "qty": 21.0, "price": 100.0})
    exit_intent = strategy.on_bar(_bar(3))
    assert exit_intent is not None  # protective exit is NEVER regime-gated
    assert exit_intent.side == "sell"


def test_nan_features_return_none_without_generate(base_config: dict[str, Any]) -> None:
    gen = ScriptedGen([1.0])
    strategy = MomentumLightGBM(_config(base_config), gen, _nan_features)
    _feed(strategy, 2)
    assert strategy.on_bar(_bar(2)) is None
    assert len(gen.values) == 1  # generate() never sees a non-finite row


def test_foreign_symbol_ignored(base_config: dict[str, Any]) -> None:
    strategy = MomentumLightGBM(_config(base_config), ScriptedGen([1.0]), _finite_features)
    for i in range(5):
        assert strategy.on_bar(_bar(i, symbol="TSLA")) is None


def test_reset_clears_state(base_config: dict[str, Any]) -> None:
    strategy = MomentumLightGBM(_config(base_config), ScriptedGen([0.5]), _finite_features)
    _feed(strategy, 3)
    strategy.on_trade_update({"symbol": "SPY", "side": "buy", "qty": 21.0, "price": 100.0})
    strategy.reset()
    assert not strategy.is_ready
    assert strategy.on_bar(_bar(0)) is None  # back inside warmup
