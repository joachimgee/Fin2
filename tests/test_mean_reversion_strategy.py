"""MeanReversionZScore strategy tests — scripted generator, no ML, no I/O."""

from __future__ import annotations

import copy
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from src.data.models import Bar
from src.strategies.mean_reversion import MeanReversionZScore


class ScriptedGen:
    def __init__(self, values: list[float]) -> None:
        self.values = list(values)

    def generate(self, features: Any) -> float:
        return self.values.pop(0)


def _finite_features(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"f": [0.0] * len(df)}, index=df.index)


def _config(base_config: dict[str, Any]) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    config["strategy"] = {
        "universe": ["SPY"],
        "warmup_bars": 3,
        "mean_reversion": {
            "entry_signal": 0.5,
            "exit_signal": 0.0,
            "max_hold_bars": 2,
            "zscore_clip": 3.0,
        },
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


def _strategy(base_config: dict[str, Any], signals: list[float]) -> MeanReversionZScore:
    return MeanReversionZScore(_config(base_config), ScriptedGen(signals), _finite_features)


def _feed(strategy: MeanReversionZScore, count: int) -> None:
    for i in range(count):
        strategy.on_bar(_bar(i))


def _fill_buy(strategy: MeanReversionZScore, qty: float = 21.0) -> None:
    strategy.on_trade_update({"symbol": "SPY", "side": "buy", "qty": qty, "price": 100.0})


def test_no_intent_before_warmup(base_config: dict[str, Any]) -> None:
    strategy = _strategy(base_config, [1.0])
    assert strategy.on_bar(_bar(0)) is None
    assert strategy.on_bar(_bar(1)) is None
    assert not strategy.is_ready


def test_enters_long_when_oversold(base_config: dict[str, Any]) -> None:
    strategy = _strategy(base_config, [0.6])
    _feed(strategy, 2)
    intent = strategy.on_bar(_bar(2, close=95.0))
    assert intent is not None
    assert (intent.side, intent.qty) == ("buy", 1.0)  # qty placeholder — risk sizes it
    assert intent.signal_strength == 0.6
    assert intent.reference_price == 95.0


def test_no_entry_when_not_oversold_enough(base_config: dict[str, Any]) -> None:
    strategy = _strategy(base_config, [0.4])
    _feed(strategy, 2)
    assert strategy.on_bar(_bar(2)) is None


def test_exits_full_qty_when_price_reverts_to_mean(base_config: dict[str, Any]) -> None:
    strategy = _strategy(base_config, [0.6, -0.1])
    _feed(strategy, 2)
    strategy.on_bar(_bar(2))  # entry intent
    _fill_buy(strategy)
    exit_intent = strategy.on_bar(_bar(3, close=101.0))
    assert exit_intent is not None
    assert (exit_intent.side, exit_intent.qty) == ("sell", 21.0)


def test_holds_while_still_below_mean(base_config: dict[str, Any]) -> None:
    strategy = _strategy(base_config, [0.6, 0.3])
    _feed(strategy, 2)
    strategy.on_bar(_bar(2))
    _fill_buy(strategy)
    assert strategy.on_bar(_bar(3)) is None  # signal 0.3 > exit 0.0 and hold 1 < max 2


def test_time_stop_forces_exit_after_max_hold(base_config: dict[str, Any]) -> None:
    # signal stays positive (never reverts) — the time stop must fire at hold 2
    strategy = _strategy(base_config, [0.6, 0.3, 0.3])
    _feed(strategy, 2)
    strategy.on_bar(_bar(2))
    _fill_buy(strategy)
    assert strategy.on_bar(_bar(3)) is None  # held 1 bar
    exit_intent = strategy.on_bar(_bar(4))  # held 2 bars == max_hold_bars
    assert exit_intent is not None
    assert (exit_intent.side, exit_intent.qty) == ("sell", 21.0)


def test_hold_counter_resets_for_next_trade(base_config: dict[str, Any]) -> None:
    strategy = _strategy(base_config, [0.6, -0.1, 0.6, 0.3])
    _feed(strategy, 2)
    strategy.on_bar(_bar(2))
    _fill_buy(strategy)
    strategy.on_bar(_bar(3))  # exit intent (reverted)
    strategy.on_trade_update({"symbol": "SPY", "side": "sell", "qty": 21.0, "price": 101.0})
    strategy.on_bar(_bar(4))  # re-entry intent
    _fill_buy(strategy)
    assert strategy.on_bar(_bar(5)) is None  # fresh trade: held 1 < max 2, no stale counter


def test_no_reentry_while_holding(base_config: dict[str, Any]) -> None:
    strategy = _strategy(base_config, [0.9, 0.9])
    _feed(strategy, 2)
    strategy.on_bar(_bar(2))
    _fill_buy(strategy)
    assert strategy.on_bar(_bar(3)) is None  # oversold again but already long


def test_foreign_symbol_ignored(base_config: dict[str, Any]) -> None:
    strategy = _strategy(base_config, [1.0])
    for i in range(5):
        assert strategy.on_bar(_bar(i, symbol="TSLA")) is None


def test_reset_clears_state(base_config: dict[str, Any]) -> None:
    strategy = _strategy(base_config, [0.6])
    _feed(strategy, 3)
    _fill_buy(strategy)
    strategy.reset()
    assert not strategy.is_ready
    assert strategy.on_bar(_bar(0)) is None  # back inside warmup
