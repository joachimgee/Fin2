"""CrossSectionalReversion tests — deterministic signals via close prices.

FakeGen returns close/100 of the last row, so a symbol's daily signal is set
directly by its close: close 90 -> signal 0.9 (most oversold), 10 -> 0.1.
"""

from __future__ import annotations

import copy
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from src.data.models import Bar
from src.strategies.xsec_reversion import CrossSectionalReversion


class FakeGen:
    def generate(self, features: Any) -> float:
        return float(features["close"].iloc[-1]) / 100.0


def _identity_features(df: pd.DataFrame) -> pd.DataFrame:
    return df[["close"]]


def _config(base_config: dict[str, Any]) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    config["strategy"] = {
        "universe": ["AAA", "BBB", "CCC", "DDD"],
        "warmup_bars": 1,
        "cross_sectional": {
            "top_n": 2,
            "rebalance_every_bars": 1,  # daily here — freeze behaviour tested separately
            "max_hold_bars": 3,
            "zscore_clip": 3.0,
            "sentiment_veto": -0.3,
            "sentiment_lookback_days": 3,
        },
    }
    return config


def _bar(symbol: str, day: int, close: float) -> Bar:
    return Bar(
        symbol=symbol,
        timestamp=datetime(2024, 1, 1 + day, 21, tzinfo=UTC),
        open=close,
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=1_000,
    )


def _strategy(base_config: dict[str, Any], **kwargs: Any) -> CrossSectionalReversion:
    return CrossSectionalReversion(_config(base_config), FakeGen(), _identity_features, **kwargs)


def _feed_day(
    strategy: CrossSectionalReversion, day: int, closes: dict[str, float]
) -> dict[str, Any]:
    return {s: strategy.on_bar(_bar(s, day, c)) for s, c in closes.items()}


DAY1 = {"AAA": 90.0, "BBB": 80.0, "CCC": 30.0, "DDD": 10.0}  # top-2 = {AAA, BBB}


def test_no_intents_on_first_day_no_complete_ranking_yet(
    base_config: dict[str, Any],
) -> None:
    intents = _feed_day(_strategy(base_config), 1, DAY1)
    assert all(intent is None for intent in intents.values())


def test_day2_enters_exactly_the_top_n_of_day1(base_config: dict[str, Any]) -> None:
    strategy = _strategy(base_config)
    _feed_day(strategy, 1, DAY1)
    intents = _feed_day(strategy, 2, DAY1)
    assert intents["AAA"] is not None and intents["AAA"].side == "buy"
    assert intents["BBB"] is not None and intents["BBB"].side == "buy"
    assert intents["CCC"] is None
    assert intents["DDD"] is None


def test_rotation_exits_symbol_leaving_target(base_config: dict[str, Any]) -> None:
    strategy = _strategy(base_config)
    _feed_day(strategy, 1, DAY1)
    _feed_day(strategy, 2, {"AAA": 90.0, "BBB": 80.0, "CCC": 95.0, "DDD": 10.0})
    strategy.on_trade_update({"symbol": "BBB", "side": "buy", "qty": 7.0, "price": 80.0})
    # day-2 ranking: CCC .95, AAA .90 -> BBB drops out of the target
    intents = _feed_day(strategy, 3, DAY1)
    assert intents["BBB"] is not None
    assert (intents["BBB"].side, intents["BBB"].qty) == ("sell", 7.0)
    assert intents["CCC"] is not None and intents["CCC"].side == "buy"


def test_no_reentry_while_holding(base_config: dict[str, Any]) -> None:
    strategy = _strategy(base_config)
    _feed_day(strategy, 1, DAY1)
    _feed_day(strategy, 2, DAY1)
    strategy.on_trade_update({"symbol": "AAA", "side": "buy", "qty": 5.0, "price": 90.0})
    intents = _feed_day(strategy, 3, DAY1)
    assert intents["AAA"] is None  # still in target, already held -> hold


def test_time_stop_exits_even_inside_target(base_config: dict[str, Any]) -> None:
    strategy = _strategy(base_config)
    _feed_day(strategy, 1, DAY1)
    _feed_day(strategy, 2, DAY1)
    strategy.on_trade_update({"symbol": "AAA", "side": "buy", "qty": 5.0, "price": 90.0})
    _feed_day(strategy, 3, DAY1)  # held 1
    _feed_day(strategy, 4, DAY1)  # held 2
    intents = _feed_day(strategy, 5, DAY1)  # held 3 == max_hold
    assert intents["AAA"] is not None and intents["AAA"].side == "sell"


def test_sentiment_veto_blocks_entry_never_exit(base_config: dict[str, Any]) -> None:
    strategy = _strategy(base_config, sentiment_fn=lambda symbol, ts: -0.9)
    _feed_day(strategy, 1, DAY1)
    intents = _feed_day(strategy, 2, DAY1)
    assert all(intent is None for intent in intents.values())  # entries all vetoed
    strategy.on_trade_update({"symbol": "AAA", "side": "buy", "qty": 5.0, "price": 90.0})
    _feed_day(strategy, 3, DAY1)
    _feed_day(strategy, 4, DAY1)
    intents = _feed_day(strategy, 5, DAY1)  # time stop
    assert intents["AAA"] is not None and intents["AAA"].side == "sell"  # never vetoed


def test_symbol_without_signal_absent_from_ranking(base_config: dict[str, Any]) -> None:
    config = _config(base_config)
    config["strategy"]["warmup_bars"] = 2  # CCC/DDD start one day later than AAA/BBB
    strategy = CrossSectionalReversion(config, FakeGen(), _identity_features)
    strategy.on_bar(_bar("AAA", 1, 30.0))
    strategy.on_bar(_bar("BBB", 1, 20.0))
    for s, c in DAY1.items():  # day 2: AAA/BBB have 2 bars (warm), CCC/DDD only 1
        strategy.on_bar(_bar(s, 2, c))
    intents = _feed_day(strategy, 3, DAY1)
    # day-2 ranking contains ONLY warm symbols {AAA .9, BBB .8} -> they are the target
    assert intents["AAA"] is not None and intents["BBB"] is not None
    assert intents["CCC"] is None and intents["DDD"] is None


def test_weekly_freeze_ignores_mid_period_rankings(base_config: dict[str, Any]) -> None:
    """rebalance_every_bars=2: the target from day 1 stays frozen through
    day 3 even though day 2's ranking would rotate CCC in; day 4 (next
    rebalance boundary) finally applies it."""
    config = _config(base_config)
    config["strategy"]["cross_sectional"]["rebalance_every_bars"] = 2
    config["strategy"]["cross_sectional"]["max_hold_bars"] = 10  # keep time stop out of frame
    strategy = CrossSectionalReversion(config, FakeGen(), _identity_features)
    _feed_day(strategy, 1, DAY1)  # ranking {AAA, BBB}
    _feed_day(strategy, 2, {"AAA": 90.0, "BBB": 80.0, "CCC": 95.0, "DDD": 10.0})
    strategy.on_trade_update({"symbol": "BBB", "side": "buy", "qty": 7.0, "price": 80.0})
    intents = _feed_day(strategy, 3, DAY1)  # boundary 2: FROZEN — no refresh
    assert intents["CCC"] is None  # day-2 ranking not applied mid-period
    assert intents["BBB"] is None  # held position not rotated out mid-period
    _feed_day(strategy, 4, DAY1)  # boundary 3: refresh from day-3 ranking {AAA,BBB}
    intents = _feed_day(strategy, 5, DAY1)
    assert intents["BBB"] is None  # back in target after refresh -> still held


def test_reset_clears_target_and_state(base_config: dict[str, Any]) -> None:
    strategy = _strategy(base_config)
    _feed_day(strategy, 1, DAY1)
    _feed_day(strategy, 2, DAY1)
    strategy.reset()
    assert not strategy.is_ready
    intents = _feed_day(strategy, 3, DAY1)
    assert all(intent is None for intent in intents.values())  # no stale target
