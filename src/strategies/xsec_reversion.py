"""Cross-sectional reversion: hold the N most oversold names of the universe.

The canonical academic construction of short-term reversal (Jegadeesh 1990,
Lehmann 1990; long leg only): instead of asking "is this stock oversold vs
its own history?" (time-series — enters the whole market at once in a
selloff), rank the cross-section every day and hold the N RELATIVELY most
oversold names. Portfolio size stays ~constant -> smooth exposure, which is
what the time-series variant lacked (Sharpe gate).

Decision timing: bars of day D are decided against the COMPLETE ranking of
day D-1 — never against a partially-arrived cross-section of day D. With the
feature shift(1) that means entries at D use prices <= D-2: conservative and
structurally lookahead-safe.

Rules (all a priori, never scanned): the target set is refreshed every
rebalance_every_bars trading days (5 = the weekly cadence of Lehmann 1990,
Quantpedia and ML4T reference implementations — daily rotation was measured
to churn the edge away in frictions) and stays FROZEN in between. Enter when
the symbol is in the target (and the FinBERT veto allows); exit when a
refresh drops it from the target or after max_hold_bars. Exits are never
vetoed. Entry qty is a placeholder — the RiskManager sizes every order.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.data.models import Bar
from src.shared.interfaces import SignalGenerator
from src.strategies.base import AbstractStrategy, OrderIntent
from src.strategies.mean_reversion import FeaturesFn, SentimentFn

_STRATEGY_ID = "xsec_reversion"


class CrossSectionalReversion(AbstractStrategy):
    def __init__(
        self,
        config: dict[str, Any],
        signal_generator: SignalGenerator,
        features_fn: FeaturesFn,
        sentiment_fn: SentimentFn | None = None,
    ) -> None:
        super().__init__(config, signal_generator)
        strategy_cfg = config["strategy"]
        xs_cfg = strategy_cfg["cross_sectional"]
        self._symbols = [str(s) for s in strategy_cfg["universe"]]
        self._warmup = int(strategy_cfg["warmup_bars"])
        self._top_n = int(xs_cfg["top_n"])
        self._rebalance_every = int(xs_cfg["rebalance_every_bars"])
        self._max_hold = int(xs_cfg["max_hold_bars"])
        self._sentiment_veto = float(xs_cfg["sentiment_veto"])
        self._sentiment_fn = sentiment_fn
        self._features_fn = features_fn
        self._bars: dict[str, list[Bar]] = {}
        self._positions: dict[str, float] = {}
        self._bars_held: dict[str, int] = {}
        self._current_day: Any = None
        self._day_count = 0
        self._today_signals: dict[str, float] = {}
        self._target: set[str] = set()  # top-N frozen at the last rebalance

    @property
    def universe(self) -> list[str]:
        return list(self._symbols)

    @property
    def is_ready(self) -> bool:
        return any(len(bars) >= self._warmup for bars in self._bars.values())

    def on_bar(self, bar: Bar) -> OrderIntent | None:
        if bar.symbol not in self._symbols:
            return None
        self._roll_day(bar.timestamp)
        buffer = self._bars.setdefault(bar.symbol, [])
        buffer.append(bar)
        if len(buffer) > self._warmup:
            del buffer[: len(buffer) - self._warmup]
        if len(buffer) < self._warmup:
            return None
        if self._positions.get(bar.symbol, 0.0) > 0.0:
            self._bars_held[bar.symbol] = self._bars_held.get(bar.symbol, 0) + 1
        signal = self._signal(buffer)
        if signal is not None:
            self._today_signals[bar.symbol] = signal
        return self._decide(bar, signal)

    def on_trade_update(self, update: dict[str, Any]) -> None:
        signed = float(update["qty"]) if update["side"] == "buy" else -float(update["qty"])
        held = self._positions.get(update["symbol"], 0.0) + signed
        self._positions[update["symbol"]] = held
        if held <= 0.0:
            self._bars_held.pop(update["symbol"], None)

    def reset(self) -> None:
        self._bars.clear()
        self._positions.clear()
        self._bars_held.clear()
        self._current_day = None
        self._day_count = 0
        self._today_signals = {}
        self._target = set()

    # --- internals ----------------------------------------------------------------

    def _roll_day(self, timestamp: Any) -> None:
        """First bar of a new day: yesterday's cross-section is now complete.
        Refresh the target top-N only on rebalance days (bootstrap on the
        first boundary, then every rebalance_every_bars); frozen in between."""
        if self._current_day is None:
            self._current_day = timestamp
            return
        if timestamp <= self._current_day:
            return
        self._day_count += 1
        if self._rebalance_every <= 1 or self._day_count % self._rebalance_every == 1:
            ranked = sorted(self._today_signals, key=lambda s: self._today_signals[s], reverse=True)
            self._target = set(ranked[: self._top_n])
        self._today_signals = {}
        self._current_day = timestamp

    def _signal(self, buffer: list[Bar]) -> float | None:
        frame = pd.DataFrame(
            {
                "open": [b.open for b in buffer],
                "high": [b.high for b in buffer],
                "low": [b.low for b in buffer],
                "close": [b.close for b in buffer],
                "volume": [b.volume for b in buffer],
            },
            index=pd.DatetimeIndex([b.timestamp for b in buffer]),
        )
        features = self._features_fn(frame)
        last = features.iloc[-1].to_numpy(dtype="float64")
        if not np.isfinite(last).all():
            return None
        return self._signal_generator.generate(features)

    def _decide(self, bar: Bar, signal: float | None) -> OrderIntent | None:
        held = self._positions.get(bar.symbol, 0.0)
        in_target = bar.symbol in self._target
        if held == 0.0:
            if not in_target:
                return None
            # sentiment veto gates NEW entries only — never exits
            if (
                self._sentiment_fn is not None
                and self._sentiment_fn(bar.symbol, bar.timestamp) < self._sentiment_veto
            ):
                return None
            return OrderIntent(
                symbol=bar.symbol,
                side="buy",
                qty=1.0,  # placeholder — RiskManager computes the real size
                signal_strength=signal if signal is not None else 1.0,
                strategy_id=_STRATEGY_ID,
                reference_price=bar.close,
            )
        timed_out = self._bars_held.get(bar.symbol, 0) >= self._max_hold
        if not in_target or timed_out:
            return OrderIntent(
                symbol=bar.symbol,
                side="sell",
                qty=held,
                signal_strength=signal if signal is not None else 0.0,
                strategy_id=_STRATEGY_ID,
                reference_price=bar.close,
            )
        return None
