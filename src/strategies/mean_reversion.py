"""Mean-reversion strategy: buy oversold, exit at the mean or after max hold.

The structurally opposite bet to momentum — entries fade weakness instead of
chasing strength. Signal arrives by INJECTION (SignalGenerator Protocol; the
composition root wires ZScoreSignalGenerator): signal >= entry_signal means
"oversold enough to enter"; signal <= exit_signal means "price is back at or
above its mean". A time stop (max_hold_bars) bounds every holding period —
a reversion thesis that has not paid within the window is wrong.

Entry quantity is a placeholder (1.0): position size is the RiskManager's
decision (Kelly + caps), never the strategy's. Exits request the full held
quantity — reducing orders pass through risk unchanged.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from src.data.models import Bar
from src.shared.interfaces import SignalGenerator
from src.strategies.base import AbstractStrategy, OrderIntent

FeaturesFn = Callable[[pd.DataFrame], pd.DataFrame]

_STRATEGY_ID = "mean_reversion"


class MeanReversionZScore(AbstractStrategy):
    def __init__(
        self,
        config: dict[str, Any],
        signal_generator: SignalGenerator,
        features_fn: FeaturesFn,
    ) -> None:
        super().__init__(config, signal_generator)
        strategy_cfg = config["strategy"]
        mr_cfg = strategy_cfg["mean_reversion"]
        self._symbols = [str(s) for s in strategy_cfg["universe"]]
        self._warmup = int(strategy_cfg["warmup_bars"])
        self._entry_signal = float(mr_cfg["entry_signal"])
        self._exit_signal = float(mr_cfg["exit_signal"])
        self._max_hold = int(mr_cfg["max_hold_bars"])
        self._features_fn = features_fn
        self._bars: dict[str, list[Bar]] = {}
        self._positions: dict[str, float] = {}
        self._bars_held: dict[str, int] = {}

    @property
    def universe(self) -> list[str]:
        return list(self._symbols)

    @property
    def is_ready(self) -> bool:
        return any(len(bars) >= self._warmup for bars in self._bars.values())

    def on_bar(self, bar: Bar) -> OrderIntent | None:
        if bar.symbol not in self._symbols:
            return None
        buffer = self._bars.setdefault(bar.symbol, [])
        buffer.append(bar)
        if len(buffer) > self._warmup:  # features only need warmup bars — O(warmup) not O(n)
            del buffer[: len(buffer) - self._warmup]
        if len(buffer) < self._warmup:
            return None
        if self._positions.get(bar.symbol, 0.0) > 0.0:
            self._bars_held[bar.symbol] = self._bars_held.get(bar.symbol, 0) + 1
        features = self._features(buffer)
        if features is None:
            return None  # indicator warmup not complete for this symbol yet
        signal = self._signal_generator.generate(features)
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

    # --- internals ----------------------------------------------------------------

    def _features(self, buffer: list[Bar]) -> pd.DataFrame | None:
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
        return features

    def _decide(self, bar: Bar, signal: float) -> OrderIntent | None:
        held = self._positions.get(bar.symbol, 0.0)
        if held == 0.0:
            if signal >= self._entry_signal:
                return OrderIntent(
                    symbol=bar.symbol,
                    side="buy",
                    qty=1.0,  # placeholder — RiskManager computes the real size
                    signal_strength=signal,
                    strategy_id=_STRATEGY_ID,
                    reference_price=bar.close,
                )
            return None
        reverted = signal <= self._exit_signal
        timed_out = self._bars_held.get(bar.symbol, 0) >= self._max_hold
        if reverted or timed_out:
            return OrderIntent(
                symbol=bar.symbol,
                side="sell",
                qty=held,
                signal_strength=signal,
                strategy_id=_STRATEGY_ID,
                reference_price=bar.close,
            )
        return None
