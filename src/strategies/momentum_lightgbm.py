"""First concrete strategy: LightGBM momentum, long-only, HMM-regime gated.

Feature computation arrives by INJECTION (features_fn) — the dependency graph
forbids strategies -> signals, so the composition root wraps
signals.features.compute_features into a plain callable. Same for the signal
generator (SignalGenerator Protocol) and the optional regime detector.

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

_STRATEGY_ID = "momentum_lightgbm"


class MomentumLightGBM(AbstractStrategy):
    def __init__(
        self,
        config: dict[str, Any],
        signal_generator: SignalGenerator,
        features_fn: FeaturesFn,
        regime_detector: Any | None = None,  # duck-typed: .current_regime(features) -> int
    ) -> None:
        super().__init__(config, signal_generator)
        strategy_cfg = config["strategy"]
        self._symbols = [str(s) for s in strategy_cfg["universe"]]
        self._threshold = float(strategy_cfg["signal_threshold"])
        self._warmup = int(strategy_cfg["warmup_bars"])
        self._hostile = {int(r) for r in strategy_cfg["hostile_regimes"]}
        self._features_fn = features_fn
        self._regime = regime_detector
        self._bars: dict[str, list[Bar]] = {}
        self._positions: dict[str, float] = {}

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
        if len(buffer) < self._warmup:
            return None
        features = self._features(buffer)
        if features is None:
            return None  # indicator warmup not complete for this symbol yet
        signal = self._signal_generator.generate(features)
        return self._decide(bar, signal, features)

    def on_trade_update(self, update: dict[str, Any]) -> None:
        signed = float(update["qty"]) if update["side"] == "buy" else -float(update["qty"])
        self._positions[update["symbol"]] = self._positions.get(update["symbol"], 0.0) + signed

    def reset(self) -> None:
        self._bars.clear()
        self._positions.clear()

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

    def _decide(self, bar: Bar, signal: float, features: pd.DataFrame) -> OrderIntent | None:
        held = self._positions.get(bar.symbol, 0.0)
        if held == 0.0:
            # regime gate blocks NEW entries only — exits must always stay possible
            if self._regime is not None and self._regime.current_regime(features) in self._hostile:
                return None
            if signal >= self._threshold:
                return OrderIntent(
                    symbol=bar.symbol,
                    side="buy",
                    qty=1.0,  # placeholder — RiskManager computes the real size
                    signal_strength=signal,
                    strategy_id=_STRATEGY_ID,
                    reference_price=bar.close,
                )
            return None
        if held > 0.0 and signal <= -self._threshold:
            return OrderIntent(
                symbol=bar.symbol,
                side="sell",
                qty=held,
                signal_strength=signal,
                strategy_id=_STRATEGY_ID,
                reference_price=bar.close,
            )
        return None
