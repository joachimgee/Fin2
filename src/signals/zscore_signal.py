"""Rule-based mean-reversion signal: the negated close z-score, normalized.

signal = clamp(-zscore / zscore_clip): +1.0 when the close sits zscore_clip
standard deviations BELOW its rolling mean (maximally oversold), -1.0 when
equally overbought, 0.0 at the mean. The zscore feature comes from
signals.features.compute_features and is already shift(1)-ed — no new
lookahead surface is introduced here.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from src.shared.exceptions import ConfigError, DataValidationError
from src.signals.base import AbstractSignalGenerator


class ZScoreSignalGenerator(AbstractSignalGenerator):
    """Deterministic, model-free — nothing to load, nothing to fit."""

    def __init__(self, zscore_clip: float) -> None:
        if zscore_clip <= 0.0:
            raise ConfigError(f"zscore_clip must be > 0, got {zscore_clip}")
        self._clip = float(zscore_clip)

    def generate(self, features: Any) -> float:
        frame: pd.DataFrame = features
        if "zscore" not in frame.columns:
            raise ConfigError("features frame has no 'zscore' column")
        z = float(frame["zscore"].iloc[-1])
        if not math.isfinite(z):
            raise DataValidationError("generate() called on non-finite zscore (warmup?)")
        return self._clamp(-z / self._clip)
