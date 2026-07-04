"""ZScoreSignalGenerator tests — hand-computed mappings, zero I/O."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.shared.exceptions import ConfigError, DataValidationError
from src.shared.interfaces import SignalGenerator
from src.signals.zscore_signal import ZScoreSignalGenerator


def _features(z: float) -> pd.DataFrame:
    return pd.DataFrame({"zscore": [0.0, z]})  # generate() must read the LAST row


def test_mapping_hand_computed() -> None:
    gen = ZScoreSignalGenerator(zscore_clip=3.0)
    assert gen.generate(_features(-3.0)) == 1.0  # maximally oversold
    assert gen.generate(_features(-1.5)) == 0.5
    assert gen.generate(_features(0.0)) == 0.0
    assert gen.generate(_features(1.5)) == -0.5
    assert gen.generate(_features(3.0)) == -1.0  # maximally overbought


def test_clamped_beyond_clip() -> None:
    gen = ZScoreSignalGenerator(zscore_clip=3.0)
    assert gen.generate(_features(-7.0)) == 1.0
    assert gen.generate(_features(7.0)) == -1.0


def test_satisfies_signal_generator_protocol() -> None:
    assert isinstance(ZScoreSignalGenerator(zscore_clip=3.0), SignalGenerator)


def test_missing_zscore_column_raises() -> None:
    with pytest.raises(ConfigError, match="zscore"):
        ZScoreSignalGenerator(zscore_clip=3.0).generate(pd.DataFrame({"rsi": [50.0]}))


def test_non_finite_zscore_raises() -> None:
    with pytest.raises(DataValidationError):
        ZScoreSignalGenerator(zscore_clip=3.0).generate(_features(float(np.nan)))


def test_non_positive_clip_rejected() -> None:
    with pytest.raises(ConfigError):
        ZScoreSignalGenerator(zscore_clip=0.0)
