"""Phase 2 — HMM regime detector tests (docs/plan/PHASE_2_SIGNALS.md)."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from src.shared.exceptions import ConfigError, DataValidationError
from src.signals.regime_hmm import RegimeDetector

_FEATURES = ["log_ret_1", "vol_short"]


class RecordingModel:
    """Records the array shape it was asked to predict on."""

    def __init__(self) -> None:
        self.received: np.ndarray | None = None

    def predict(self, x: np.ndarray) -> np.ndarray:
        self.received = np.asarray(x)
        return np.arange(len(x)) % 3


class TransformOnlyScaler:
    """transform() passes through; fit() must never be called at inference."""

    def __init__(self) -> None:
        self.transformed: bool = False

    def transform(self, x: np.ndarray) -> np.ndarray:
        self.transformed = True
        return np.asarray(x)

    def fit(self, x: np.ndarray) -> None:
        raise AssertionError("scaler.fit() called at inference time")


@pytest.fixture
def artifact_dir(tmp_path: Path) -> Path:
    out = tmp_path / "hmm"
    out.mkdir()
    joblib.dump(RecordingModel(), out / "model.pkl")
    joblib.dump(TransformOnlyScaler(), out / "scaler.pkl")
    (out / "features.json").write_text(json.dumps(_FEATURES), encoding="utf-8")
    return out


def _feature_df(n: int = 100, warmup_nan: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    df = pd.DataFrame(rng.normal(size=(n, 2)), columns=_FEATURES)
    df["unrelated_extra"] = 0.0  # detector must select its own columns
    if warmup_nan:
        df.iloc[:warmup_nan, 0] = np.nan
    return df


def test_missing_artifact_fails_fast(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"model\.pkl"):
        RegimeDetector(tmp_path)


def test_full_sequence_passed_to_predict(artifact_dir: Path) -> None:
    """The whole window goes through Viterbi — never a single reshaped row."""
    detector = RegimeDetector(artifact_dir)
    regime = detector.current_regime(_feature_df(100))
    assert detector._model.received is not None
    assert detector._model.received.shape == (100, 2)  # full sequence, own columns
    assert regime == 99 % 3  # states[-1] of the recorded sequence


def test_scaler_transform_applied(artifact_dir: Path) -> None:
    detector = RegimeDetector(artifact_dir)
    detector.current_regime(_feature_df(50))
    assert detector._scaler.transformed  # normalized before predict (never fit)


def test_warmup_rows_dropped(artifact_dir: Path) -> None:
    detector = RegimeDetector(artifact_dir)
    detector.current_regime(_feature_df(100, warmup_nan=30))
    assert detector._model.received.shape == (70, 2)


def test_missing_feature_column_raises(artifact_dir: Path) -> None:
    detector = RegimeDetector(artifact_dir)
    with pytest.raises(ConfigError, match="vol_short"):
        detector.current_regime(pd.DataFrame({"log_ret_1": [0.1]}))


def test_all_nan_window_raises(artifact_dir: Path) -> None:
    detector = RegimeDetector(artifact_dir)
    with pytest.raises(DataValidationError, match="no complete"):
        detector.current_regime(_feature_df(10, warmup_nan=10))
