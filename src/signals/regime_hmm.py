"""Market regime detection — Gaussian HMM (trained by scripts/train_hmm.py).

CRITICAL (src/signals/CLAUDE.md <hmm_critical>): features are normalized with
the persisted StandardScaler before ANY predict — without normalization vol
(~0.015) dominates log returns (~0.001) and the HMM silently collapses to one
state. Prediction runs Viterbi on the FULL sequence, then takes states[-1] —
never a single reshaped observation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.shared.exceptions import ConfigError, DataValidationError


def _load_artifact(path: Path) -> Any:
    if not path.exists():
        raise ConfigError(f"missing model artifact: {path}")
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return joblib.load(path)


class RegimeDetector:
    """Fail-fast artifact loading in __init__; pure computation afterwards."""

    def __init__(self, artifact_path: Path) -> None:
        self._model = _load_artifact(artifact_path / "model.pkl")
        self._scaler = _load_artifact(artifact_path / "scaler.pkl")
        features = _load_artifact(artifact_path / "features.json")
        if not isinstance(features, list) or not features:
            raise ConfigError(f"features.json must be a non-empty list: {artifact_path}")
        self._features: list[str] = features

    def current_regime(self, feature_df: pd.DataFrame) -> int:
        missing = set(self._features) - set(feature_df.columns)
        if missing:
            raise ConfigError(f"regime features missing from input: {sorted(missing)}")
        window = feature_df[self._features].dropna()
        if window.empty:
            raise DataValidationError("current_regime() called with no complete feature rows")
        x = self._scaler.transform(window.to_numpy(dtype="float64"))  # transform only
        states = self._model.predict(x)  # full sequence — Viterbi, not a single row
        return int(states[-1])
