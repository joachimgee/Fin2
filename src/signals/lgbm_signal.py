"""LightGBM directional signal generator.

Loads the artifact triple produced by scripts/train_lgbm.py:
    models/{name}_{timestamp}/
      model.pkl     (lgb.LGBMClassifier)
      scaler.pkl    (StandardScaler fitted on train only)
      features.json (exact feature order used at training time)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.shared.exceptions import ConfigError, DataValidationError
from src.signals.base import AbstractSignalGenerator


def _load_artifact(path: Path) -> Any:
    if not path.exists():
        raise ConfigError(f"missing model artifact: {path}")
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return joblib.load(path)


class LightGBMSignalGenerator(AbstractSignalGenerator):
    """Fail-fast at startup: all three artifacts load in __init__ or the
    process never starts. generate() is pure computation — no I/O."""

    def __init__(self, artifact_path: Path) -> None:
        self._model = _load_artifact(artifact_path / "model.pkl")
        self._scaler = _load_artifact(artifact_path / "scaler.pkl")
        features = _load_artifact(artifact_path / "features.json")
        if not isinstance(features, list) or not features:
            raise ConfigError(f"features.json must be a non-empty list: {artifact_path}")
        self._features: list[str] = features
        classes = list(self._model.classes_)
        if 1 not in classes:
            raise ConfigError(f"model at {artifact_path} has no positive class: {classes}")
        self._up_index = classes.index(1)

    def generate(self, features: Any) -> float:
        frame: pd.DataFrame = features
        got, expected = set(frame.columns), set(self._features)
        if got != expected:
            raise ConfigError(
                f"feature set mismatch: missing={sorted(expected - got)} "
                f"extra={sorted(got - expected)}"
            )
        row = frame[self._features].iloc[-1:]  # reorder to training order, last bar
        values = row.to_numpy(dtype="float64")
        if not np.isfinite(values).all():
            raise DataValidationError("generate() called on non-finite feature row (warmup?)")
        scaled = self._scaler.transform(values)  # transform only — never fit here
        p_up = float(self._model.predict_proba(scaled)[0][self._up_index])
        return self._clamp(2.0 * p_up - 1.0)
