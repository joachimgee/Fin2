"""LightGBM directional signal generator.

Loads the artifact triple produced by training (src/signals/CLAUDE.md):
    models/{strategy_name}_{timestamp}/
      model.pkl     (lgb.LGBMClassifier)
      scaler.pkl    (StandardScaler fitted on train only)
      features.json (exact feature order used at training time)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.signals.base import AbstractSignalGenerator


class LightGBMSignalGenerator(AbstractSignalGenerator):
    """TODO(Phase 2):
    - __init__(artifact_path: Path): load all three artifact files;
      fail-fast at startup if any is missing or feature list mismatches.
    - generate(features): reorder columns to features.json order,
      scaler.transform (never fit), predict_proba -> map to [-1, 1],
      return self._clamp(signal).  Deterministic, < 5 ms, no I/O.
    """

    def __init__(self, artifact_path: Path) -> None:
        raise NotImplementedError("Phase 2 — LightGBM signal")

    def generate(self, features: Any) -> float:
        raise NotImplementedError("Phase 2")
