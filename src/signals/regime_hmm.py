"""Market regime detection — Gaussian HMM over SPY features (3 states).

CRITICAL (src/signals/CLAUDE.md <hmm_critical>): features MUST be normalized
with StandardScaler before model.fit(). Without it, vol (~0.015) dominates
log_return (~0.001) and the HMM silently collapses to a single state.
Prediction uses Viterbi on the FULL sequence, then takes states[-1] —
never model.predict() on a single reshaped observation.
"""

from __future__ import annotations

from pathlib import Path


class RegimeDetector:
    """TODO(Phase 2):
      - __init__(artifact_path: Path): load model.pkl + scaler.pkl + features.json,
        fail-fast at startup if any file is missing.
      - current_regime(feature_df) -> int:
          X_scaled = self._scaler.transform(X)   # transform only, never fit here
          states = self._model.predict(X_scaled) # full sequence (Viterbi)
          return int(states[-1])
      - train(...) lives in a separate offline script, scaler fit on train only.
    """

    def __init__(self, artifact_path: Path) -> None:
        raise NotImplementedError("Phase 2 — HMM regime detector")
