"""Phase 2 — LightGBM signal generator tests (docs/plan/PHASE_2_SIGNALS.md)."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler
from src.shared.exceptions import ConfigError, DataValidationError
from src.signals.lgbm_signal import LightGBMSignalGenerator

_COLS = ["f1", "f2", "f3"]


@pytest.fixture
def artifact_dir(tmp_path: Path) -> Path:
    """Tiny REAL artifacts: LGBM trained on a learnable rule (y = f1 > 0)."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 3))
    y = (x[:, 0] > 0).astype(int)
    scaler = StandardScaler().fit(x)
    model = lgb.LGBMClassifier(
        n_estimators=20, min_child_samples=5, random_state=0, verbosity=-1
    ).fit(scaler.transform(x), y)
    out = tmp_path / "artifacts"
    out.mkdir()
    joblib.dump(model, out / "model.pkl")
    joblib.dump(scaler, out / "scaler.pkl")
    (out / "features.json").write_text(json.dumps(_COLS), encoding="utf-8")
    return out


def _frame(values: list[float], columns: list[str] | None = None) -> pd.DataFrame:
    return pd.DataFrame([values], columns=columns or _COLS)


def test_missing_artifact_fails_fast(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"model\.pkl"):
        LightGBMSignalGenerator(tmp_path)


def test_generate_in_minus1_plus1_and_directional(artifact_dir: Path) -> None:
    gen = LightGBMSignalGenerator(artifact_dir)
    up = gen.generate(_frame([2.0, 0.0, 0.0]))
    down = gen.generate(_frame([-2.0, 0.0, 0.0]))
    assert -1.0 <= down <= 1.0 and -1.0 <= up <= 1.0
    assert up > down  # the model learned y = f1 > 0


def test_generate_deterministic(artifact_dir: Path) -> None:
    gen = LightGBMSignalGenerator(artifact_dir)
    frame = _frame([0.5, -0.3, 1.2])
    assert gen.generate(frame) == gen.generate(frame.copy())


def test_wrong_feature_set_raises(artifact_dir: Path) -> None:
    gen = LightGBMSignalGenerator(artifact_dir)
    with pytest.raises(ConfigError, match=r"missing.*f3"):
        gen.generate(_frame([1.0, 2.0], columns=["f1", "f2"]))
    with pytest.raises(ConfigError, match=r"extra.*f9"):
        gen.generate(_frame([1.0, 2.0, 3.0, 4.0], columns=[*_COLS, "f9"]))


def test_column_order_enforced_not_assumed(artifact_dir: Path) -> None:
    """Shuffled input columns must give the identical signal — the generator
    reorders to features.json order instead of trusting input order."""
    gen = LightGBMSignalGenerator(artifact_dir)
    canonical = _frame([0.7, -1.1, 0.4])
    shuffled = canonical[["f3", "f1", "f2"]]
    assert gen.generate(canonical) == gen.generate(shuffled)


def test_non_finite_row_raises(artifact_dir: Path) -> None:
    gen = LightGBMSignalGenerator(artifact_dir)
    with pytest.raises(DataValidationError, match="non-finite"):
        gen.generate(_frame([np.nan, 0.0, 0.0]))
