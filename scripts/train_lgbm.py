"""Offline LightGBM training — produces the artifact dir consumed by
LightGBMSignalGenerator. Composition-side: may import data/signals/shared,
never execution/.

Usage: python scripts/train_lgbm.py --symbol SPY --name momentum_lgbm
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from src.data.storage import BarStorage
from src.shared.config import load_config
from src.signals.features import compute_features

log = logging.getLogger(__name__)


def make_labels(close: pd.Series, horizon_bars: int) -> pd.Series:
    """1 if close rises over the next horizon_bars, else 0. Labels look
    forward BY DESIGN — legitimate in training only, never as a feature."""
    return (close.shift(-horizon_bars) > close).astype("float64")


def temporal_split(n: int, train_frac: float, valid_frac: float) -> tuple[slice, slice, slice]:
    """Chronological split — never shuffled (TimeSeriesSplit semantics)."""
    train_end = int(n * train_frac)
    valid_end = int(n * (train_frac + valid_frac))
    return slice(0, train_end), slice(train_end, valid_end), slice(valid_end, n)


def train(args: argparse.Namespace) -> Path:
    config = load_config(Path(args.config))
    t = config["signals"]["training"]

    bars = BarStorage(Path(config["data"]["db_path"])).get_bars(
        args.symbol,
        datetime.fromisoformat(args.start).replace(tzinfo=UTC),
        datetime.fromisoformat(args.end).replace(tzinfo=UTC),
    )
    bars = bars.set_index("timestamp").sort_index()

    features = compute_features(bars, config)
    labels = make_labels(bars["close"], t["label_horizon_bars"])
    valid_rows = features.notna().all(axis=1) & labels.notna()
    x_all, y_all = features[valid_rows], labels[valid_rows]

    tr, va, te = temporal_split(len(x_all), t["train_frac"], t["valid_frac"])
    scaler = StandardScaler().fit(x_all.iloc[tr])  # fit on train ONLY
    model = lgb.LGBMClassifier(
        n_estimators=t["n_estimators"],
        learning_rate=t["learning_rate"],
        num_leaves=t["num_leaves"],
        random_state=42,
        verbosity=-1,
    )
    model.fit(scaler.transform(x_all.iloc[tr]), y_all.iloc[tr])

    metrics: dict[str, Any] = {}
    for name, split in {"valid": va, "test": te}.items():
        proba = model.predict_proba(scaler.transform(x_all.iloc[split]))[:, 1]
        metrics[f"{name}_accuracy"] = float(accuracy_score(y_all.iloc[split], proba > 0.5))
        metrics[f"{name}_auc"] = float(roc_auc_score(y_all.iloc[split], proba))

    out = Path(args.output_root) / f"{args.name}_{datetime.now(tz=UTC):%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=False)
    joblib.dump(model, out / "model.pkl")
    joblib.dump(scaler, out / "scaler.pkl")
    (out / "features.json").write_text(json.dumps(list(x_all.columns)), encoding="utf-8")
    (out / "metadata.yaml").write_text(
        yaml.safe_dump(
            {
                "symbol": args.symbol,
                "training_period": [str(x_all.index[tr][0]), str(x_all.index[tr][-1])],
                "oos_period": [str(x_all.index[te][0]), str(x_all.index[te][-1])],
                "label_horizon_bars": t["label_horizon_bars"],
                "metrics": metrics,
                "data_source": config["data"]["data_source"],
            }
        ),
        encoding="utf-8",
    )
    log.info("lgbm_training_completed", extra={"artifact_dir": str(out), **metrics})
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--output-root", default="models")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
