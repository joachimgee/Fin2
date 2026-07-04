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
from src.signals.labels import ewma_volatility, triple_barrier_labels

log = logging.getLogger(__name__)


def temporal_split(n: int, train_frac: float, valid_frac: float) -> tuple[slice, slice, slice]:
    """Chronological split — never shuffled (TimeSeriesSplit semantics)."""
    train_end = int(n * train_frac)
    valid_end = int(n * (train_frac + valid_frac))
    return slice(0, train_end), slice(train_end, valid_end), slice(valid_end, n)


def train_lgbm_artifacts(
    bars: pd.DataFrame,
    config: dict[str, Any],
    output_dir: Path,
    train_start: Any | None = None,
) -> Path:
    """Fit scaler + LGBM on the given bars ONLY and write the artifact triple.

    Multi-symbol aware (features computed per symbol, then stacked in time
    order). train_start: rows before it are a warmup LEAD-IN — they feed
    feature computation but are excluded from the training set. This is the
    per-window trainer used by the WFO (the model never sees anything
    outside the bars it is given).
    """
    t = config["signals"]["training"]
    frames = []
    for _, symbol_bars in bars.groupby("symbol"):
        indexed = symbol_bars.sort_values("timestamp").set_index("timestamp")
        features = compute_features(indexed, config)
        labels = triple_barrier_labels(
            indexed["high"],
            indexed["low"],
            indexed["close"],
            ewma_volatility(indexed["close"], t["label_vol_span"]),
            t["label_pt_mult"],
            t["label_sl_mult"],
            t["label_horizon_bars"],
        )
        frame = features.assign(_label=labels)
        if train_start is not None:
            frame = frame[frame.index >= train_start]
        frames.append(frame)
    stacked = pd.concat(frames).sort_index(kind="stable")
    usable = stacked.drop(columns="_label").notna().all(axis=1) & stacked["_label"].notna()
    stacked = stacked[usable]
    x_all, y_all = stacked.drop(columns="_label"), stacked["_label"]

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

    metrics: dict[str, Any] = {"n_training_rows": int(tr.stop)}
    for name, split in {"valid": va, "test": te}.items():
        x_split, y_split = x_all.iloc[split], y_all.iloc[split]
        if len(x_split) == 0 or y_split.nunique() < 2:
            continue  # split too small for meaningful metrics (per-window training)
        proba = model.predict_proba(scaler.transform(x_split))[:, 1]
        metrics[f"{name}_accuracy"] = float(accuracy_score(y_split, proba > 0.5))
        metrics[f"{name}_auc"] = float(roc_auc_score(y_split, proba))

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "model.pkl")
    joblib.dump(scaler, output_dir / "scaler.pkl")
    (output_dir / "features.json").write_text(json.dumps(list(x_all.columns)), encoding="utf-8")
    (output_dir / "metadata.yaml").write_text(
        yaml.safe_dump(
            {
                "symbols": sorted(bars["symbol"].unique().tolist()),
                "training_period": [str(x_all.index[tr][0]), str(x_all.index[tr][-1])],
                "label_horizon_bars": t["label_horizon_bars"],
                "label_pt_mult": t["label_pt_mult"],
                "label_sl_mult": t["label_sl_mult"],
                "label_vol_span": t["label_vol_span"],
                "metrics": metrics,
                "data_source": config["data"]["data_source"],
            }
        ),
        encoding="utf-8",
    )
    log.info("lgbm_training_completed", extra={"artifact_dir": str(output_dir), **metrics})
    return output_dir


def train(args: argparse.Namespace) -> Path:
    config = load_config(Path(args.config))
    storage = BarStorage(Path(config["data"]["db_path"]))
    start = datetime.fromisoformat(args.start).replace(tzinfo=UTC)
    end = datetime.fromisoformat(args.end).replace(tzinfo=UTC)
    bars = pd.concat(
        [storage.get_bars(symbol, start, end) for symbol in config["strategy"]["universe"]]
    )
    out = Path(args.output_root) / f"{args.name}_{datetime.now(tz=UTC):%Y%m%d_%H%M%S}"
    return train_lgbm_artifacts(bars, config, out)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default=f"{datetime.now(tz=UTC):%Y-%m-%d}")
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--output-root", default="models")
    train(parser.parse_args())  # trains on the full config strategy.universe


if __name__ == "__main__":
    main()
