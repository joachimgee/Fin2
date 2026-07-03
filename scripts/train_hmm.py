"""Offline HMM regime-model training — produces the artifact dir consumed by
RegimeDetector. Composition-side: may import data/signals/shared, never
execution/.

Scaler and HMM are fit on the TRAIN fraction only. Features are the shifted
(lookahead-safe) columns from compute_features, so a regime at T uses only
data <= T-1 — exactly what live sees.

Usage: python scripts/train_hmm.py --symbol SPY --name regime_hmm
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import joblib
import yaml
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from src.data.storage import BarStorage
from src.shared.config import load_config
from src.signals.features import compute_features

log = logging.getLogger(__name__)

_REGIME_FEATURES = ["log_ret_1", "vol_short"]


def train(args: argparse.Namespace) -> Path:
    config = load_config(Path(args.config))
    t = config["signals"]["training"]

    bars = BarStorage(Path(config["data"]["db_path"])).get_bars(
        args.symbol,
        datetime.fromisoformat(args.start).replace(tzinfo=UTC),
        datetime.fromisoformat(args.end).replace(tzinfo=UTC),
    )
    bars = bars.set_index("timestamp").sort_index()

    window = compute_features(bars, config)[_REGIME_FEATURES].dropna()
    train_end = int(len(window) * t["train_frac"])
    train_x = window.iloc[:train_end].to_numpy(dtype="float64")

    # CRITICAL: scale before fit — unscaled features silently collapse the
    # HMM to one state (vol ~0.015 dominates log_ret ~0.001).
    scaler = StandardScaler().fit(train_x)  # fit on train ONLY
    model = GaussianHMM(
        n_components=t["hmm_states"], covariance_type="full", n_iter=200, random_state=42
    )
    model.fit(scaler.transform(train_x))

    out = Path(args.output_root) / f"{args.name}_{datetime.now(tz=UTC):%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=False)
    joblib.dump(model, out / "model.pkl")
    joblib.dump(scaler, out / "scaler.pkl")
    (out / "features.json").write_text(json.dumps(_REGIME_FEATURES), encoding="utf-8")
    (out / "metadata.yaml").write_text(
        yaml.safe_dump(
            {
                "symbol": args.symbol,
                "n_states": t["hmm_states"],
                "training_period": [str(window.index[0]), str(window.index[train_end - 1])],
                "converged": bool(model.monitor_.converged),
                "data_source": config["data"]["data_source"],
            }
        ),
        encoding="utf-8",
    )
    log.info("hmm_training_completed", extra={"artifact_dir": str(out)})
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
