"""A/B: does the FinBERT news-sentiment feature carry predictive information?

Same WFO window slicing as run_backtest, but the measurement is OOS AUC of
the LightGBM classifier — the sharpest instrument for "is there information
here": if AUC does not move, no downstream engine plumbing can save it.

Per window: train variant A (base features) and variant B (base +
news_sentiment merged from DuckDB, no-news days = 0.0 neutral) on the SAME
in-sample bars, then score both on the SAME untouched out-of-sample rows.
Paired per-window comparison, summary written to backtest_results/.

Usage: python3 -m scripts.ab_sentiment --start 2020-07-01
"""

from __future__ import annotations

import argparse
import logging
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

import joblib
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score
from src.backtest.wfo import make_windows
from src.data.storage import BarStorage
from src.monitoring.logging_setup import setup_logging
from src.shared.config import load_config, load_universe
from src.signals.features import compute_features
from src.signals.labels import ewma_volatility, triple_barrier_labels

from scripts.train_lgbm import train_lgbm_artifacts

log = logging.getLogger(__name__)


def merge_sentiment(bars: pd.DataFrame, storage: BarStorage) -> pd.DataFrame:
    """Attach the news_sentiment column per (symbol, UTC calendar date).
    Days without stored headlines are 0.0 — neutral BY DEFINITION (no news
    is absence of signal, not missing market data)."""
    frames = []
    for symbol, symbol_bars in bars.groupby("symbol"):
        sentiment = storage.get_daily_sentiment(str(symbol))
        merged = symbol_bars.assign(date=symbol_bars["timestamp"].dt.date)
        if len(sentiment):
            sentiment = sentiment.assign(date=pd.to_datetime(sentiment["date"]).dt.date)
            merged = merged.merge(
                sentiment[["date", "score"]].rename(columns={"score": "news_sentiment"}),
                on="date",
                how="left",
            )
        else:
            merged["news_sentiment"] = 0.0
        merged["news_sentiment"] = merged["news_sentiment"].fillna(0.0)
        frames.append(merged.drop(columns="date"))
    return pd.concat(frames).sort_values("timestamp")


def oos_auc(artifact_dir: Path, oos_bars: pd.DataFrame, config: dict[str, Any]) -> float | None:
    """Apply saved scaler+model to OOS feature rows; None if AUC undefined."""
    model = joblib.load(artifact_dir / "model.pkl")
    scaler = joblib.load(artifact_dir / "scaler.pkl")
    t = config["signals"]["training"]
    frames = []
    for _, symbol_bars in oos_bars.groupby("symbol"):
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
        frames.append(features.assign(_label=labels))
    stacked = pd.concat(frames).sort_index(kind="stable")
    usable = stacked.drop(columns="_label").notna().all(axis=1) & stacked["_label"].notna()
    stacked = stacked[usable]
    if stacked.empty or stacked["_label"].nunique() < 2:
        return None
    x, y = stacked.drop(columns="_label"), stacked["_label"]
    proba = model.predict_proba(scaler.transform(x))[:, 1]
    return float(roc_auc_score(y, proba))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default=f"{datetime.now(tz=UTC):%Y-%m-%d}")
    parser.add_argument("--universe", default=None, help="YAML universe override file")
    parser.add_argument("--output-dir", default="backtest_results")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    setup_logging(str(config["monitoring"]["log_level"]))
    override = load_universe(args.universe)
    if override is not None:
        config["strategy"]["universe"] = override
    storage = BarStorage(Path(config["data"]["db_path"]))
    start = datetime.fromisoformat(args.start).replace(tzinfo=UTC)
    end = datetime.fromisoformat(args.end).replace(tzinfo=UTC)
    bars_a = pd.concat(
        [storage.get_bars(symbol, start, end) for symbol in config["strategy"]["universe"]]
    ).sort_values("timestamp")
    bars_b = merge_sentiment(bars_a, storage)

    warmup = int(config["strategy"]["warmup_bars"])
    day_list = sorted(bars_a["timestamp"].unique())
    days = pd.DataFrame({"timestamp": day_list})
    day_position = {day: i for i, day in enumerate(day_list)}
    wfo_cfg = config["wfo"]
    windows = make_windows(len(days), int(wfo_cfg["min_windows"]), int(wfo_cfg["is_oos_ratio"]))
    work_dir = Path(args.output_dir) / f"ab_sentiment_{datetime.now(tz=UTC):%Y%m%d_%H%M%S}"

    def span(bars: pd.DataFrame, day_slice: slice, lead_in: bool) -> tuple[pd.DataFrame, Any]:
        span_days = days.iloc[day_slice]
        span_start, span_end = span_days["timestamp"].iloc[0], span_days["timestamp"].iloc[-1]
        lead_start = day_list[max(0, day_position[span_start] - warmup)] if lead_in else span_start
        mask = (bars["timestamp"] >= lead_start) & (bars["timestamp"] <= span_end)
        return bars[mask].sort_values("timestamp"), span_start

    rows = []
    for w, (is_slice, oos_slice) in enumerate(windows):
        aucs: dict[str, float | None] = {}
        for variant, bars in (("a", bars_a), ("b", bars_b)):
            is_bars, train_start = span(bars, is_slice, lead_in=True)
            artifact_dir = train_lgbm_artifacts(
                is_bars, config, work_dir / f"w{w}_{variant}", train_start=train_start
            )
            oos_bars, _ = span(bars, oos_slice, lead_in=True)
            aucs[variant] = oos_auc(artifact_dir, oos_bars, config)
        rows.append({"window": w, "auc_base": aucs["a"], "auc_sentiment": aucs["b"]})
        log.info("ab_window", extra=rows[-1])

    defined = [r for r in rows if r["auc_base"] is not None and r["auc_sentiment"] is not None]
    summary = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "universe_size": int(bars_a["symbol"].nunique()),
        "windows": rows,
        "mean_auc_base": mean(r["auc_base"] for r in defined) if defined else None,
        "mean_auc_sentiment": mean(r["auc_sentiment"] for r in defined) if defined else None,
        "b_wins_windows": sum(1 for r in defined if r["auc_sentiment"] > r["auc_base"]),
    }
    out = work_dir / "summary.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")
    log.info(
        "ab_summary",
        extra={k: v for k, v in summary.items() if k not in ("windows", "generated_at")},
    )


if __name__ == "__main__":
    main()
