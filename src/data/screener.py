"""Universe screener — pure funnel stages (docs/plan/PHASE_9_SCREENER.md).

Three stages, each cheap enough to pay for the next:
  1. apply_static_filters:    ~12k assets -> ~2-3k   (metadata only)
  2. apply_liquidity_filters: ~2-3k -> ~300-500      (DuckDB daily bars only)
  3. rank_universe:           ~300-500 -> universe_size

Point-in-time discipline: apply_liquidity_filters takes as_of and uses only
bars <= as_of — ranking at time T is invariant to any data after T (same
truncation-invariance property as features, and tested the same way).
Asset metadata arrives as plain dicts; this module never imports alpaca-py.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)


def apply_static_filters(
    assets: list[dict[str, Any]], screener_cfg: dict[str, Any]
) -> list[dict[str, Any]]:
    """Stage 1: active, tradable, listed on an allowed exchange. No market data."""
    exchanges = {str(x) for x in screener_cfg["exchanges"]}
    kept = [
        asset
        for asset in assets
        if str(asset.get("status", "")).lower() == "active"
        and bool(asset.get("tradable"))
        and str(asset.get("exchange", "")) in exchanges
    ]
    log.info("screener_static_filtered", extra={"in": len(assets), "out": len(kept)})
    return kept


def apply_liquidity_filters(
    daily: pd.DataFrame, screener_cfg: dict[str, Any], as_of: pd.Timestamp | None = None
) -> pd.DataFrame:
    """Stage 2: liquidity/quality metrics per symbol from daily bars.

    Returns a symbol-indexed frame with the ranking inputs
    (close, median_dollar_volume, atr_pct, momentum, history_days).
    Only bars <= as_of are consulted — the point-in-time guarantee.
    """
    if as_of is not None:
        daily = daily[daily["timestamp"] <= as_of]
    min_price = float(screener_cfg["min_price"])
    min_mdv = float(screener_cfg["min_median_dollar_volume"])
    dv_window = int(screener_cfg["dollar_volume_window"])
    min_history = int(screener_cfg["min_history_days"])
    max_atr_pct = float(screener_cfg["max_atr_pct"])
    momentum_window = int(screener_cfg["rank_momentum_window"])

    rows: list[dict[str, Any]] = []
    for symbol, bars in daily.sort_values("timestamp").groupby("symbol"):
        if len(bars) < max(min_history, momentum_window + 1):
            continue
        last_close = float(bars["close"].iloc[-1])
        if last_close < min_price:
            continue
        tail = bars.tail(dv_window)
        median_dollar_volume = float((tail["close"] * tail["volume"]).median())
        if median_dollar_volume < min_mdv:
            continue
        atr_pct = float(((tail["high"] - tail["low"]) / tail["close"]).mean())
        if atr_pct > max_atr_pct:
            continue
        momentum = last_close / float(bars["close"].iloc[-(momentum_window + 1)]) - 1.0
        rows.append(
            {
                "symbol": symbol,
                "close": last_close,
                "median_dollar_volume": median_dollar_volume,
                "atr_pct": atr_pct,
                "momentum": momentum,
                "history_days": len(bars),
            }
        )
    candidates = pd.DataFrame(rows).set_index("symbol") if rows else pd.DataFrame()
    log.info(
        "screener_liquidity_filtered",
        extra={"in": int(daily["symbol"].nunique()) if len(daily) else 0, "out": len(candidates)},
    )
    return candidates


def rank_universe(
    candidates: pd.DataFrame, sector_map: dict[str, str], screener_cfg: dict[str, Any]
) -> list[str]:
    """Stage 3: composite rank (momentum + liquidity + inverse volatility),
    greedy selection under max_per_sector. Symbols without sector information
    are not sector-capped (no data is not a reason to exclude). Fully
    deterministic: ties broken by symbol."""
    if candidates.empty:
        return []
    universe_size = int(screener_cfg["universe_size"])
    max_per_sector = int(screener_cfg["max_per_sector"])
    score = (
        candidates["momentum"].rank()
        + candidates["median_dollar_volume"].rank()
        + (-candidates["atr_pct"]).rank()
    )
    ordered = (
        pd.DataFrame({"score": score.to_numpy(), "symbol": score.index.to_numpy()})
        .sort_values(["score", "symbol"], ascending=[False, True], kind="stable")["symbol"]
        .tolist()
    )
    selected: list[str] = []
    per_sector: dict[str, int] = {}
    for symbol in ordered:
        sector = sector_map.get(symbol)
        if sector is not None and per_sector.get(sector, 0) >= max_per_sector:
            continue
        selected.append(symbol)
        if sector is not None:
            per_sector[sector] = per_sector.get(sector, 0) + 1
        if len(selected) == universe_size:
            break
    log.info("screener_ranked", extra={"in": len(candidates), "out": len(selected)})
    return selected
