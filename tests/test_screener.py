"""Phase 9 — screener funnel tests (docs/plan/PHASE_9_SCREENER.md)."""

from __future__ import annotations

from typing import Any

import pandas as pd
from src.data.screener import apply_liquidity_filters, apply_static_filters, rank_universe

_CFG: dict[str, Any] = {
    "exchanges": ["NYSE", "NASDAQ", "ARCA"],
    "min_price": 5.0,
    "min_median_dollar_volume": 5_000_000,
    "dollar_volume_window": 20,
    "min_history_days": 50,
    "max_atr_pct": 0.05,
    "rank_momentum_window": 20,
    "universe_size": 3,
    "max_per_sector": 2,
    "rebalance": "weekly",
}


def _asset(symbol: str, **overrides: Any) -> dict[str, Any]:
    asset = {"symbol": symbol, "exchange": "NYSE", "status": "active", "tradable": True}
    asset.update(overrides)
    return asset


def _daily(specs: dict[str, dict[str, Any]], days: int = 80) -> pd.DataFrame:
    """Synthetic daily bars: price drift and volume control each filter."""
    timestamps = pd.date_range("2024-01-02", periods=days, freq="B", tz="UTC")
    rows: list[dict[str, Any]] = []
    for symbol, spec in specs.items():
        n = int(spec.get("days", days))
        price = float(spec.get("price", 100.0))
        drift = float(spec.get("drift", 0.0))
        volume = int(spec.get("volume", 1_000_000))
        range_pct = float(spec.get("range_pct", 0.02))
        close = price
        for ts in timestamps[-n:]:
            close *= 1.0 + drift
            rows.append(
                {
                    "symbol": symbol,
                    "timestamp": ts,
                    "open": close,
                    "high": close * (1 + range_pct / 2),
                    "low": close * (1 - range_pct / 2),
                    "close": close,
                    "volume": volume,
                }
            )
    return pd.DataFrame(rows)


# --- stage 1: static ---------------------------------------------------------------


def test_static_filters_drop_otc_and_inactive() -> None:
    assets = [
        _asset("GOOD"),
        _asset("OTCX", exchange="OTC"),
        _asset("DEAD", status="inactive"),
        _asset("HALT", tradable=False),
        _asset("NASD", exchange="NASDAQ"),
    ]
    kept = {a["symbol"] for a in apply_static_filters(assets, _CFG)}
    assert kept == {"GOOD", "NASD"}


# --- stage 2: liquidity/quality -----------------------------------------------------


def test_min_price_filter() -> None:
    daily = _daily({"PENNY": {"price": 2.0, "volume": 50_000_000}, "OK": {}})
    assert list(apply_liquidity_filters(daily, _CFG).index) == ["OK"]


def test_median_dollar_volume_filter() -> None:
    daily = _daily({"THIN": {"volume": 10_000}, "OK": {}})  # 10k sh x $100 = $1M < $5M
    assert list(apply_liquidity_filters(daily, _CFG).index) == ["OK"]


def test_min_history_filter() -> None:
    daily = _daily({"IPO": {"days": 30}, "OK": {}})
    assert list(apply_liquidity_filters(daily, _CFG).index) == ["OK"]


def test_max_atr_filter() -> None:
    daily = _daily({"WILD": {"range_pct": 0.15}, "OK": {}})
    assert list(apply_liquidity_filters(daily, _CFG).index) == ["OK"]


# --- stage 3: ranking ----------------------------------------------------------------


def _candidates(n: int = 6) -> pd.DataFrame:
    # every metric strictly improves with i: momentum, liquidity, and lower range
    daily = _daily(
        {
            f"S{i}": {
                "drift": 0.001 * i,
                "volume": 1_000_000 + 100_000 * i,
                "range_pct": 0.03 - 0.002 * i,
            }
            for i in range(n)
        }
    )
    return apply_liquidity_filters(daily, _CFG)


def test_ranking_respects_universe_size() -> None:
    universe = rank_universe(_candidates(), {}, _CFG)
    assert len(universe) == _CFG["universe_size"]
    assert universe == ["S5", "S4", "S3"]  # highest momentum + liquidity first


def test_ranking_respects_max_per_sector() -> None:
    sector_map = {"S5": "tech", "S4": "tech", "S3": "tech", "S2": "energy"}
    universe = rank_universe(_candidates(), sector_map, _CFG)
    assert universe == ["S5", "S4", "S2"]  # third tech name displaced by the cap


def test_unmapped_symbols_are_not_sector_capped() -> None:
    universe = rank_universe(_candidates(), {}, _CFG)  # all unknown sectors
    assert len(universe) == 3  # no phantom "unknown" sector cap


# --- point-in-time + determinism ------------------------------------------------------


def test_point_in_time_no_future_data() -> None:
    daily = _daily({"A": {"drift": 0.002}, "B": {"drift": 0.001}, "LATE": {"drift": 0.0}})
    as_of = daily["timestamp"].sort_values().unique()[59]
    # LATE moons AFTER as_of — a point-in-time screener must not see it
    surge = daily[daily["symbol"] == "LATE"].copy()
    surge = surge[surge["timestamp"] > as_of]
    surge[["open", "high", "low", "close"]] *= 10.0
    full = pd.concat([daily[~daily.index.isin(surge.index)], surge])

    ranked_full = rank_universe(apply_liquidity_filters(full, _CFG, as_of=as_of), {}, _CFG)
    truncated = full[full["timestamp"] <= as_of]
    ranked_trunc = rank_universe(apply_liquidity_filters(truncated, _CFG, as_of=as_of), {}, _CFG)
    assert ranked_full == ranked_trunc  # future data changed nothing
    assert ranked_full[0] != "LATE"  # the post-as_of surge bought no rank


def test_funnel_deterministic_given_same_inputs() -> None:
    daily = _daily({f"S{i}": {"drift": 0.0005 * i} for i in range(8)})
    first = rank_universe(apply_liquidity_filters(daily, _CFG), {}, _CFG)
    second = rank_universe(
        apply_liquidity_filters(daily.sample(frac=1.0, random_state=1), _CFG), {}, _CFG
    )
    assert first == second  # input row order is irrelevant
