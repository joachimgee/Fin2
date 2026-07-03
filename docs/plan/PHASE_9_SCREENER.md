# Phase 9 — Universe screener (12k Alpaca assets → tradable universe)

> Context to load: this file + `src/data/storage.py` + `src/data/screener.py`
> (new) + `scripts/run_screener.py` (new). Prerequisite: Phase 8 GATE green.

## Objective

Turn ~12,000 Alpaca assets into a point-in-time universe of `universe_size`
symbols, offline and on a schedule — never at runtime. The live system only
ever sees the current universe file; the WebSocket subscribes to those
symbols only.

## The funnel (each stage cheap enough to pay for the next)

| Stage | Input → output | Cost | Criteria (all from config `screener:`) |
|---|---|---|---|
| 1. Static | ~12k → ~2-3k | 1 REST call, zero market data | `tradable`, `status=active`, exchange in {NYSE, NASDAQ, ARCA}, not OTC, `easy_to_borrow`/`shortable` if strategy shorts |
| 2. Liquidity/quality | ~2-3k → ~300-500 | DuckDB only | `min_price` (no penny stocks), `min_median_dollar_volume` (e.g. $5M/day over 60d), `min_history_days` (≥ 1y listed), `max_atr_pct` (untradable gappers out) |
| 3. Ranking | ~300-500 → `universe_size` (30-50) | DuckDB only | composite rank (momentum + liquidity + inverse vol) with `max_per_sector` diversification — feeds the risk sector cap |

**The data trick that makes stage 2 affordable**: never fetch 12k symbols
individually. Polygon's **grouped daily** endpoint
(`/v2/aggs/grouped/locale/us/market/stocks/{date}`) returns the ENTIRE
market's OHLCV for one day in ONE request. Two years of daily history for
all 12k symbols ≈ 500 requests total, upserted into the same DuckDB `bars`
table. `PolygonClient` gains a `fetch_grouped_daily(date)` method.

## Point-in-time discipline (the screener's lookahead trap)

- The screener runs on a schedule (weekly, config `rebalance`) and writes
  `universe/{run_date}.yaml`: symbols + the criteria values that selected them.
- Live sessions load the LATEST file dated ≤ today.
- **Backtest/WFO must reconstitute the universe as it was at each point in
  time** — trading period T uses the universe file dated ≤ start of T, never
  today's. Screening on today's survivors and backtesting yesterday is
  selection bias: the same silent inflation as yfinance's missing Enron.
- Selection criteria use only data ≤ run_date (same truncation-invariance
  test pattern as features).

## Files

1. `src/data/screener.py` — PURE functions (data → shared only, no alpaca-py):
   `apply_static_filters(assets: list[dict], cfg)`,
   `apply_liquidity_filters(daily: pd.DataFrame, cfg)`,
   `rank_universe(candidates: pd.DataFrame, sector_map, cfg) -> list[str]`.
   Asset metadata arrives as plain dicts — fetched by the script, not here.
2. `src/execution/broker.py` — add `list_assets()` to `AlpacaBrokerClient`
   (concrete-only; the ABC keeps the strategy/backtest surface minimal),
   normalized to plain dicts.
3. `src/data/polygon_client.py` — add `fetch_grouped_daily(date)`.
4. `scripts/run_screener.py` — composition: assets via broker, grouped-daily
   sync into DuckDB, funnel, write `universe/{date}.yaml`, log a funnel
   summary (counts per stage).
5. `config/base.yaml` — new `screener:` section (+ `_SCHEMA`): `min_price`,
   `min_median_dollar_volume`, `dollar_volume_window`, `min_history_days`,
   `max_atr_pct`, `universe_size`, `max_per_sector`, `rebalance`.

## Tests

`tests/test_screener.py`: `test_static_filters_drop_otc_and_inactive`,
`test_min_price_filter`, `test_median_dollar_volume_filter`,
`test_min_history_filter`, `test_ranking_respects_universe_size`,
`test_ranking_respects_max_per_sector`,
`test_point_in_time_no_future_data` (truncation invariance on the ranking),
`test_funnel_deterministic_given_same_inputs`.

## GATE 9

```bash
make check
python scripts/run_screener.py --dry-run   # full funnel on synthetic DuckDB fixtures
```

## Pitfalls

- Do not stream or fetch quotes for 12k symbols "to be thorough" — the
  funnel exists precisely so the expensive stages see hundreds, not thousands.
- Alpaca's asset list is TODAY's list — it contains no delisted names. That
  is fine for selecting what to trade NOW, but backtests still price
  everything from Polygon (which includes delisted stocks) so performance
  numbers stay honest (ADR-004).
- A symbol leaving the universe does NOT force-close its position — exits
  belong to the strategy; the screener only stops NEW entries.
