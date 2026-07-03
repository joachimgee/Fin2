# Phase 1 — Data (validation, DuckDB storage, Polygon sync)

> Context to load: this file + `src/data/*.py` + `src/shared/exceptions.py`.
> Prerequisite: Phase 0 GATE green.

## Objective

Every bar entering the system passes one validation gate; historical bars live
in DuckDB; Polygon.io is contacted only by the sync job. After this phase the
whole offline column of the architecture diagram has its data layer.

## Allowed imports

`data/` → `shared` only (+ stdlib, pandas/polars, duckdb, httpx).

## Files

### 1. `src/data/models.py` — implement `validate_bar(bar)`

Checks, in order; first failure raises `DataValidationError` with
`f"{bar.symbol}@{bar.timestamp}: <reason>"`:
1. No NaN/inf in open/high/low/close/vwap (`math.isfinite`).
2. All prices > 0; volume ≥ 0.
3. `low ≤ min(open, close)` and `max(open, close) ≤ high`.
4. `timestamp.tzinfo` is not None (must be tz-aware UTC).

Pure function: returns the same frozen `Bar`, mutates nothing.

### 2. `src/data/storage.py` — implement `BarStorage`

- `__init__`: `duckdb.connect(str(db_path))`; `CREATE TABLE IF NOT EXISTS bars(
  symbol VARCHAR, timestamp TIMESTAMPTZ, open DOUBLE, high DOUBLE, low DOUBLE,
  close DOUBLE, volume BIGINT, vwap DOUBLE, PRIMARY KEY(symbol, timestamp))`.
- `insert_bars(bars)`: `INSERT OR REPLACE`, parameterized `executemany`;
  returns count. Idempotent by design (re-sync overwrites, never duplicates).
- `get_bars(symbol, start, end)`: parameterized SELECT, `ORDER BY timestamp ASC`,
  returns pandas DataFrame. (Switch to polars only if a result exceeds ~500k rows.)

### 3. `src/data/polygon_client.py` — implement `PolygonClient.fetch_bars`

1. `httpx.AsyncClient` GET
   `/v2/aggs/ticker/{symbol}/range/1/{span}/{start}/{end}` with
   `adjusted=true` (non-negotiable — unadjusted data corrupts every backtest).
2. Map each result: `t` (ms epoch) → UTC datetime, `o/h/l/c/v/vw` → `Bar`.
3. Run every bar through `validate_bar` before returning — fail fast.
4. Retry 429/5xx with exponential backoff (1s, 2s, 4s, 8s, 16s max 5 tries);
   any other 4xx raises immediately. Timeframe mapping `"1Day"` → `1/day`.

## Tests

`tests/test_data_models.py`: `test_valid_bar_passes`, `test_nan_price_raises`,
`test_negative_price_raises`, `test_low_above_open_raises`,
`test_close_above_high_raises`, `test_naive_timestamp_raises`,
`test_zero_volume_ok`.

`tests/test_storage.py` (use `tmp_db_path`): `test_insert_then_get_roundtrip`,
`test_insert_same_bar_twice_one_row`, `test_get_bars_sorted_ascending`,
`test_get_bars_filters_symbol_and_range`.

`tests/test_polygon_client.py` (httpx `MockTransport`, zero network):
`test_maps_polygon_fields_to_bar`, `test_requests_adjusted_true`,
`test_retries_on_429_then_succeeds`, `test_invalid_bar_from_api_raises`,
`test_4xx_raises_no_retry`.

## GATE 1

```bash
make check
python - <<'EOF'   # storage smoke: insert 3 synthetic bars, read 3 back, sorted
# build 3 Bars by hand, BarStorage(tmp), insert, assert len==3
EOF
```

## Definition of done

- [ ] No `NotImplementedError` left in `src/data/`
- [ ] 16 tests above pass; `make check` green
- [ ] Progress ticked; committed `Phase 1: data layer`

## Pitfalls

- Never interpolate SQL strings — parameterized queries only.
- Polygon `t` is **milliseconds**; dividing by the wrong factor shifts every
  timestamp by ~50 years and validation won't catch it. Assert one known
  fixture timestamp exactly in the mapping test.
- Do not add yfinance "as a fallback". There is no fallback (ADR-004).
