# Phase 3 — Risk (Kelly, circuit breakers, exposure, validate())

> Context to load: this file + `src/risk/*.py` (its CLAUDE.md loads
> automatically) + `src/strategies/base.py` (OrderIntent).
> Prerequisite: Phase 2 GATE green.

## Objective

The mandatory gatekeeper becomes real: `validate()` runs the 7-step pipeline
in order, synchronous, < 1 ms, zero I/O. When in doubt, reject.

## Allowed imports

`risk/` → `strategies.base` (OrderIntent ONLY) + `shared`. No monitoring/, no
execution/, no network clients. `test_risk_only_imports_strategies_base`
already enforces the first rule.

## Files

### 1. `src/risk/kelly.py` — implement `kelly_fraction()`

Guards first, math second:
1. `fraction > 0.50` → raise `InvalidKellyFractionError` (hard cap: half-Kelly).
2. `avg_loss <= 0` or `avg_win <= 0` → return `0.0`.
3. `full = win_rate − (1 − win_rate) / (avg_win / avg_loss)`.
4. `full <= 0` (no edge) → return `0.0`.
5. Return `full * fraction`.

### 2. `src/risk/circuit_breaker.py` — implement `CircuitBreaker`

- `__init__(thresholds: dict, on_trip: Callable[[str, dict], None])` —
  thresholds from `config risk.circuit_breakers`; **alert dispatch is an
  injected callback** because risk/ may not import monitoring/ (DIP — the
  composition root wires it to Telegram in Phase 8).
- State: `_halted`, `_start_of_day_equity`, `_peak_equity`, `_consecutive_losses`.
- `on_equity_update(equity)`: update peak; trip on `equity < (1 − max_drawdown_pct) × peak`
  or daily P&L `< −daily_loss_pct × start_of_day`.
- `on_trade_closed(pnl)`: losing streak counter; trip at `consecutive_losses`.
- Trip = set `_halted`, `log.critical("circuit_breaker_tripped", extra={reason, values})`,
  call `on_trip`. Never auto-cancel orders. No auto-reset of any kind —
  `reset_circuit_breaker()` is the only way back, called by a human.

### 3. `src/risk/exposure_tracker.py` — implement `ExposureTracker`

- All public methods under one `threading.Lock` (stream thread + validate()).
- `on_fill(event)`: update qty/avg-price per symbol, realized P&L.
- `sync_from_api(positions, equity)`: full state replacement (startup/reconnect only).
- Reads for validate(): `position_value(symbol)`, `total_exposure()`,
  `sector_exposure(sector)`, `equity` — all O(1)/O(n_positions), no I/O.

### 4. `src/risk/manager.py` — implement `RiskManager.validate()`

`__init__(config, tracker, breaker, correlation_provider, sector_map)`:
- `correlation_provider.get(sym_a, sym_b) -> float | None` — injected;
  correlations are precomputed offline (validate() itself never computes them).
- `sector_map: dict[symbol, sector]` from YAML config.

Pipeline — exact order, first failure returns immediately:
1. `breaker.trading_halted` → reject `"circuit_breaker"`.
2. Kelly → `adjusted_qty` from `intent.signal_strength` and strategy stats.
3. Risk-per-trade cap (`max_risk_per_trade_pct`) → **reduce** qty.
4. Position cap (`max_position_pct`) → **reduce** qty.
5. Total exposure cap (`max_total_exposure_pct`) → **reject** if still breached.
6. Sector cap (`max_sector_exposure_pct`) → **reject**.
7. Correlation guard (`max_correlation`) → **reject** if > threshold vs any
   open position (unknown correlation → treat as 0, do not reject).

Return `ValidationResult(approved, adjusted_qty, reason)`; `reason` = step name.
Also implement `on_fill(event)` forwarding to tracker + breaker.

## Tests

`tests/test_kelly.py`: `test_above_half_kelly_raises`, `test_no_edge_returns_zero`,
`test_zero_avg_loss_returns_zero`, `test_quarter_kelly_known_value` (hand-computed).

`tests/test_circuit_breaker.py`: `test_daily_loss_trips`, `test_drawdown_trips`,
`test_five_consecutive_losses_trip`, `test_win_resets_loss_streak`,
`test_no_auto_reset_on_new_day`, `test_manual_reset_restores`,
`test_trip_calls_injected_callback`.

`tests/test_exposure_tracker.py`: `test_on_fill_updates_position`,
`test_sync_replaces_all_state`, `test_concurrent_fills_consistent`
(N threads × M fills, final qty exact).

`tests/test_risk_manager.py`: `test_halted_rejected_before_anything_else`,
`test_pipeline_stops_at_first_failure`, `test_caps_reduce_not_reject`,
`test_exposure_cap_rejects`, `test_sector_cap_rejects`,
`test_correlation_above_080_rejects`, `test_unknown_correlation_passes`,
`test_result_carries_adjusted_qty`, `test_validate_has_no_awaits`
(inspect: `validate` is not a coroutine function).

## GATE 3

```bash
make check
```

## Definition of done

- [ ] No `NotImplementedError` left in `src/risk/`
- [ ] All tests above pass; `make check` green
- [ ] Progress ticked; committed `Phase 3: risk gatekeeper`

## Pitfalls

- Steps 3–4 reduce; steps 5–7 reject. Mixing these semantics up produces a
  gatekeeper that either over-trades or never trades.
- All seven thresholds come from the config dict — a literal `0.02` in
  manager.py is warn-list violation #6.
- No `async def validate` ever. If something needs I/O it belongs outside
  the pipeline (precomputed and injected).
