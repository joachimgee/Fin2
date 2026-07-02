# Phase 6 — Backtest engine + Walk-Forward Optimization

> Context to load: this file + `src/backtest/*.py` + `src/strategies/base.py`
> + `src/shared/interfaces.py`. Prerequisite: Phase 5 GATE green.

## Objective

Event-driven replay through the SAME strategy→risk path as live, and the WFO
gate that decides `cleared_for_paper`. This phase proves the Liskov claim:
one strategy class, two engines, zero changes.

## Allowed imports

`backtest/` → strategies + signals + data + risk + shared. NEVER execution/ —
`SimulatedBroker` implements `shared.interfaces.AbstractBrokerClient` (ADR-001).

## Files

### 1. `src/backtest/engine.py` — implement `SimulatedBroker` + `BacktestEngine`

`SimulatedBroker(AbstractBrokerClient)`:
- Orders fill at the NEXT bar's open (never the signal bar's close — filling
  on the same bar is execution-side lookahead), adjusted by
  `slippage_bps` and `commission_per_share` from a new `backtest:` config
  section (add it to `config/base.yaml` + `_SCHEMA`).
- Emits synthetic fill events in the same dict shape the live TradingStream
  produces — strategies and risk must not be able to tell the difference.

`BacktestEngine.run()` loop per bar, in order:
1. `intent = strategy.on_bar(bar)`; `None` → next bar.
2. `result = risk.validate(intent)`; not approved → log, next bar. The
   engine physically cannot submit an unvalidated order — same invariant as live.
3. `SimulatedBroker.submit_order(...)` with `result.adjusted_qty`.
4. On next bar: generate fill → `strategy.on_trade_update(fill)` +
   `risk.on_fill(fill)`.
5. Track equity curve, trade ledger.

Output: metrics dict — `sharpe`, `max_drawdown`, `profit_factor`, `n_trades`,
equity series. Metrics helpers are pure functions in this module.

### 2. `src/backtest/wfo.py` — implement `run_wfo`

1. Split history into ≥ `wfo.min_windows` rolling (IS, OOS) windows —
   temporal order preserved, zero overlap between IS and its OOS.
2. Per window: Optuna optimizes params on IS ONLY (≤ 10 params — more is
   overfitting surface); run OOS once with the chosen params.
3. `wfe = mean(oos_metric) / mean(is_metric)` (annualized return or Sharpe —
   pick one, record it in the YAML).
4. Gates (all from `config wfo.*`): windows ≥ min, `wfe ≥ min_wfe`,
   OOS trades ≥ `min_oos_trades`, OOS Sharpe ≥ `min_oos_sharpe`,
   OOS max DD ≤ `max_oos_drawdown_pct`.
5. Write `backtest_results/{strategy}_{ts}.yaml`: every gate value, pass/fail
   per gate, `data_source: polygon`, and `cleared_for_paper: <all gates true>`.
6. WFE < 0.50 → the answer is abandon, not re-tune. `run_wfo` returns the
   result dict; it never loops "one more optimization".

## Tests

`tests/test_backtest_engine.py`: `test_fill_at_next_bar_open_not_signal_bar`,
`test_slippage_and_commission_applied`, `test_rejected_intent_never_fills`,
`test_fill_event_shape_matches_live_contract`,
`test_same_strategy_instance_reusable_after_reset`,
`test_equity_curve_matches_hand_computed_two_trade_case`.

`tests/test_wfo.py`: `test_windows_are_temporal_no_overlap`,
`test_optimization_never_sees_oos` (spy on data passed to the objective),
`test_wfe_computation_known_values`, `test_all_gates_pass_sets_cleared`,
`test_wfe_below_min_not_cleared`, `test_results_yaml_contains_all_gates`.

`tests/test_strategy_parity.py`: `test_liskov_same_intents_backtest_vs_replay` —
feed identical bars to one strategy through BacktestEngine and through a
minimal live-loop harness (mocked broker); assert identical intent sequence.

## GATE 6

```bash
make check
pytest tests/test_strategy_parity.py -v   # the Liskov proof, seen passing
```

## Definition of done

- [ ] No `NotImplementedError` left in `src/backtest/`
- [ ] Parity test passes; `make check` green
- [ ] Progress ticked; committed `Phase 6: backtest + WFO`

## Pitfalls

- Same-bar fills are the backtest twin of lookahead bias — the dedicated
  test must stay red until fills land on bar t+1.
- Optuna objective must be built ONLY from IS slices; passing the full frame
  "for convenience" invalidates every result silently.
- `cleared_for_paper` is computed, never hand-edited. The Phase 8 CLI trusts it.
