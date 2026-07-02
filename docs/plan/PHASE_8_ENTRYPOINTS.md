# Phase 8 — Entrypoints (composition root)

> Context to load: this file + `scripts/` + `Makefile` + `config/base.yaml`.
> Prerequisite: Phase 7 GATE green. This is the LAST phase — everything it
> wires already exists and is tested.

## Objective

The composition root: the only place where concrete classes are constructed,
credentials are read, and dependencies are injected. Everything before this
phase depends on abstractions; this phase supplies the implementations (the
"D" of SOLID, made physical).

## Location rule

Entrypoints live in `scripts/` — OUTSIDE `src/`. The dependency graph governs
`src/` internals; the composition root legitimately imports every module and
therefore cannot live inside any of them.

## Files

### 1. `scripts/sync_data.py` (`make data-sync`)

`load_config` → `require_env("POLYGON_API_KEY")` → `PolygonClient` →
`fetch_bars` per symbol from config universe → `BarStorage.insert_bars` →
log summary (symbol, rows, span).

### 2. `scripts/run_backtest.py` (`make wfo S=<strategy>`)

Build storage → features/generators from artifacts → strategy → fresh
`ExposureTracker`/`CircuitBreaker`/`RiskManager` → `run_wfo` → print gate
table and the `cleared_for_paper` verdict; exit code 0 only if cleared.

### 3. `scripts/run_paper.py` (`make paper-trade S=<strategy>`)

Startup sequence — exact order:
1. `setup_logging(config)`.
2. **Clearance check**: load latest `backtest_results/{strategy}_*.yaml`;
   `cleared_for_paper: true` required or exit with a clear error. No result
   file = not cleared. This gate is not skippable by flag.
3. `paper = not config["execution"]["live_mode"]` — the ONLY line in the
   codebase where paper/live is decided (ADR-007).
4. Construct, in dependency order: redis client → `ExposureTracker` →
   `CircuitBreaker(thresholds, on_trip=dispatch_alert)` → correlation
   provider + sector map → `RiskManager` → `AlpacaBrokerClient(..., paper=paper)` →
   generators from artifacts → strategy (generator injected) → `StreamManager`.
5. Initial sync BEFORE any signal: `get_positions()` + `get_account()` →
   `tracker.sync_from_api(...)`.
6. `await stream_manager.start(strategy.universe)`; consume Redis bars →
   `strategy.on_bar` → `risk.validate` → `broker.submit_order` (this loop —
   the live engine — lives here or in a small `scripts/_live_loop.py` helper).
7. Graceful shutdown on SIGINT/SIGTERM: stop consuming, log final state.
   Open orders are left alone.

### 4. `Makefile` — point the three workflow targets at these scripts.

## Tests — `tests/test_entrypoints.py` (all wiring mocked)

| Test | Asserts |
|---|---|
| `test_paper_refuses_without_clearance` | no results YAML → SystemExit, no broker constructed |
| `test_paper_refuses_when_not_cleared` | `cleared_for_paper: false` → SystemExit |
| `test_paper_flag_is_not_live_mode` | `live_mode: false` → broker called with `paper=True` (and inverse) |
| `test_sync_runs_before_stream_start` | call-order recorded on mocks |
| `test_breaker_wired_to_telegram_dispatch` | `on_trip` is `dispatch_alert` |
| `test_wiring_smoke` | full construction with mocks raises nothing |

## GATE 8 — final

```bash
make check
python scripts/run_paper.py --strategy momentum_lightgbm   # without clearance file:
# must exit non-zero with "not cleared for paper" — that failure IS the pass criterion
```

## Definition of done

- [ ] All Progress rows in CODING_PLAN.md ticked
- [ ] `make check` green; clearance refusal demonstrated
- [ ] Committed `Phase 8: composition root + entrypoints`

## Pitfalls

- If a `src/` module "needs" something constructed here, pass it in — never
  let construction leak back into `src/` (that is how import cycles start).
- No `--live` CLI flag, ever. Live mode exists only as YAML (ADR-007).
- The order sync-before-stream (step 5 before 6) prevents validating against
  an empty tracker during the first seconds — do not "parallelize" it away.
