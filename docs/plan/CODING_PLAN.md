# CODING_PLAN.md — Master index

Guides an AI session through building finbot phase by phase, without context
overload. The scaffold already exists: every file below is a stub with
signatures and TODOs — the work is **implementing the TODOs**, never redesigning
the contracts.

## Session protocol (follow exactly)

1. Read the **Progress** table below to find the current phase.
2. Open ONLY `docs/plan/PHASE_<N>_*.md` for that phase. Do NOT read other
   phase files. Root `CLAUDE.md` and the module's `CLAUDE.md` load automatically.
3. Read only the source files listed in that phase file (plus anything they import).
4. Implement, write the phase's tests, run the GATE.
5. GATE green → tick the Progress row (edit this file), commit
   `Phase N: <summary>`, push.
6. Next phase = next session (or `/compact` first). One phase per session.

A red GATE is never carried into the next phase. If a gate fails, fix it
within the phase. If a contract seems wrong, stop and say so — do not
work around it (see `ARCHITECTURE.md` ADRs for why contracts are where they are).

## Progress

| Phase | Doc | Delivers | GATE | Done |
|---|---|---|---|---|
| 0 | PHASE_0_FOUNDATIONS.md | config loader, test fixtures | `make check` | ☑ |
| 1 | PHASE_1_DATA.md | Bar validation, DuckDB storage, Polygon sync | `make check` | ☑ |
| 2 | PHASE_2_SIGNALS.md | features (lookahead-safe), LightGBM, HMM, training scripts | `make check` + anti-lookahead test | ☑ |
| 3 | PHASE_3_RISK.md | Kelly, circuit breakers, ExposureTracker, validate() | `make check` | ☑ |
| 4 | PHASE_4_EXECUTION.md | AlpacaBrokerClient, StreamManager | `make check` | ☑ |
| 5 | PHASE_5_LLM_FEATURES.md | Claude sentiment feature + Redis cache | `make check` | ☐ |
| 6 | PHASE_6_BACKTEST_WFO.md | SimulatedBroker, engine, WFO gates | `make check` | ☐ |
| 7 | PHASE_7_MONITORING.md | JSON logging, Telegram alerts | `make check` | ☐ |
| 8 | PHASE_8_ENTRYPOINTS.md | composition root: sync/backtest/paper CLIs | `make check` + dry-run | ☐ |
| 9 | PHASE_9_SCREENER.md | 12k-asset funnel → point-in-time universe files | `make check` + dry-run | ☐ |

Phase order is dependency order: each phase only imports modules completed in
earlier phases, so every phase is testable the moment it is written.

## Global rules (recap — full versions in CLAUDE.md)

- Import graph is law; `tests/test_architecture.py` enforces it in every GATE.
- All numeric parameters from `config/base.yaml`; credentials via `require_env()`.
- Structured logging only: `log.info("event_name", extra={...})`. No `print`.
- Async for all I/O. `RiskManager.validate()` stays sync, <1 ms, zero I/O.
- Tests define correctness, not the solution: never hardcode to pass a test.

## SOLID map (where each principle is enforced)

| Principle | Concrete enforcement |
|---|---|
| S | One module = one reason to change; phase boundaries match module boundaries |
| O | New strategy/signal = new file implementing `AbstractStrategy` / `SignalGenerator`; zero edits to engines |
| L | Any `AbstractStrategy` runs unchanged in `BacktestEngine` and the live engine (Phase 6 parity test) |
| I | `AbstractStrategy` = `on_bar, on_trade_update, is_ready, universe, reset` — nothing else is ever added |
| D | All concrete classes are constructed ONLY in the composition root (Phase 8) and injected: broker via `AbstractBrokerClient`, generators via `SignalGenerator` Protocol, alert dispatch via callback |

## Fin v1 harvest map (github.com/joachimgee/fin — reference quarry, not a base)

Fin v1 contains ~20k lines of working domain logic. Port the pieces below
through this project's gates — every port gets `.shift(1)` discipline, params
moved to YAML, and the dependency graph applied. Never copy a file wholesale.

| Phase | Consult in v1 | Take | Leave behind |
|---|---|---|---|
| 1 | `data/alpaca_ingestion.py` | retry/validation edge cases | Alpaca as backtest data source (v2 = Polygon, ADR-004) |
| 2 | `data/feature_engine.py`, `models/tabular.py`, `models/cv.py`, `models/calibration.py`, `monitoring/regime_detection.py` | 30+ indicator formulas, LightGBM training patterns, calibration | **unshifted features** (v1 computes feature[t] from close[t] — lookahead by v2 rules), NaN→0.0 silent fills, input mutation |
| 3 | `models/risk.py`, `alpaca/risk_monitor.py`, `docs/RISK_PARAMETERS_RESEARCH.md` | parameter research, monitoring ideas | v1 has NO order gatekeeper (`validate_order` checks format only; `risk/` package is empty) — the v2 pipeline is not in v1 |
| 4 | `execution/alpaca_executor.py`, `reliability/__init__.py` | order-request building, API error taxonomy, call-level circuit breaker/retry patterns | three parallel executor implementations — pick patterns, not structure |
| 6 | `backtest/engine.py`, `backtest/metrics.py`, `backtest/monte_carlo.py`, `models/walkforward.py`, `docs/WALK_FORWARD_*.md` | metric formulas, Monte Carlo robustness checks, WFO window logic | any historical performance claims (produced with unshifted features) |
| 7 | `monitoring/trade_logger.py`, `paper_trading_monitor.py`, `drift_monitor.py` | drift detection, trade logging schema | — |

Global v1 exclusions: `yfinance` in the production signal path
(`scheduler/signal_generator.py`), committed `.env`, vendored `ressources/`.

### Port protocol — MANDATORY for any v1-derived code

Nothing from v1 enters this codebase without passing all seven steps. "It
worked in v1" is not evidence — v1's backtests ran on unshifted features.

1. **Re-derive, don't copy.** Read the v1 implementation, then write the v2
   version from the formula/definition. Compare afterwards. Copy-paste
   imports v1's bugs invisibly.
2. **Cross-check the math twice**: (a) a hand-computed fixture in the test
   (≤ 10 rows, expected values derived manually in a comment), and
   (b) where a reference implementation exists, assert agreement on the same
   input within tolerance — the `ta` library for indicators,
   `quantstats` for performance metrics (dev-dependency only).
3. **Lookahead pass**: every ported feature column ends `.shift(1)` with a
   `# lookahead-safe:` comment, and the truncation-invariance test must
   cover the new columns (parametrize it over ALL feature columns, not a
   hardcoded list).
4. **No silent numerics**: v1 replaces NaN/Inf with 0.0 — forbidden here.
   Warmup NaN stays NaN; unexpected NaN/Inf raises `DataValidationError`.
5. **Purity**: no mutation of input DataFrames (v1's `compute_features`
   mutates its `bars` argument); add an input-unchanged assertion to the test.
6. **Params to YAML**: every window/threshold/constant extracted to
   `config/base.yaml` and registered in `_SCHEMA`.
7. **Provenance in the commit message**: which v1 file/lines inspired the
   port and what was changed (e.g. "ported RSI from v1 feature_engine.py:
   added shift(1), NaN policy, config window").

## Reference

- `ARCHITECTURE.md` — system diagram, ADR-001…007, control gates
- `CLAUDE.md` (root, `src/signals/`, `src/risk/`, `src/execution/`) — behavioral rules
- alpaca-py docs: https://alpaca.markets/sdks/python/ · Polygon aggregates: https://polygon.io/docs/rest/stocks/aggregates
- Reference repos: nautechsystems/nautilus_trader (event-driven parity), microsoft/qlib (WFO discipline), polakowo/vectorbt (vectorized research)
