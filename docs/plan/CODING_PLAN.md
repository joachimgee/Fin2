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
| 0 | PHASE_0_FOUNDATIONS.md | config loader, test fixtures | `make check` | ☐ |
| 1 | PHASE_1_DATA.md | Bar validation, DuckDB storage, Polygon sync | `make check` | ☐ |
| 2 | PHASE_2_SIGNALS.md | features (lookahead-safe), LightGBM, HMM, training scripts | `make check` + anti-lookahead test | ☐ |
| 3 | PHASE_3_RISK.md | Kelly, circuit breakers, ExposureTracker, validate() | `make check` | ☐ |
| 4 | PHASE_4_EXECUTION.md | AlpacaBrokerClient, StreamManager | `make check` | ☐ |
| 5 | PHASE_5_LLM_FEATURES.md | Claude sentiment feature + Redis cache | `make check` | ☐ |
| 6 | PHASE_6_BACKTEST_WFO.md | SimulatedBroker, engine, WFO gates | `make check` | ☐ |
| 7 | PHASE_7_MONITORING.md | JSON logging, Telegram alerts | `make check` | ☐ |
| 8 | PHASE_8_ENTRYPOINTS.md | composition root: sync/backtest/paper CLIs | `make check` + dry-run | ☐ |

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

## Reference

- `ARCHITECTURE.md` — system diagram, ADR-001…007, control gates
- `CLAUDE.md` (root, `src/signals/`, `src/risk/`, `src/execution/`) — behavioral rules
- alpaca-py docs: https://alpaca.markets/sdks/python/ · Polygon aggregates: https://polygon.io/docs/rest/stocks/aggregates
- Reference repos: nautechsystems/nautilus_trader (event-driven parity), microsoft/qlib (WFO discipline), polakowo/vectorbt (vectorized research)
