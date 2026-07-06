# ARCHITECTURE.md — finbot

Reference document for cross-module decisions. The behavioral rules live in
`CLAUDE.md` (root + per-module); this file explains **why** the structure is
what it is, and records the decisions (ADRs) that make the dependency graph
satisfiable.

## 1. System overview

Two execution modes, one codebase. The strategy and risk code that ran in the
backtest is byte-for-byte the code that runs live — parity is structural.

```
        OFFLINE (periodic)                      RUNTIME (continuous)
┌─────────────────────────────┐    ┌──────────────────────────────────────────┐
│ Polygon.io ──► data/ sync   │    │ Alpaca WS (1 conn) ──► StreamManager     │
│        │ validate_bar()     │    │        │ (execution/)                    │
│        ▼                    │    │        ▼                                 │
│    DuckDB (bars)            │    │   Redis pub/sub  channel:bars:{sym}      │
│        │                    │    │        │                                 │
│        ▼                    │    │        ▼                                 │
│ signals/ training           │    │ strategy.on_bar(bar)      (strategies/)  │
│  features (.shift(1))       │    │        │ signal via injected generator   │
│  LightGBM + HMM + scaler    │    │        ▼                                 │
│        │ artifacts          │    │ OrderIntent ──► RiskManager.validate()   │
│        ▼                    │    │        │ <1ms, in-memory only  (risk/)   │
│ backtest/ WFO (5 windows)   │    │        ▼ adjusted_qty                    │
│  gates: WFE≥.5, ≥200 OOS    │    │ AlpacaBrokerClient.submit_order()        │
│        │                    │    │        │ (execution/, paper=True)        │
│        ▼                    │    │        ▼                                 │
│ cleared_for_paper: true ────┼───►│ fills via TradingStream ──► on_fill()    │
└─────────────────────────────┘    │  ExposureTracker + CircuitBreaker        │
                                   │  monitoring/: JSON logs, Telegram        │
                                   └──────────────────────────────────────────┘
```

## 2. Dependency graph (normative)

Defined in `CLAUDE.md <dependency_graph>`, enforced by
`tests/test_architecture.py` (runs in CI and as a pre-commit hook on every
`src/**.py` change). The test parses imports with `ast` — violations fail the
build, so the graph cannot rot.

## 3. Contract placement — where shared types live and why

The graph forbids several "natural" imports; contracts are placed so the graph
still holds:

| Contract | Lives in | Consumed by | Why there |
|---|---|---|---|
| `AbstractBrokerClient` | `shared/interfaces.py` | execution (Alpaca impl), backtest (sim impl), engines | backtest/ may not import execution/ — the abstraction must sit below both |
| `SignalGenerator` (Protocol) | `shared/interfaces.py` | strategies (injection target), signals (implements structurally) | strategies/ may not import signals/ — DI via structural typing |
| `Bar` | `data/models.py` | strategies, signals, backtest | canonical market-data model; SDK types never leak past their module |
| `OrderIntent` | `strategies/base.py` | risk, execution (via risk re-export) | produced by strategies; this is the one sanctioned risk→strategies/base edge |
| `ValidationResult` | `risk/manager.py` | execution | produced by risk; execution→risk is an allowed edge |

## 4. Architecture Decision Records

### ADR-001 — Broker abstraction lives in shared/, not execution/
The SOLID-D rule ("depend on AbstractBrokerClient, never AlpacaBrokerClient")
is unsatisfiable if the ABC lives in `execution/`: backtest/ and strategies/
are forbidden from importing execution/. Moving the ABC to
`shared/interfaces.py` lets execution/ and backtest/ each implement it while
nobody outside execution/ ever sees alpaca-py.

### ADR-002 — Strategies receive signal generators by injection
`strategies → signals` is forbidden (a strategy must be testable with a stub
generator, and signals/ must stay reusable across strategies). Strategies type
their dependency against the `SignalGenerator` Protocol in shared/; the
concrete generator (LightGBM, HMM-gated, …) is constructed at composition
root and injected. Same mechanism swaps a constant stub in unit tests.

### ADR-003 — OrderIntent ≠ Order
Strategies emit *intents*; only `RiskManager.validate()` turns an intent into
something executable, possibly with a reduced quantity (`adjusted_qty`).
Execution uses `result.adjusted_qty`, never `intent.qty`. This makes
"no order without validate()" a type-level habit, not a convention.

### ADR-004 — Backtest data: Polygon.io only, via DuckDB
yfinance excludes delisted stocks (Enron, Lehman) → survivorship bias makes
strategies look better than they were. Polygon is synced once into DuckDB;
backtests never hit an API. Every backtest config documents
`data_source: polygon`.

### ADR-005 — LLM outputs are features, never signals
Claude API → numeric score → one feature among many in LightGBM → signal.
The model consuming the feature passed WFO on OOS data; a raw LLM opinion did
not. Responses are Redis-cached by SHA256(headline); API errors degrade to
0.0 (neutral), never into the signal path.

### ADR-006 — One WebSocket, fan-out over Redis
Alpaca allows one WS connection per endpoint per account; a second silently
kills the first. `StreamManager` is the sole consumer and republishes on
Redis pub/sub so N strategies subscribe without touching Alpaca.

### ADR-007 — paper=True default, overridden only by YAML
`execution.live_mode: true` in config is the single path to real orders.
Never inferred from env/branch names, never a CLI flag: one accidental
`paper=False` submits real orders.

### ADR-008 — Long history for fixed universes: Alpaca Market Data (IEX)
Polygon's free tier caps history at ~2 years — too short for walk-forward
validation (paid upgrade is not an option). Alpaca Market Data's free plan
serves IEX-feed daily bars back to ~2016 with `adjustment=all`
(split+dividend adjusted). Sanctioned **only for fixed research universes**
(`scripts.sync_data --source alpaca`): IEX volumes are a fraction of the
consolidated tape (a consistent bias, harmless for price-derived features but
wrong for absolute liquidity filters), and closes can differ marginally from
the official auction close. Full-market screening stays on Polygon (ADR-004).
Implemented as raw REST (`src/data/alpaca_data.py`, httpx) — the alpaca-py
SDK remains exclusive to execution/ per the dependency graph.

### ADR-009 — A point Sharpe is not a decision: bootstrap its interval
The WFO reports ONE OOS Sharpe per strategy, but "1.17 < 1.50" is only a
verdict if 1.50 sits outside the estimate's sampling band. `backtest/bootstrap.py`
resamples the strategy's realized daily returns and recomputes the Sharpe to
produce a confidence interval and P(Sharpe ≥ gate). Method: STATIONARY
bootstrap (Politis & Romano 1994), not the IID bootstrap Fin v1 used —
IID resampling destroys volatility clustering and serial dependence, which
falsely tightens the interval; geometric blocks preserve it. This does not
change the gate (still the point Sharpe ≥ 1.5) — it tells us whether a
near-miss is a real shortfall or noise, so "abandon the family" is a
measured call, not a coin flip on one number.

## 5. Control gates (when code lands)

| Gate | Where | Blocks |
|---|---|---|
| Dependency graph test | pre-commit + CI | any merge |
| ruff (incl. T20 print ban) + mypy strict | pre-commit + CI | any merge |
| Anti-lookahead feature tests (Phase 2) | CI | signals changes |
| WFO gates: ≥5 windows, WFE ≥ 0.50, ≥200 OOS trades, Sharpe ≥ 1.5, DD ≤ 20% | `backtest/wfo.py` → results YAML | paper trading start |
| `cleared_for_paper: true` check | paper-trade CLI startup | live/paper session |
| Circuit breakers (runtime) | `risk/circuit_breaker.py` | all new orders |

## 6. Build order (phases)

Each phase is testable in isolation before the next exists. Detailed
per-phase specs (files, tests, gates) live in `docs/plan/` — start from
`docs/plan/CODING_PLAN.md`:

0. `shared/` config loader + exceptions, conftest fixtures
1. `data/` — models, validation, DuckDB storage, Polygon sync
2. `signals/` — features (lookahead-safe), LightGBM artifact loading, HMM regime
3. `risk/` — Kelly, validate() pipeline, circuit breakers, ExposureTracker
4. `execution/` — AlpacaBrokerClient, StreamManager (paper)
5. `signals/llm_sentiment` — Claude API feature + Redis cache
6. `backtest/` — SimulatedBroker, engine, WFO
7. `monitoring/` — JSON logging, Telegram, daily report
8. CLI entrypoints (last — they only compose what already exists)
