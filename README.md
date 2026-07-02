# finbot

Algorithmic trading system — backtest and live share one codebase.
Broker: Alpaca Markets (**paper by default**). Backtest data: Polygon.io via DuckDB.

## Status

Architecture scaffold. All module contracts (interfaces, dataclasses, pipelines)
are defined with signatures and invariants; implementations follow the
phase-by-phase guide in [docs/plan/CODING_PLAN.md](docs/plan/CODING_PLAN.md)
(one phase per session, gated). The dependency graph is already enforced by
`tests/test_architecture.py`.

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
make install                 # pip install -e ".[dev]" + pre-commit hooks
cp .env.example .env         # fill in keys (paper keys!)
make check                   # lint + typecheck + tests
```

## Layout

```
CLAUDE.md              ← rules loaded by Claude Code every session
ARCHITECTURE.md        ← why the structure is what it is (ADRs, gates, phases)
config/base.yaml       ← ALL numeric parameters (nothing numeric in src/)
src/
  shared/     contracts: AbstractBrokerClient, SignalGenerator, config, exceptions
  data/       Bar model, validation, DuckDB storage, Polygon sync
  signals/    features (.shift(1)), LightGBM, HMM regime, LLM sentiment   [CLAUDE.md]
  strategies/ AbstractStrategy + OrderIntent, concrete strategies
  risk/       validate() pipeline, Kelly, circuit breakers, exposure      [CLAUDE.md]
  execution/  Alpaca client + StreamManager — sole alpaca-py importer     [CLAUDE.md]
  backtest/   simulated broker, event-driven engine, WFO gates
  monitoring/ JSON logging, Telegram alerts
tests/
  test_architecture.py ← import-graph enforcement (pre-commit + CI)
```

## Non-negotiables

- No order reaches the broker without `RiskManager.validate()` first.
- `paper=False` comes exclusively from `config/base.yaml` (`execution.live_mode`).
- Every feature is `.shift(1)` lookahead-safe, with a `# lookahead-safe:` comment.
- WFO gates (≥5 windows, WFE ≥ 0.50, ≥200 OOS trades) before any paper trading.
- LLM output is a feature into a validated model — never a trading decision.
