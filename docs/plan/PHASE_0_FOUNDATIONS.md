# Phase 0 — Foundations (shared/ config + test fixtures)

> Context to load: this file + `src/shared/config.py` + `src/shared/exceptions.py`
> + `config/base.yaml` + `tests/conftest.py`. Nothing else.

## Objective

Make configuration loading real and fail-fast, and give every later phase its
shared test fixtures. After this phase, `load_config()` is the single door
through which all numeric parameters enter the system.

## Prerequisites

None (first phase). Scaffold commit exists; `make install` has been run.

## Allowed imports

`shared/` imports nothing from `src/` (stdlib + pyyaml only).

## Files

### 1. `src/shared/config.py` — implement `load_config(path)`

Algorithm:
1. `yaml.safe_load` the file; raise `ConfigError` if unreadable or not a mapping.
2. Validate against an explicit module-level schema
   `_SCHEMA: dict[str, set[str]]` = section → allowed keys, mirroring
   `config/base.yaml` exactly. Unknown section or key → `ConfigError`
   naming it (a typo'd risk cap must fail loudly, never be silently ignored).
3. Bounds (values still come from YAML — these are ceilings, not defaults):
   - `risk.kelly_fraction` > 0.50 → `ConfigError`
   - `risk.max_risk_per_trade_pct` > 0.05 → `ConfigError`
   - `execution.live_mode` must be `bool`; absent → default `False`.
4. Return the dict unchanged otherwise. No global state, no caching.

`require_env()` is already implemented — do not touch it.

### 2. `tests/conftest.py` — fixtures

- `sample_bars` → `pd.DataFrame` of 300 valid OHLCV rows, seeded
  (`np.random.default_rng(42)`), tz-aware UTC timestamps, `low ≤ open/close ≤ high`.
- `base_config` → `dict` loaded from `config/base.yaml` via `load_config`.
- `tmp_db_path` → `tmp_path / "test.duckdb"` (Path only; storage comes in Phase 1).

## Tests — `tests/test_config.py`

| Test | Asserts |
|---|---|
| `test_require_env_missing_raises` | unset var → `ConfigError` |
| `test_require_env_returns_value` | set var → value |
| `test_load_base_yaml_valid` | real `config/base.yaml` loads clean |
| `test_unknown_key_raises` | extra key `risk.max_risk_per_trad_pct` (typo) → `ConfigError` naming it |
| `test_kelly_above_half_raises` | `kelly_fraction: 0.6` → `ConfigError` |
| `test_risk_per_trade_above_5pct_raises` | `0.06` → `ConfigError` |
| `test_live_mode_defaults_false` | section without `live_mode` → `False` |
| `test_live_mode_non_bool_raises` | `live_mode: "yes"` → `ConfigError` |

## GATE 0

```bash
make check          # ruff + mypy + all tests incl. test_architecture.py
```

## Definition of done

- [ ] `load_config` implemented, no `NotImplementedError` left in `shared/`
- [ ] All 8 tests above pass; `make check` green
- [ ] Progress row ticked in CODING_PLAN.md; committed `Phase 0: config loader + fixtures`

## Pitfalls

- Do not add default values for numeric params "to be helpful" — a missing
  risk cap must crash, not default.
- Do not validate credentials here; `require_env` is called at composition
  root (Phase 8), not at config load.
