# Phase 2 — Signals (features, LightGBM, HMM, training scripts)

> Context to load: this file + `src/signals/*.py` (its CLAUDE.md loads
> automatically) + `src/data/models.py`. Prerequisite: Phase 1 GATE green.

## Objective

Lookahead-safe features, artifact-loading signal generators, and the offline
training scripts that produce those artifacts. The anti-lookahead test written
here guards the single most dangerous silent bug in the project.

## Allowed imports

`signals/` → `data` + `shared` (+ pandas, numpy, sklearn, lightgbm, hmmlearn,
joblib). NEVER strategies/, NEVER execution/.

## Files

### 1. `config/base.yaml` — add a `signals:` section

```yaml
signals:
  features:
    sma_window: 20
    rsi_window: 14
    vol_window: 20
```
Register the new section in `_SCHEMA` (`src/shared/config.py`). Windows are
numeric params → YAML, never literals in `features.py`.

### 2. `src/signals/features.py` — implement `compute_features(df, config)`

- Signature becomes `compute_features(df: pd.DataFrame, config: dict) -> pd.DataFrame`.
- Features (every line ends `.shift(1)` + `# lookahead-safe:` comment):
  `log_ret_1`, `sma_ratio` (close / rolling mean), `rsi`, `vol`
  (rolling std of log returns), `zscore` ((close − sma) / rolling std).
- Stationary transforms only — never raw prices as feature values.
- Keep warmup rows as NaN (callers check `is_ready`); same index as input;
  input DataFrame never mutated. Column order is deterministic and documented:
  it becomes `features.json`.

### 3. `src/signals/lgbm_signal.py` — implement `LightGBMSignalGenerator`

- `__init__`: load `model.pkl`, `scaler.pkl`, `features.json`; any missing →
  `ConfigError` naming the path (fail-fast at startup, not on first bar).
- `generate(features)`: take the LAST row; verify column set equals
  `features.json` exactly (extra/missing → `ConfigError`); reorder to that
  order; `scaler.transform` (never fit); `p = predict_proba[:, 1]`; map
  `signal = 2p − 1`; `return self._clamp(signal)`. No I/O, deterministic.

### 4. `src/signals/regime_hmm.py` — implement `RegimeDetector`

- `__init__`: same artifact loading + fail-fast rules.
- `current_regime(feature_df)`: `X = scaler.transform(window)` on the FULL
  window, `states = model.predict(X)` (full sequence — Viterbi),
  `return int(states[-1])`. Never predict on a single reshaped row.

### 5. `scripts/train_lgbm.py` and `scripts/train_hmm.py` (offline, new files)

- Read bars from DuckDB (`BarStorage`), compute features, temporal split
  (train/valid/test — `TimeSeriesSplit` semantics, never KFold/shuffle).
- Scaler `fit` on train ONLY; `transform` elsewhere.
- HMM: 3 states on scaled `[log_ret, vol]` — unscaled input collapses to one
  state silently (signals/CLAUDE.md <hmm_critical>).
- Dump artifact dir `models/{name}_{ts}/`: `model.pkl`, `scaler.pkl`,
  `features.json`, `metadata.yaml` (training_period, oos_period, metrics,
  `data_source: polygon`).
- `scripts/` is composition-side: it may import data+signals+shared, still
  never execution/.

## Tests

`tests/test_features.py`:
- `test_no_lookahead_truncation_invariance` — **the critical test**: compute on
  full df, then on `df.iloc[:t]` for several t; feature rows at t−1 must be
  identical. Any feature reading the future fails this.
- `test_warmup_rows_nan`, `test_column_order_stable`, `test_input_not_mutated`,
  `test_windows_come_from_config` (change config → output changes).

`tests/test_lgbm_signal.py` (build tiny real artifacts in a tmp fixture):
`test_missing_artifact_fails_fast`, `test_generate_in_minus1_plus1`,
`test_generate_deterministic`, `test_wrong_feature_set_raises`,
`test_column_order_enforced_not_assumed`.

`tests/test_regime_hmm.py`: `test_missing_artifact_fails_fast`,
`test_full_sequence_passed_to_predict` (mock model records received shape),
`test_scaler_transform_called_never_fit`.

## GATE 2

```bash
make check
pytest tests/test_features.py::test_no_lookahead_truncation_invariance -v  # must be seen passing
```

## Definition of done

- [ ] No `NotImplementedError` left in `src/signals/` except `llm_sentiment.py` (Phase 5)
- [ ] Anti-lookahead test passes; `make check` green
- [ ] Progress ticked; committed `Phase 2: signals + training scripts`

## Pitfalls

- `.rolling(n).mean().shift(1)` ≠ `.shift(1).rolling(n).mean()` — both are
  lookahead-safe but produce different columns; pick the first form and stay
  consistent between training and live.
- `scaler.fit_transform` anywhere except the train split is an automatic
  review rejection (root CLAUDE.md warn-list #5).
- Do not persist the scaler and model from different runs — one training run
  writes one artifact dir atomically.
