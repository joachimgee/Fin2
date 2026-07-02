# CLAUDE.md — src/signals/

<!-- Chargé automatiquement par Claude Code quand tu travailles dans src/signals/ -->

<role>
Produces numeric signals consumed by strategies. Output contract: float in [-1.0, 1.0].
Knows nothing about brokers, orders, or risk limits.
</role>

<invariants>
1. generate() always returns float in [-1.0, 1.0]. Always call self._clamp() before return.
2. Every feature at time T uses only data from times ≤ T-1. Always .shift(1).
3. generate() is deterministic and < 5ms. No network calls, no DB reads.
</invariants>

<lookahead_rule>
The most dangerous silent bug in this module.
A feature using current bar's data to predict current bar's return looks great in backtest,
fails completely in live.

Correct:
  df["sma_20"] = df["close"].rolling(20).mean().shift(1)  # lookahead-safe: uses bars [t-21..t-2]
  df["rsi_14"] = compute_rsi(df["close"], 14).shift(1)    # lookahead-safe: same

Wrong:
  df["sma_20"] = df["close"].rolling(20).mean()           # uses close[t] — lookahead bias

Add comment on every feature: # lookahead-safe: [reason]
</lookahead_rule>

<stationarity>
Raw prices are non-stationary. Never use as direct model features.
Use: log_returns, z-scores, momentum ratios, ATR, RSI.
A model trained on 2015-2020 prices will behave unpredictably on 2024 prices if trained on raw prices.
</stationarity>

<scaler_rule>
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit here and only here
X_val_scaled   = scaler.transform(X_val)          # transform only
X_test_scaled  = scaler.transform(X_test)         # transform only
joblib.dump(scaler, f"{artifact_path}/scaler.pkl") # always persist alongside model
</scaler_rule>

<hmm_critical>
HMM features MUST be normalized with StandardScaler before model.fit().
Without normalization: vol (≈0.015) dominates log_return (≈0.001) → HMM collapses to one state.
This failure is silent — the model appears to train but predicts nothing useful.

Predict using Viterbi on full sequence, not single last observation:
  states = model.predict(X_scaled)   # correct — full sequence
  state  = states[-1]                # current regime
  # NOT: model.predict(X_scaled[-1:].reshape(1,-1))  ← ignores temporal dependencies
</hmm_critical>

<llm_features>
Claude API sentiment → numeric feature → LightGBM. Never LLM → order.
Cache every response in Redis (SHA256 key). Same headline must never be sent twice.
On API error: return 0.0 (neutral), never propagate exception.
Confidence-weighted output: return score * confidence.
</llm_features>

<artifact_structure>
models/{strategy_name}_{timestamp}/
  model.pkl       ← lgb.LGBMClassifier
  scaler.pkl      ← StandardScaler (fitted on train only)
  features.json   ← list[str] in exact training order
  metadata.yaml   ← training_period, oos_period, metrics, data_source

Load all three files in __init__(). Fail-fast at startup if any is missing.
</artifact_structure>

<warn_about>
- Feature missing .shift(1) without # lookahead-safe comment
- Raw prices used as direct model features (not returns or z-scores)
- StandardScaler fit on validation or test data
- HMM features not normalized before model.fit()
- HMM prediction from single observation instead of full sequence
- LLM output routed to signal without validated ML model
- LLM responses not cached (same headline sent twice to API)
- generate() returns value without self._clamp()
</warn_about>
