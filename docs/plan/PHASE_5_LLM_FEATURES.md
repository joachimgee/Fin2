# Phase 5 — LLM sentiment feature (Claude API + Redis cache)

> Context to load: this file + `src/signals/llm_sentiment.py` (signals
> CLAUDE.md loads automatically). Prerequisite: Phase 4 GATE green.

## Objective

Claude API → cached numeric feature. The LLM never decides a trade (ADR-005);
its output is one column consumed by the validated LightGBM model.

## Allowed imports

Still `signals/` → `data` + `shared` (+ anthropic, redis, hashlib). No new edges.

## Files

### 1. `src/signals/llm_sentiment.py` — implement `get_sentiment_feature`

Refactor to a small class so clients are injected (testability, DIP):

```
class SentimentFeature:
    def __init__(self, anthropic_client, redis_client, config)  # llm section
    async def get(self, headline: str, ticker: str) -> float
```

`get()` algorithm:
1. `key = "sent:" + sha256(f"{ticker}:{headline}".encode()).hexdigest()`.
2. Redis GET — hit → return cached float. The same headline is NEVER sent
   to the API twice.
3. Call `messages.create` with `model=config["llm"]["model"]`, `max_tokens≈200`,
   system prompt demanding ONLY strict JSON:
   `{"sentiment": -1|0|1, "confidence": 0.0-1.0, "catalyst_type": ..., "time_horizon": ...}`.
4. Parse; `value = float(sentiment) * float(confidence)`; clamp to [-1, 1].
5. Redis SET with `ex=config["llm"]["cache_ttl_s"]`; return value.
6. `except (anthropic.APIError, json.JSONDecodeError, KeyError, ValueError)`:
   `log.warning("sentiment_feature_failed", extra={ticker, error})`,
   **return 0.0** (neutral). An LLM outage must never take down the signal
   path — degraded, not broken. Never a bare `except`.

## Tests — `tests/test_llm_sentiment.py` (mock anthropic client + fakeredis)

| Test | Asserts |
|---|---|
| `test_cache_hit_skips_api` | second call with same headline → 0 API calls |
| `test_cache_key_is_sha256_of_ticker_and_headline` | exact key format |
| `test_response_cached_with_ttl` | SET called with configured `ex` |
| `test_api_error_returns_neutral_zero` | APIError → 0.0, no raise |
| `test_invalid_json_returns_neutral_zero` | garbage response → 0.0 |
| `test_output_is_sentiment_times_confidence` | (1, 0.7) → 0.7 |
| `test_output_clamped` | malformed (2, 1.0) → ≤ 1.0 |
| `test_different_tickers_different_keys` | same headline, 2 tickers → 2 entries |

## GATE 5

```bash
make check
```

## Definition of done

- [ ] No `NotImplementedError` left in `src/signals/`
- [ ] 8 tests pass; `make check` green
- [ ] Progress ticked; committed `Phase 5: LLM sentiment feature`

## Pitfalls

- The neutral fallback is 0.0 — not the last cached value of a DIFFERENT
  headline, not a retry loop inside the signal path.
- Do not "upgrade" this into a trading signal. Its consumer is
  `compute_features`/the model, never a strategy directly.
- API key via `require_env("ANTHROPIC_API_KEY")` at composition root; this
  module receives a constructed client.
