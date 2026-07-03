"""Claude API sentiment -> numeric feature. NEVER LLM -> order.

Pattern (root CLAUDE.md <trading_rules>): the LLM produces a number that
becomes ONE feature in a validated ML model. It never decides a trade.

Rules (src/signals/CLAUDE.md <llm_features>):
  - Cache every response in Redis under SHA256(ticker:headline) — the same
    headline is never sent to the API twice.
  - On any API/parse error: return 0.0 (neutral) — an LLM outage degrades the
    signal path, it never breaks it. A Redis outage degrades to cache-miss.
  - Output is confidence-weighted: sentiment * confidence, clamped [-1.0, 1.0].

Clients are injected (constructed at the composition root from
require_env("ANTHROPIC_API_KEY") / REDIS_URL) — this module holds no secrets.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

import anthropic
from redis.exceptions import RedisError

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a financial news analyzer. Return ONLY valid JSON, no other text.\n"
    'Schema: {"sentiment": -1 | 0 | 1, "confidence": <float 0.0-1.0>,\n'
    '         "catalyst_type": "earnings|macro|sector|m&a|regulatory|other",\n'
    '         "time_horizon": "intraday|swing|long_term"}'
)


def cache_key(ticker: str, headline: str) -> str:
    digest = hashlib.sha256(f"{ticker}:{headline}".encode()).hexdigest()
    return f"sent:{digest}"


class SentimentFeature:
    def __init__(self, anthropic_client: Any, redis_client: Any, config: dict[str, Any]) -> None:
        llm_cfg = config["llm"]
        self._model = str(llm_cfg["model"])
        self._max_tokens = int(llm_cfg["max_tokens"])
        self._cache_ttl_s = int(llm_cfg["cache_ttl_s"])
        self._anthropic = anthropic_client
        self._redis = redis_client

    async def get(self, headline: str, ticker: str) -> float:
        key = cache_key(ticker, headline)
        try:
            cached = await self._redis.get(key)
        except RedisError as exc:
            log.warning("sentiment_cache_read_failed", extra={"error": str(exc)})
            cached = None
        if cached is not None:
            return float(cached)

        try:
            value = await self._score(headline, ticker)
        except (anthropic.APIError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            log.warning("sentiment_feature_failed", extra={"ticker": ticker, "error": str(exc)})
            return 0.0  # neutral — never propagate into the signal path

        try:
            await self._redis.set(key, str(value), ex=self._cache_ttl_s)
        except RedisError as exc:
            log.warning("sentiment_cache_write_failed", extra={"error": str(exc)})
        return value

    async def _score(self, headline: str, ticker: str) -> float:
        response = await self._anthropic.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=_SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": f"Analyze this headline for {ticker}: {headline}"}
            ],
        )
        parsed = json.loads(response.content[0].text)
        value = float(parsed["sentiment"]) * float(parsed["confidence"])
        return max(-1.0, min(1.0, value))
