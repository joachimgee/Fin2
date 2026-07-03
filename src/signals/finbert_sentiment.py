"""FinBERT sentiment -> numeric feature. Local, free, no API key.

Same contract as SentimentFeature (llm_sentiment.py): async get(headline,
ticker) -> float in [-1, 1], SHA256 Redis cache, neutral 0.0 on any failure.
The consumer cannot tell the providers apart — swapping is a config change
(llm.provider), nothing else moves.

The classifier is INJECTED (a callable returning [{"label", "score"}]), so
tests run with fakes and torch/transformers stay an optional extra:
    pip install -e ".[finbert]"
Inference is CPU-bound and runs in a worker thread — the event loop and the
stream callbacks never block.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

from redis.exceptions import RedisError

from src.shared.exceptions import ConfigError
from src.signals.llm_sentiment import cache_key

log = logging.getLogger(__name__)

Classifier = Callable[[str], list[dict[str, Any]]]

_LABEL_SIGNS = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}


def load_finbert(model_name: str) -> Classifier:
    """Build the real HuggingFace pipeline. Composition-root only.

    Fail-fast with an actionable message when the optional extra is missing —
    the trading process must not discover this on its first headline.
    """
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise ConfigError(
            "llm.provider is 'finbert' but transformers is not installed — "
            "run: pip install -e '.[finbert]'"
        ) from exc
    classifier: Classifier = pipeline("text-classification", model=model_name)
    log.info("finbert_loaded", extra={"model": model_name})
    return classifier


class FinBertSentiment:
    def __init__(self, classifier: Classifier, redis_client: Any, config: dict[str, Any]) -> None:
        self._classifier = classifier
        self._redis = redis_client
        self._cache_ttl_s = int(config["llm"]["cache_ttl_s"])

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
            value = await asyncio.to_thread(self._score, headline)
        except (RuntimeError, ValueError, KeyError, IndexError, TypeError) as exc:
            log.warning("sentiment_feature_failed", extra={"ticker": ticker, "error": str(exc)})
            return 0.0  # neutral — a model failure never breaks the signal path

        try:
            await self._redis.set(key, str(value), ex=self._cache_ttl_s)
        except RedisError as exc:
            log.warning("sentiment_cache_write_failed", extra={"error": str(exc)})
        return value

    def _score(self, headline: str) -> float:
        result = self._classifier(headline)[0]
        label = str(result["label"]).lower()
        sign = _LABEL_SIGNS.get(label)
        if sign is None:
            raise ValueError(f"unknown FinBERT label: {label!r}")
        return max(-1.0, min(1.0, sign * float(result["score"])))
