"""Sentiment provider selection — one config key (llm.provider) decides.

Providers share the contract async get(headline, ticker) -> float in [-1, 1];
downstream code (feature pipeline) never knows which one is wired. Secrets
stay at the composition root: the anthropic client arrives pre-built.
"""

from __future__ import annotations

from typing import Any, Protocol

from src.shared.exceptions import ConfigError
from src.signals.finbert_sentiment import FinBertSentiment, load_finbert
from src.signals.llm_sentiment import SentimentFeature


class SentimentProvider(Protocol):
    async def get(self, headline: str, ticker: str) -> float: ...


def create_sentiment_provider(
    config: dict[str, Any], redis_client: Any, anthropic_client: Any | None = None
) -> SentimentProvider | None:
    provider = str(config["llm"]["provider"])
    if provider == "none":
        return None
    if provider == "finbert":
        classifier = load_finbert(str(config["llm"]["finbert_model"]))
        return FinBertSentiment(classifier, redis_client, config)
    if provider == "anthropic":
        if anthropic_client is None:
            raise ConfigError("llm.provider is 'anthropic' but no anthropic client was provided")
        return SentimentFeature(anthropic_client, redis_client, config)
    raise ConfigError(f"unknown llm.provider: {provider!r} (finbert | anthropic | none)")
