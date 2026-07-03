"""FinBERT sentiment provider tests — fake classifier, no torch, no network."""

from __future__ import annotations

from typing import Any

import pytest
from src.shared.exceptions import ConfigError
from src.signals.finbert_sentiment import FinBertSentiment
from src.signals.llm_sentiment import SentimentFeature, cache_key
from src.signals.sentiment_factory import create_sentiment_provider


class FakeClassifier:
    def __init__(self, label: str = "positive", score: float = 0.9) -> None:
        self.calls: list[str] = []
        self._label = label
        self._score = score
        self.error: Exception | None = None

    def __call__(self, text: str) -> list[dict[str, Any]]:
        self.calls.append(text)
        if self.error is not None:
            raise self.error
        return [{"label": self._label, "score": self._score}]


class FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.ttls: dict[str, int | None] = {}

    async def get(self, key: str) -> str | None:
        return self.store.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        self.store[key] = value
        self.ttls[key] = ex


def _feature(
    base_config: dict[str, Any], label: str = "positive", score: float = 0.9
) -> tuple[FinBertSentiment, FakeClassifier, FakeRedis]:
    classifier, redis = FakeClassifier(label, score), FakeRedis()
    return FinBertSentiment(classifier, redis, base_config), classifier, redis


async def test_positive_label_signed_score(base_config: dict[str, Any]) -> None:
    feature, _, _ = _feature(base_config, "positive", 0.9)
    assert await feature.get("Apple beats earnings", "AAPL") == pytest.approx(0.9)


async def test_negative_label_signed_score(base_config: dict[str, Any]) -> None:
    feature, _, _ = _feature(base_config, "negative", 0.8)
    assert await feature.get("Apple misses badly", "AAPL") == pytest.approx(-0.8)


async def test_neutral_label_is_zero(base_config: dict[str, Any]) -> None:
    feature, _, _ = _feature(base_config, "neutral", 0.99)
    assert await feature.get("Apple holds annual meeting", "AAPL") == 0.0


async def test_cache_hit_skips_classifier(base_config: dict[str, Any]) -> None:
    feature, classifier, redis = _feature(base_config)
    redis.store[cache_key("AAPL", "old news")] = "0.42"
    assert await feature.get("old news", "AAPL") == pytest.approx(0.42)
    assert classifier.calls == []  # same headline never scored twice


async def test_result_cached_with_configured_ttl(base_config: dict[str, Any]) -> None:
    feature, classifier, redis = _feature(base_config)
    first = await feature.get("fresh news", "AAPL")
    second = await feature.get("fresh news", "AAPL")
    assert first == second
    assert len(classifier.calls) == 1
    assert redis.ttls[cache_key("AAPL", "fresh news")] == base_config["llm"]["cache_ttl_s"]


async def test_classifier_failure_returns_neutral_not_cached(
    base_config: dict[str, Any],
) -> None:
    feature, classifier, redis = _feature(base_config)
    classifier.error = RuntimeError("CUDA out of memory")
    assert await feature.get("headline", "AAPL") == 0.0
    assert redis.store == {}  # a failure is never cached as a real value


async def test_unknown_label_returns_neutral(base_config: dict[str, Any]) -> None:
    feature, _, _ = _feature(base_config, label="bullish??")
    assert await feature.get("headline", "AAPL") == 0.0


# --- provider factory ----------------------------------------------------------


def test_factory_none_returns_none(base_config: dict[str, Any]) -> None:
    config = {**base_config, "llm": {**base_config["llm"], "provider": "none"}}
    assert create_sentiment_provider(config, FakeRedis()) is None


def test_factory_finbert_builds_finbert(
    base_config: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.signals.sentiment_factory as factory_mod

    monkeypatch.setattr(factory_mod, "load_finbert", lambda model: FakeClassifier())
    provider = create_sentiment_provider(base_config, FakeRedis())  # default: finbert
    assert isinstance(provider, FinBertSentiment)


def test_factory_anthropic_requires_client(base_config: dict[str, Any]) -> None:
    config = {**base_config, "llm": {**base_config["llm"], "provider": "anthropic"}}
    with pytest.raises(ConfigError, match="anthropic client"):
        create_sentiment_provider(config, FakeRedis())
    provider = create_sentiment_provider(config, FakeRedis(), anthropic_client=object())
    assert isinstance(provider, SentimentFeature)


def test_factory_unknown_provider_raises(base_config: dict[str, Any]) -> None:
    config = {**base_config, "llm": {**base_config["llm"], "provider": "vibes"}}
    with pytest.raises(ConfigError, match="vibes"):
        create_sentiment_provider(config, FakeRedis())
