"""Phase 5 — LLM sentiment feature tests (docs/plan/PHASE_5_LLM_FEATURES.md)."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from typing import Any

import anthropic
import pytest
from src.signals.llm_sentiment import SentimentFeature, cache_key

_GOOD_JSON = json.dumps(
    {"sentiment": 1, "confidence": 0.7, "catalyst_type": "earnings", "time_horizon": "swing"}
)


class _BoomAPIError(anthropic.APIError):
    def __init__(self) -> None:  # bypass APIError's request/body plumbing
        Exception.__init__(self, "api down")


class FakeAnthropic:
    """messages.create recorder with scripted text or error."""

    def __init__(self, text: str = _GOOD_JSON, error: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._text = text
        self._error = error
        self.messages = self  # so client.messages.create resolves to self.create

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return SimpleNamespace(content=[SimpleNamespace(text=self._text)])


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
    base_config: dict[str, Any], text: str = _GOOD_JSON, error: Exception | None = None
) -> tuple[SentimentFeature, FakeAnthropic, FakeRedis]:
    api, redis = FakeAnthropic(text, error), FakeRedis()
    return SentimentFeature(api, redis, base_config), api, redis


def test_cache_key_is_sha256_of_ticker_and_headline() -> None:
    key = cache_key("AAPL", "Apple beats earnings")
    digest = hashlib.sha256(b"AAPL:Apple beats earnings").hexdigest()
    assert key == f"sent:{digest}"


def test_different_tickers_different_keys() -> None:
    assert cache_key("AAPL", "same headline") != cache_key("MSFT", "same headline")


async def test_output_is_sentiment_times_confidence(base_config: dict[str, Any]) -> None:
    feature, _, _ = _feature(base_config)
    assert await feature.get("Apple beats earnings", "AAPL") == pytest.approx(0.7)


async def test_cache_hit_skips_api(base_config: dict[str, Any]) -> None:
    feature, api, redis = _feature(base_config)
    redis.store[cache_key("AAPL", "old news")] = "0.42"
    assert await feature.get("old news", "AAPL") == pytest.approx(0.42)
    assert api.calls == []  # the same headline is never sent twice


async def test_second_call_uses_cache(base_config: dict[str, Any]) -> None:
    feature, api, _ = _feature(base_config)
    first = await feature.get("fresh news", "AAPL")
    second = await feature.get("fresh news", "AAPL")
    assert first == second
    assert len(api.calls) == 1


async def test_response_cached_with_configured_ttl(base_config: dict[str, Any]) -> None:
    feature, _, redis = _feature(base_config)
    await feature.get("fresh news", "AAPL")
    key = cache_key("AAPL", "fresh news")
    assert float(redis.store[key]) == pytest.approx(0.7)
    assert redis.ttls[key] == base_config["llm"]["cache_ttl_s"]


async def test_api_error_returns_neutral_and_not_cached(base_config: dict[str, Any]) -> None:
    feature, _, redis = _feature(base_config, error=_BoomAPIError())
    assert await feature.get("headline", "AAPL") == 0.0
    assert redis.store == {}  # a failure is never cached as a real value


async def test_invalid_json_returns_neutral(base_config: dict[str, Any]) -> None:
    feature, _, _ = _feature(base_config, text="the market feels bullish today")
    assert await feature.get("headline", "AAPL") == 0.0


async def test_missing_field_returns_neutral(base_config: dict[str, Any]) -> None:
    feature, _, _ = _feature(base_config, text=json.dumps({"sentiment": 1}))
    assert await feature.get("headline", "AAPL") == 0.0


async def test_output_clamped(base_config: dict[str, Any]) -> None:
    malformed = json.dumps({"sentiment": 5, "confidence": 1.0})  # out-of-schema model reply
    feature, _, _ = _feature(base_config, text=malformed)
    assert await feature.get("headline", "AAPL") == 1.0  # clamped, not 5.0


async def test_uses_model_and_max_tokens_from_config(base_config: dict[str, Any]) -> None:
    feature, api, _ = _feature(base_config)
    await feature.get("headline", "AAPL")
    assert api.calls[0]["model"] == base_config["llm"]["model"]
    assert api.calls[0]["max_tokens"] == base_config["llm"]["max_tokens"]
