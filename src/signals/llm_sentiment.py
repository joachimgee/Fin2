"""Claude API sentiment -> numeric feature. NEVER LLM -> order.

Pattern (root CLAUDE.md <trading_rules>): the LLM produces a number that
becomes ONE feature in a validated ML model. It never decides a trade.

Rules (src/signals/CLAUDE.md <llm_features>):
  - Cache every response in Redis under SHA256(headline) — the same headline
    is never sent to the API twice.
  - On any API error: return 0.0 (neutral). Never propagate the exception
    into the signal path.
  - Output is confidence-weighted: score * confidence, in [-1.0, 1.0].
"""

from __future__ import annotations


async def get_sentiment_feature(headline: str, ticker: str) -> float:
    """TODO(Phase 5):
    1. key = sha256(f"{ticker}:{headline}") ; hit Redis first, return cached.
    2. Call Claude API with a strict-JSON system prompt
       {"sentiment": -1|0|1, "confidence": 0..1, ...}.
    3. value = sentiment * confidence ; cache it ; return it.
    4. except (APIError, ValidationError): log WARNING, return 0.0.
    """
    raise NotImplementedError("Phase 5 — LLM sentiment feature")
